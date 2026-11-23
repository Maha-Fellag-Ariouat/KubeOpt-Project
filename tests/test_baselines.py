"""Tests B3 : baselines RR/FF/FFD, équivalence avec le baselines.py du projet, métrique Bal."""
from types import SimpleNamespace

import numpy as np
import pytest

from kubeopt import (
    BASELINES, DataValidationError, Instance, InstanceType, Scenario, Workload, cluster_nodes,
    compute_scenario_metrics, evaluate, generate_cluster_instance, run_all_baselines,
    run_baseline, server_lower_bound, validate_solution,
)
from kubeopt.validator import overflowed_pods_that_fit

OVF = -1


# --------------------------------------------------------------------------- #
# Copie FIDÈLE du baselines.py du projet (référence de non-régression)
# --------------------------------------------------------------------------- #
def _ref_fits(L, K, j, dem):
    return np.all(L[j] + dem <= K[j] + 1e-9)


def ref_round_robin(inst):
    a = -np.ones(inst.n, dtype=int)
    L = np.zeros_like(inst.K)
    ptr = 0
    for i in range(inst.n):
        for t in range(inst.m):
            j = (ptr + t) % inst.m
            if _ref_fits(L, inst.K, j, inst.d[i]):
                a[i] = j; L[j] += inst.d[i]; ptr = (j + 1) % inst.m
                break
    return a


def ref_first_fit(inst, order=None):
    a = -np.ones(inst.n, dtype=int)
    L = np.zeros_like(inst.K)
    for i in (range(inst.n) if order is None else order):
        for j in range(inst.m):
            if _ref_fits(L, inst.K, j, inst.d[i]):
                a[i] = j; L[j] += inst.d[i]
                break
    return a


def ref_first_fit_decreasing(inst):
    size = (inst.d / inst.K[0]).max(1)
    return ref_first_fit(inst, order=np.argsort(-size, kind="stable"))


def as_project_instance(instance, scenario):
    """Vue « n, m, d, K » attendue par le code du projet, pour un scénario d'une instance option A."""
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    m = int(instance.n_max_total or max(x.n_pods for x in instance.scenarios))
    cap = instance.allocatable[0][act]
    return SimpleNamespace(n=s.n_pods, m=m, d=s.demand[:, act].copy(), K=np.tile(cap, (m, 1)))


# --------------------------------------------------------------------------- #
# Équivalence avec le code du projet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [False, True])
@pytest.mark.parametrize("seed", range(15))
def test_baselines_identical_to_project_code(seed, bw):
    inst = generate_cluster_instance(seed=seed, n_containers=40, m=15, server_vcpu=8, bandwidth_active=bw,
                                     cpu_median=1.0, heterogeneity=0.9, n_variable_states=2)
    nodes = cluster_nodes(inst)
    for s in inst.scenarios:
        ref = as_project_instance(inst, s)
        expected = {"round_robin": ref_round_robin(ref), "first_fit": ref_first_fit(ref),
                    "first_fit_decreasing": ref_first_fit_decreasing(ref)}
        for name in BASELINES:
            got = run_baseline(inst, name, nodes).solution.assignments[inst.scenarios.index(s)]
            assert got.tolist() == expected[name].tolist(), (name, s.name)


# --------------------------------------------------------------------------- #
# Exemples calculés à la main (serveurs 10 vCPU, RAM non contraignante)
# --------------------------------------------------------------------------- #
def _cpu_instance(cpus, m, **kw):
    t = InstanceType("srv", 1.0, 10, 1000, **kw)
    ws = tuple(Workload(f"p{i}", c, 0.1) for i, c in enumerate(cpus))
    sc = Scenario.from_base_requests("s", ws, range(len(cpus)), 0.0)
    return Instance(types=(t,), workloads=ws, scenarios=(sc,), n_max_total=m)


def test_ffd_beats_ff_on_classic_example():
    inst = _cpu_instance([4, 4, 6, 6], m=4)
    ff = run_baseline(inst, "first_fit").evaluation.by_scenario("s")
    ffd = run_baseline(inst, "first_fit_decreasing").evaluation.by_scenario("s")
    assert ff.active_nodes == 3                      # (4,4) (6) (6)
    assert ffd.active_nodes == 2                     # (6,4) (6,4)
    assert server_lower_bound(inst, cluster_nodes(inst), "s") == 2      # FFD atteint le plancher


def test_round_robin_spreads_over_all_servers():
    inst = _cpu_instance([4, 4, 6, 6], m=3)
    rr = run_baseline(inst, "round_robin")
    assert rr.solution.assignments[0].tolist() == [0, 1, 2, 0]          # le 4e (6) saute s1 (4+6=10 passe : ptr=0)
    assert rr.evaluation.by_scenario("s").active_nodes == 3


def test_round_robin_skips_full_servers():
    inst = _cpu_instance([8, 8, 3, 3], m=3)
    a = run_baseline(inst, "round_robin").solution.assignments[0]
    # 8->s0, 8->s1, 3->s2 (ptr -> 0) ; 4e conteneur : s0 (8+3 > 10) saute, s1 saute, s2 (3+3 <= 10)
    assert a.tolist() == [0, 1, 2, 2]


def test_round_robin_overflows_when_no_server_fits():
    inst = _cpu_instance([8, 8, 3], m=2)
    run = run_baseline(inst, "round_robin")
    assert run.solution.assignments[0].tolist() == [0, 1, OVF]          # 8+3 > 10 sur les deux serveurs
    assert validate_solution(inst, run.solution)


# --------------------------------------------------------------------------- #
# Garanties : validité, maximalité, plancher, déterminisme
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("bw", [False, True])
def test_all_baselines_valid_and_lower_bounded(seed, bw):
    inst = generate_cluster_instance(seed=seed, n_containers=60, m=30, server_vcpu=8, bandwidth_active=bw,
                                     cpu_median=1.2, n_variable_states=1)
    nodes = cluster_nodes(inst)
    for name, res in run_all_baselines(inst, nodes).items():
        assert validate_solution(inst, res.solution), name
        for k, s in enumerate(inst.scenarios):
            a = res.solution.assignments[k]
            m = res.evaluation.scenarios[k]
            assert overflowed_pods_that_fit(inst, nodes, s, a) == [], (name, s.name)
            if m.overflow_pods == 0:                                      # placement complet
                assert m.active_nodes >= server_lower_bound(inst, nodes, s), (name, s.name)
            assert m.active_nodes <= nodes.size


def test_baselines_are_deterministic():
    inst = generate_cluster_instance(seed=4, n_containers=50, m=20, server_vcpu=8, cpu_median=1.0)
    a = run_all_baselines(inst)
    b = run_all_baselines(inst)
    for name in BASELINES:
        assert all(np.array_equal(x, y) for x, y in zip(a[name].solution.assignments, b[name].solution.assignments))


def test_default_cluster_size_is_always_sufficient():
    inst = generate_cluster_instance(seed=2, n_containers=30)          # m = max |I_s| : un pod par serveur suffit
    assert cluster_nodes(inst).size == max(s.n_pods for s in inst.scenarios)
    for res in run_all_baselines(inst).values():
        assert all(m.overflow_pods == 0 for m in res.evaluation.scenarios)


# --------------------------------------------------------------------------- #
# Cas limites
# --------------------------------------------------------------------------- #
def test_cluster_too_small_overflows_and_is_flagged_not_feasible():
    inst = _cpu_instance([6, 6, 6], m=2)                               # 18 vCPU demandés, 20 disponibles mais indivisibles
    for name in BASELINES:
        res = run_baseline(inst, name)
        m = res.evaluation.by_scenario("s")
        assert m.overflow_pods == 1 and m.violation > 0 and res.evaluation.violation > 0
        assert validate_solution(inst, res.solution)                   # capacité jamais violée


def test_item_larger_than_server_always_overflows():
    inst = _cpu_instance([11, 3], m=3)
    for name in BASELINES:
        a = run_baseline(inst, name).solution.assignments[0]
        assert a[0] == OVF and a[1] >= 0


def test_empty_cluster():
    inst = _cpu_instance([3, 3], m=None)                     # N_max >= 1 exigé : cluster vide passé explicitement
    assert cluster_nodes(inst, 0).size == 0
    for name in BASELINES:
        res = run_baseline(inst, name, node_types=[])
        m = res.evaluation.by_scenario("s")
        assert m.overflow_pods == 2 and m.active_nodes == 0 and m.balance == 0.0


def test_max_pods_limit_respected_by_all_baselines():
    inst = _cpu_instance([1, 1, 1, 1, 1], m=2, max_pods=2)
    for name in BASELINES:
        res = run_baseline(inst, name)
        assert validate_solution(inst, res.solution)
        assert res.evaluation.by_scenario("s").overflow_pods == 1      # 2 serveurs x 2 pods < 5
    assert server_lower_bound(inst, cluster_nodes(inst), "s") == 3     # ceil(5 / 2)


def test_cluster_nodes_requires_single_type_and_valid_m():
    t1, t2 = InstanceType("a", 1.0, 4, 8), InstanceType("b", 1.0, 8, 16)
    w = Workload("w", 1, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    two = Instance(types=(t1, t2), workloads=(w,), scenarios=(sc,))
    with pytest.raises(DataValidationError):
        cluster_nodes(two)
    one = Instance(types=(t1,), workloads=(w,), scenarios=(sc,))
    assert cluster_nodes(one, 5).tolist() == [0] * 5
    with pytest.raises(DataValidationError):
        cluster_nodes(one, -1)
    with pytest.raises(DataValidationError):
        server_lower_bound(two, [0, 1], "s")                           # serveurs non identiques
    with pytest.raises(DataValidationError):
        server_lower_bound(one, [], "s")
    with pytest.raises(DataValidationError):
        run_baseline(one, "best_fit")


def test_baseline_exceeding_quota_is_rejected_by_validator():
    inst = _cpu_instance([3, 3], m=2)
    with pytest.raises(Exception, match="violation"):
        run_baseline(inst, "first_fit", node_types=[0, 0, 0])          # 3 serveurs > N_max = 2


def test_identical_instances_for_all_baselines_and_timing_recorded():
    inst = generate_cluster_instance(seed=0, n_containers=30, m=30)
    results = run_all_baselines(inst)
    assert set(results) == set(BASELINES)
    assert all(r.elapsed_s >= 0 and r.evaluation.elapsed_s == r.elapsed_s for r in results.values())
    assert all(r.solution.node_types.tolist() == results["first_fit"].solution.node_types.tolist()
               for r in results.values())


# --------------------------------------------------------------------------- #
# Métrique Bal et clé lexicographique (B1-ter)
# --------------------------------------------------------------------------- #
def test_balance_hand_computed_active_servers_only():
    t = InstanceType("srv", 1.0, 10, 10)
    ws = (Workload("a", 10, 5), Workload("b", 5, 5))
    sc = Scenario.from_base_requests("s", ws, [0, 1], 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(sc,))
    m = compute_scenario_metrics(inst, [0, 0, 0], "s", [0, 1])         # 3e serveur vide : exclu de Bal
    # u_cpu = (1.0, 0.5) std = 0.25 ; u_ram = (0.5, 0.5) std = 0 ; Bal = (0.25 + 0) / 2
    assert m.active_nodes == 2 and m.balance == pytest.approx(0.125)
    # utilisation des serveurs ACTIFS : CPU 15/20, RAM 10/20 ; sur tout le cluster : 15/30 et 10/30
    assert m.active_utilization.tolist() == pytest.approx([0.75, 0.5])
    assert m.utilization.tolist() == pytest.approx([0.5, 1 / 3])
    # sigma (B1 §5.2) pondère sur TOUS les nœuds, Bal non : les deux métriques sont distinctes
    assert m.sigma[0] != pytest.approx(m.balance)


def test_balance_single_server_is_zero_and_bounded():
    inst = generate_cluster_instance(seed=3, n_containers=40, m=40, server_vcpu=8, cpu_median=1.0)
    for res in run_all_baselines(inst).values():
        for m in res.evaluation.scenarios:
            assert np.all((m.active_utilization >= 0) & (m.active_utilization <= 1 + 1e-9))
            assert np.all(m.active_utilization + 1e-12 >= m.utilization)         # actifs seuls : >= cluster entier
            assert 0.0 <= m.balance <= 0.5 + 1e-12                     # écart-type d'une grandeur dans [0,1]
    one = _cpu_instance([3], m=1)
    assert run_baseline(one, "first_fit").evaluation.by_scenario("s").balance == 0.0


def test_objective_key_is_lexicographic_cost_then_balance():
    inst = generate_cluster_instance(seed=3, n_containers=40, m=40, server_vcpu=8, cpu_median=1.0)
    res = run_all_baselines(inst)
    keys = {n: r.evaluation.objective_key() for n, r in res.items()}
    assert all(k[0] == 0.0 for k in keys.values())                     # V = 0 : tout est placé
    for n, r in res.items():                                           # serveurs identiques : C = prix x serveurs
        assert keys[n][1] == pytest.approx(float(inst.prices[0]) * r.evaluation.n_nodes)
        assert keys[n][2] == r.evaluation.worst_balance
    # priorité stricte : un coût plus faible gagne TOUJOURS, quel que soit l'équilibre
    assert (0.0, 2.0, 0.49) < (0.0, 3.0, 0.0)
    assert (0.0, 2.0, 0.1) < (0.0, 2.0, 0.2)


def test_log_record_has_objective_fields():
    inst = generate_cluster_instance(seed=0, n_containers=20, m=20)
    rec = run_baseline(inst, "first_fit").evaluation.log_record()
    assert "active_nodes[stable]" in rec and "Bal[stable]" in rec and "time_s" in rec


def test_baselines_performance():
    import time
    inst = generate_cluster_instance(seed=0, n_containers=2000, m=2000, server_vcpu=16,
                                     bandwidth_active=True, n_variable_states=2)
    start = time.perf_counter()
    res = run_all_baselines(inst)
    assert time.perf_counter() - start < 30.0
    assert all(m.overflow_pods == 0 for m in res["first_fit_decreasing"].evaluation.scenarios)
