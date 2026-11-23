"""Tests B3b : FF/FFD avec ouverture du type le moins cher, fusion de flotte, RR mono-type."""
import numpy as np
import pytest

from kubeopt import (
    DataValidationError, GeneratorConfig, Instance, InstanceType, InvalidPlacementError, OPEN_BASELINES,
    Scenario, Workload, cluster_nodes, evaluate, generate_cluster_instance, generate_instance,
    merge_runs, mono_type_fleet, open_assignment, paper_example, run_all_open_baselines,
    run_baseline, run_open_baseline, validate_solution,
)
from kubeopt.baselines import first_fit_assignment, first_fit_decreasing_assignment
from kubeopt.validator import overflowed_pods_that_fit

OVF = -1


# --------------------------------------------------------------------------- #
# Exemple B1 §11 : traces calculées à la main
# --------------------------------------------------------------------------- #
def test_ff_open_paper_example_hand_trace():
    inst = paper_example()
    nt, a = open_assignment(inst, "stable", "ff")
    # w1 (3;2) -> T2 (seul type éligible assez gros parmi les moins chers) ; w2 y tient ; w3 -> T1 ;
    # w4 -> nouveau T1 ; w5 tient dans ce dernier
    assert nt.tolist() == [1, 0, 0] and a.tolist() == [0, 0, 1, 2, 2]
    assert inst.cost([2, 1, 0]) == pytest.approx(0.44)
    nt, a = open_assignment(inst, "pic", "ff")
    assert nt.tolist() == [1, 0, 0, 0, 0] and a.tolist() == [0, 0, 1, 2, 2, 3, 4]
    assert float(inst.prices[nt].sum()) == pytest.approx(0.68)


def test_ffd_open_paper_example_hand_trace():
    inst = paper_example()
    nt, a = open_assignment(inst, "stable", "ffd")        # ordre w1, w2, w5, w3, w4
    assert nt.tolist() == [1, 0, 0] and a.tolist() == [0, 0, 2, 1, 1]
    assert float(inst.prices[nt].sum()) == pytest.approx(0.44)


def test_merge_is_max_by_type_and_all_valid():
    inst = paper_example()
    for name, expected_cost in (("first_fit_open", 0.68), ("first_fit_decreasing_open", 0.68)):
        res = run_open_baseline(inst, name)
        assert res.evaluation.composition.tolist() == [4, 1, 0]          # max(1,4) T1, max(1,1) T2
        assert res.evaluation.cost == pytest.approx(expected_cost)
        assert validate_solution(inst, res.solution)
        assert all(m.overflow_pods == 0 for m in res.evaluation.scenarios)
        # le scénario stable n'utilise que 3 des 5 nœuds de la flotte commune
        assert res.evaluation.by_scenario("stable").active_nodes == 3
        assert res.evaluation.by_scenario("pic").active_nodes == 5


def test_rr_mono_paper_example():
    inst = paper_example()
    nt = mono_type_fleet(inst)
    assert len(set(nt.tolist())) == 1                                      # un seul type
    # T1 ne peut pas héberger w1 (3 vCPU) : candidats T2 et T3 ; T3 x 2 = 0,72 $/h est le moins cher
    assert nt.tolist() == [2, 2]
    res = run_open_baseline(inst, "round_robin_mono")
    assert res.evaluation.cost == pytest.approx(0.72)
    assert validate_solution(inst, res.solution)


def test_rr_mono_slack_nodes():
    inst = paper_example()
    # sans marge : 2 x T3 = 0,72 $/h (T2 en demande 4 : 0,80). Avec +1 nœud : 3 x T3 = 1,08 > 5 x T2 = 1,00
    assert mono_type_fleet(inst).tolist() == [2, 2]
    assert mono_type_fleet(inst, slack_nodes=1).tolist() == [1, 1, 1, 1, 1]
    with pytest.raises(DataValidationError):
        mono_type_fleet(inst, slack_nodes=-1)


# --------------------------------------------------------------------------- #
# Règle d'ouverture : type éligible le moins cher qui accueille le pod
# --------------------------------------------------------------------------- #
def _catalog_instance(types, workloads, **kw):
    sc = Scenario.from_base_requests("s", workloads, range(len(workloads)), 0.0)
    return Instance(types=tuple(types), workloads=tuple(workloads), scenarios=(sc,), **kw)


def test_opens_cheapest_fitting_type_not_cheapest_per_vcpu():
    types = (InstanceType("big", 1.0, 16, 64),            # 0,0625 $/vCPU : meilleur prix par vCPU
             InstanceType("small", 0.3, 4, 16))           # plus cher par vCPU mais moins cher en absolu
    inst = _catalog_instance(types, [Workload("p", 2, 2)])
    nt, a = open_assignment(inst, "s", "ff")
    assert nt.tolist() == [1]                              # small, pas big
    inst = _catalog_instance(types, [Workload("p", 8, 2)])
    assert open_assignment(inst, "s", "ff")[0].tolist() == [0]   # seul big convient


def test_opening_ties_go_to_smallest_index():
    types = (InstanceType("a", 1.0, 4, 8), InstanceType("b", 1.0, 8, 16))
    inst = _catalog_instance(types, [Workload("p", 1, 1)])
    assert open_assignment(inst, "s", "ff")[0].tolist() == [0]


def test_opening_respects_eligibility():
    types = (InstanceType("cheap", 0.1, 8, 16, arch="arm64"), InstanceType("dear", 1.0, 8, 16))
    inst = _catalog_instance(types, [Workload("p", 1, 1, arch="x86_64")])
    assert open_assignment(inst, "s", "ff")[0].tolist() == [1]


def test_opening_respects_type_quota_and_total_limit():
    types = (InstanceType("cheap", 0.1, 4, 8, max_count=1), InstanceType("dear", 1.0, 4, 8))
    ws = [Workload(f"p{i}", 3, 1) for i in range(3)]       # un pod par nœud (3 + 3 > 4)
    inst = _catalog_instance(types, ws)
    nt, a = open_assignment(inst, "s", "ff")
    assert sorted(nt.tolist()) == [0, 1, 1] and (a >= 0).all()   # un seul 'cheap' (quota)
    inst = _catalog_instance(types, ws, n_max_total=2)
    nt, a = open_assignment(inst, "s", "ff")
    assert nt.size == 2 and (a == OVF).sum() == 1                # N_max = 2 : le 3e pod déborde


def test_pod_larger_than_any_type_overflows():
    types = (InstanceType("a", 1.0, 4, 8),)
    inst = _catalog_instance(types, [Workload("big", 9, 1), Workload("ok", 1, 1)])
    nt, a = open_assignment(inst, "s", "ffd")
    assert a.tolist() == [OVF, 0] and nt.tolist() == [0]
    for name in ("first_fit_open", "first_fit_decreasing_open"):
        res = run_open_baseline(inst, name)
        assert res.evaluation.violation > 0 and validate_solution(inst, res.solution)
    with pytest.raises(DataValidationError):                      # RR mono impossible : un pod ne tient nulle part
        run_open_baseline(inst, "round_robin_mono")


def test_max_pods_respected_when_opening():
    types = (InstanceType("a", 1.0, 100, 100, max_pods=2),)
    inst = _catalog_instance(types, [Workload(f"p{i}", 1, 1) for i in range(5)])
    nt, a = open_assignment(inst, "s", "ff")
    assert nt.size == 3 and np.bincount(a).tolist() == [2, 2, 1]


def test_bandwidth_drives_opening():
    types = (InstanceType("lowbw", 0.1, 8, 16, bw_mbps=100.0), InstanceType("highbw", 0.5, 8, 16, bw_mbps=1000.0))
    inst = _catalog_instance(types, [Workload("p", 1, 1, bw_mbps=500.0)], bandwidth_active=True)
    assert open_assignment(inst, "s", "ff")[0].tolist() == [1]    # lowbw ne suffit pas (BW)
    off = _catalog_instance(types, [Workload("p", 1, 1, bw_mbps=500.0)], bandwidth_active=False)
    assert open_assignment(off, "s", "ff")[0].tolist() == [0]


def test_empty_scenario_opens_nothing():
    types = (InstanceType("a", 1.0, 4, 8),)
    w = Workload("w", 1, 1)
    empty = Scenario("vide", np.array([], dtype=int), np.zeros((0, 3)), 0.0)
    inst = Instance(types=types, workloads=(w,), scenarios=(empty,))
    nt, a = open_assignment(inst, "vide", "ff")
    assert nt.size == 0 and a.size == 0
    res = run_open_baseline(inst, "first_fit_open")
    assert res.evaluation.cost == 0.0 and res.evaluation.n_nodes == 0


def test_invalid_arguments():
    inst = paper_example()
    with pytest.raises(DataValidationError):
        open_assignment(inst, "stable", "best")
    with pytest.raises(DataValidationError):
        open_assignment(inst, "stable", "ff", allowed_types=[True])
    with pytest.raises(DataValidationError):
        run_open_baseline(inst, "first_fit")


# --------------------------------------------------------------------------- #
# Équivalence avec B3 (type unique) et avec baselines.py du projet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_open_equals_cluster_baselines_on_single_type(seed, bw):
    inst = generate_cluster_instance(seed=seed, n_containers=40, server_vcpu=8, bandwidth_active=bw,
                                     cpu_median=1.0, heterogeneity=0.9, n_variable_states=2)
    nodes = cluster_nodes(inst)                                     # m = max |I_s| : jamais limitant
    for s in inst.scenarios:
        for policy, ref in (("ff", first_fit_assignment), ("ffd", first_fit_decreasing_assignment)):
            nt, a = open_assignment(inst, s, policy)
            expected = ref(inst, nodes, s)
            assert a.tolist() == expected.tolist(), (policy, s.name)
            assert nt.size == np.unique(expected[expected >= 0]).size


# --------------------------------------------------------------------------- #
# Propriétés sur catalogues hétérogènes (2 et 3 dimensions)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_open_baselines_valid_maximal_and_complete(seed, bw):
    inst = generate_instance(GeneratorConfig(seed=seed, bandwidth_active=bw, n_variable_states=2,
                                             n_base_workloads=50, cpu_median=1.5, heterogeneity=1.0))
    results = run_all_open_baselines(inst)
    for name in ("first_fit_open", "first_fit_decreasing_open"):
        res = results[name]
        assert validate_solution(inst, res.solution), name
        assert all(m.overflow_pods == 0 for m in res.evaluation.scenarios), name   # tout placé (ouverture libre)
        for s, a in zip(inst.scenarios, res.solution.assignments):
            assert overflowed_pods_that_fit(inst, res.solution.node_types, s, a) == []
        # chaque nœud de la flotte fusionnée sert dans au moins un scénario (pas de nœud inutile)
        used = np.zeros(res.solution.n_nodes, dtype=bool)
        for a in res.solution.assignments:
            used[a[a >= 0]] = True
        assert used.all()
        assert res.evaluation.cost == pytest.approx(float(inst.prices[res.solution.node_types].sum()))
    rr = results["round_robin_mono"]
    assert validate_solution(inst, rr.solution)
    assert len(set(rr.solution.node_types.tolist())) == 1


@pytest.mark.parametrize("seed", range(6))
def test_per_scenario_cost_bounded_by_dedicated_node_per_pod(seed):
    """Majorant valide : FF n'ouvre jamais plus cher que « un nœud dédié par pod au type le moins cher »."""
    inst = generate_instance(GeneratorConfig(seed=seed, n_base_workloads=40, n_variable_states=1))
    act = list(inst.active_dims)
    for s in inst.scenarios:
        nt, a = open_assignment(inst, s, "ff")
        cheapest_fit = []
        for r, i in enumerate(s.members):
            ok = inst.eligibility[i] & np.all(s.demand[r, act] <= inst.allocatable[:, act] + 1e-9, axis=1)
            cheapest_fit.append(inst.prices[ok].min())
        assert float(inst.prices[nt].sum()) <= sum(cheapest_fit) + 1e-9


def test_deterministic():
    inst = generate_instance(GeneratorConfig(seed=5, n_base_workloads=40, n_variable_states=2))
    a, b = run_all_open_baselines(inst), run_all_open_baselines(inst)
    for name in OPEN_BASELINES:
        assert np.array_equal(a[name].solution.node_types, b[name].solution.node_types)
        assert all(np.array_equal(x, y) for x, y in zip(a[name].solution.assignments, b[name].solution.assignments))


def test_merge_runs_preserves_loads_and_handles_empty():
    inst = paper_example()
    runs = [open_assignment(inst, s, "ff") for s in inst.scenarios]
    merged = merge_runs(inst, runs)
    for s, (nt, a), am in zip(inst.scenarios, runs, merged.assignments):
        for b in range(nt.size):                                    # même type de nœud avant/après remappage
            idx = np.flatnonzero(a == b)
            if idx.size:
                assert len(set(am[idx].tolist())) == 1
                assert merged.node_types[am[idx[0]]] == nt[b]
        assert ((a >= 0) == (am >= 0)).all()
    empty = merge_runs(inst, [(np.zeros(0, dtype=np.int64), np.full(5, OVF)), (np.zeros(0, dtype=np.int64), np.full(7, OVF))])
    assert empty.n_nodes == 0 and (empty.assignments[0] == OVF).all()


def test_merge_exceeding_total_limit_is_rejected():
    """Chaque scénario tient dans N_max, mais le maximum type par type le dépasse : rejet explicite."""
    types = (InstanceType("a", 0.1, 4, 8), InstanceType("b", 0.2, 4, 8, labels={"x=1"}))
    ws = (Workload("p0", 3, 1), Workload("p1", 3, 1),
          Workload("q0", 3, 1, node_selector={"x=1"}), Workload("q1", 3, 1, node_selector={"x=1"}))
    sA = Scenario.from_base_requests("A", ws, [0, 1], 0.0)
    sB = Scenario.from_base_requests("B", ws, [2, 3], 0.0)
    inst = Instance(types=types, workloads=ws, scenarios=(sA, sB), n_max_total=3)
    assert open_assignment(inst, "A", "ff")[0].size == 2 and open_assignment(inst, "B", "ff")[0].size == 2
    with pytest.raises(InvalidPlacementError, match="N_max"):
        run_open_baseline(inst, "first_fit_open")


def test_objective_key_with_heterogeneous_costs():
    inst = paper_example()
    ev = run_open_baseline(inst, "first_fit_open").evaluation
    v, c, bal = ev.objective_key()
    assert v == 0.0 and c == pytest.approx(0.68) and bal == pytest.approx(ev.worst_balance)
    assert ev.objective_key() < (0.0, 0.72, 0.0)                    # coût plus bas gagne quel que soit Bal


def test_performance_open_baselines():
    import time
    inst = generate_instance(GeneratorConfig(seed=0, n_base_workloads=1500, n_variable_states=2,
                                             bandwidth_active=True, cpu_median=1.0))
    start = time.perf_counter()
    res = run_all_open_baselines(inst)
    assert time.perf_counter() - start < 60.0
    assert all(m.overflow_pods == 0 for m in res["first_fit_decreasing_open"].evaluation.scenarios)
