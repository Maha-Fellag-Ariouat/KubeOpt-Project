"""Tests B4 : recuit simulé (phase 1) -- cohérence de l'état, validité, comparaison au solveur exact."""
import dataclasses
import time

import numpy as np
import pytest

from kubeopt import (
    DataValidationError, GeneratorConfig, Instance, InstanceType, SAConfig, Scenario, Workload, evaluate,
    generate_instance, initial_solution, multistart_annealing, paper_example, simulated_annealing,
    validate_solution,
)
from kubeopt.annealing import MOVES, _State
from exact_reference import exact_min_cost


def tiny(seed, nb=6, extra=0.34, nmax=6):
    return generate_instance(GeneratorConfig(
        seed=seed, families=(("a", 3.0, 0.05),), sizes_vcpu=(4, 8, 16), n_base_workloads=nb,
        peak_extra_fraction=extra, n_variable_states=1, cpu_median=1.5, heterogeneity=0.6,
        price_noise=0.15, n_max_total=nmax))


def strict(inst):
    """Même instance avec eps = 0 dans tous les scénarios."""
    return dataclasses.replace(inst, scenarios=tuple(Scenario(s.name, s.members, s.demand, 0.0)
                                                     for s in inst.scenarios))


def with_eps(inst, eps):
    return dataclasses.replace(inst, scenarios=tuple(Scenario(s.name, s.members, s.demand, eps)
                                                     for s in inst.scenarios))


def make_state(inst):
    sol = initial_solution(inst)
    try:
        c_ub = inst.cost_upper_bound()
    except DataValidationError:
        c_ub = float(inst.prices.max())
    return _State(inst, sol.node_types, sol.assignments, c_ub)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    dict(iterations=-1), dict(initial_acceptance=0.01, final_acceptance=0.5), dict(final_acceptance=0.0),
    dict(initial_acceptance=1.0), dict(p_open=1.5), dict(patience=0), dict(time_limit_s=0),
    dict(move_weights=(("teleport", 1.0),)), dict(move_weights=(("relocate", 0.0),)),
    dict(move_weights=(("relocate", 1.0), ("relocate", 1.0))), dict(log_every=0), dict(calibration_samples=-1),
    dict(final_reference="median"),
])
def test_config_validation(kwargs):
    with pytest.raises(DataValidationError):
        simulated_annealing(paper_example(), SAConfig(**kwargs))


def test_all_moves_are_declared():
    assert set(n for n, _ in SAConfig().move_weights) == set(MOVES)


# --------------------------------------------------------------------------- #
# Cohérence de l'état incrémental et réversibilité des mouvements
# --------------------------------------------------------------------------- #
def _fuzz(inst, seed, steps=400):
    rng = np.random.default_rng(seed)
    cfg = SAConfig(p_open=0.2)
    st = make_state(inst)
    applied = {m: 0 for m in MOVES}
    for step in range(steps):
        name = MOVES[int(rng.integers(len(MOVES)))]
        before = (st.node_type.copy(), st.alive.copy(), [a.copy() for a in st.assign],
                  st.type_count.copy(), st.fsum.copy(), st.cost())
        if st.try_move(name, rng, cfg):
            applied[name] += 1
            if rng.random() < 0.5:
                st.commit()
            else:
                st.revert()                                   # annulation : retour exact à l'état précédent
                assert np.array_equal(st.node_type, before[0]) and np.array_equal(st.alive, before[1])
                assert all(np.array_equal(a, b) for a, b in zip(st.assign, before[2]))
                assert np.array_equal(st.type_count, before[3])
                assert np.allclose(st.fsum, before[4], atol=1e-9) and st.cost() == pytest.approx(before[5])
        else:                                                  # mouvement irréalisable : état inchangé
            assert np.array_equal(st.node_type, before[0]) and all(np.array_equal(a, b) for a, b in zip(st.assign, before[2]))
        if step % 20 == 0:
            st.check_consistency()
    st.check_consistency()
    assert validate_solution(inst, st.compact()).valid or st.violation() > 0   # capacités toujours respectées
    return applied


@pytest.mark.parametrize("seed", range(4))
def test_state_consistency_and_reversibility(seed):
    inst = generate_instance(GeneratorConfig(seed=seed, n_base_workloads=25, n_variable_states=2,
                                             cpu_median=1.2, bandwidth_active=bool(seed % 2)))
    _fuzz(inst, seed)


def test_every_move_type_is_exercised_with_overflow_tolerance():
    inst = with_eps(generate_instance(GeneratorConfig(seed=3, n_base_workloads=30, n_variable_states=2,
                                                      cpu_median=1.5)), 0.15)
    total = {m: 0 for m in MOVES}
    for seed in range(3):
        for m, c in _fuzz(inst, seed, steps=600).items():
            total[m] += c
    assert all(total[m] > 0 for m in MOVES), total


def test_moves_never_violate_capacity_even_with_quotas_and_max_pods():
    types = (InstanceType("a", 0.3, 4, 8, max_count=3, max_pods=3), InstanceType("b", 0.6, 8, 16, max_pods=4))
    ws = tuple(Workload(f"p{i}", 0.5 + 0.2 * (i % 5), 1.0 + 0.3 * (i % 4)) for i in range(14))
    sc = [Scenario.from_base_requests("s1", ws, range(10), 0.0), Scenario.from_base_requests("s2", ws, range(14), 0.0)]
    inst = Instance(types=types, workloads=ws, scenarios=tuple(sc), n_max_total=8)
    r = simulated_annealing(inst, SAConfig(iterations=3000, seed=0))
    assert validate_solution(inst, r.solution).valid
    assert inst.respects_limits(r.evaluation.composition)


# --------------------------------------------------------------------------- #
# Propriétés du résultat
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(4))
def test_result_is_valid_and_never_worse_than_initial(seed):
    inst = generate_instance(GeneratorConfig(seed=seed, n_base_workloads=30, n_variable_states=2,
                                             cpu_median=1.2, bandwidth_active=True))
    r = simulated_annealing(inst, SAConfig(iterations=3000, seed=seed))
    assert validate_solution(inst, r.solution).valid
    assert r.evaluation.violation == 0.0
    assert r.evaluation.cost <= r.initial_evaluation.cost + 1e-12
    assert all(m.overflow_pods == 0 for m in r.evaluation.scenarios)          # eps = 0 : tout placé
    bests = [h["best_E1"] for h in r.history]
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(bests, bests[1:]))           # le meilleur ne se dégrade jamais
    assert len(r.history) == 3000 // 500
    assert r.temperatures["T0"] > r.temperatures["T_end"] > 0 and 0 < r.temperatures["alpha"] < 1


def test_same_seed_same_result_and_seed_changes_trajectory():
    inst = generate_instance(GeneratorConfig(seed=5, n_base_workloads=30, n_variable_states=1, cpu_median=1.2))
    a = simulated_annealing(inst, SAConfig(iterations=2000, seed=11))
    b = simulated_annealing(inst, SAConfig(iterations=2000, seed=11))
    c = simulated_annealing(inst, SAConfig(iterations=2000, seed=12))
    assert np.array_equal(a.solution.node_types, b.solution.node_types)
    assert all(np.array_equal(x, y) for x, y in zip(a.solution.assignments, b.solution.assignments))
    assert a.history == b.history and a.stats["accepted"] == b.stats["accepted"]
    assert a.history != c.history


def test_zero_iterations_returns_the_initial_quality():
    inst = generate_instance(GeneratorConfig(seed=1, n_base_workloads=20))
    r = simulated_annealing(inst, SAConfig(iterations=0))
    assert r.stop_reason == "iterations" and r.evaluation.cost <= r.initial_evaluation.cost + 1e-12
    assert r.history == ()


def test_patience_and_time_limit_stop_reasons():
    inst = generate_instance(GeneratorConfig(seed=2, n_base_workloads=25, cpu_median=1.2))
    r = simulated_annealing(inst, SAConfig(iterations=50000, patience=300, seed=0))
    assert r.stop_reason == "patience" and r.stats["iterations_done"] < 50000
    start = time.perf_counter()
    r = simulated_annealing(inst, SAConfig(iterations=10 ** 7, time_limit_s=0.5, seed=0))
    assert r.stop_reason == "time_limit" and time.perf_counter() - start < 10
    assert validate_solution(inst, r.solution).valid


def test_periodic_resync_keeps_state_exact():
    inst = generate_instance(GeneratorConfig(seed=2, n_base_workloads=25, cpu_median=1.2))
    r = simulated_annealing(inst, SAConfig(iterations=3000, resync_every=500, seed=3))
    assert r.stats["resyncs"] == 6 and validate_solution(inst, r.solution).valid


# --------------------------------------------------------------------------- #
# Comparaison au solveur EXACT (petites instances)
# --------------------------------------------------------------------------- #
def test_paper_example_matches_B1_optimum():
    """Exact : C* = 0,48 (flotte (1,0,1)) avec eps_pic = 0,15, comme annoncé par B1 §11.6."""
    inst = paper_example()
    assert exact_min_cost(inst, 6) == (pytest.approx(0.48), (1, 0, 1))
    hits = 0
    for seed in range(6):
        r = simulated_annealing(inst, SAConfig(iterations=10000, seed=seed))
        assert r.evaluation.violation == 0.0
        hits += abs(r.evaluation.cost - 0.48) < 1e-9
        assert r.evaluation.by_scenario("pic").overflow_rate <= 0.15 + 1e-12
    assert hits >= 5


def test_paper_example_strict_service_optimum():
    """Avec eps = 0 l'optimum exact est 0,56 (0,1,1) : le recuit ne peut pas faire mieux."""
    inst = strict(paper_example())
    cost, comp = exact_min_cost(inst, 6)
    assert cost == pytest.approx(0.56) and comp == (0, 1, 1)
    results = [simulated_annealing(inst, SAConfig(iterations=10000, seed=s)) for s in range(6)]
    assert all(r.evaluation.cost >= cost - 1e-9 and r.evaluation.violation == 0 for r in results)
    assert all(m.overflow_pods == 0 for r in results for m in r.evaluation.scenarios)
    assert sum(abs(r.evaluation.cost - cost) < 1e-9 for r in results) >= 3     # mesuré : 12/20 en moyenne


def test_never_below_exact_and_mostly_optimal_on_random_tiny_instances():
    panel = [tiny(sd) for sd in (0, 1, 2, 3, 4, 5, 6, 7)]
    hits = total = 0
    for inst in panel:
        exact, _ = exact_min_cost(inst, 6)
        for seed in range(2):
            r = simulated_annealing(inst, SAConfig(iterations=10000, seed=seed))
            assert r.evaluation.cost >= exact - 1e-9, "le recuit ne peut pas battre l'optimum exact"
            assert r.evaluation.violation == 0.0
            total += 1
            hits += abs(r.evaluation.cost - exact) < 1e-9
    assert hits / total >= 0.8, (hits, total)


def test_multistart_is_at_least_as_good_as_every_run():
    inst = tiny(1003)                                           # instance où certaines graines restent piégées
    best, runs = multistart_annealing(inst, SAConfig(iterations=10000, seed=0), n_runs=4)
    assert len(runs) == 4 and best in runs
    assert best.evaluation.cost <= min(r.evaluation.cost for r in runs) + 1e-12
    assert best.evaluation.cost >= exact_min_cost(inst, 6)[0] - 1e-9
    with pytest.raises(DataValidationError):
        multistart_annealing(inst, n_runs=0)


# --------------------------------------------------------------------------- #
# Tolérance au débordement, cas limites
# --------------------------------------------------------------------------- #
def test_overflow_tolerance_can_lower_cost_but_respects_epsilon():
    base = generate_instance(GeneratorConfig(seed=4, n_base_workloads=30, cpu_median=1.5, n_variable_states=1))
    strict_r = simulated_annealing(strict(base), SAConfig(iterations=4000, seed=0))
    loose_r = simulated_annealing(with_eps(base, 0.2), SAConfig(iterations=4000, seed=0))
    assert strict_r.evaluation.violation == 0 and loose_r.evaluation.violation == 0
    assert all(m.overflow_pods == 0 for m in strict_r.evaluation.scenarios)
    assert all(m.overflow_rate <= 0.2 + 1e-12 for m in loose_r.evaluation.scenarios)
    assert loose_r.evaluation.cost <= strict_r.evaluation.cost + 1e-9        # relâcher ne peut pas coûter plus cher (exact)


def test_unplaceable_pod_keeps_service_violation_without_crashing():
    types = (InstanceType("a", 1.0, 4, 8),)
    ws = (Workload("huge", 9, 1), Workload("p1", 1, 1), Workload("p2", 1, 1))
    sc = Scenario.from_base_requests("s", ws, range(3), 0.0)
    inst = Instance(types=types, workloads=ws, scenarios=(sc,))
    r = simulated_annealing(inst, SAConfig(iterations=500, seed=0))
    assert r.evaluation.violation > 0 and r.evaluation.by_scenario("s").overflow_pods == 1
    assert validate_solution(inst, r.solution).valid
    assert r.evaluation.cost == pytest.approx(1.0)                           # p1 et p2 partagent un seul nœud


def test_empty_and_trivial_instances():
    t = InstanceType("a", 1.0, 4, 8)
    w = Workload("w", 1, 1)
    empty = Scenario("vide", np.array([], dtype=int), np.zeros((0, 3)), 0.0)
    r = simulated_annealing(Instance(types=(t,), workloads=(w,), scenarios=(empty,)), SAConfig(iterations=200))
    assert r.evaluation.cost == 0.0 and r.solution.n_nodes == 0
    one = Instance(types=(t,), workloads=(w,), scenarios=(Scenario.from_base_requests("s", [w], [0], 0.0),))
    r = simulated_annealing(one, SAConfig(iterations=300))
    assert r.evaluation.cost == pytest.approx(1.0) and r.stats["calibration_fallback"] is True


def test_quota_forces_use_of_other_types():
    types = (InstanceType("cheap", 0.1, 4, 8, max_count=1), InstanceType("dear", 1.0, 4, 8))
    ws = tuple(Workload(f"p{i}", 3, 1) for i in range(3))                     # un pod par nœud
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(3), 0.0),))
    r = simulated_annealing(inst, SAConfig(iterations=1500, seed=0))
    assert r.evaluation.composition.tolist() == [1, 2] and r.evaluation.cost == pytest.approx(2.1)
    assert exact_min_cost(inst, 3)[0] == pytest.approx(2.1)


def test_bandwidth_dimension_is_respected_by_the_search():
    types = (InstanceType("lowbw", 0.1, 8, 16, bw_mbps=500.0), InstanceType("highbw", 0.3, 8, 16, bw_mbps=2000.0))
    ws = tuple(Workload(f"p{i}", 1, 1, bw_mbps=400.0) for i in range(4))     # 1600 Mbps au total
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(4), 0.0),),
                    bandwidth_active=True)
    r = simulated_annealing(inst, SAConfig(iterations=2000, seed=0))
    assert r.evaluation.cost == pytest.approx(0.3)                            # 1 highbw (0,3) < 4 lowbw (0,4)
    assert exact_min_cost(inst, 4)[0] == pytest.approx(0.3)
    off = dataclasses.replace(inst, bandwidth_active=False)
    assert simulated_annealing(off, SAConfig(iterations=2000, seed=0)).evaluation.cost == pytest.approx(0.1)


def test_custom_initial_solution_is_validated():
    inst = paper_example()
    bad = initial_solution(inst)
    from kubeopt import Solution
    over = Solution(np.array([0]), (np.zeros(5, dtype=int), np.zeros(7, dtype=int)))   # tout sur un T1
    with pytest.raises(Exception):
        simulated_annealing(inst, SAConfig(iterations=10), init=over)
    ok = simulated_annealing(inst, SAConfig(iterations=500, seed=0), init=bad)
    assert ok.evaluation.cost <= evaluate(inst, bad).cost + 1e-12


def test_performance_reasonable():
    inst = generate_instance(GeneratorConfig(seed=0, n_base_workloads=80, n_variable_states=2, cpu_median=1.2,
                                             bandwidth_active=True))
    start = time.perf_counter()
    r = simulated_annealing(inst, SAConfig(iterations=3000, seed=0))
    assert time.perf_counter() - start < 30.0
    assert r.evaluation.violation == 0.0


def _only_moves(inst, names, steps=500, seed=0):
    """Applique uniquement les mouvements `names` (tous acceptés) et valide l'état après chacun."""
    rng = np.random.default_rng(seed)
    cfg = SAConfig(p_open=0.3)
    st = make_state(inst)
    applied = 0
    for _ in range(steps):
        if st.try_move(names[int(rng.integers(len(names)))], rng, cfg):
            st.commit()
            applied += 1
            sol = st.compact()
            rep = validate_solution(inst, sol)
            assert rep.valid, rep.violations
            assert inst.respects_limits(sol.composition(inst.k))
    st.check_consistency()
    return applied


def test_split_and_merge_respect_eligibility():
    types = (InstanceType("a", 0.2, 8, 16), InstanceType("b", 0.2, 8, 16, taints={"t=1"}),
             InstanceType("c", 0.1, 4, 8))
    ws = tuple(Workload(f"p{i}", 1.0, 1.0) for i in range(6)) + \
        tuple(Workload(f"q{i}", 1.0, 1.0, tolerations={"t=1"}) for i in range(2))
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(8), 0.0),))
    assert not inst.eligibility[0, 1] and inst.eligibility[6, 1]            # p0 ne tolère pas le taint de b
    assert _only_moves(inst, ["split", "merge", "retype"]) > 20


def test_split_and_merge_respect_type_quotas_and_total_limit():
    types = (InstanceType("a", 0.1, 4, 8, max_count=1), InstanceType("b", 0.3, 8, 16, max_count=2),
             InstanceType("c", 0.6, 16, 32))
    ws = tuple(Workload(f"p{i}", 0.8, 1.0) for i in range(10))
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(10), 0.0),),
                    n_max_total=4)
    assert _only_moves(inst, ["split", "merge", "retype", "relocate"]) > 20
    assert _only_moves(inst, ["split", "merge"], seed=1) > 5


def test_merge_never_exceeds_a_full_type_quota():
    """Deux nœuds b (charge 3 chacun) tiendraient dans un type a (cap 8), mais a est déjà au quota."""
    types = (InstanceType("a", 0.1, 8, 16, max_count=1), InstanceType("b", 0.3, 4, 8))
    ws = tuple(Workload(f"p{i}", 3.0, 1.0) for i in range(4))
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(4), 0.0),))
    st = make_state(inst)
    assert st.type_count.tolist() == [1, 2]                                   # a plein, deux b à 3 vCPU
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert not st.try_move("merge", rng, SAConfig()) or inst.respects_limits(st.compact().composition(2))
        st.commit()
    assert st.type_count.tolist() == [1, 2]                                   # aucune fusion possible : état inchangé
