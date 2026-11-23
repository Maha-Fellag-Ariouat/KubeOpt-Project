"""Tests B5 : phase 2 (équilibre à coût <= (1 + eta) C*) -- contraintes, cohérence des métriques, solveur exact."""
import dataclasses

import numpy as np
import pytest

from kubeopt import (
    DataValidationError, GeneratorConfig, Instance, InstanceType, SAConfig, Scenario, Workload,
    balance_annealing, compute_scenario_metrics, generate_instance, initial_solution, paper_example,
    simulated_annealing, two_phase_annealing, validate_solution,
)
from kubeopt.annealing import MOVES
from test_annealing import make_state, strict, tiny, with_eps
from exact_reference import exact_min_balance, exact_min_cost


# --------------------------------------------------------------------------- #
# Cohérence de l'énergie de phase 2
# --------------------------------------------------------------------------- #
def test_state_balance_matches_metrics_after_random_moves():
    inst = generate_instance(GeneratorConfig(seed=2, n_base_workloads=25, n_variable_states=2, cpu_median=1.2,
                                             bandwidth_active=True))
    st = make_state(inst)
    rng = np.random.default_rng(0)
    cfg = SAConfig(p_open=0.2)
    checked = 0
    for _ in range(300):
        if st.try_move(MOVES[int(rng.integers(len(MOVES)))], rng, cfg):
            st.commit()
            sol = st.compact()
            for s, (sc, a) in enumerate(zip(inst.scenarios, sol.assignments)):
                m = compute_scenario_metrics(inst, sol.node_types, sc, a)
                assert st.balances()[s] == pytest.approx(m.balance, abs=1e-10)
            checked += 1
    assert checked > 50


def test_energy2_semantics():
    inst = tiny(2)
    st = make_state(inst)
    cost = st.cost()
    e_ok, g_ok = st.energy2(cost, 0.0)                         # C = C* : contrainte respectée
    assert 0.0 <= e_ok <= 0.5 and e_ok == pytest.approx(st.balance_worst())
    assert g_ok == pytest.approx(float(st.balances().mean())) and g_ok <= e_ok + 1e-12
    e_bad, _ = st.energy2(cost / 2, 0.0)                       # C = 2 C* : violation de coût de 100 %
    assert e_bad == pytest.approx(1.0 + 1.0)
    assert st.energy2(cost / 2, 1.0)[0] == pytest.approx(e_ok)  # eta = 100 % : (1 + eta) C* = C


# --------------------------------------------------------------------------- #
# Contraintes de la phase 2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eta", [0.0, 0.05, 0.2])
@pytest.mark.parametrize("seed", range(3))
def test_phase2_respects_cost_service_and_never_worsens_balance(seed, eta):
    inst = generate_instance(GeneratorConfig(seed=seed, n_base_workloads=30, n_variable_states=2, cpu_median=1.2,
                                             bandwidth_active=bool(seed % 2)))
    p1 = simulated_annealing(inst, SAConfig(iterations=2500, seed=seed))
    r = balance_annealing(inst, p1.solution, SAConfig(seed=seed), eta, 2500)
    assert validate_solution(inst, r.solution).valid and r.evaluation.violation == 0.0
    assert r.c_star <= p1.evaluation.cost + 1e-12                              # C* ne fait que baisser
    assert r.evaluation.cost <= (1.0 + eta) * r.c_star * (1 + 1e-12)           # contrainte de coût (B1 éq. 30)
    assert r.evaluation.worst_balance <= p1.evaluation.worst_balance + 1e-12   # jamais moins équilibré
    assert all(m.overflow_pods == 0 for m in r.evaluation.scenarios)


def test_phase2_with_eta_zero_does_not_increase_cost():
    inst = generate_instance(GeneratorConfig(seed=5, n_base_workloads=30, n_variable_states=1, cpu_median=1.2))
    p1 = simulated_annealing(inst, SAConfig(iterations=3000, seed=0))
    r = balance_annealing(inst, p1.solution, SAConfig(seed=0), 0.0, 3000)
    assert r.evaluation.cost <= p1.evaluation.cost + 1e-12


def test_phase2_is_deterministic_and_zero_iterations_is_neutral():
    inst = generate_instance(GeneratorConfig(seed=3, n_base_workloads=25, cpu_median=1.2))
    p1 = simulated_annealing(inst, SAConfig(iterations=2000, seed=0))
    a = balance_annealing(inst, p1.solution, SAConfig(seed=7), 0.0, 1500)
    b = balance_annealing(inst, p1.solution, SAConfig(seed=7), 0.0, 1500)
    assert a.history == b.history and np.array_equal(a.solution.node_types, b.solution.node_types)
    z = balance_annealing(inst, p1.solution, SAConfig(seed=7), 0.0, 0)
    assert z.evaluation.worst_balance == pytest.approx(p1.evaluation.worst_balance)
    assert z.history == () and z.evaluation.cost <= p1.evaluation.cost + 1e-12


def test_phase2_keeps_epsilon_tolerance_of_service():
    inst = paper_example()                                                       # eps_pic = 0,15
    p1 = simulated_annealing(inst, SAConfig(iterations=6000, seed=0))
    r = balance_annealing(inst, p1.solution, SAConfig(seed=0), 0.0, 3000)
    assert r.evaluation.violation == 0.0
    assert r.evaluation.by_scenario("pic").overflow_rate <= 0.15 + 1e-12
    assert r.evaluation.cost <= p1.evaluation.cost + 1e-12


@pytest.mark.parametrize("eta", [-0.1, float("nan"), float("inf"), True, "0.1"])
def test_eta_validation(eta):
    inst = tiny(1)
    sol = initial_solution(inst)
    with pytest.raises(DataValidationError):
        balance_annealing(inst, sol, SAConfig(), eta, 10)
    with pytest.raises(DataValidationError):
        balance_annealing(inst, sol, SAConfig(), 0.0, -1)


def test_phase2_skipped_when_service_cannot_be_met_or_cost_is_zero():
    types = (InstanceType("a", 1.0, 4, 8),)
    ws = (Workload("huge", 9, 1), Workload("p", 1, 1))
    inst = Instance(types=types, workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(2), 0.0),))
    p1 = simulated_annealing(inst, SAConfig(iterations=200, seed=0))
    r = balance_annealing(inst, p1.solution, SAConfig(), 0.0, 500)
    assert r.stop_reason == "skipped" and r.evaluation.violation > 0
    assert r.solution is p1.solution
    empty = Instance(types=types, workloads=(Workload("w", 1, 1),),
                     scenarios=(Scenario("vide", np.array([], dtype=int), np.zeros((0, 3)), 0.0),))
    assert balance_annealing(empty, initial_solution(empty), SAConfig(), 0.0, 100).stop_reason == "skipped"


def test_single_active_node_has_zero_balance_and_stays():
    t = InstanceType("a", 1.0, 16, 64)
    ws = tuple(Workload(f"p{i}", 1, 1) for i in range(4))
    inst = Instance(types=(t,), workloads=ws, scenarios=(Scenario.from_base_requests("s", ws, range(4), 0.0),))
    r = two_phase_annealing(inst, SAConfig(iterations=500, seed=0), phase2_iterations=500)
    assert r.evaluation.worst_balance == 0.0 and r.evaluation.cost == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Comparaison au solveur EXACT (petites instances)
# --------------------------------------------------------------------------- #
def test_balance_never_below_exact_and_mostly_optimal_with_restarts():
    hits = total = 0
    for sd in range(4):
        inst = tiny(sd)
        c_star, _ = exact_min_cost(inst, 6)
        bal_exact = exact_min_balance(inst, c_star, 6)
        r = two_phase_annealing(inst, SAConfig(iterations=8000, seed=0), 0.0, phase2_iterations=5000, phase2_runs=4)
        assert r.phase1.evaluation.cost >= c_star - 1e-9
        assert r.evaluation.worst_balance <= r.phase1.evaluation.worst_balance + 1e-12
        if abs(r.evaluation.cost - c_star) < 1e-9:              # même coût optimal : comparable à l'exact
            assert r.evaluation.worst_balance >= bal_exact - 1e-9, "le recuit ne peut pas battre l'optimum exact"
            total += 1
            hits += abs(r.evaluation.worst_balance - bal_exact) < 1e-6
    assert total >= 3 and hits / total >= 0.5, (hits, total)


def test_eta_relaxation_cannot_beat_exact_balance_under_relaxed_cost():
    inst = tiny(1)
    c_star, _ = exact_min_cost(inst, 6)
    relaxed = exact_min_balance(inst, 1.2 * c_star, 6)
    strict_bal = exact_min_balance(inst, c_star, 6)
    assert relaxed <= strict_bal + 1e-12                          # relâcher le coût ne peut pas dégrader l'optimum
    r = two_phase_annealing(inst, SAConfig(iterations=8000, seed=0), 0.2, phase2_iterations=5000, phase2_runs=3)
    assert r.evaluation.cost <= 1.2 * r.phase2.c_star * (1 + 1e-12)
    assert r.evaluation.worst_balance >= relaxed - 1e-9


# --------------------------------------------------------------------------- #
# API deux phases
# --------------------------------------------------------------------------- #
def test_two_phase_api_and_restarts():
    inst = generate_instance(GeneratorConfig(seed=4, n_base_workloads=25, n_variable_states=1, cpu_median=1.2))
    r = two_phase_annealing(inst, SAConfig(iterations=2000, seed=0), 0.0, phase2_iterations=1500,
                            phase1_runs=2, phase2_runs=3)
    assert len(r.phase2_runs) == 3 and r.phase2 in r.phase2_runs
    assert r.evaluation.worst_balance <= min(x.evaluation.worst_balance for x in r.phase2_runs) + 1e-12
    v, c, b = r.evaluation.objective_key()
    assert v == 0.0 and c == pytest.approx(r.evaluation.cost) and b == r.evaluation.worst_balance
    assert r.solution is r.phase2.solution
    for kw in (dict(phase1_runs=0), dict(phase2_runs=0), dict(phase2_runs=True)):
        with pytest.raises(DataValidationError):
            two_phase_annealing(inst, SAConfig(iterations=10), **kw)


def test_ratchet_tightens_cost_reference_when_a_cheaper_state_is_found():
    """Départ volontairement médiocre (FFD avec ouverture) : la phase 2 trouve moins cher et resserre C*."""
    inst = generate_instance(GeneratorConfig(seed=7, n_base_workloads=60, n_variable_states=2, cpu_median=1.2,
                                             heterogeneity=0.7, bandwidth_active=True))
    start = initial_solution(inst)
    r = balance_annealing(inst, start, SAConfig(seed=0), 0.0, 3000)
    assert r.stats["ratchets"] > 0
    assert r.c_star < r.start_evaluation.cost - 0.1                              # mesuré : 7,79 -> 7,10
    assert r.evaluation.cost == pytest.approx(r.c_star, rel=1e-9)                # eta = 0 : coût = C* final
    assert r.evaluation.violation == 0.0 and validate_solution(inst, r.solution).valid


def test_metropolis_actually_rejects_moves_in_phase2():
    inst = generate_instance(GeneratorConfig(seed=7, n_base_workloads=60, n_variable_states=2, cpu_median=1.2,
                                             heterogeneity=0.7, bandwidth_active=True))
    r = balance_annealing(inst, initial_solution(inst), SAConfig(seed=1), 0.0, 3000)
    accepted, applied = sum(r.stats["accepted"].values()), sum(r.stats["applied"].values())
    assert 0 < accepted < 0.5 * applied                                          # mesuré : environ 12 %
    assert r.temperatures["T0"] > r.temperatures["T_end"] > 0
