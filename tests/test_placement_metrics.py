"""Tests B2-b : Place, validateur indépendant, métriques O/F/U/sigma/V/Delta, évaluation."""
import itertools
import time

import numpy as np
import pytest

from kubeopt import (
    GeneratorConfig, Instance, InstanceType, InvalidPlacementError, PAPER_FLEETS, Scenario, Solution,
    Workload, change_metrics, compute_scenario_metrics, evaluate, fleet_violation, generate_instance,
    node_types_from_composition, paper_example, place, place_all, validate_placement, validate_solution,
)
from kubeopt.validator import overflowed_pods_that_fit

OVF = -1
# Placements « à la main » de B1 §11 (rangs : w1..w7 = 0..6 ; types : 0=T1, 1=T2, 2=T3)
HAND = {
    ("Xa", "stable"): ([0, 0, 0, 0], [OVF, 0, 2, 1, 1]),            # N1{w2} N2{w5,w4} N3{w3}, w1 déborde
    ("Xb", "stable"): ([1, 1], [0, 0, 1, OVF, 1]),                   # N1{w1,w2} N2{w3,w5}, w4 déborde
    ("Xc", "stable"): ([1, 1, 1], [0, 0, 1, 2, 1]),                  # N1{w1,w2} N2{w3,w5} N3{w4}
    ("Xd", "stable"): ([0, 2], [1, 1, 1, 1, 0]),                     # T3{w1..w4} T1{w5}
    ("Xd", "pic"): ([0, 2], [1, 1, 1, 0, 0, OVF, 1]),                # T1{w4,w5} T3{w1,w2,w3,w7}, w6 déborde
    ("Xc", "pic"): ([1, 1, 1], [0, 1, 2, 1, 0, OVF, 2]),             # {w1,w5} {w2,w4} {w3,w7}, w6 déborde
}


@pytest.fixture(scope="module")
def paper():
    return paper_example()


def metrics_of(paper, fleet, scen):
    nt, a = HAND[(fleet, scen)]
    assert node_types_from_composition(PAPER_FLEETS[fleet]).tolist() == nt
    assert validate_placement(paper, nt, scen, a), validate_placement(paper, nt, scen, a).violations
    return compute_scenario_metrics(paper, nt, scen, a)


# --------------------------------------------------------------------------- #
# Reproduction exacte de B1 §11
# --------------------------------------------------------------------------- #
def test_xa_stable_overflow_and_violation(paper):
    m = metrics_of(paper, "Xa", "stable")
    assert m.overflow.tolist() == [3.0, 2.0] and m.overflow_pods == 1
    assert m.overflow_rate == pytest.approx(0.2431, abs=1e-4)
    assert m.violation == pytest.approx(0.2431, abs=1e-4)             # eps_stable = 0
    assert m.placed.tolist() == [5.0, 16.0]
    assert m.active_nodes == 3                                         # N4 reste vide


def test_xb_stable_overflow(paper):
    m = metrics_of(paper, "Xb", "stable")
    assert m.overflow_rate == pytest.approx(0.1181, abs=1e-4)
    assert m.violation > 0


def test_xc_stable_B1_values(paper):
    m = metrics_of(paper, "Xc", "stable")
    assert m.overflow_rate == 0.0 and m.violation == 0.0
    assert m.utilization.tolist() == pytest.approx([8 / 12, 18 / 24])
    assert m.unused.tolist() == [4.0, 6.0]
    assert m.fragmentation == pytest.approx(0.0556, abs=1e-4)          # 1/2 (1,333/12)
    assert m.fragmentation == pytest.approx(0.5 * (4 / 3) / 12)
    # sigma^CPU calculé à la main : nœuds (4,3,1)/4 -> u = (1, .75, .25), U = 2/3
    u = np.array([1.0, 0.75, 0.25])
    assert m.sigma[0] == pytest.approx(float(np.sqrt(np.mean((u - 2 / 3) ** 2))))


def test_xd_stable_B1_values(paper):
    m = metrics_of(paper, "Xd", "stable")
    assert m.overflow_rate == 0.0
    assert m.utilization.tolist() == pytest.approx([0.8, 0.75])
    assert m.unused.tolist() == [2.0, 6.0]
    assert m.fragmentation == pytest.approx(0.03125)
    assert m.node_utilization.shape == (2, 2)


def test_xd_pic_B1_values(paper):
    m = metrics_of(paper, "Xd", "pic")
    assert m.overflow_rate == pytest.approx(0.125)
    assert m.violation == 0.0                                          # 0,125 <= eps_pic = 0,15
    assert m.utilization.tolist() == pytest.approx([1.0, 22 / 24])     # CPU 100 %, RAM 91,7 %


def test_fragmentation_bounded_by_unused_fraction(paper):
    """B1 §5.3 : F <= 1 - moyenne(U) (F est une partie de l'inutilisé)."""
    for key in HAND:
        m = metrics_of(paper, *key)
        assert 0.0 <= m.fragmentation <= 1.0 - m.utilization.mean() + 1e-12


def test_change_metrics_B1_s11_5(paper):
    xc = change_metrics(paper, PAPER_FLEETS["Xc"])
    assert (xc.delta_plus, xc.delta_minus, xc.l1) == (2, 2, 4)
    assert xc.delta_normalized == pytest.approx(4 / 9, abs=1e-12)
    assert xc.delta_cost == pytest.approx(0.16)
    assert xc.weighted_rotation == pytest.approx(0.64)
    xd = change_metrics(paper, PAPER_FLEETS["Xd"])
    assert (xd.delta_plus, xd.delta_minus, xd.l1) == (1, 2, 3)
    assert xd.delta_normalized == pytest.approx(3 / 9)
    assert xd.delta_cost == pytest.approx(0.04)
    assert xd.weighted_rotation == pytest.approx(0.68)
    assert xd.theta is None                                            # gammas non fournis : rien d'inventé


def test_change_metrics_theta_and_missing_x0(paper):
    xd = change_metrics(paper, PAPER_FLEETS["Xd"], gamma_plus=2.0, gamma_minus=1.0)
    assert xd.theta == pytest.approx(2 * 1 + 1 * 2)
    assert xd.theta_normalized == pytest.approx(4 / (2 * 6 + 1 * 3))   # Theta_max = g+ Nmax + g- sum n0
    t = InstanceType("a", 1.0, 4, 8)
    w = Workload("w", 1, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    no_x0 = Instance(types=(t,), workloads=(w,), scenarios=(sc,))
    assert change_metrics(no_x0, (1,)) is None                         # B1 §3.4 : terme neutralisé


def test_violation_formula_and_budget(paper):
    ms = [metrics_of(paper, "Xd", "stable"), metrics_of(paper, "Xd", "pic")]
    assert fleet_violation(paper, ms) == 0.0
    # eps_pic = 0,15 : une valeur O = 0,5 donnerait (0,5 - 0,15) / 0,85
    assert (0.5 - 0.15) / (1 - 0.15) == pytest.approx(0.41176, abs=1e-5)
    ms_bad = [metrics_of(paper, "Xa", "stable")]
    assert fleet_violation(paper, ms_bad) == pytest.approx(0.2431, abs=1e-4)
    cost = paper.cost(PAPER_FLEETS["Xd"])
    assert fleet_violation(paper, ms, cost=cost, budget=0.40) == pytest.approx((0.48 - 0.40) / 2.16)
    with pytest.raises(Exception):
        fleet_violation(paper, ms, budget=0.4)                         # cost manquant


# --------------------------------------------------------------------------- #
# Vérification exhaustive des affirmations de B1 §11 (indépendante de Place)
# --------------------------------------------------------------------------- #
def brute_min_overflow(paper, fleet, scen):
    nt = node_types_from_composition(PAPER_FLEETS[fleet])
    s = paper.get_scenario(scen)
    cap = paper.allocatable[nt][:, :2]
    dem = s.demand[:, :2]
    total = dem.sum(axis=0)
    best = np.inf
    for a in itertools.product(range(-1, len(nt)), repeat=s.n_pods):
        a = np.array(a)
        load = np.zeros_like(cap)
        for r, b in enumerate(a):
            if b >= 0:
                load[b] += dem[r]
        if np.any(load > cap + 1e-9):
            continue
        best = min(best, float((dem[a < 0].sum(axis=0) / total).sum() / 2))
    return best


@pytest.mark.parametrize("fleet,scen,expected", [
    ("Xa", "stable", 0.2431), ("Xb", "stable", 0.1181), ("Xc", "stable", 0.0), ("Xd", "stable", 0.0),
    ("Xd", "pic", 0.125),
])
def test_B1_claims_confirmed_by_exhaustive_search(paper, fleet, scen, expected):
    assert brute_min_overflow(paper, fleet, scen) == pytest.approx(expected, abs=1e-4)


def test_B1_s11_4_xc_pic_overflow_is_not_the_minimum(paper):
    """B1 §11.4 donne O_pic(Xc) = 0,125 (débordement de w6) : placement VALIDE mais non optimal.

    Le minimum exact est 0,0833 (débordement de w4 seul). La preuve de B1 (« au moins un pod
    déborde ») reste vraie ; seule la valeur 0,125 n'est pas le minimum. Conséquence : B1 §11.6
    (« même débordement au pic » pour Xc et Xd) est inexact, Xc déborde moins que Xd.
    """
    assert metrics_of(paper, "Xc", "pic").overflow_rate == pytest.approx(0.125)       # placement de B1
    best = brute_min_overflow(paper, "Xc", "pic")
    assert best > 0                                                                    # la preuve tient
    assert best == pytest.approx(0.0833, abs=1e-4)
    assert best < brute_min_overflow(paper, "Xd", "pic")


@pytest.mark.parametrize("fleet", ["Xa", "Xb", "Xc", "Xd"])
@pytest.mark.parametrize("scen", ["stable", "pic"])
@pytest.mark.parametrize("policy", ["ff", "ffd"])
def test_place_is_valid_and_never_beats_optimum(paper, fleet, scen, policy):
    nt = node_types_from_composition(PAPER_FLEETS[fleet])
    a = place(paper, nt, scen, policy)
    assert validate_placement(paper, nt, scen, a)
    assert overflowed_pods_that_fit(paper, nt, scen, a) == []          # maximalité
    o = compute_scenario_metrics(paper, nt, scen, a).overflow_rate
    if len(nt) <= 3 or scen == "stable":
        assert o >= brute_min_overflow(paper, fleet, scen) - 1e-9


def test_ffd_reproduces_B1_stable_overflows(paper):
    for fleet, expected in (("Xa", 0.2431), ("Xb", 0.1181), ("Xc", 0.0), ("Xd", 0.0)):
        nt = node_types_from_composition(PAPER_FLEETS[fleet])
        m = compute_scenario_metrics(paper, nt, "stable", place(paper, nt, "stable"))
        assert m.overflow_rate == pytest.approx(expected, abs=1e-4)


def test_place_heuristic_gap_is_visible_at_peak(paper):
    """Mesuré : FFD sur Xd au pic déborde w7 (O = 0,1667) alors que l'optimum est 0,125."""
    nt = node_types_from_composition(PAPER_FLEETS["Xd"])
    a = place(paper, nt, "pic", "ffd")
    assert compute_scenario_metrics(paper, nt, "pic", a).overflow_rate == pytest.approx(1 / 6, abs=1e-4)


def test_place_is_deterministic_and_does_not_open_nodes(paper):
    nt = node_types_from_composition((1, 1, 1))
    first = place_all(paper, nt)
    second = place_all(paper, nt)
    assert all(np.array_equal(x, y) for x, y in zip(first.assignments, second.assignments))
    assert first.n_nodes == 3 == second.n_nodes


def test_place_rejects_bad_inputs(paper):
    with pytest.raises(Exception):
        place(paper, [0, 5], "stable")                                  # type inexistant
    with pytest.raises(Exception):
        place(paper, [0], "stable", policy="best")
    with pytest.raises(Exception):
        place(paper, [0], "inconnu")


# --------------------------------------------------------------------------- #
# Validateur indépendant
# --------------------------------------------------------------------------- #
def test_validator_detects_cpu_and_ram_violations(paper):
    nt = [0]                                                           # un T1 (2 vCPU, 8 GiB)
    rep = validate_placement(paper, nt, "stable", [0, 0, 0, 0, 0])     # tout sur un nœud
    assert not rep.valid and any("(7)" in v for v in rep.violations) and any("(8)" in v for v in rep.violations)
    with pytest.raises(InvalidPlacementError):
        rep.raise_if_invalid()


def test_validator_exact_equality_is_feasible(paper):
    nt = [1]                                                           # T2 (4 vCPU, 8 GiB)
    assert validate_placement(paper, nt, "stable", [0, 0, OVF, OVF, OVF]).valid      # w1+w2 = (4;8) pile


def test_validator_malformed_inputs(paper):
    nt = [0, 0]
    assert not validate_placement(paper, nt, "stable", [0, 0, 0]).valid              # mauvaise longueur
    assert not validate_placement(paper, nt, "stable", [0, 0, 0, 0, 7]).valid        # nœud inexistant
    assert not validate_placement(paper, nt, "stable", [0, 0, 0, 0, -2]).valid
    assert not validate_placement(paper, nt, "stable", np.array([0.0, 0, 0, 0, 0])).valid   # non entier
    assert not validate_placement(paper, [9], "stable", [OVF] * 5).valid             # type de nœud invalide
    assert validate_placement(paper, [], "stable", [OVF] * 5).valid                  # flotte vide


def test_validator_fleet_limits(paper):
    sol = Solution(np.zeros(7, dtype=int), (np.full(5, OVF), np.full(7, OVF)))       # 7 nœuds > N_max = 6
    rep = validate_solution(paper, sol)
    assert not rep.valid and any("N_max" in v for v in rep.violations)
    with pytest.raises(InvalidPlacementError):
        evaluate(paper, sol)
    assert not validate_solution(paper, Solution([0], (np.full(5, OVF),))).valid     # 1 affectation pour 2 scénarios


def _special_instance(bandwidth=False):
    types = (
        InstanceType("a", 1.0, 4, 8, bw_mbps=1000.0, max_pods=2, labels={"x=1"}),
        InstanceType("b", 2.0, 8, 16, bw_mbps=2000.0, max_count=1),
    )
    ws = (
        Workload("p0", 0.1, 0.1, bw_mbps=400.0), Workload("p1", 0.1, 0.1, bw_mbps=400.0),
        Workload("p2", 0.1, 0.1, bw_mbps=400.0), Workload("sel", 0.1, 0.1, node_selector={"x=1"}),
    )
    sc = Scenario.from_base_requests("s", ws, range(4), 0.0)
    return Instance(types=types, workloads=ws, scenarios=(sc,), bandwidth_active=bandwidth)


def test_validator_max_pods_eligibility_and_type_quota():
    inst = _special_instance()
    assert not validate_placement(inst, [0], "s", [0, 0, 0, OVF]).valid              # 3 pods > pi = 2
    assert validate_placement(inst, [0], "s", [0, 0, OVF, OVF]).valid
    rep = validate_placement(inst, [1], "s", [OVF, OVF, OVF, 0])                     # 'sel' non éligible à b
    assert not rep.valid and any("(9)" in v for v in rep.violations)
    sol = Solution([1, 1], (np.full(4, OVF),))                                       # quota n_b <= 1
    assert not validate_solution(inst, sol).valid


def test_bandwidth_constraint_8bis_in_validator_and_place():
    off, on = _special_instance(False), _special_instance(True)
    nt = [1]                                                                         # un type b (BW 2000)
    assert validate_placement(off, nt, "s", [0, 0, 0, OVF]).valid                    # BW ignorée si inactive (sel non éligible à b)
    nt = [0]                                                                         # un type a (BW 1000, 2 pods max)
    assert validate_placement(on, nt, "s", [0, 0, OVF, OVF]).valid                   # 800 <= 1000
    types3 = [0, 0]
    assert validate_placement(on, types3, "s", [0, 0, 1, OVF]).valid
    # 3 pods de 400 Mbps sur un type b (2000 Mbps) : OK ; mais sur deux pods + un trop gros -> test dédié
    t = (InstanceType("a", 1.0, 4, 8, bw_mbps=1000.0),)
    ws = tuple(Workload(f"p{i}", 0.1, 0.1, bw_mbps=400.0) for i in range(3))
    sc = Scenario.from_base_requests("s", ws, range(3), 0.0)
    inst = Instance(types=t, workloads=ws, scenarios=(sc,), bandwidth_active=True)
    rep = validate_placement(inst, [0], "s", [0, 0, 0])                              # 1200 > 1000
    assert not rep.valid and any("(8bis)" in v for v in rep.violations)
    a = place(inst, [0], "s")                                                        # Place respecte (8bis)
    assert (a == OVF).sum() == 1 and validate_placement(inst, [0], "s", a).valid


def test_place_respects_max_pods_and_eligibility():
    inst = _special_instance()
    a = place(inst, [0, 0], "s", "ff")                                               # 2 nœuds a (2 pods max chacun)
    assert validate_placement(inst, [0, 0], "s", a).valid
    assert (a >= 0).sum() == 4                                                       # 2 + 2
    a = place(inst, [1], "s", "ff")                                                  # seul b : 'sel' (rang 3) déborde
    assert a.tolist() == [0, 0, 0, OVF]


def test_validator_matches_independent_numpy_check_on_random_assignments():
    """Fuzz : le validateur coïncide avec un recalcul NumPy indépendant (7)-(10)."""
    inst = _special_instance(bandwidth=True)
    rng = np.random.default_rng(0)
    s = inst.get_scenario("s")
    for _ in range(400):
        nt = rng.integers(0, 2, size=int(rng.integers(0, 4)))
        a = rng.integers(-1, len(nt), size=s.n_pods) if len(nt) else np.full(s.n_pods, OVF)
        ok = True
        for b in range(len(nt)):
            idx = np.flatnonzero(a == b)
            j = nt[b]
            ok &= bool(np.all(s.demand[idx][:, :3].sum(axis=0) <= inst.allocatable[j] + 1e-9)) if idx.size else True
            ok &= bool(np.all(inst.eligibility[s.members[idx], j])) if idx.size else True
            ok &= idx.size <= inst.max_pods[j]
        assert validate_placement(inst, nt, "s", a).valid == ok


# --------------------------------------------------------------------------- #
# Cas limites des métriques
# --------------------------------------------------------------------------- #
def test_empty_fleet_everything_overflows(paper):
    nt = np.zeros(0, dtype=int)
    sol = place_all(paper, nt)
    ev = evaluate(paper, sol)
    m = ev.by_scenario("stable")
    assert ev.cost == 0.0 and ev.n_nodes == 0
    assert m.overflow_rate == 1.0 and m.overflow_pods == 5
    assert m.utilization.tolist() == [0.0, 0.0] and m.fragmentation == 0.0
    assert m.sigma.tolist() == [0.0, 0.0] and m.active_nodes == 0
    assert ev.violation == 1.0                                                       # eps_stable = 0 -> v = O = 1


def test_empty_scenario_and_zero_demand_conventions():
    t = InstanceType("a", 1.0, 4, 8)
    ws = (Workload("idle1", 0.0, 0.0), Workload("idle2", 0.0, 0.0))
    empty = Scenario("vide", np.array([], dtype=int), np.zeros((0, 3)), 0.0)
    zero = Scenario.from_base_requests("zero", ws, [0, 1], 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(empty, zero))
    for name in ("vide", "zero"):
        a = place(inst, [0], name)
        m = compute_scenario_metrics(inst, [0], name, a)
        assert m.overflow_rate == 0.0 and np.isnan(m.fragmentation)                  # kappa non défini
        assert m.utilization.tolist() == [0.0, 0.0] and m.violation == 0.0
    assert compute_scenario_metrics(inst, [0], "zero", place(inst, [0], "zero")).active_nodes == 1


def test_unused_empty_nodes_count_in_fragmentation(paper):
    """B1 §5.3 somme sur tous les nœuds de B(X), y compris vides."""
    nt = node_types_from_composition((0, 3, 0))
    a = [0, 0, 1, 2, 1]
    base = compute_scenario_metrics(paper, nt, "stable", a).fragmentation
    assert base == pytest.approx(0.5 * (4 / 3) / 12)
    # + un T1 vide (2 vCPU, 8 GiB) : q = min(2, 8/2.25) = 2 -> RAM échouée 8 - 4,5 = 3,5 GiB
    with_empty = compute_scenario_metrics(paper, list(nt) + [0], "stable", a).fragmentation
    assert with_empty == pytest.approx(0.5 * ((4 / 3) / 14 + 3.5 / 32))


def test_fragmentation_with_three_dimensions_hand_computed():
    t = InstanceType("t", 1.0, 4, 8, bw_mbps=1000.0)
    w = Workload("p1", 1, 2, bw_mbps=100.0)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    inst = Instance(types=(t,), workloads=(w,), scenarios=(sc,), bandwidth_active=True)
    m = compute_scenario_metrics(inst, [0], "s", [0])
    # kappa = (1, 2, 100) ; r = (3, 6, 900) ; q = min(3, 3, 9) = 3 ; g = (0, 0, 600) ; F = (1/3)(600/1000)
    assert m.fragmentation == pytest.approx(0.2)
    assert m.dims == ("cpu", "ram", "bw")
    assert m.utilization.tolist() == pytest.approx([0.25, 0.25, 0.1])


def test_margin_excess(paper):
    nt, a = HAND[("Xd", "pic")]
    m = compute_scenario_metrics(paper, nt, "pic", a, u_star=0.9)
    assert m.margin_excess.tolist() == pytest.approx([0.1, 22 / 24 - 0.9])
    with pytest.raises(Exception):
        compute_scenario_metrics(paper, nt, "pic", a, u_star=1.0)
    assert compute_scenario_metrics(paper, nt, "pic", a).margin_excess is None


def test_evaluate_rejects_invalid_and_logs_record(paper):
    bad = Solution([0], (np.zeros(5, dtype=int), np.zeros(7, dtype=int)))            # tout sur un T1 : capacité violée
    with pytest.raises(InvalidPlacementError, match="violation"):
        evaluate(paper, bad)
    sol = place_all(paper, node_types_from_composition(PAPER_FLEETS["Xd"]))
    rec = evaluate(paper, sol, elapsed_s=0.01).log_record()
    for key in ("X", "C", "V", "O[stable]", "F[pic]", "U[stable]", "sigma[pic]", "L1", "time_s"):
        assert key in rec
    assert rec["X"] == [1, 0, 1] and rec["C"] == pytest.approx(0.48)


# --------------------------------------------------------------------------- #
# Propriétés sur instances aléatoires (2 ou 3 dimensions)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bw", [False, True])
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("policy", ["ff", "ffd"])
def test_random_instances_invariants(seed, bw, policy):
    inst = generate_instance(GeneratorConfig(seed=seed, bandwidth_active=bw, n_variable_states=2,
                                             heterogeneity=1.0, generate_x0=True))
    rng = np.random.default_rng(seed)
    ref, _ = inst.reference_fleet()
    comp = np.maximum(ref - rng.integers(0, 2, size=inst.k), 0)
    nt = node_types_from_composition(comp)
    sol = place_all(inst, nt, policy)
    assert validate_solution(inst, sol)
    ev = evaluate(inst, sol)
    for s, a, m in zip(inst.scenarios, sol.assignments, ev.scenarios):
        assert overflowed_pods_that_fit(inst, nt, s, a) == []
        assert np.allclose(m.placed + m.overflow, m.demand)
        loads = compute_scenario_metrics(inst, nt, s, a).node_utilization * inst.allocatable[nt][:, list(inst.active_dims)]
        assert np.allclose(loads.sum(axis=0), m.placed)
        assert 0.0 <= m.overflow_rate <= 1.0 and 0.0 <= m.overflow_pods_rate <= 1.0
        assert np.all((m.utilization >= 0) & (m.utilization <= 1 + 1e-9))
        assert np.all(m.sigma >= 0) and np.all(m.node_utilization <= 1 + 1e-9)
        assert 0.0 <= m.fragmentation <= 1.0 - m.utilization.mean() + 1e-9
        assert m.dims == (("cpu", "ram", "bw") if bw else ("cpu", "ram"))
    assert ev.cost == pytest.approx(float(inst.prices @ comp))
    assert ev.changes is not None and ev.changes.l1 == int(np.abs(comp - inst.x0).sum())


def test_ffd_never_worse_than_ff_in_count_is_not_assumed():
    """Aucune supériorité de FFD n'est supposée : on vérifie seulement que les deux sont valides."""
    inst = generate_instance(GeneratorConfig(seed=3))
    nt = node_types_from_composition(inst.reference_fleet()[0])
    for policy in ("ff", "ffd"):
        assert validate_solution(inst, place_all(inst, nt, policy))


def test_insufficient_capacity_produces_overflow():
    t = InstanceType("a", 1.0, 4, 8)
    ws = tuple(Workload(f"w{i}", 3, 1) for i in range(3))
    sc = Scenario.from_base_requests("s", ws, range(3), 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(sc,))
    a = place(inst, [0, 0], "s")
    m = compute_scenario_metrics(inst, [0, 0], "s", a)
    assert (a == OVF).sum() == 1 and m.overflow_rate == pytest.approx((3 / 9 + 1 / 3) / 2)


def test_performance_large_instance():
    inst = generate_instance(GeneratorConfig(seed=0, n_base_workloads=2000, n_variable_states=2,
                                             bandwidth_active=True))
    nt = node_types_from_composition(inst.reference_fleet()[0])
    start = time.perf_counter()
    sol = place_all(inst, nt, "ffd")
    ev = evaluate(inst, sol)
    assert time.perf_counter() - start < 15.0
    assert ev.n_nodes == nt.size


def test_overflow_rate_three_dimensions_hand_computed():
    """O_s = (1/|D|) sum_d o^d / D^d avec |D| = 3 (B1-bis) ; bande passante seule cause du débordement."""
    t = InstanceType("a", 1.0, 4, 8, bw_mbps=1000.0)
    ws = (Workload("p1", 1, 2, bw_mbps=500.0), Workload("p2", 1, 2, bw_mbps=500.0),
          Workload("p3", 2, 1, bw_mbps=500.0))
    sc = Scenario.from_base_requests("s", ws, range(3), 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(sc,), bandwidth_active=True)
    a = place(inst, [0], "s", "ff")
    assert a.tolist() == [0, 0, OVF]                                  # p3 : BW 1500 > 1000
    m = compute_scenario_metrics(inst, [0], "s", a)
    # D = (4; 5; 1500) ; o = (2; 1; 500) -> O = (1/3)(2/4 + 1/5 + 500/1500)
    assert m.overflow_rate == pytest.approx((0.5 + 0.2 + 1 / 3) / 3)
    assert m.violation == pytest.approx(m.overflow_rate)              # eps = 0
    off = Instance(types=(t,), workloads=ws, scenarios=(sc,), bandwidth_active=False)
    m2 = compute_scenario_metrics(off, [0], "s", place(off, [0], "s", "ff"))
    assert m2.overflow_rate == 0.0 and m2.dims == ("cpu", "ram")      # sans BW : (4;5) <= (4;8) tout tient
