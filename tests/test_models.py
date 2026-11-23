"""Tests des modèles (B2-a) : éq. (1), (5), (11), (12), kappa_s, C_ub, validations et cas limites."""
from dataclasses import replace

import numpy as np
import pytest

from kubeopt import (
    BW, CPU, RAM, DataValidationError, Instance, InstanceType, PAPER_FLEETS, Reservation,
    Scenario, Workload, paper_example,
)


# --------------------------------------------------------------------------- #
# Éq. (1) : capacité utilisable
# --------------------------------------------------------------------------- #
def test_allocatable_equation_1():
    res = Reservation(cpu_fixed=0.1, cpu_rate=0.05, ram_fixed=0.25, ram_rate=0.06)
    t = InstanceType("t", price=1.0, vcpu=4, ram_gib=16, reservation=res)
    a = t.allocatable(daemonset_overhead=(0.2, 0.5, 0.0))
    assert a[CPU] == pytest.approx(4 - (0.1 + 0.05 * 4) - 0.2)         # 3.5
    assert a[RAM] == pytest.approx(16 - (0.25 + 0.06 * 16) - 0.5)      # 14.29


def test_allocatable_not_above_nominal():
    t = InstanceType("t", price=1.0, vcpu=4, ram_gib=16, reservation=Reservation(cpu_fixed=0.5))
    assert np.all(t.allocatable()[:2] <= t.nominal()[:2])


def test_nonpositive_allocatable_rejected():
    t = InstanceType("t", price=1.0, vcpu=2, ram_gib=8, reservation=Reservation(cpu_fixed=2.0))
    w = Workload("w", 0.5, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    with pytest.raises(DataValidationError, match="capacité utilisable"):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,))


def test_bandwidth_active_requires_positive_bw_capacity():
    t = InstanceType("t", price=1.0, vcpu=2, ram_gib=8, bw_mbps=0.0)
    w = Workload("w", 0.5, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,), bandwidth_active=True)


@pytest.mark.parametrize("kwargs", [
    dict(cpu_rate=1.0), dict(cpu_fixed=-1.0), dict(ram_rate=-0.1), dict(bw_fixed=float("nan")),
])
def test_reservation_validation(kwargs):
    with pytest.raises(DataValidationError):
        Reservation(**kwargs)


# --------------------------------------------------------------------------- #
# Validation des types, workloads, scénarios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    dict(price=0.0), dict(price=-1.0), dict(vcpu=0), dict(ram_gib=float("inf")),
    dict(arch="ppc64"), dict(max_pods=0), dict(max_count=-1), dict(gpus=-1),
])
def test_instance_type_validation(kwargs):
    base = dict(name="t", price=1.0, vcpu=2, ram_gib=8)
    base.update(kwargs)
    with pytest.raises(DataValidationError):
        InstanceType(**base)


@pytest.mark.parametrize("kwargs", [
    dict(cpu=-1), dict(ram_gib=float("nan")), dict(bw_mbps=-5), dict(arch="mips"), dict(gpus=-2),
])
def test_workload_validation(kwargs):
    base = dict(name="w", cpu=1, ram_gib=1)
    base.update(kwargs)
    with pytest.raises(DataValidationError):
        Workload(**base)


def test_scenario_validation():
    ok = dict(name="s", members=np.array([0, 1]), demand=np.ones((2, 3)), epsilon=0.1)
    Scenario(**ok)
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "epsilon": 1.0})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "epsilon": -0.1})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "members": np.array([0, 0])})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "demand": np.ones((3, 3))})            # mauvaise dimension
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "demand": np.ones((2, 2))})            # il manque la colonne BW
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "demand": np.array([[1, 1, 1], [np.nan, 1, 1]])})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "demand": np.array([[1, 1, 1], [-1, 1, 1]])})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "members": np.array([0.5, 1.0])})
    with pytest.raises(DataValidationError):
        Scenario(**{**ok, "tau_hours": 0})


def test_scenario_arrays_are_immutable_and_copied():
    demand = np.ones((2, 3))
    sc = Scenario("s", np.array([0, 1]), demand, 0.0)
    demand[0, 0] = 99.0
    assert sc.demand[0, 0] == 1.0
    with pytest.raises(ValueError):
        sc.demand[0, 0] = 5.0


def test_empty_scenario_allowed_but_kappa_undefined():
    inst = paper_example()
    empty = Scenario("vide", np.array([], dtype=int), np.zeros((0, 3)), 0.0)
    inst2 = replace(inst, scenarios=inst.scenarios + (empty,))
    assert inst2.get_scenario("vide").total_demand().tolist() == [0.0, 0.0, 0.0]
    assert inst2.aggregate_capacity_ok((0, 0, 0), "vide")          # demande nulle, flotte vide : OK
    assert inst2.unplaceable_workloads("vide").size == 0
    with pytest.raises(DataValidationError, match="kappa"):
        inst2.reference_block("vide")


def test_instance_structural_validation():
    t = InstanceType("t", price=1.0, vcpu=4, ram_gib=8)
    w = Workload("w", 1, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    with pytest.raises(DataValidationError):
        Instance(types=(), workloads=(w,), scenarios=(sc,))
    with pytest.raises(DataValidationError):
        Instance(types=(t, t), workloads=(w,), scenarios=(sc,))                 # noms dupliqués
    bad = Scenario("s2", np.array([3]), np.ones((1, 3)), 0.0)                   # indice hors bornes
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(bad,))
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,), x0=np.array([1, 2]))   # longueur
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,), x0=np.array([-1]))
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,), n_max_total=0)
    with pytest.raises(DataValidationError):
        Instance(types=(t,), workloads=(w,), scenarios=(sc,), daemonset_overhead=(0, 0))


def test_from_base_requests_out_of_range():
    with pytest.raises(DataValidationError):
        Scenario.from_base_requests("s", [Workload("w", 1, 1)], [1], 0.0)


# --------------------------------------------------------------------------- #
# Éligibilité e_ij (éq. 11)
# --------------------------------------------------------------------------- #
def _eligibility_instance(workload):
    types = (
        InstanceType("x86", 1.0, 4, 8, arch="x86_64", labels={"disk=ssd"}),
        InstanceType("arm", 1.0, 4, 8, arch="arm64", taints={"dedicated=arm"}),
        InstanceType("gpu", 2.0, 4, 8, arch="x86_64", gpus=1),
    )
    sc = Scenario.from_base_requests("s", [workload], [0], 0.0)
    return Instance(types=types, workloads=(workload,), scenarios=(sc,))


def test_eligibility_default_is_all_ones():
    inst = _eligibility_instance(Workload("w", 1, 1))
    # taint de « arm » non toléré -> seul facteur actif ici
    assert inst.eligibility.tolist() == [[True, False, True]]


def test_eligibility_arch_selector_taint_gpu():
    assert _eligibility_instance(Workload("w", 1, 1, arch="arm64", tolerations={"dedicated=arm"})
                                 ).eligibility.tolist() == [[False, True, False]]
    assert _eligibility_instance(Workload("w", 1, 1, node_selector={"disk=ssd"})
                                 ).eligibility.tolist() == [[True, False, False]]
    assert _eligibility_instance(Workload("w", 1, 1, gpus=1)
                                 ).eligibility.tolist() == [[False, False, True]]


def test_workload_with_no_eligible_type_is_data_error():
    with pytest.raises(DataValidationError, match="aucun type éligible"):
        _eligibility_instance(Workload("w", 1, 1, node_selector={"zone=nowhere"}))


# --------------------------------------------------------------------------- #
# Exemple B1 §11 : jeu de non-régression
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def paper():
    return paper_example()


def test_paper_example_demands_and_kappa(paper):
    d_stable = paper.get_scenario("stable").total_demand()
    d_pic = paper.get_scenario("pic").total_demand()
    assert d_stable[[CPU, RAM]].tolist() == [8.0, 18.0]
    assert d_pic[[CPU, RAM]].tolist() == [12.0, 24.0]
    assert paper.reference_block("stable").tolist() == pytest.approx([1.0, 2.25])
    assert paper.reference_block("pic").tolist() == pytest.approx([1.0, 2.0])
    # Les scénarios ne s'additionnent jamais : aucune demande de 20/42 n'apparaît.
    assert d_stable[CPU] + d_pic[CPU] == 20.0 and d_pic[CPU] != 20.0


def test_paper_example_costs(paper):
    expected = {"Xa": 0.48, "Xb": 0.40, "Xc": 0.60, "Xd": 0.48}
    for name, fleet in PAPER_FLEETS.items():
        assert paper.cost(fleet) == pytest.approx(expected[name]), name
    assert paper.cost((2, 1, 0)) == pytest.approx(0.44)                  # X0


def test_paper_example_cub_and_normalised_cost(paper):
    assert paper.cost_upper_bound() == pytest.approx(2.16)
    assert paper.cost(PAPER_FLEETS["Xc"]) / paper.cost_upper_bound() == pytest.approx(0.2778, abs=1e-4)
    assert paper.cost(PAPER_FLEETS["Xd"]) / paper.cost_upper_bound() == pytest.approx(0.2222, abs=1e-4)


def test_paper_example_total_capacity(paper):
    expected = {"Xa": (8, 32), "Xb": (8, 16), "Xc": (12, 24), "Xd": (10, 24)}
    for name, fleet in PAPER_FLEETS.items():
        assert paper.total_capacity(fleet)[[CPU, RAM]].tolist() == list(expected[name]), name
    assert paper.total_capacity(PAPER_FLEETS["Xc"])[BW] == 0.0            # BW neutralisée


def test_paper_example_aggregate_condition(paper):
    stable = {"Xa": True, "Xb": False, "Xc": True, "Xd": True}
    pic = {"Xa": False, "Xb": False, "Xc": True, "Xd": False}
    for name, fleet in PAPER_FLEETS.items():
        assert paper.aggregate_capacity_ok(fleet, "stable") is stable[name], name
        assert paper.aggregate_capacity_ok(fleet, "pic") is pic[name], name
    # (5) est nécessaire mais pas suffisante : Xa et Xc passent le test agrégé alors que le
    # placement échoue (B1 §11.2 et §11.4) ; ce point sera vérifié par Place en B2-b.


def test_paper_example_no_unplaceable_pod(paper):
    for s in ("stable", "pic"):
        assert paper.unplaceable_workloads(s).size == 0


def test_paper_example_reference_fleet(paper):
    fleet, excluded = paper.reference_fleet()
    assert fleet.tolist() == [6, 1, 0]            # w1 -> T2 (CPU 3) ; les 6 autres pods -> T1
    assert excluded.size == 0
    for s in ("stable", "pic"):
        assert paper.aggregate_capacity_ok(fleet, s)
    no_nmax = replace(paper, n_max_total=None)
    assert no_nmax.cost_upper_bound() == pytest.approx(6 * 0.12 + 0.20)


def test_paper_example_limits_and_x0(paper):
    assert paper.x0.tolist() == [2, 1, 0]
    assert paper.respects_limits((2, 2, 2))
    assert not paper.respects_limits((3, 3, 1))                           # somme 7 > N_max = 6


def test_paper_example_is_immutable(paper):
    with pytest.raises(ValueError):
        paper.allocatable[0, 0] = 0.0
    with pytest.raises(ValueError):
        paper.x0[0] = 9


# --------------------------------------------------------------------------- #
# Compositions invalides
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [(1, 2), (1, 2, 3, 4), (-1, 0, 0), (0.5, 0, 0), (float("nan"), 0, 0)])
def test_invalid_compositions_rejected(paper, bad):
    with pytest.raises(DataValidationError):
        paper.cost(bad)


def test_float_integers_accepted_and_per_type_quota():
    t1 = InstanceType("a", 1.0, 4, 8, max_count=2)
    t2 = InstanceType("b", 2.0, 8, 16)
    w = Workload("w", 1, 1)
    sc = Scenario.from_base_requests("s", [w], [0], 0.0)
    inst = Instance(types=(t1, t2), workloads=(w,), scenarios=(sc,))
    assert inst.cost(np.array([1.0, 2.0])) == pytest.approx(5.0)
    assert inst.respects_limits((2, 9))
    assert not inst.respects_limits((3, 0))


# --------------------------------------------------------------------------- #
# Cas limites de placement individuel
# --------------------------------------------------------------------------- #
def test_container_exceeding_every_server_is_reported_not_error():
    types = (InstanceType("a", 1.0, 4, 8), InstanceType("b", 2.0, 8, 16))
    ws = (Workload("big", 9, 1), Workload("small", 1, 1), Workload("ram-hog", 1, 17))
    sc = Scenario.from_base_requests("s", ws, [0, 1, 2], 0.0)
    inst = Instance(types=types, workloads=ws, scenarios=(sc,))
    assert inst.unplaceable_workloads("s").tolist() == [0, 2]
    fleet, excluded = inst.reference_fleet()
    assert excluded.tolist() == [0, 2]
    assert fleet.tolist() == [1, 0]


def test_insufficient_total_capacity_detected():
    t = InstanceType("a", 1.0, 4, 8)
    ws = tuple(Workload(f"w{i}", 3, 1) for i in range(3))          # 9 vCPU demandés
    sc = Scenario.from_base_requests("s", ws, [0, 1, 2], 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(sc,))
    assert not inst.aggregate_capacity_ok((2,), "s")                # 8 < 9
    assert inst.aggregate_capacity_ok((3,), "s")                    # (5) vrai, mais 1 pod/nœud (cf. indivisibilité)
    assert not inst.aggregate_capacity_ok((0,), "s")                # flotte vide


def test_zero_demand_pods_and_zero_utilisation():
    t = InstanceType("a", 1.0, 4, 8)
    ws = (Workload("idle", 0.0, 0.0), Workload("w", 1, 1))
    sc = Scenario.from_base_requests("s", ws, [0, 1], 0.0)
    inst = Instance(types=(t,), workloads=ws, scenarios=(sc,))
    assert inst.unplaceable_workloads("s").size == 0
    assert inst.aggregate_capacity_ok((1,), "s")


def test_highly_heterogeneous_demands():
    types = (InstanceType("s", 0.1, 2, 4), InstanceType("l", 3.0, 64, 256))
    ws = (Workload("tiny", 0.001, 0.001), Workload("huge", 60, 250))
    sc = Scenario.from_base_requests("s", ws, [0, 1], 0.0)
    inst = Instance(types=types, workloads=ws, scenarios=(sc,))
    assert inst.unplaceable_workloads("s").size == 0
    fleet, _ = inst.reference_fleet()
    assert fleet.tolist() == [1, 1]


def test_unplaceable_matches_bruteforce():
    rng = np.random.default_rng(5)
    types = tuple(InstanceType(f"t{j}", 1.0 + j, float(2 + 2 * j), float(4 + 4 * j)) for j in range(4))
    ws = tuple(Workload(f"w{i}", float(rng.uniform(0.1, 12)), float(rng.uniform(0.1, 40))) for i in range(60))
    sc = Scenario.from_base_requests("s", ws, range(60), 0.0)
    inst = Instance(types=types, workloads=ws, scenarios=(sc,))
    expected = [i for i, w in enumerate(ws)
                if not any(w.cpu <= inst.allocatable[j, CPU] and w.ram_gib <= inst.allocatable[j, RAM]
                           for j in range(inst.k))]
    assert inst.unplaceable_workloads("s").tolist() == expected


# --------------------------------------------------------------------------- #
# Extension bande passante (B1-bis)
# --------------------------------------------------------------------------- #
def _bw_instance(active: bool, bw_demand: float = 100.0):
    types = (InstanceType("a", 1.0, 4, 8, bw_mbps=1000.0), InstanceType("b", 2.0, 8, 16, bw_mbps=2000.0))
    ws = (Workload("w1", 1, 1, bw_mbps=bw_demand), Workload("w2", 1, 1, bw_mbps=bw_demand))
    sc = Scenario.from_base_requests("s", ws, [0, 1], 0.0)
    return Instance(types=types, workloads=ws, scenarios=(sc,), bandwidth_active=active)


def test_bandwidth_inactive_reduces_to_two_dimensions():
    inst = _bw_instance(False, bw_demand=10 ** 9)
    assert inst.active_dims == (CPU, RAM) and inst.n_dims == 2
    assert inst.aggregate_capacity_ok((1, 0), "s")                  # la BW énorme est ignorée
    assert inst.allocatable[:, BW].tolist() == [0.0, 0.0]
    assert inst.reference_block("s").shape == (2,)
    assert inst.unplaceable_workloads("s").size == 0


def test_bandwidth_active_adds_third_constraint():
    inst = _bw_instance(True, bw_demand=600.0)                      # 1200 Mbps demandés
    assert inst.active_dims == (CPU, RAM, BW) and inst.n_dims == 3
    assert not inst.aggregate_capacity_ok((1, 0), "s")              # 1000 < 1200
    assert inst.aggregate_capacity_ok((0, 1), "s")                  # 2000 >= 1200
    assert inst.reference_block("s").tolist() == pytest.approx([1.0, 1.0, 600.0])


def test_bandwidth_makes_pod_unplaceable():
    types = (InstanceType("a", 1.0, 4, 8, bw_mbps=1000.0),)
    ws = (Workload("w", 1, 1, bw_mbps=5000.0),)
    sc = Scenario.from_base_requests("s", ws, [0], 0.0)
    assert Instance(types=types, workloads=ws, scenarios=(sc,)).unplaceable_workloads("s").size == 0
    inst = Instance(types=types, workloads=ws, scenarios=(sc,), bandwidth_active=True)
    assert inst.unplaceable_workloads("s").tolist() == [0]
