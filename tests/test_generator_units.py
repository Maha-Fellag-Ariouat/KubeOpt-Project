"""Tests du générateur (reproductibilité, validité, performance) et de la vue entière."""
import time

import numpy as np
import pytest

from kubeopt import (
    BW, CPU, RAM, DataValidationError, GeneratorConfig, generate_instance, paper_example, to_integer_view,
)
from kubeopt.units import ceil_scaled, floor_scaled


def _signature(inst):
    return (
        tuple(t.price for t in inst.types),
        tuple((w.cpu, w.ram_gib, w.bw_mbps) for w in inst.workloads),
        tuple((s.name, s.members.tolist(), s.demand.tolist(), s.epsilon) for s in inst.scenarios),
        None if inst.x0 is None else inst.x0.tolist(),
    )


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #
def test_same_seed_same_instance():
    cfg = GeneratorConfig(seed=42, n_variable_states=3, generate_x0=True, bandwidth_active=True)
    assert _signature(generate_instance(cfg)) == _signature(generate_instance(cfg))


def test_different_seeds_differ():
    a = generate_instance(GeneratorConfig(seed=1))
    b = generate_instance(GeneratorConfig(seed=2))
    assert _signature(a) != _signature(b)


def test_streams_are_independent():
    """Changer le nombre d'états « variable » ne modifie ni catalogue ni workloads."""
    a = generate_instance(GeneratorConfig(seed=7, n_variable_states=1))
    b = generate_instance(GeneratorConfig(seed=7, n_variable_states=4))
    assert _signature(a)[:2] == _signature(b)[:2]
    assert [s.name for s in b.scenarios] == ["stable", "pic", "variable_1", "variable_2", "variable_3", "variable_4"]


def test_bandwidth_flag_does_not_shift_other_draws():
    a = generate_instance(GeneratorConfig(seed=3, bandwidth_active=False))
    b = generate_instance(GeneratorConfig(seed=3, bandwidth_active=True))
    assert [w.cpu for w in a.workloads] == [w.cpu for w in b.workloads]
    assert [w.ram_gib for w in a.workloads] == [w.ram_gib for w in b.workloads]


# --------------------------------------------------------------------------- #
# Validité structurelle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("bw", [False, True])
def test_generated_instances_are_valid_and_placeable(seed, bw):
    inst = generate_instance(GeneratorConfig(seed=seed, bandwidth_active=bw, n_variable_states=2,
                                             heterogeneity=1.2))
    for s in inst.scenarios:
        assert inst.unplaceable_workloads(s).size == 0
        assert np.all(s.demand[:, [CPU, RAM]] > 0)
    # Le scénario stable est inclus dans le pic ; scénarios jamais additionnés.
    stable, pic = inst.get_scenario("stable"), inst.get_scenario("pic")
    assert set(stable.members) <= set(pic.members)
    # La flotte de référence satisfait la condition agrégée (5) dans chaque scénario.
    fleet, excluded = inst.reference_fleet()
    assert excluded.size == 0
    for s in inst.scenarios:
        assert inst.aggregate_capacity_ok(fleet, s)
    assert 0 < inst.cost_upper_bound() < np.inf


def test_bandwidth_inactive_gives_zero_bw_demand_and_two_dims():
    inst = generate_instance(GeneratorConfig(seed=0, bandwidth_active=False))
    assert inst.n_dims == 2
    assert all(w.bw_mbps == 0.0 for w in inst.workloads)


def test_bandwidth_active_gives_positive_bw():
    inst = generate_instance(GeneratorConfig(seed=0, bandwidth_active=True))
    assert inst.n_dims == 3
    assert all(w.bw_mbps > 0 for w in inst.workloads)
    assert np.all(inst.allocatable[:, BW] > 0)


def test_demands_vary_between_scenarios():
    inst = generate_instance(GeneratorConfig(seed=11, n_variable_states=1))
    stable, var = inst.get_scenario("stable"), inst.get_scenario("variable")
    common = np.intersect1d(stable.members, var.members)
    d_s = stable.demand[np.searchsorted(stable.members, common)]
    d_v = var.demand[np.searchsorted(var.members, common)]
    assert not np.array_equal(d_s, d_v)                     # d_{i,s} dépend de s


def test_heterogeneity_increases_dispersion():
    def spread(h):
        inst = generate_instance(GeneratorConfig(seed=9, n_base_workloads=200, heterogeneity=h))
        cpus = np.array([w.cpu for w in inst.workloads])
        return cpus.std() / cpus.mean()
    assert spread(1.5) > spread(0.1)


def test_default_epsilons_are_zero_no_invented_values():
    inst = generate_instance(GeneratorConfig(seed=0))
    assert all(s.epsilon == 0.0 for s in inst.scenarios)


def test_optional_x0_and_nmax():
    inst = generate_instance(GeneratorConfig(seed=0, generate_x0=True, n_max_total=40))
    assert inst.x0 is not None and inst.x0.size == inst.k
    assert inst.cost_upper_bound() == pytest.approx(40 * inst.prices.max())
    assert generate_instance(GeneratorConfig(seed=0)).x0 is None


def test_reservations_applied_in_generated_catalog():
    from kubeopt import Reservation
    cfg = GeneratorConfig(seed=0, reservation=Reservation(cpu_fixed=0.1, ram_rate=0.05),
                          daemonset_overhead=(0.2, 0.5, 0.0))
    inst = generate_instance(cfg)
    t = inst.types[0]
    assert inst.allocatable[0, CPU] == pytest.approx(t.vcpu - 0.1 - 0.2)
    assert inst.allocatable[0, RAM] == pytest.approx(t.ram_gib * 0.95 - 0.5)


@pytest.mark.parametrize("kwargs", [
    dict(seed=-1), dict(n_base_workloads=0), dict(peak_extra_fraction=-0.1), dict(max_pod_fraction=0.0),
    dict(variable_scale_range=(1.3, 0.7)), dict(peak_scale=0), dict(sizes_vcpu=()), dict(cpu_median=0),
    dict(heterogeneity=-1), dict(n_variable_states=-1),
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(DataValidationError):
        generate_instance(GeneratorConfig(**kwargs))


def test_catalog_too_small_for_grid_rejected():
    with pytest.raises(DataValidationError, match="Plafond"):
        generate_instance(GeneratorConfig(sizes_vcpu=(1,), max_pod_fraction=0.01))


# --------------------------------------------------------------------------- #
# Performance (borne large, non flaky)
# --------------------------------------------------------------------------- #
def test_generation_performance():
    start = time.perf_counter()
    inst = generate_instance(GeneratorConfig(seed=0, n_base_workloads=2000, n_variable_states=3,
                                             bandwidth_active=True))
    inst.reference_fleet()
    for s in inst.scenarios:
        inst.unplaceable_workloads(s)
    assert time.perf_counter() - start < 5.0
    assert inst.n_workloads == 2800


# --------------------------------------------------------------------------- #
# Vue entière (CP-SAT)
# --------------------------------------------------------------------------- #
def test_scaled_rounding_is_conservative_and_tolerant():
    assert floor_scaled(0.1, 1000) == 100 and ceil_scaled(0.1, 1000) == 100     # bruit flottant absorbé
    assert floor_scaled(2.3, 1024) == 2355 and ceil_scaled(2.3, 1024) == 2356   # 2355.2
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(DataValidationError):
            floor_scaled(bad, 1000)
        with pytest.raises(DataValidationError):
            ceil_scaled(bad, 1000)


def test_integer_view_paper_example_exact():
    view = to_integer_view(paper_example())
    assert view.capacity[0].tolist() == [2000, 8192, 0]
    assert view.capacity[2].tolist() == [8000, 16384, 0]
    assert view.demands[0][0].tolist() == [3000, 2048, 0]                       # w1 stable
    assert view.price_micro.tolist() == [120000, 200000, 360000]
    assert view.active_dims == (CPU, RAM)


def test_integer_view_is_conservative_on_random_instances():
    for seed in range(10):
        inst = generate_instance(GeneratorConfig(seed=seed, bandwidth_active=True, n_variable_states=2))
        view = to_integer_view(inst)
        factors = np.array([1000.0, 1024.0, 1.0])
        assert np.all(view.capacity <= inst.allocatable * factors + 1e-6)
        for s, dem in zip(inst.scenarios, view.demands):
            assert np.all(dem >= s.demand * factors - 1e-6)
            assert dem.shape == s.demand.shape
        assert np.all(view.price_micro > 0)
