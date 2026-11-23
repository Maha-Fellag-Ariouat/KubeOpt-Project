"""Générateur d'instances de test reproductibles (B1 §14) et exemple numérique de B1 §11.

TOUTES les données produites ici sont SYNTHÉTIQUES ou FICTIVES : les prix, tailles et
distributions de demande ne proviennent d'aucune source AWS ni d'aucune trace réelle.
Aucun résultat obtenu sur ces instances ne doit être présenté comme un résultat sur des
données réelles.

Reproductibilité : la graine est éclatée en flux indépendants (SeedSequence.spawn) pour le
catalogue, les workloads, les scénarios et X0. Changer un paramètre d'un flux (ex. le
nombre d'états « variable ») ne décale donc pas les tirages des autres.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .errors import DataValidationError
from .models import (
    BW, CPU, RAM, Instance, InstanceType, Reservation, Scenario, Workload,
)

# --------------------------------------------------------------------------- #
# Exemple numérique de B1 §11 (entièrement fictif ; capacités déjà allocatable)
# --------------------------------------------------------------------------- #
PAPER_FLEETS: Dict[str, Tuple[int, int, int]] = {
    "Xa": (4, 0, 0),
    "Xb": (0, 2, 0),
    "Xc": (0, 3, 0),
    "Xd": (1, 0, 1),
}


def paper_example() -> Instance:
    """Instance fictive de B1 §11 : jeu de test de non-régression (bande passante désactivée)."""
    types = (
        InstanceType("T1", price=0.12, vcpu=2, ram_gib=8),
        InstanceType("T2", price=0.20, vcpu=4, ram_gib=8),
        InstanceType("T3", price=0.36, vcpu=8, ram_gib=16),
    )
    requests = [(3, 2), (1, 6), (2, 3), (1, 2), (1, 5), (2, 2), (2, 4)]  # w1..w7 (vCPU, GiB)
    workloads = tuple(Workload(f"w{i + 1}", cpu=c, ram_gib=r) for i, (c, r) in enumerate(requests))
    stable = Scenario.from_base_requests("stable", workloads, range(0, 5), epsilon=0.0)
    pic = Scenario.from_base_requests("pic", workloads, range(0, 7), epsilon=0.15)
    return Instance(
        types=types,
        workloads=workloads,
        scenarios=(stable, pic),
        n_max_total=6,
        x0=np.array([2, 1, 0]),
        name="paper_example_B1_s11",
        provenance="fictif (B1 §11)",
    )


# --------------------------------------------------------------------------- #
# Générateur aléatoire
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorConfig:
    """Paramètres du générateur. Tous les défauts sont FICTIFS et à régler selon l'étude."""

    seed: int = 0
    # Catalogue synthétique : (famille, GiB par vCPU, $/vCPU·h de base) -- valeurs fictives
    families: Tuple[Tuple[str, float, float], ...] = (("c", 2.0, 0.040), ("m", 4.0, 0.048), ("r", 8.0, 0.063))
    sizes_vcpu: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    price_noise: float = 0.05              # sigma du bruit log-normal sur les prix
    bw_per_vcpu_mbps: float = 500.0        # bande passante nominale par vCPU (fictive)
    reservation: Reservation = Reservation()
    daemonset_overhead: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bandwidth_active: bool = False
    # Workloads
    n_base_workloads: int = 30             # workloads du scénario stable
    peak_extra_fraction: float = 0.4       # workloads supplémentaires du pic / n_base
    cpu_median: float = 0.5                # vCPU
    heterogeneity: float = 0.8             # sigma log-normal du CPU (hétérogénéité des demandes)
    ram_per_cpu_median: float = 2.0        # GiB/vCPU
    ram_ratio_sigma: float = 0.5
    bw_median_mbps: float = 50.0
    bw_sigma: float = 0.8
    max_pod_fraction: float = 0.5          # un pod demande au plus cette fraction du plus gros nœud
    # Scénarios
    eps_stable: float = 0.0
    eps_peak: float = 0.0                  # aucune valeur choisie : à fournir [À valider]
    eps_variable: float = 0.0
    peak_scale: float = 1.0                # facteur de demande du pic sur tous les pods
    n_variable_states: int = 1
    variable_scale_range: Tuple[float, float] = (0.7, 1.3)
    # Limites et flotte actuelle
    n_max_total: Optional[int] = None
    generate_x0: bool = False

    def validate(self) -> None:
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise DataValidationError("seed doit être un entier >= 0")
        if self.n_base_workloads < 1:
            raise DataValidationError("n_base_workloads doit être >= 1")
        if self.peak_extra_fraction < 0:
            raise DataValidationError("peak_extra_fraction doit être >= 0")
        if self.n_variable_states < 0:
            raise DataValidationError("n_variable_states doit être >= 0")
        if not self.families or not self.sizes_vcpu:
            raise DataValidationError("families et sizes_vcpu ne doivent pas être vides")
        if not 0 < self.max_pod_fraction <= 1:
            raise DataValidationError("max_pod_fraction doit être dans ]0, 1]")
        lo, hi = self.variable_scale_range
        if not 0 < lo <= hi:
            raise DataValidationError("variable_scale_range doit vérifier 0 < min <= max")
        if self.peak_scale <= 0:
            raise DataValidationError("peak_scale doit être > 0")
        for label, v in (("cpu_median", self.cpu_median), ("ram_per_cpu_median", self.ram_per_cpu_median),
                         ("bw_median_mbps", self.bw_median_mbps)):
            if v <= 0:
                raise DataValidationError(f"{label} doit être > 0")
        for label, v in (("heterogeneity", self.heterogeneity), ("ram_ratio_sigma", self.ram_ratio_sigma),
                         ("bw_sigma", self.bw_sigma), ("price_noise", self.price_noise)):
            if v < 0:
                raise DataValidationError(f"{label} doit être >= 0")


#: Pas de quantification des demandes (50 m, 0,125 GiB, 5 Mbps) et minimum = un pas.
_GRID = np.array([0.05, 0.125, 5.0])


def _quantize(values: np.ndarray, cap: np.ndarray, active_bw: bool) -> np.ndarray:
    """Écrête à cap, arrondit vers le bas sur la grille, impose au moins un pas (BW=0 si inactive)."""
    clipped = np.minimum(values, cap)
    out = np.maximum(np.floor(clipped / _GRID + 1e-9) * _GRID, _GRID)
    out = np.round(out, 9)
    if not active_bw:
        out[..., BW] = 0.0
    return out


def synthetic_catalog(config: GeneratorConfig, rng: np.random.Generator) -> Tuple[InstanceType, ...]:
    """Catalogue SYNTHÉTIQUE : familles x tailles, prix ~ vCPU avec bruit log-normal."""
    types = []
    for family, ram_ratio, price_per_vcpu in config.families:
        for vcpu in config.sizes_vcpu:
            noise = float(np.exp(config.price_noise * rng.standard_normal()))
            types.append(
                InstanceType(
                    name=f"{family}-{vcpu}",
                    price=round(price_per_vcpu * vcpu * noise, 4),
                    vcpu=float(vcpu),
                    ram_gib=float(vcpu) * ram_ratio,
                    bw_mbps=float(vcpu) * config.bw_per_vcpu_mbps if config.bandwidth_active else 0.0,
                    reservation=config.reservation,
                    family=family,
                )
            )
    return tuple(types)


def generate_instance(config: GeneratorConfig = GeneratorConfig()) -> Instance:
    """Génère une instance synthétique reproductible à partir de config.seed."""
    config.validate()
    ss = np.random.SeedSequence(int(config.seed))
    rng_cat, rng_wl, rng_sc, rng_x0 = (np.random.default_rng(c) for c in ss.spawn(4))

    types = synthetic_catalog(config, rng_cat)
    delta = np.asarray(config.daemonset_overhead, dtype=float)

    # Plafond par pod : fraction du plus gros nœud (CPU puis RAM) -> tout pod est placeable.
    allocs = np.vstack([t.allocatable(delta) for t in types])
    big = max(range(len(types)), key=lambda j: (allocs[j, CPU], allocs[j, RAM]))
    cap = config.max_pod_fraction * allocs[big]
    if not config.bandwidth_active:
        cap[BW] = np.inf
    if cap[CPU] < _GRID[CPU] or cap[RAM] < _GRID[RAM] or (config.bandwidth_active and cap[BW] < _GRID[BW]):
        raise DataValidationError("Plafond de pod inférieur au pas de quantification : agrandir le catalogue")

    n_base = config.n_base_workloads
    n_extra = int(round(config.peak_extra_fraction * n_base))
    n_total = n_base + n_extra

    cpu = config.cpu_median * np.exp(config.heterogeneity * rng_wl.standard_normal(n_total))
    ram = cpu * config.ram_per_cpu_median * np.exp(config.ram_ratio_sigma * rng_wl.standard_normal(n_total))
    bw = config.bw_median_mbps * np.exp(config.bw_sigma * rng_wl.standard_normal(n_total))
    base = _quantize(np.column_stack([cpu, ram, bw]), cap, config.bandwidth_active)
    workloads = tuple(
        Workload(f"pod-{i:03d}", cpu=float(base[i, CPU]), ram_gib=float(base[i, RAM]), bw_mbps=float(base[i, BW]))
        for i in range(n_total)
    )

    scenarios = [
        Scenario("stable", np.arange(n_base), base[:n_base], epsilon=config.eps_stable),
        Scenario("pic", np.arange(n_total), _quantize(base * config.peak_scale, cap, config.bandwidth_active),
                 epsilon=config.eps_peak),
    ]
    lo, hi = config.variable_scale_range
    for t in range(config.n_variable_states):
        extras_in = np.flatnonzero(rng_sc.random(n_extra) < 0.5) + n_base
        members = np.concatenate([np.arange(n_base), extras_in]).astype(np.int64)
        scale = rng_sc.uniform(lo, hi, size=(members.size, 1))
        name = "variable" if config.n_variable_states == 1 else f"variable_{t + 1}"
        scenarios.append(
            Scenario(name, members, _quantize(base[members] * scale, cap, config.bandwidth_active),
                     epsilon=config.eps_variable)
        )

    x0 = rng_x0.integers(0, 3, size=len(types)) if config.generate_x0 else None
    return Instance(
        types=types,
        workloads=workloads,
        scenarios=tuple(scenarios),
        daemonset_overhead=tuple(float(v) for v in delta),
        bandwidth_active=config.bandwidth_active,
        n_max_total=config.n_max_total,
        x0=x0,
        name=f"synthetic_seed{config.seed}",
        provenance="synthétique (générateur KubeOpt, données fictives)",
    )


def generate_cluster_instance(seed: int = 0, n_containers: int = 30, m: Optional[int] = None,
                              server_vcpu: int = 16, ram_per_vcpu: float = 4.0,
                              bandwidth_active: bool = False, **overrides) -> Instance:
    """Instance « option A » (B1-ter) : un seul type de serveur, m serveurs identiques disponibles.

    Le prix est sans objet pour l'objectif (nombre de serveurs actifs) ; il est fixé à 1 $/h.
    Données SYNTHÉTIQUES : à utiliser comme banc d'essai, pas comme résultat sur données réelles.
    `overrides` : tout autre champ de GeneratorConfig (heterogeneity, n_variable_states, eps_*...).
    """
    config = GeneratorConfig(
        seed=seed, n_base_workloads=n_containers,
        families=(("srv", float(ram_per_vcpu), 1.0 / server_vcpu),),
        sizes_vcpu=(int(server_vcpu),), price_noise=0.0,
        n_max_total=m, bandwidth_active=bandwidth_active, **overrides,
    )
    return generate_instance(config)
