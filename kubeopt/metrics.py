"""Métriques d'évaluation (B1 §5, §9.1, §10), généralisées à |D| dimensions actives.

|D| = 2 (CPU, RAM) redonne EXACTEMENT les formules de B1 ; |D| = 3 ajoute la bande passante
(amendement B1-bis). Les facteurs 1/2 de B1 deviennent 1/|D|.

    O_s   = (1/|D|) sum_d o_s^d / D_s^d                       (éq. 20)
    U_s^d = P_s^d / K^d                                       (éq. 13)
    u_b^d = load_b^d / cap_b^d                                (éq. 14)
    R_s^d = K^d - P_s^d                                       (éq. 15)
    sigma_s^d = sqrt( sum_b w_b^d (u_b^d - U_s^d)^2 ), w_b^d = cap_b^d / K^d      (éq. 16)
    q_b   = min_d r_b^d / kappa^d        (kappa^CPU = 1)      (éq. 17)
    g_b^d = r_b^d - kappa^d q_b                               (stranded)
    F_s   = (1/|D|) sum_d (sum_b g_b^d) / K^d                 (éq. 18)
    v_s   = max(0, O_s - eps_s) / (1 - eps_s) ; V = max_s v_s (éq. 31)

Conventions de bord (documentées, testées) :
  - une dimension de demande totale nulle contribue 0 à O_s ;
  - K^d = 0 (flotte vide) : U^d = 0, sigma^d = 0, contribution nulle à F_s ;
  - F_s = NaN si la demande CPU du scénario est nulle (kappa_s non défini) ;
  - une dimension de kappa^d = 0 (aucune demande) n'entre pas dans q_b ; son résiduel est
    entièrement « échoué » (le bloc de référence ne la consomme pas) ;
  - les résiduels sont écrêtés à >= 0 (bruit flottant).
sigma et U sont des métriques de RAPPORT, jamais des termes d'optimisation (B1 §5.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .errors import DataValidationError
from .models import DIMENSIONS, Instance, ScenarioKey


@dataclass(frozen=True, eq=False)
class ScenarioMetrics:
    """Métriques d'un placement dans un scénario. Les vecteurs suivent l'ordre `dims`."""

    scenario: str
    dims: Tuple[str, ...]
    demand: np.ndarray               # D_s
    placed: np.ndarray               # P_s
    overflow: np.ndarray             # o_s
    capacity: np.ndarray             # K(X)
    utilization: np.ndarray          # U_s
    unused: np.ndarray               # R_s
    node_utilization: np.ndarray     # (N, |D|) u_b
    sigma: np.ndarray                # sigma_s par dimension
    overflow_rate: float             # O_s
    overflow_pods: int               # N_ovf
    overflow_pods_rate: float        # O_pods = N_ovf / |I_s|
    fragmentation: float             # F_s
    active_nodes: int                # N_act : nœuds portant au moins un pod
    active_utilization: np.ndarray   # P_s / capacité des seuls nœuds actifs (0 s'il n'y en a pas)
    balance: float                   # Bal : moyenne sur les dimensions de l'écart-type de u_b (nœuds actifs)
    violation: float                 # v_s
    margin_excess: Optional[np.ndarray]  # H_s (si u* fourni)


def node_loads(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey,
               assignment: Sequence[int]) -> np.ndarray:
    """Charge (N, |D|) de chaque nœud (dimensions actives)."""
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    nt = np.asarray(node_types, dtype=np.int64)
    a = np.asarray(assignment, dtype=np.int64)
    if a.shape != (s.n_pods,):
        raise DataValidationError(f"affectation de forme {a.shape}, attendu ({s.n_pods},)")
    if a.size and (a.min() < -1 or a.max() >= nt.size):
        raise DataValidationError("affectation : indice de nœud hors bornes")
    loads = np.zeros((nt.size, len(act)))
    placed = a >= 0
    np.add.at(loads, a[placed], s.demand[placed][:, act])
    return loads


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def compute_scenario_metrics(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey,
                             assignment: Sequence[int], u_star: Optional[float] = None) -> ScenarioMetrics:
    """Calcule O_s, U_s, R_s, sigma_s, F_s, v_s pour un placement (supposé valide)."""
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    d = len(act)
    nt = np.asarray(node_types, dtype=np.int64)
    a = np.asarray(assignment, dtype=np.int64)
    loads = node_loads(instance, nt, s, a)
    alloc = instance.allocatable[nt][:, act]                      # (N, d)
    capacity = alloc.sum(axis=0)

    placed_mask = a >= 0
    dem = s.demand[:, act]
    demand = dem.sum(axis=0)
    placed = dem[placed_mask].sum(axis=0)
    overflow = dem[~placed_mask].sum(axis=0)

    overflow_rate = float(_safe_div(overflow, demand).sum() / d)
    utilization = _safe_div(placed, capacity)
    unused = capacity - placed
    node_util = _safe_div(loads, alloc) if nt.size else np.zeros((0, d))

    sigma = np.zeros(d)
    if nt.size:
        for k in range(d):
            if capacity[k] > 0:
                w = alloc[:, k] / capacity[k]
                sigma[k] = float(np.sqrt(np.sum(w * (node_util[:, k] - utilization[k]) ** 2)))

    # Fragmentation (éq. 17-18)
    if demand[0] <= 0:
        fragmentation = float("nan")
    else:
        kappa = demand / demand[0]                                # (d,), kappa[0] = 1
        resid = np.maximum(alloc - loads, 0.0)
        if nt.size:
            ratios = np.full(resid.shape, np.inf)
            pos = kappa > 0
            ratios[:, pos] = resid[:, pos] / kappa[pos]
            q = ratios.min(axis=1)
            stranded = np.maximum(resid - q[:, None] * kappa[None, :], 0.0)
            fragmentation = float(_safe_div(stranded.sum(axis=0), capacity).sum() / d)
        else:
            fragmentation = 0.0

    active_mask = np.zeros(nt.size, dtype=bool)
    active_mask[np.unique(a[placed_mask])] = True
    active_capacity = alloc[active_mask].sum(axis=0)
    active_utilization = _safe_div(placed, active_capacity)
    # Bal (B1-ter, définition de sa.py) : dispersion entre serveurs ACTIFS seulement (ddof = 0)
    balance = float(node_util[active_mask].std(axis=0).mean()) if active_mask.any() else 0.0

    eps = s.epsilon
    violation = max(0.0, overflow_rate - eps) / (1.0 - eps)
    margin = None
    if u_star is not None:
        if not 0 < u_star < 1:
            raise DataValidationError("u_star doit être dans ]0, 1[")
        margin = np.maximum(utilization - u_star, 0.0)
    n_ovf = int((~placed_mask).sum())
    return ScenarioMetrics(
        scenario=s.name, dims=tuple(DIMENSIONS[k] for k in act), demand=demand, placed=placed,
        overflow=overflow, capacity=capacity, utilization=utilization, unused=unused,
        node_utilization=node_util, sigma=sigma, overflow_rate=overflow_rate,
        overflow_pods=n_ovf, overflow_pods_rate=(n_ovf / s.n_pods if s.n_pods else 0.0),
        fragmentation=fragmentation, active_nodes=int(active_mask.sum()),
        active_utilization=active_utilization, balance=balance,
        violation=float(violation), margin_excess=margin,
    )


def fleet_violation(instance: Instance, scenario_metrics: Sequence[ScenarioMetrics],
                    cost: Optional[float] = None, budget: Optional[float] = None) -> float:
    """V(X) = max_s v_s (éq. 31) ; si un budget B est actif : max(V, max(0, C - B) / C_ub)."""
    v = max((m.violation for m in scenario_metrics), default=0.0)
    if budget is not None:
        if cost is None:
            raise DataValidationError("cost est requis quand un budget est fourni")
        v = max(v, max(0.0, cost - budget) / instance.cost_upper_bound())
    return float(v)


@dataclass(frozen=True)
class ChangeMetrics:
    """Changements par rapport à X0 (éq. 21-25)."""

    delta_plus: int
    delta_minus: int
    l1: int
    delta_normalized: Optional[float]       # L1 / (N_max + sum n0), si N_max connu (éq. 24)
    delta_cost: float                       # C(X) - C(X0)
    weighted_rotation: float                # sum_j c_j (a+_j + a-_j)
    theta: Optional[float]                  # gamma+ D+ + gamma- D-, si gammas fournis (éq. 25)
    theta_normalized: Optional[float]       # Theta / Theta_max, Theta_max = g+ N_max + g- sum n0


def change_metrics(instance: Instance, n: Sequence[int], gamma_plus: Optional[float] = None,
                   gamma_minus: Optional[float] = None) -> Optional[ChangeMetrics]:
    """Métriques de changement ; None si X0 est inconnu (B1 §3.4 : terme neutralisé)."""
    if instance.x0 is None:
        return None
    x = instance.check_composition(n)
    x0 = np.asarray(instance.x0, dtype=np.int64)
    a_plus = np.maximum(x - x0, 0)
    a_minus = np.maximum(x0 - x, 0)
    dp, dm = int(a_plus.sum()), int(a_minus.sum())
    denom = None if instance.n_max_total is None else instance.n_max_total + int(x0.sum())
    theta = theta_norm = None
    if gamma_plus is not None and gamma_minus is not None:
        if gamma_plus < 0 or gamma_minus < 0:
            raise DataValidationError("gamma+ et gamma- doivent être >= 0")
        theta = float(gamma_plus * dp + gamma_minus * dm)
        if instance.n_max_total is not None:
            theta_max = gamma_plus * instance.n_max_total + gamma_minus * int(x0.sum())
            theta_norm = theta / theta_max if theta_max > 0 else 0.0
    return ChangeMetrics(
        delta_plus=dp, delta_minus=dm, l1=dp + dm,
        delta_normalized=None if denom is None or denom == 0 else (dp + dm) / denom,
        delta_cost=instance.cost(x) - instance.cost(x0),
        weighted_rotation=float(instance.prices @ (a_plus + a_minus)),
        theta=theta, theta_normalized=theta_norm,
    )
