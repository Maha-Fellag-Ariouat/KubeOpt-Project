"""Procédure de placement déterministe Place(X, s) sur une flotte DONNÉE (B1 §8).

Propriétés imposées par B1 :
  - tout placement retourné respecte (7)-(10) et (8bis) (vérifié en test par le validateur) ;
  - déterminisme : aucune source aléatoire ;
  - un échec de placement ne prouve pas l'infaisabilité exacte (heuristique).

Politiques :
  "ff"  : First Fit, pods pris dans l'ordre du scénario ;
  "ffd" : First Fit Decreasing, pods triés par taille décroissante
          size(i) = max_d d_i^d / D_s^d (dimension dominante normalisée par la demande totale) ;
  "ffd_cap" : First Fit Decreasing de baselines.py (projet) : size(i) = max_d d_i^d / cap_0^d,
          normalisée par la capacité utilisable du nœud 0 (serveurs identiques), tri stable.

Cette procédure ne OUVRE jamais de nœud : un pod qui ne tient dans aucun nœud existant
déborde (z = 1). L'ouverture de nœuds relève des baselines (B3) et du recuit (B4).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .errors import DataValidationError
from .models import FLOAT_TOL, Instance, ScenarioKey
from .solution import OVERFLOW, Solution

POLICIES = ("ff", "ffd", "ffd_cap")


def _check_node_types(instance: Instance, node_types: Sequence[int]) -> np.ndarray:
    nt = np.asarray(node_types)
    if nt.size == 0:
        return np.zeros(0, dtype=np.int64)
    if nt.ndim != 1 or not np.issubdtype(nt.dtype, np.integer):
        raise DataValidationError("node_types doit être un vecteur d'entiers")
    if nt.min() < 0 or nt.max() >= instance.k:
        raise DataValidationError(f"node_types doit être dans [0, {instance.k})")
    return nt.astype(np.int64)


def pod_order(instance: Instance, scenario: ScenarioKey, policy: str = "ffd",
              reference_capacity: Optional[np.ndarray] = None) -> np.ndarray:
    """Ordre de traitement des pods (rangs 0..m-1 du scénario).

    `reference_capacity` (vecteur sur les dimensions actives) est requis par "ffd_cap".
    """
    if policy not in POLICIES:
        raise DataValidationError(f"politique inconnue : {policy!r} (attendu : {POLICIES})")
    s = instance.get_scenario(scenario)
    if policy == "ff" or s.n_pods == 0:
        return np.arange(s.n_pods)
    act = list(instance.active_dims)
    if policy == "ffd_cap":
        if reference_capacity is None:
            return np.arange(s.n_pods)                     # flotte vide : aucun nœud, ordre sans effet
        size = (s.demand[:, act] / np.asarray(reference_capacity, dtype=float)).max(axis=1)
        return np.argsort(-size, kind="stable")
    total = s.total_demand()[act]
    ratio = np.divide(s.demand[:, act], total, out=np.zeros((s.n_pods, len(act))), where=total > 0)
    return np.argsort(-ratio.max(axis=1), kind="stable")


def place(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey,
          policy: str = "ffd") -> np.ndarray:
    """Place les pods du scénario sur la flotte ; retourne l'affectation (nœud ou -1)."""
    nt = _check_node_types(instance, node_types)
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    cap = instance.allocatable[nt][:, act]                 # (N, d)
    order = pod_order(instance, s, policy, reference_capacity=cap[0] if nt.size else None)

    max_pods = instance.max_pods[nt]                       # (N,)
    eligible = instance.eligibility[s.members][:, nt]      # (m, N)
    dem = s.demand[:, act]
    loads = np.zeros_like(cap)
    counts = np.zeros(nt.size)
    assignment = np.full(s.n_pods, OVERFLOW, dtype=np.int64)
    for r in order:
        ok = eligible[r] & (counts < max_pods) & np.all(loads + dem[r] <= cap + FLOAT_TOL, axis=1)
        if ok.any():
            b = int(ok.argmax())                           # premier nœud admissible
            assignment[r] = b
            loads[b] += dem[r]
            counts[b] += 1
    return assignment


def place_all(instance: Instance, node_types: Sequence[int], policy: str = "ffd") -> Solution:
    """Applique `place` à chaque scénario (placements indépendants sur la même flotte)."""
    nt = _check_node_types(instance, node_types)
    return Solution(nt, tuple(place(instance, nt, s, policy) for s in instance.scenarios))
