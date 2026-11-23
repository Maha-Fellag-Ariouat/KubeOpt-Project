"""Baselines classiques : Round Robin, First Fit, First Fit Decreasing (B3 et B3b).

B3b (catalogue de types avec prix, objectif coût) : FF et FFD « avec ouverture » ouvrent, quand
un conteneur ne tient dans aucun nœud ouvert, le type ÉLIGIBLE LE MOINS CHER qui peut l'accueillir
seul (ex æquo : plus petit indice) ; RR, qui n'ouvre rien, tourne sur une flotte mono-type
dimensionnée par FFD. Chaque scénario est exécuté isolément ; les flottes sont fusionnées par
maximum type par type (flotte statique commune, B1 §5.6) puis les affectations sont réindexées.

Adaptation à la structure KubeOpt des algorithmes de `baselines.py` du projet. Cadre B1-ter,
option A : un cluster de m serveurs IDENTIQUES est donné ; on cherche à placer tous les
conteneurs en minimisant le nombre de serveurs ACTIFS, puis à équilibrer la charge.

    round_robin           : RR sur les m serveurs, saute un serveur où le conteneur ne tient pas
                            (active donc potentiellement tous les serveurs : référence naïve) ;
    first_fit             : premier serveur (par indice) où le conteneur tient ;
    first_fit_decreasing  : FF après tri par taille décroissante
                            size(i) = max_d d_i^d / cap_0^d, tri stable (placement.py, "ffd_cap").

Contraintes respectées à chaque pas, comme dans `place` : capacité de chaque dimension active
(CPU, RAM, BW), éligibilité (9) et limite de pods (10). Un conteneur qui ne tient nulle part
déborde (z = 1) : la solution est alors de service incomplet (V > 0), jamais « faisable ».
Chaque scénario est traité isolément, sur le même cluster.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .errors import DataValidationError
from .evaluation import Evaluation, evaluate
from .models import FLOAT_TOL, Instance, ScenarioKey
from .placement import _check_node_types, place
from .solution import OVERFLOW, Solution, node_types_from_composition

BASELINES: Tuple[str, ...] = ("round_robin", "first_fit", "first_fit_decreasing")
OPEN_BASELINES: Tuple[str, ...] = ("round_robin_mono", "first_fit_open", "first_fit_decreasing_open")


def cluster_nodes(instance: Instance, m: Optional[int] = None) -> np.ndarray:
    """Cluster de m serveurs identiques (instance à un seul type).

    m par défaut : `n_max_total` s'il est défini, sinon le plus grand nombre de pods d'un scénario
    (borne toujours suffisante si chaque pod tient seul sur un serveur).
    """
    if instance.k != 1:
        raise DataValidationError("cluster_nodes exige une instance à un seul type de serveur (option A)")
    if m is None:
        m = instance.n_max_total if instance.n_max_total is not None else max(s.n_pods for s in instance.scenarios)
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 0:
        raise DataValidationError("m doit être un entier >= 0")
    return np.zeros(int(m), dtype=np.int64)


def server_lower_bound(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey) -> int:
    """Borne inférieure du nombre de serveurs pour placer TOUT le scénario (serveurs identiques).

    max( ceil(max_d D^d / cap^d), ceil(|I_s| / pi) ). Ce n'est pas l'optimum : seulement un
    plancher valable, utile pour situer un résultat (écart au plancher).
    """
    nt = _check_node_types(instance, node_types)
    if nt.size == 0 or np.unique(nt).size != 1:
        raise DataValidationError("server_lower_bound exige un cluster non vide de serveurs identiques")
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    cap = instance.allocatable[nt[0]][act]
    total = s.total_demand()[act]
    bound = int(np.ceil(float((total / cap).max()) - FLOAT_TOL)) if s.n_pods else 0
    pods_limit = instance.max_pods[nt[0]]
    if np.isfinite(pods_limit):
        bound = max(bound, int(np.ceil(s.n_pods / pods_limit - FLOAT_TOL)))
    return max(bound, 0)


def round_robin_assignment(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey) -> np.ndarray:
    """Round Robin : un pointeur circulaire ; saute les serveurs où le conteneur ne tient pas."""
    nt = _check_node_types(instance, node_types)
    s = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    cap = instance.allocatable[nt][:, act]
    max_pods = instance.max_pods[nt]
    eligible = instance.eligibility[s.members][:, nt]
    dem = s.demand[:, act]
    loads = np.zeros_like(cap)
    counts = np.zeros(nt.size)
    assignment = np.full(s.n_pods, OVERFLOW, dtype=np.int64)
    ptr = 0
    for r in range(s.n_pods):
        if nt.size == 0:
            break
        ok = eligible[r] & (counts < max_pods) & np.all(loads + dem[r] <= cap + FLOAT_TOL, axis=1)
        if ok.any():
            j = (ptr + int(np.roll(ok, -ptr).argmax())) % nt.size   # premier admissible à partir de ptr
            assignment[r] = j
            loads[j] += dem[r]
            counts[j] += 1
            ptr = (j + 1) % nt.size
    return assignment


def first_fit_assignment(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey) -> np.ndarray:
    """First Fit : conteneurs dans l'ordre du scénario, premier serveur admissible."""
    return place(instance, node_types, scenario, "ff")


def first_fit_decreasing_assignment(instance: Instance, node_types: Sequence[int],
                                    scenario: ScenarioKey) -> np.ndarray:
    """First Fit Decreasing : tri par dimension dominante normalisée par la capacité du serveur 0."""
    return place(instance, node_types, scenario, "ffd_cap")


_ASSIGNERS = {
    "round_robin": round_robin_assignment,
    "first_fit": first_fit_assignment,
    "first_fit_decreasing": first_fit_decreasing_assignment,
}


@dataclass(frozen=True, eq=False)
class BaselineResult:
    """Résultat d'une baseline : solution, évaluation et temps de placement (tous scénarios)."""

    name: str
    solution: Solution
    evaluation: Evaluation
    elapsed_s: float


def run_baseline(instance: Instance, name: str, node_types: Optional[Sequence[int]] = None) -> BaselineResult:
    """Exécute une baseline sur chaque scénario ; la solution est validée par le validateur indépendant."""
    if name not in _ASSIGNERS:
        raise DataValidationError(f"baseline inconnue : {name!r} (attendu : {BASELINES})")
    nt = cluster_nodes(instance) if node_types is None else _check_node_types(instance, node_types)
    start = time.perf_counter()
    assignments = tuple(_ASSIGNERS[name](instance, nt, s) for s in instance.scenarios)
    elapsed = time.perf_counter() - start
    solution = Solution(nt, assignments)
    return BaselineResult(name, solution, evaluate(instance, solution, elapsed_s=elapsed), elapsed)


def run_all_baselines(instance: Instance, node_types: Optional[Sequence[int]] = None) -> Dict[str, BaselineResult]:
    """Les trois baselines sur la MÊME instance et le MÊME cluster (comparaison équitable)."""
    return {name: run_baseline(instance, name, node_types) for name in BASELINES}


# --------------------------------------------------------------------------- #
# B3b : baselines avec ouverture de nœuds (catalogue hétérogène, objectif coût)
# --------------------------------------------------------------------------- #
def open_assignment(instance: Instance, scenario: ScenarioKey, policy: str = "ff",
                    allowed_types: Optional[Sequence[bool]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """First Fit avec ouverture pour UN scénario ; retourne (types des nœuds ouverts, affectation).

    policy "ff"  : ordre du scénario ;
    policy "ffd" : tri par taille décroissante max_d d_i^d / Cref^d, Cref = capacité utilisable
                   maximale du catalogue par dimension (= capacité du serveur si type unique,
                   donc identique au FFD de baselines.py du projet).
    Un pod que ni un nœud ouvert ni l'ouverture d'un type éligible (capacité, quotas n_j^max et
    N_max) ne peut accueillir déborde (-1).
    """
    if policy not in ("ff", "ffd"):
        raise DataValidationError(f"politique inconnue : {policy!r} (attendu : 'ff' ou 'ffd')")
    sc = instance.get_scenario(scenario)
    act = list(instance.active_dims)
    m, k = sc.n_pods, instance.k
    dem = sc.demand[:, act]
    type_cap = instance.allocatable[:, act]                                   # (k, d)
    allowed = np.ones(k, dtype=bool) if allowed_types is None else np.asarray(allowed_types, dtype=bool)
    if allowed.shape != (k,):
        raise DataValidationError(f"allowed_types doit avoir la longueur {k}")
    if policy == "ffd" and m:
        order = np.argsort(-(dem / type_cap.max(axis=0)).max(axis=1), kind="stable")
    else:
        order = np.arange(m)
    elig = instance.eligibility[sc.members]                                   # (m, k)
    node_type = np.full(m, -1, dtype=np.int64)
    cap = np.zeros((m, len(act)))
    loads = np.zeros((m, len(act)))
    counts = np.zeros(m)
    node_max_pods = np.zeros(m)
    type_count = np.zeros(k)
    assignment = np.full(m, OVERFLOW, dtype=np.int64)
    n_open = 0
    for r in order:
        if n_open:
            ok = (elig[r, node_type[:n_open]] & (counts[:n_open] < node_max_pods[:n_open])
                  & np.all(loads[:n_open] + dem[r] <= cap[:n_open] + FLOAT_TOL, axis=1))
            if ok.any():
                b = int(ok.argmax())
                assignment[r] = b
                loads[b] += dem[r]
                counts[b] += 1
                continue
        if instance.n_max_total is not None and n_open >= instance.n_max_total:
            continue
        can = elig[r] & allowed & np.all(dem[r] <= type_cap + FLOAT_TOL, axis=1) & (type_count < instance.n_max_per_type)
        if not can.any():
            continue
        j = int(np.where(can, instance.prices, np.inf).argmin())               # le moins cher, indice le plus petit
        node_type[n_open], cap[n_open] = j, type_cap[j]
        node_max_pods[n_open] = instance.max_pods[j]
        type_count[j] += 1
        assignment[r] = n_open
        loads[n_open] += dem[r]
        counts[n_open] += 1
        n_open += 1
    return node_type[:n_open].copy(), assignment


def merge_runs(instance: Instance, runs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Solution:
    """Fusionne des exécutions par scénario (types ouverts, affectation) en une flotte statique commune.

    n_j = max_s n_j^(s) ; le r-ième nœud de type j du scénario s devient le r-ième nœud de type j de
    la flotte fusionnée. Les charges de chaque scénario sont inchangées : un placement valide reste valide.
    """
    k = instance.k
    counts = np.zeros(k, dtype=np.int64)
    for nt, _ in runs:
        counts = np.maximum(counts, np.bincount(nt, minlength=k))
    fleet = node_types_from_composition(counts)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    assignments = []
    for nt, a in runs:
        if nt.size == 0:
            assignments.append(a.copy())
            continue
        rank = np.zeros(nt.size, dtype=np.int64)
        seen = np.zeros(k, dtype=np.int64)
        for b, j in enumerate(nt):
            rank[b] = seen[j]
            seen[j] += 1
        mapping = start[nt] + rank
        assignments.append(np.where(a >= 0, mapping[np.maximum(a, 0)], OVERFLOW))
    return Solution(fleet, tuple(assignments))


def mono_type_fleet(instance: Instance, slack_nodes: int = 0) -> np.ndarray:
    """Flotte mono-type pour Round Robin : type unique le moins cher, dimensionné par FFD.

    Pour chaque type j pouvant accueillir seul tous les conteneurs de tous les scénarios, n_j est le
    plus grand nombre de nœuds ouverts par FFD restreint à j ; on retient le type de coût
    (n_j + slack_nodes) * c_j minimal (ex æquo : plus petit indice). `slack_nodes` = 0 par défaut
    (aucune marge inventée) ; un RR peut alors déborder, ce qui est reporté (V > 0).
    """
    if isinstance(slack_nodes, bool) or not isinstance(slack_nodes, (int, np.integer)) or slack_nodes < 0:
        raise DataValidationError("slack_nodes doit être un entier >= 0")
    best = None
    for j in range(instance.k):
        only = np.zeros(instance.k, dtype=bool)
        only[j] = True
        counts, complete = [], True
        for sc in instance.scenarios:
            nt, a = open_assignment(instance, sc, "ffd", allowed_types=only)
            complete &= bool((a >= 0).all())
            counts.append(nt.size)
        n = (max(counts) if counts else 0) + int(slack_nodes)
        if not complete or n > instance.n_max_per_type[j]:
            continue
        if instance.n_max_total is not None and n > instance.n_max_total:
            continue
        cost = n * float(instance.prices[j])
        if best is None or cost < best[0] - 1e-12:
            best = (cost, j, n)
    if best is None:
        raise DataValidationError("aucun type unique ne peut héberger tous les conteneurs (quotas, éligibilité, capacité)")
    return np.full(best[2], best[1], dtype=np.int64)


def run_open_baseline(instance: Instance, name: str, slack_nodes: int = 0) -> BaselineResult:
    """Exécute une baseline B3b ; la solution fusionnée est validée par le validateur indépendant.

    Si la fusion dépasse N_max (chaque scénario respectait N_max mais pas leur maximum type par
    type), l'évaluation lève InvalidPlacementError : la baseline ne tient pas dans N_max.
    """
    if name not in OPEN_BASELINES:
        raise DataValidationError(f"baseline inconnue : {name!r} (attendu : {OPEN_BASELINES})")
    start = time.perf_counter()
    if name == "round_robin_mono":
        nt = mono_type_fleet(instance, slack_nodes)
        solution = Solution(nt, tuple(round_robin_assignment(instance, nt, sc) for sc in instance.scenarios))
    else:
        policy = "ff" if name == "first_fit_open" else "ffd"
        solution = merge_runs(instance, [open_assignment(instance, sc, policy) for sc in instance.scenarios])
    elapsed = time.perf_counter() - start
    return BaselineResult(name, solution, evaluate(instance, solution, elapsed_s=elapsed), elapsed)


def run_all_open_baselines(instance: Instance, slack_nodes: int = 0) -> Dict[str, BaselineResult]:
    """Les trois baselines B3b sur la MÊME instance (comparaison équitable)."""
    return {name: run_open_baseline(instance, name, slack_nodes) for name in OPEN_BASELINES}
