"""Validateur INDÉPENDANT des contraintes dures (B1 §4) -- aucun code partagé avec `placement`.

Il recalcule tout depuis les données brutes avec des boucles Python explicites (lentes mais
évidemment correctes) : c'est l'arbitre final. Aucune solution ne doit être déclarée faisable
sans passer ici.

Contraintes vérifiées, par scénario s et nœud b :
  (4)  n_j <= n_j^max, sum_j n_j <= N_max, types de nœuds dans [0, k)
  (6)  chaque pod est soit sur exactement un nœud, soit en débordement (garanti par la forme
       du vecteur d'affectation : une seule valeur par pod) ; indices de nœuds valides
  (7)  CPU, (8) RAM, (8bis) BW (si active) : charge du nœud <= capacité utilisable
  (9)  éligibilité e_{i,j(b)} = 1
  (10) nombre de pods <= pi_j(b)

Le débordement z est libre (relâchement) : un pod débordé n'est pas une violation ; c'est V
(éq. 31) qui juge l'écart de service. `overflowed_pods_that_fit` signale les débordements
non justifiés (information, pas une violation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .errors import InvalidPlacementError
from .models import DIMENSIONS, Instance, ScenarioKey
from .solution import Solution

_TOL = 1e-9  # même valeur que models.FLOAT_TOL, volontairement redéfinie ici


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    violations: Tuple[str, ...]

    def __bool__(self) -> bool:
        return self.valid

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise InvalidPlacementError(self.violations)


def _fleet_violations(instance: Instance, node_types: np.ndarray) -> List[str]:
    out: List[str] = []
    k = len(instance.types)
    counts = [0] * k
    for b, j in enumerate(node_types.tolist()):
        if j < 0 or j >= k:
            out.append(f"nœud {b} : type {j} hors de [0, {k})")
        else:
            counts[j] += 1
    for j, t in enumerate(instance.types):
        if t.max_count is not None and counts[j] > t.max_count:
            out.append(f"(4) type {t.name} : {counts[j]} nœuds > n_max = {t.max_count}")
    if instance.n_max_total is not None and len(node_types) > instance.n_max_total:
        out.append(f"(4) {len(node_types)} nœuds > N_max = {instance.n_max_total}")
    return out


def _scenario_violations(instance: Instance, node_types: np.ndarray, scenario: ScenarioKey,
                         assignment: Sequence[int]) -> List[str]:
    s = instance.get_scenario(scenario)
    tag = f"[{s.name}]"
    out: List[str] = []
    assign = np.asarray(assignment)
    if assign.ndim != 1 or assign.size != s.n_pods:
        return [f"{tag} affectation de longueur {assign.size} (attendu {s.n_pods})"]
    if assign.size and not np.issubdtype(assign.dtype, np.integer):
        return [f"{tag} affectation non entière"]
    k = len(instance.types)
    n_nodes = len(node_types)
    if any(j < 0 or j >= k for j in node_types.tolist()):
        return [f"{tag} flotte invalide : type de nœud hors de [0, {k})"]

    act = list(instance.active_dims)
    members = s.members.tolist()
    demand = s.demand.tolist()
    load = [[0.0] * 3 for _ in range(n_nodes)]
    count = [0] * n_nodes
    types = node_types.tolist()
    for r, b in enumerate(assign.tolist()):
        if b == -1:
            continue                                  # débordement : pas une violation
        if b < -1 or b >= n_nodes:
            out.append(f"{tag} pod {instance.workloads[members[r]].name} : nœud {b} inexistant")
            continue
        j = types[b]
        if not bool(instance.eligibility[members[r], j]):
            out.append(f"{tag} (9) pod {instance.workloads[members[r]].name} non éligible au type "
                       f"{instance.types[j].name} (nœud {b})")
        count[b] += 1
        for d in range(3):
            load[b][d] += demand[r][d]
    for b in range(n_nodes):
        j = types[b]
        max_pods = instance.types[j].max_pods
        if max_pods is not None and count[b] > max_pods:
            out.append(f"{tag} (10) nœud {b} ({instance.types[j].name}) : {count[b]} pods > {max_pods}")
        for d in act:
            cap = float(instance.allocatable[j][d])
            if load[b][d] > cap + _TOL:
                label = {0: "(7)", 1: "(8)", 2: "(8bis)"}[d]
                out.append(f"{tag} {label} nœud {b} ({instance.types[j].name}) : charge {DIMENSIONS[d]} "
                           f"{load[b][d]:.6g} > capacité {cap:.6g}")
    return out


def validate_placement(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey,
                       assignment: Sequence[int]) -> ValidationReport:
    """Valide un placement pour un scénario (contraintes (6)-(10), (8bis)) ; ne vérifie pas (4)."""
    nt = np.asarray(node_types, dtype=np.int64)
    viol = _scenario_violations(instance, nt, scenario, assignment)
    return ValidationReport(not viol, tuple(viol))


def validate_solution(instance: Instance, solution: Solution) -> ValidationReport:
    """Valide la flotte (4) et chaque placement de la solution."""
    viol = _fleet_violations(instance, solution.node_types)
    if len(solution.assignments) != len(instance.scenarios):
        viol.append(f"{len(solution.assignments)} affectations pour {len(instance.scenarios)} scénarios")
    else:
        for s, assignment in zip(instance.scenarios, solution.assignments):
            viol.extend(_scenario_violations(instance, solution.node_types, s, assignment))
    return ValidationReport(not viol, tuple(viol))


def overflowed_pods_that_fit(instance: Instance, node_types: Sequence[int], scenario: ScenarioKey,
                             assignment: Sequence[int]) -> List[int]:
    """Indices globaux des pods débordés qui tiendraient pourtant dans le résiduel d'un nœud.

    Information : B1 §4.5 définit z = 1 « si aucun nœud ne peut accueillir i sous la politique
    retenue ». Une liste non vide signifie que le placement n'est pas maximal (pas invalide).
    """
    s = instance.get_scenario(scenario)
    nt = np.asarray(node_types, dtype=np.int64).tolist()
    act = list(instance.active_dims)
    assign = np.asarray(assignment).tolist()
    load = [[0.0] * 3 for _ in nt]
    count = [0] * len(nt)
    for r, b in enumerate(assign):
        if b >= 0:
            count[b] += 1
            for d in range(3):
                load[b][d] += float(s.demand[r][d])
    fits: List[int] = []
    for r, b in enumerate(assign):
        if b != -1:
            continue
        i = int(s.members[r])
        for nb, j in enumerate(nt):
            t = instance.types[j]
            if not bool(instance.eligibility[i, j]):
                continue
            if t.max_pods is not None and count[nb] + 1 > t.max_pods:
                continue
            if all(load[nb][d] + float(s.demand[r][d]) <= float(instance.allocatable[j][d]) + _TOL for d in act):
                fits.append(i)
                break
    return fits
