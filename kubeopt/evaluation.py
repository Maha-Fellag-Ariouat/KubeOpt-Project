"""Évaluation complète d'une solution (fitness vectorielle de B1 §9, sans énergie scalaire).

`evaluate` valide d'abord la solution avec le validateur indépendant : une solution qui viole
une contrainte dure lève InvalidPlacementError, elle n'est JAMAIS évaluée comme faisable.
Les énergies E1/E2 (§9.2-9.3) relèvent de l'étape B4/B5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .metrics import (
    ChangeMetrics, ScenarioMetrics, change_metrics, compute_scenario_metrics, fleet_violation,
)
from .models import Instance
from .solution import Solution
from .validator import validate_solution


@dataclass(frozen=True, eq=False)
class Evaluation:
    composition: np.ndarray                  # X dérivé de la flotte
    n_nodes: int
    cost: float                              # C(X), $/h (éq. 12)
    scenarios: Tuple[ScenarioMetrics, ...]
    violation: float                         # V(X) (éq. 31)
    changes: Optional[ChangeMetrics]
    elapsed_s: Optional[float] = None

    @property
    def worst_balance(self) -> float:
        """Bal le plus défavorable sur les scénarios (pire cas, R3 de B1 §5.6)."""
        return max((m.balance for m in self.scenarios), default=0.0)

    def objective_key(self) -> Tuple[float, float, float]:
        """Clé lexicographique de l'objectif : (V, C(X), Bal pire cas), à minimiser dans cet ordre.

        Priorité stricte par comparaison de tuples : aucun poids. La phase 2 avec tolérance
        eta > 0 (C <= (1 + eta) C*) relève du recuit (B5) ; ici eta = 0.
        """
        return (self.violation, self.cost, self.worst_balance)

    def by_scenario(self, name: str) -> ScenarioMetrics:
        for m in self.scenarios:
            if m.scenario == name:
                return m
        raise KeyError(name)

    def log_record(self) -> Dict[str, object]:
        """Enregistrement à journaliser à chaque évaluation (B1 §9)."""
        rec: Dict[str, object] = {
            "X": self.composition.tolist(), "n_nodes": self.n_nodes, "C": self.cost, "V": self.violation,
            "feasible_service": self.violation == 0.0, "time_s": self.elapsed_s,
        }
        for m in self.scenarios:
            rec[f"O[{m.scenario}]"] = m.overflow_rate
            rec[f"F[{m.scenario}]"] = m.fragmentation
            rec[f"U[{m.scenario}]"] = dict(zip(m.dims, m.utilization.tolist()))
            rec[f"sigma[{m.scenario}]"] = dict(zip(m.dims, m.sigma.tolist()))
            rec[f"active_nodes[{m.scenario}]"] = m.active_nodes
            rec[f"Bal[{m.scenario}]"] = m.balance
            rec[f"U_active[{m.scenario}]"] = dict(zip(m.dims, m.active_utilization.tolist()))
        if self.changes is not None:
            rec.update({"delta_plus": self.changes.delta_plus, "delta_minus": self.changes.delta_minus,
                        "L1": self.changes.l1})
        return rec


def evaluate(instance: Instance, solution: Solution, *, u_star: Optional[float] = None,
             budget: Optional[float] = None, gamma_plus: Optional[float] = None,
             gamma_minus: Optional[float] = None, validate: bool = True,
             elapsed_s: Optional[float] = None) -> Evaluation:
    """Évalue une solution. `validate=False` est réservé aux boucles internes déjà sûres."""
    if validate:
        validate_solution(instance, solution).raise_if_invalid()
    comp = solution.composition(instance.k)
    cost = instance.cost(comp)
    metrics = tuple(
        compute_scenario_metrics(instance, solution.node_types, s, a, u_star)
        for s, a in zip(instance.scenarios, solution.assignments)
    )
    return Evaluation(
        composition=comp, n_nodes=solution.n_nodes, cost=cost, scenarios=metrics,
        violation=fleet_violation(instance, metrics, cost=cost, budget=budget),
        changes=change_metrics(instance, comp, gamma_plus, gamma_minus), elapsed_s=elapsed_s,
    )
