"""Représentation d'une solution (B1-bis) : flotte de nœuds ouverts + affectation par scénario.

    node_types[b] = j(b) : type du nœud b (b = 0..N-1)
    assignments[s][r]    : nœud hébergeant le pod de rang r du scénario s, ou -1 s'il déborde
                           (z_{i,s} = 1). Le rang r suit l'ordre de Scenario.members.

La flotte est commune à tous les scénarios ; les placements sont indépendants entre scénarios.
La composition X = (n_1..n_k) est DÉRIVÉE : n_j = nombre de nœuds de type j.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .errors import DataValidationError

OVERFLOW = -1


def _int_vector(values, label: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.ndim != 1 or not np.issubdtype(arr.dtype, np.integer):
        raise DataValidationError(f"{label} doit être un vecteur d'entiers")
    out = arr.astype(np.int64, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class Solution:
    """Flotte (types des nœuds ouverts) + une affectation par scénario de l'instance."""

    node_types: np.ndarray
    assignments: Tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        nt = _int_vector(self.node_types, "node_types")
        nt.setflags(write=False)
        object.__setattr__(self, "node_types", nt)
        object.__setattr__(
            self, "assignments",
            tuple(_int_vector(a, f"assignments[{i}]") for i, a in enumerate(self.assignments)),
        )

    @property
    def n_nodes(self) -> int:
        return int(self.node_types.size)

    def composition(self, k: int) -> np.ndarray:
        """X = (n_1..n_k) dérivé de la flotte."""
        if self.node_types.size and (self.node_types.min() < 0 or self.node_types.max() >= k):
            raise DataValidationError("node_types contient un type hors de [0, k)")
        return np.bincount(self.node_types, minlength=k).astype(np.int64)


def node_types_from_composition(n: Sequence[int]) -> np.ndarray:
    """Flotte canonique associée à X : n_j nœuds de type j, triés par type croissant."""
    arr = np.asarray(n)
    if arr.ndim != 1 or (arr.size and not np.issubdtype(arr.dtype, np.integer)):
        raise DataValidationError("n doit être un vecteur d'entiers")
    if np.any(arr < 0):
        raise DataValidationError("n doit être >= 0")
    return np.repeat(np.arange(arr.size, dtype=np.int64), arr)
