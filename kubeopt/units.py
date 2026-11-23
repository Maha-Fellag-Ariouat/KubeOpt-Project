"""Conversions d'unités vers des entiers (nécessaires au solveur CP-SAT, B1 §12).

Unités entières : CPU en millicores, RAM en MiB, bande passante en Mbps.

Arrondis CONSERVATEURS : les capacités sont arrondies vers le bas, les demandes vers le
haut. Une solution faisable en entiers est donc toujours faisable en flottants (la réciproque
peut être fausse de moins d'une unité entière). Le coût est mis à l'échelle en micro-dollars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .errors import DataValidationError
from .models import BW, CPU, RAM, Instance

MILLICORES_PER_VCPU = 1000
MIB_PER_GIB = 1024
MICRO_PER_UNIT = 1_000_000
#: Tolérance absolue (en unités entières) contre le bruit flottant (ex. 0.1 * 1000 = 100.00000000000001).
_TOL = 1e-9


def floor_scaled(value: float, factor: float) -> int:
    """Arrondi vers le bas de value * factor (capacités)."""
    if not math.isfinite(value) or value < 0:
        raise DataValidationError(f"valeur invalide pour la conversion : {value!r}")
    return int(math.floor(value * factor + _TOL))


def ceil_scaled(value: float, factor: float) -> int:
    """Arrondi vers le haut de value * factor (demandes)."""
    if not math.isfinite(value) or value < 0:
        raise DataValidationError(f"valeur invalide pour la conversion : {value!r}")
    return int(math.ceil(value * factor - _TOL))


@dataclass(frozen=True, eq=False)
class IntegerView:
    """Vue entière d'une instance : capacités, demandes par scénario, prix en micro-$/h."""

    capacity: np.ndarray                 # (k, 3) int64 : millicores, MiB, Mbps
    demands: Tuple[np.ndarray, ...]      # par scénario : (|I_s|, 3) int64
    price_micro: np.ndarray              # (k,) int64
    active_dims: Tuple[int, ...]


_FACTORS = np.array([MILLICORES_PER_VCPU, MIB_PER_GIB, 1.0])


def to_integer_view(instance: Instance) -> IntegerView:
    """Construit la vue entière (capacités floor, demandes ceil) d'une instance."""
    k = instance.k
    cap = np.zeros((k, 3), dtype=np.int64)
    for j in range(k):
        for d in (CPU, RAM, BW):
            cap[j, d] = floor_scaled(max(float(instance.allocatable[j, d]), 0.0), _FACTORS[d])
    demands = []
    for s in instance.scenarios:
        dem = np.zeros((s.n_pods, 3), dtype=np.int64)
        for r in range(s.n_pods):
            for d in (CPU, RAM, BW):
                dem[r, d] = ceil_scaled(float(s.demand[r, d]), _FACTORS[d])
        demands.append(dem)
    price_micro = np.array([int(round(float(p) * MICRO_PER_UNIT)) for p in instance.prices], dtype=np.int64)
    cap.setflags(write=False)
    price_micro.setflags(write=False)
    for dem in demands:
        dem.setflags(write=False)
    return IntegerView(capacity=cap, demands=tuple(demands), price_micro=price_micro,
                       active_dims=instance.active_dims)
