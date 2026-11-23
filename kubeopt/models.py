"""Modèles de données KubeOpt (B2-a).

Correspondance avec la formulation B1 (+ amendement B1-bis, bande passante) :

    J, c_j, C_j, M_j, B_j  -> InstanceType, Instance.allocatable, Instance.prices   (éq. 1)
    I, d_i                 -> Workload
    e_ij                   -> Instance.eligibility                                  (éq. 11)
    S, I_s, d_{i,s}, eps_s -> Scenario
    X0, N_max, n_j^max     -> Instance.x0, n_max_total, n_max_per_type
    C(X) = sum c_j n_j     -> Instance.cost                                         (éq. 12)
    K(X), condition (5)    -> Instance.total_capacity, Instance.aggregate_capacity_ok
    kappa_s                -> Instance.reference_block
    C_ub                   -> Instance.cost_upper_bound                             (§9.2)

Unités : CPU en vCPU, mémoire en GiB, bande passante en Mbps, coûts en $/h.
Colonnes des tableaux de ressources : 0 = CPU, 1 = RAM, 2 = BW.

Si la bande passante est désactivée, la colonne BW est ignorée partout
(dimensions actives = {CPU, RAM}) et le modèle se réduit exactement à B1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import DataValidationError

CPU, RAM, BW = 0, 1, 2
DIMENSIONS: Tuple[str, str, str] = ("cpu", "ram", "bw")
UNITS: Tuple[str, str, str] = ("vCPU", "GiB", "Mbps")
VALID_ARCHS = frozenset({"x86_64", "arm64"})

#: Tolérance absolue pour les comparaisons de sommes flottantes (capacité >= demande).
FLOAT_TOL = 1e-9


# --------------------------------------------------------------------------- #
# Utilitaires de validation
# --------------------------------------------------------------------------- #
def _check_real(label: str, value: object, *, minimum: float = 0.0, strict: bool = False) -> float:
    """Vérifie qu'une valeur est un réel fini >= minimum (ou > minimum si strict)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DataValidationError(f"{label} doit être un nombre (reçu : {value!r})")
    v = float(value)
    if not math.isfinite(v):
        raise DataValidationError(f"{label} doit être fini (reçu : {value!r})")
    if (strict and v <= minimum) or (not strict and v < minimum):
        op = ">" if strict else ">="
        raise DataValidationError(f"{label} doit être {op} {minimum} (reçu : {value!r})")
    return v


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


# --------------------------------------------------------------------------- #
# Réservations et types d'instances
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Reservation:
    """Capacité réservée r_j par nœud (kube-reserved, system-reserved, éviction).

    Pour chaque dimension : r = fixe + taux * capacité nominale.
    Les valeurs réelles sont inconnues [À valider] : tout est paramétrable,
    par défaut aucune réservation.
    """

    cpu_fixed: float = 0.0
    cpu_rate: float = 0.0
    ram_fixed: float = 0.0
    ram_rate: float = 0.0
    bw_fixed: float = 0.0
    bw_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cpu_fixed", "ram_fixed", "bw_fixed"):
            _check_real(f"Reservation.{name}", getattr(self, name))
        for name in ("cpu_rate", "ram_rate", "bw_rate"):
            v = _check_real(f"Reservation.{name}", getattr(self, name))
            if v >= 1.0:
                raise DataValidationError(f"Reservation.{name} doit être < 1 (reçu : {v})")

    def reserved(self, nominal: Sequence[float]) -> np.ndarray:
        """Vecteur (3,) des quantités réservées pour une capacité nominale donnée."""
        nominal_arr = np.asarray(nominal, dtype=float)
        fixed = np.array([self.cpu_fixed, self.ram_fixed, self.bw_fixed])
        rate = np.array([self.cpu_rate, self.ram_rate, self.bw_rate])
        return fixed + rate * nominal_arr


@dataclass(frozen=True)
class InstanceType:
    """Type EC2 candidat j in J (après filtrage de disponibilité)."""

    name: str
    price: float                       # c_j, $/h On-Demand
    vcpu: float                        # C_j^nom
    ram_gib: float                     # M_j^nom
    bw_mbps: float = 0.0               # B_j^nom (utile seulement si bande passante active)
    reservation: Reservation = Reservation()
    arch: str = "x86_64"
    family: str = ""
    max_pods: Optional[int] = None     # pi_j (limite ENI/IP), None = non contraint
    max_count: Optional[int] = None    # n_j^max (quota), None = non borné
    gpus: int = 0
    labels: frozenset = frozenset()    # étiquettes "clé=valeur" du nœud
    taints: frozenset = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataValidationError("InstanceType.name doit être une chaîne non vide")
        _check_real(f"{self.name}.price", self.price, strict=True)
        _check_real(f"{self.name}.vcpu", self.vcpu, strict=True)
        _check_real(f"{self.name}.ram_gib", self.ram_gib, strict=True)
        _check_real(f"{self.name}.bw_mbps", self.bw_mbps)
        if self.arch not in VALID_ARCHS:
            raise DataValidationError(f"{self.name}.arch invalide : {self.arch!r}")
        if self.max_pods is not None and (not isinstance(self.max_pods, (int, np.integer)) or self.max_pods < 1):
            raise DataValidationError(f"{self.name}.max_pods doit être un entier >= 1 ou None")
        if self.max_count is not None and (not isinstance(self.max_count, (int, np.integer)) or self.max_count < 0):
            raise DataValidationError(f"{self.name}.max_count doit être un entier >= 0 ou None")
        if not isinstance(self.gpus, (int, np.integer)) or self.gpus < 0:
            raise DataValidationError(f"{self.name}.gpus doit être un entier >= 0")
        object.__setattr__(self, "labels", frozenset(self.labels))
        object.__setattr__(self, "taints", frozenset(self.taints))

    def nominal(self) -> np.ndarray:
        """Capacité nominale (3,) = (vCPU, GiB, Mbps)."""
        return np.array([self.vcpu, self.ram_gib, self.bw_mbps], dtype=float)

    def allocatable(self, daemonset_overhead: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        """Capacité utilisable, éq. (1) : C_j = C_nom - r_j - delta (idem RAM, BW)."""
        nominal = self.nominal()
        return nominal - self.reservation.reserved(nominal) - np.asarray(daemonset_overhead, dtype=float)


# --------------------------------------------------------------------------- #
# Workloads
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Workload:
    """Unité de placement i in I (pod ou groupe indivisible) avec ses requests de base.

    Les requests de base servent de valeurs par défaut ; la demande effective par
    scénario d_{i,s} est portée par Scenario.demand. La consommation observée u_i
    n'intervient pas dans la faisabilité (B1, H1) et n'est donc pas modélisée ici.
    """

    name: str
    cpu: float
    ram_gib: float
    bw_mbps: float = 0.0
    arch: Optional[str] = None             # None = indifférent
    node_selector: frozenset = frozenset()  # sous-ensemble des labels exigés
    tolerations: frozenset = frozenset()    # taints tolérés
    gpus: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataValidationError("Workload.name doit être une chaîne non vide")
        _check_real(f"{self.name}.cpu", self.cpu)
        _check_real(f"{self.name}.ram_gib", self.ram_gib)
        _check_real(f"{self.name}.bw_mbps", self.bw_mbps)
        if self.arch is not None and self.arch not in VALID_ARCHS:
            raise DataValidationError(f"{self.name}.arch invalide : {self.arch!r}")
        if not isinstance(self.gpus, (int, np.integer)) or self.gpus < 0:
            raise DataValidationError(f"{self.name}.gpus doit être un entier >= 0")
        object.__setattr__(self, "node_selector", frozenset(self.node_selector))
        object.__setattr__(self, "tolerations", frozenset(self.tolerations))

    def requests(self) -> np.ndarray:
        return np.array([self.cpu, self.ram_gib, self.bw_mbps], dtype=float)


# --------------------------------------------------------------------------- #
# Scénarios
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Scenario:
    """Scénario s in S : état alternatif du système (jamais additionné aux autres).

    members : indices (dans Instance.workloads) des workloads présents I_s.
    demand  : tableau (|I_s|, 3) des demandes d_{i,s} (CPU, RAM, BW).
    epsilon : seuil de débordement eps_s in [0, 1) (R4, B1 §5.6).
    tau_hours : durée optionnelle tau_s (inconnue par défaut, [À valider]).
    """

    name: str
    members: np.ndarray
    demand: np.ndarray
    epsilon: float
    tau_hours: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise DataValidationError("Scenario.name doit être une chaîne non vide")
        members = np.asarray(self.members)
        if members.size == 0:
            members = np.zeros(0, dtype=np.int64)
        elif members.ndim != 1 or not np.issubdtype(members.dtype, np.integer):
            raise DataValidationError(f"[{self.name}] members doit être un vecteur d'entiers")
        members = members.astype(np.int64, copy=True)
        if np.unique(members).size != members.size:
            raise DataValidationError(f"[{self.name}] members contient des doublons")
        demand = np.array(self.demand, dtype=float, copy=True)
        if demand.size == 0:
            demand = demand.reshape(0, 3)
        if demand.ndim != 2 or demand.shape != (members.size, 3):
            raise DataValidationError(
                f"[{self.name}] demand doit avoir la forme ({members.size}, 3), reçu {demand.shape}"
            )
        if not np.all(np.isfinite(demand)) or np.any(demand < 0):
            raise DataValidationError(f"[{self.name}] demand doit être fini et >= 0")
        eps = _check_real(f"[{self.name}] epsilon", self.epsilon)
        if eps >= 1.0:
            raise DataValidationError(f"[{self.name}] epsilon doit être < 1 (reçu : {eps})")
        if self.tau_hours is not None:
            _check_real(f"[{self.name}] tau_hours", self.tau_hours, strict=True)
        object.__setattr__(self, "members", _readonly(members))
        object.__setattr__(self, "demand", _readonly(demand))
        object.__setattr__(self, "epsilon", eps)

    @property
    def n_pods(self) -> int:
        return int(self.members.size)

    def total_demand(self) -> np.ndarray:
        """Demande agrégée D_s (3,) : somme sur les workloads du scénario (jamais sur S)."""
        return self.demand.sum(axis=0)

    @classmethod
    def from_base_requests(
        cls,
        name: str,
        workloads: Sequence[Workload],
        members: Sequence[int],
        epsilon: float,
        tau_hours: Optional[float] = None,
        scale: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    ) -> "Scenario":
        """Construit un scénario à partir des requests de base (optionnellement mises à l'échelle)."""
        members_arr = np.asarray(list(members), dtype=np.int64)
        if members_arr.size and (members_arr.min() < 0 or members_arr.max() >= len(workloads)):
            raise DataValidationError(f"[{name}] indice de workload hors bornes")
        base = (
            np.array([workloads[i].requests() for i in members_arr], dtype=float)
            if members_arr.size
            else np.zeros((0, 3))
        )
        if scale is not None:
            s = np.asarray(scale, dtype=float)
            if s.ndim == 1:
                s = s[:, None]
            base = base * s
        return cls(name=name, members=members_arr, demand=base, epsilon=epsilon, tau_hours=tau_hours)


# --------------------------------------------------------------------------- #
# Instance du problème
# --------------------------------------------------------------------------- #
ScenarioKey = Union[str, int, Scenario]


@dataclass(frozen=True, eq=False)
class Instance:
    """Instance complète du problème (données validées + grandeurs dérivées).

    Les grandeurs dérivées (allocatable, prix, éligibilité...) sont calculées une
    fois à la construction et rendues en lecture seule.
    """

    types: Tuple[InstanceType, ...]
    workloads: Tuple[Workload, ...]
    scenarios: Tuple[Scenario, ...]
    daemonset_overhead: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # delta (CPU, RAM, BW)
    bandwidth_active: bool = False
    n_max_total: Optional[int] = None             # N_max
    x0: Optional[np.ndarray] = None               # X0 (flotte actuelle)
    name: str = "instance"
    provenance: str = "synthetic"

    # Grandeurs dérivées
    prices: np.ndarray = field(init=False, repr=False)
    allocatable: np.ndarray = field(init=False, repr=False)       # (k, 3), éq. (1)
    eligibility: np.ndarray = field(init=False, repr=False)       # (n, k) bool, éq. (11)
    max_pods: np.ndarray = field(init=False, repr=False)          # (k,), +inf si non contraint
    n_max_per_type: np.ndarray = field(init=False, repr=False)    # (k,), +inf si non borné

    def __post_init__(self) -> None:
        types = tuple(self.types)
        workloads = tuple(self.workloads)
        scenarios = tuple(self.scenarios)
        if not types:
            raise DataValidationError("Au moins un type d'instance est requis")
        if not workloads:
            raise DataValidationError("Au moins un workload est requis")
        if not scenarios:
            raise DataValidationError("Au moins un scénario est requis")
        for kind, items in (("types", types), ("workloads", workloads), ("scenarios", scenarios)):
            names = [it.name for it in items]
            if len(set(names)) != len(names):
                raise DataValidationError(f"Noms dupliqués dans {kind}")
        if not all(isinstance(t, InstanceType) for t in types):
            raise DataValidationError("types doit contenir des InstanceType")
        if not all(isinstance(w, Workload) for w in workloads):
            raise DataValidationError("workloads doit contenir des Workload")
        if not all(isinstance(s, Scenario) for s in scenarios):
            raise DataValidationError("scenarios doit contenir des Scenario")

        delta = np.asarray(self.daemonset_overhead, dtype=float)
        if delta.shape != (3,) or not np.all(np.isfinite(delta)) or np.any(delta < 0):
            raise DataValidationError("daemonset_overhead doit être un triplet fini >= 0 (CPU, RAM, BW)")
        if self.n_max_total is not None and (
            isinstance(self.n_max_total, bool)
            or not isinstance(self.n_max_total, (int, np.integer))
            or self.n_max_total < 1
        ):
            raise DataValidationError("n_max_total doit être un entier >= 1 ou None")

        k, n_w = len(types), len(workloads)
        active = list(self.active_dims)

        # Éq. (1) : capacité utilisable
        alloc = np.vstack([t.allocatable(delta) for t in types])
        for j, t in enumerate(types):
            for d in active:
                if alloc[j, d] <= 0:
                    raise DataValidationError(
                        f"Type {t.name} : capacité utilisable {DIMENSIONS[d]} = {alloc[j, d]:.6g} {UNITS[d]} <= 0 "
                        f"(réservations >= capacité nominale, ou bande passante nominale nulle)"
                    )
        if not self.bandwidth_active:
            alloc[:, BW] = 0.0  # colonne neutralisée

        # Éq. (11) : éligibilité e_ij = arch * sélecteur * taints * gpu
        elig = np.ones((n_w, k), dtype=bool)
        for i, w in enumerate(workloads):
            for j, t in enumerate(types):
                elig[i, j] = (
                    (w.arch is None or w.arch == t.arch)
                    and w.node_selector <= t.labels
                    and t.taints <= w.tolerations
                    and t.gpus >= w.gpus
                )
            if not elig[i].any():
                raise DataValidationError(
                    f"Workload {w.name} : aucun type éligible (sum_j e_ij = 0) -> erreur de données (B1 §3.2)"
                )

        for s in scenarios:
            if s.members.size and (s.members.min() < 0 or s.members.max() >= n_w):
                raise DataValidationError(f"[{s.name}] indice de workload hors bornes")

        x0 = None
        if self.x0 is not None:
            x0 = self._as_composition(self.x0, k, "x0")

        object.__setattr__(self, "types", types)
        object.__setattr__(self, "workloads", workloads)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "daemonset_overhead", tuple(float(v) for v in delta))
        object.__setattr__(self, "x0", None if x0 is None else _readonly(x0))
        object.__setattr__(self, "prices", _readonly(np.array([t.price for t in types], dtype=float)))
        object.__setattr__(self, "allocatable", _readonly(alloc))
        object.__setattr__(self, "eligibility", _readonly(elig))
        object.__setattr__(
            self, "max_pods",
            _readonly(np.array([np.inf if t.max_pods is None else t.max_pods for t in types], dtype=float)),
        )
        object.__setattr__(
            self, "n_max_per_type",
            _readonly(np.array([np.inf if t.max_count is None else t.max_count for t in types], dtype=float)),
        )

    # ------------------------------------------------------------------ #
    # Propriétés simples
    # ------------------------------------------------------------------ #
    @property
    def k(self) -> int:
        return len(self.types)

    @property
    def n_workloads(self) -> int:
        return len(self.workloads)

    @property
    def active_dims(self) -> Tuple[int, ...]:
        """Dimensions actives : (CPU, RAM) ou (CPU, RAM, BW) si bande passante activée."""
        return (CPU, RAM, BW) if self.bandwidth_active else (CPU, RAM)

    @property
    def n_dims(self) -> int:
        return len(self.active_dims)

    def get_scenario(self, key: ScenarioKey) -> Scenario:
        if isinstance(key, Scenario):
            if not any(key is s for s in self.scenarios):
                raise DataValidationError(f"Scénario {key.name!r} n'appartient pas à cette instance")
            return key
        if isinstance(key, str):
            for s in self.scenarios:
                if s.name == key:
                    return s
            raise DataValidationError(f"Scénario inconnu : {key!r}")
        if isinstance(key, (int, np.integer)) and 0 <= key < len(self.scenarios):
            return self.scenarios[int(key)]
        raise DataValidationError(f"Clé de scénario invalide : {key!r}")

    # ------------------------------------------------------------------ #
    # Compositions X
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_composition(n: Sequence[float], k: int, label: str = "n") -> np.ndarray:
        arr = np.asarray(n)
        if arr.ndim != 1 or arr.size != k:
            raise DataValidationError(f"{label} doit être un vecteur de longueur {k} (reçu forme {arr.shape})")
        if not np.all(np.isfinite(arr.astype(float))):
            raise DataValidationError(f"{label} doit être fini")
        if not np.issubdtype(arr.dtype, np.integer):
            rounded = np.rint(arr.astype(float))
            if not np.allclose(arr.astype(float), rounded, atol=0.0, rtol=0.0):
                raise DataValidationError(f"{label} doit être entier")
            arr = rounded
        arr = arr.astype(np.int64, copy=True)
        if np.any(arr < 0):
            raise DataValidationError(f"{label} doit être >= 0 (éq. 2)")
        return arr

    def check_composition(self, n: Sequence[float]) -> np.ndarray:
        """Valide la forme, l'intégralité et la positivité de X = (n_1..n_k) ; retourne un int64."""
        return self._as_composition(n, self.k)

    def respects_limits(self, n: Sequence[float]) -> bool:
        """Contraintes de bornes (4) : n_j <= n_j^max et sum_j n_j <= N_max."""
        arr = self.check_composition(n)
        if np.any(arr > self.n_max_per_type):
            return False
        return self.n_max_total is None or int(arr.sum()) <= int(self.n_max_total)

    # ------------------------------------------------------------------ #
    # Coût, capacités, conditions agrégées
    # ------------------------------------------------------------------ #
    def cost(self, n: Sequence[float]) -> float:
        """Coût horaire C(X) = sum_j c_j n_j, éq. (12)."""
        return float(self.prices @ self.check_composition(n))

    def total_capacity(self, n: Sequence[float]) -> np.ndarray:
        """Capacités totales K(X) (3,) = (sum C_j n_j, sum M_j n_j, sum B_j n_j) ; BW=0 si inactive."""
        return self.allocatable.T @ self.check_composition(n).astype(float)

    def aggregate_capacity_ok(self, n: Sequence[float], scenario: ScenarioKey) -> bool:
        """Condition nécessaire (5) : K^d(X) >= D_s^d pour chaque dimension active.

        Nécessaire mais NON suffisante : elle ignore l'indivisibilité des pods,
        l'équilibre CPU/RAM par nœud, l'éligibilité et la limite de pods.
        """
        s = self.get_scenario(scenario)
        act = list(self.active_dims)
        return bool(np.all(self.total_capacity(n)[act] >= s.total_demand()[act] - FLOAT_TOL))

    def reference_block(self, scenario: ScenarioKey) -> np.ndarray:
        """Bloc de référence (1 vCPU, kappa_RAM GiB[, kappa_BW Mbps]) du scénario (B1 §5.3).

        kappa^d_s = D^d_s / D^CPU_s ; lève une erreur si la demande CPU est nulle.
        """
        s = self.get_scenario(scenario)
        d = s.total_demand()
        if d[CPU] <= 0:
            raise DataValidationError(f"[{s.name}] demande CPU nulle : kappa_s non défini")
        block = d[list(self.active_dims)] / d[CPU]
        return block

    # ------------------------------------------------------------------ #
    # Placement individuel : pods impossibles à loger
    # ------------------------------------------------------------------ #
    def unplaceable_workloads(self, scenario: ScenarioKey) -> np.ndarray:
        """Indices globaux des workloads du scénario qu'aucun type éligible ne peut accueillir seuls.

        Un tel pod déborde forcément (z_{i,s} = 1) quelle que soit la flotte : ce n'est pas
        une erreur de données (B1) mais un fait structurel à signaler.
        """
        s = self.get_scenario(scenario)
        if s.n_pods == 0:
            return np.zeros(0, dtype=np.int64)
        act = list(self.active_dims)
        fits = np.all(s.demand[:, None, act] <= self.allocatable[None, :, act] + FLOAT_TOL, axis=2)
        ok = (fits & self.eligibility[s.members]).any(axis=1)
        return s.members[~ok].copy()

    def reference_fleet(self) -> Tuple[np.ndarray, np.ndarray]:
        """Flotte de référence trivialement faisable (pour C_ub si N_max est inconnu, B1 §9.2).

        Chaque workload reçoit un nœud dédié du type éligible le moins cher pouvant l'accueillir
        seul (sur le maximum de ses demandes entre scénarios) ; n_j = max_s (nombre de nœuds de
        type j nécessaires dans s). Retourne (X_ref, indices des workloads inplaçables, exclus).
        """
        n_w, k = self.n_workloads, self.k
        act = list(self.active_dims)
        max_demand = np.zeros((n_w, 3))
        present = np.zeros(n_w, dtype=bool)
        for s in self.scenarios:
            if s.n_pods:
                np.maximum.at(max_demand, s.members, s.demand)
                present[s.members] = True
        fits = np.all(max_demand[:, None, act] <= self.allocatable[None, :, act] + FLOAT_TOL, axis=2)
        fits &= self.eligibility
        price_matrix = np.where(fits, self.prices[None, :], np.inf)
        chosen = price_matrix.argmin(axis=1)
        valid = np.isfinite(price_matrix.min(axis=1))
        excluded = np.flatnonzero(present & ~valid)
        fleet = np.zeros(k, dtype=np.int64)
        for s in self.scenarios:
            members = s.members[valid[s.members]]
            counts = np.bincount(chosen[members], minlength=k)
            fleet = np.maximum(fleet, counts)
        return fleet, excluded

    def cost_upper_bound(self) -> float:
        """C_ub (§9.2) : N_max * max_j c_j si N_max est connu, sinon coût de la flotte de référence."""
        if self.n_max_total is not None:
            return float(self.n_max_total) * float(self.prices.max())
        fleet, _ = self.reference_fleet()
        value = self.cost(fleet)
        if value <= 0:
            raise DataValidationError("C_ub nul : aucune flotte de référence (aucun workload placeable)")
        return value

    def summary(self) -> Dict[str, object]:
        """Résumé compact (journalisation)."""
        return {
            "name": self.name,
            "provenance": self.provenance,
            "k_types": self.k,
            "n_workloads": self.n_workloads,
            "scenarios": {s.name: {"pods": s.n_pods, "epsilon": s.epsilon} for s in self.scenarios},
            "active_dims": [DIMENSIONS[d] for d in self.active_dims],
            "n_max_total": self.n_max_total,
            "has_x0": self.x0 is not None,
        }
