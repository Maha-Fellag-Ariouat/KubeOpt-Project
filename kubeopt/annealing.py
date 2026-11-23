"""Recuit simulé sur le PLACEMENT, phase 1 (B4) : minimiser le coût sous service garanti.

Problème (B1 §6-7, objectif de l'énoncé : réduire les coûts d'hébergement) :
    min C(X) = sum_j c_j n_j   s.c.   V = 0   (tous les seuils de service eps_s respectés).

Représentation : flotte de nœuds ouverts (type par nœud) + une affectation par scénario
(nœud, ou -1 = débordement). X est DÉRIVÉ de la flotte. Les placements restent TOUJOURS
respectueux des capacités (CPU, RAM, BW), de l'éligibilité et de la limite de pods : tout
mouvement qui les violerait est rejeté (espace faisable). Le débordement est la variable de
relâchement (B1 §4.5) mesurée par V ; l'état final est re-validé par le validateur indépendant.

Énergie à deux niveaux (B1 §9.2 et §6C) : AUCUN POIDS.
  niveau 1  E1 = C(X)/C_ub si V = 0, sinon 1 + V        (dominance structurelle de la faisabilité)
  niveau 2  g  = 1 - moyenne_s Falkenauer_s, utilisé SEULEMENT si deux états ont le même E1
            (plateaux de coût) : Falkenauer_s = sum_b c_b fill_{b,s}^2 / C(X), fill = charge
            de la dimension la plus chargée ; guide vers la concentration de la charge, donc
            vers des nœuds vidables (idée de sa.py du projet, pondérée par le prix).
Critère de Metropolis lexicographique (B1 §6C, recommandé) : Metropolis sur dE1 si |dE1| > tol,
sinon Metropolis sur dg. Un gain de coût l'emporte donc toujours sur g, sans constante.

Mouvements : relocate, swap, dissolve (évacue un nœud puis le ferme), retype (redimensionne
un nœud), insert (réintègre un pod débordé), split (scinde un nœud en deux types) et merge
(fusionne deux nœuds en un seul de type adapté). split et merge sont des mouvements COMPOSÉS : sans
eux, passer d'un gros nœud à deux nœuds moyens (ou l'inverse) exige plusieurs relocations
successives, que le guidage de niveau 2 (concentration de la charge) bloque. La relocation peut
ouvrir un nouveau nœud (p_open).
Un nœud vide dans tous les scénarios est fermé automatiquement (le coût baisse).

Refroidissement géométrique T <- alpha T. T0 et T_end sont CALIBRÉES sur des écarts d'énergie
échantillonnés : T = -mean(dE+) / ln(p), p = initial_acceptance puis final_acceptance (une
variation montante moyenne est acceptée avec probabilité p). Même principe pour le niveau 2.
Aucune température absolue n'est fixée à la main.

Phase 2 (B5, `balance_annealing` / `two_phase_annealing`) : à coût C <= (1 + eta) C* (eta = 0 par
défaut), minimiser Bal pire cas = max_s Bal_s, Bal_s = moyenne sur les dimensions de l'écart-type de
l'utilisation entre nœuds actifs. Énergie E2 (B1 éq. 33-34) : 1 + V2 si V2 > 0, sinon Bal, avec
V2 = max(V, max(0, C - (1 + eta) C*) / C*). Niveau 2 lexicographique (Metropolis sur la moyenne des Bal_s
à E2 égal) : le maximum est plat hors du pire scénario. C* est ramené au meilleur coût vu si la phase 2
trouve moins cher.

Initialisation : FFD avec ouverture (B3b), fusionné par maximum type par type. Le meilleur état
(clé lexicographique (E1, g)) est mémorisé : le résultat n'est JAMAIS pire que l'initial.
Reproductibilité : tout le hasard passe par numpy.random.default_rng(seed).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import merge_runs, open_assignment
from .errors import DataValidationError
from .evaluation import Evaluation, evaluate
from .models import FLOAT_TOL, Instance
from .solution import OVERFLOW, Solution
from .validator import validate_solution

E_TOL = 1e-12                      # égalité de coût (bruit flottant)
MOVES: Tuple[str, ...] = ("relocate", "swap", "dissolve", "retype", "insert", "split", "merge")


@dataclass(frozen=True)
class SAConfig:
    """Hyperparamètres de la recherche (choix d'implémentation, hors modèle d'optimisation)."""

    iterations: int = 20000
    seed: int = 0
    initial_acceptance: float = 0.3        # probabilité d'accepter une montée moyenne à T0
    final_acceptance: float = 1e-3         # idem à T_end
    calibration_samples: int = 300
    final_reference: str = "mean"          # écart de référence de T_end : "mean" ou "min" des montées échantillonnées
    move_weights: Tuple[Tuple[str, float], ...] = (
        ("relocate", 0.40), ("swap", 0.15), ("dissolve", 0.10), ("retype", 0.10), ("insert", 0.05),
        ("split", 0.10), ("merge", 0.10),
    )
    p_open: float = 0.05                   # probabilité qu'une relocation ouvre un nouveau nœud
    p_low: float = 0.30                    # probabilité de viser le nœud le moins rempli
    patience: Optional[int] = None         # arrêt si le meilleur n'a pas progressé depuis n itérations
    time_limit_s: Optional[float] = None
    log_every: int = 500
    resync_every: int = 2000               # recalcul exact des charges (dérive flottante)

    def validate(self) -> None:
        if self.iterations < 0 or self.calibration_samples < 0 or self.log_every < 1 or self.resync_every < 1:
            raise DataValidationError("iterations, calibration_samples >= 0 ; log_every, resync_every >= 1")
        if not 0 < self.final_acceptance < self.initial_acceptance < 1:
            raise DataValidationError("il faut 0 < final_acceptance < initial_acceptance < 1")
        if not (0 <= self.p_open <= 1 and 0 <= self.p_low <= 1):
            raise DataValidationError("p_open et p_low doivent être dans [0, 1]")
        if self.final_reference not in ("mean", "min"):
            raise DataValidationError("final_reference doit valoir 'mean' ou 'min'")
        if self.patience is not None and self.patience < 1:
            raise DataValidationError("patience doit être >= 1")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise DataValidationError("time_limit_s doit être > 0")
        names = [n for n, _ in self.move_weights]
        if not names or any(n not in MOVES for n in names) or len(set(names)) != len(names):
            raise DataValidationError(f"move_weights doit référencer des mouvements de {MOVES}")
        if any(w < 0 for _, w in self.move_weights) or sum(w for _, w in self.move_weights) <= 0:
            raise DataValidationError("les poids de mouvements doivent être >= 0 et de somme > 0")


# --------------------------------------------------------------------------- #
# État mutable de la recherche
# --------------------------------------------------------------------------- #
class _State:
    """État incrémental : flotte (emplacements de nœuds) + affectations + agrégats de l'énergie."""

    def __init__(self, instance: Instance, node_types: Sequence[int], assignments: Sequence[Sequence[int]],
                 c_ub: float) -> None:
        self.inst = instance
        act = list(instance.active_dims)
        scs = instance.scenarios
        self.d, self.S, self.k = len(act), len(scs), instance.k
        self.dem = [sc.demand[:, act].astype(float) for sc in scs]
        self.elig = [instance.eligibility[sc.members] for sc in scs]
        self.P = [sc.n_pods for sc in scs]
        self.tot = [dm.sum(axis=0) for dm in self.dem]
        self.eps = [sc.epsilon for sc in scs]
        self.type_cap = instance.allocatable[:, act]
        self.prices = instance.prices
        self.max_pods = instance.max_pods
        self.quota = instance.n_max_per_type
        self.n_max_total = instance.n_max_total
        self.c_ub = c_ub
        nt = np.asarray(node_types, dtype=np.int64)
        self.M = int(sum(self.P) + nt.size + 4)
        self.node_type = np.full(self.M, -1, dtype=np.int64)
        self.node_type[: nt.size] = nt
        self.alive = self.node_type >= 0
        self.cap = np.zeros((self.M, self.d))
        if nt.size:
            self.cap[: nt.size] = self.type_cap[nt]
        self.assign = [np.array(a, dtype=np.int64) for a in assignments]
        self.log: List[tuple] = []
        self.touched: set = set()
        self._rec = False
        self.rebuild()
        self.cleanup_all()

    # ---- recalcul exact à partir des affectations ---------------------- #
    def rebuild(self) -> None:
        M, d, S = self.M, self.d, self.S
        self.load = [np.zeros((M, d)) for _ in range(S)]
        self.cnt = [np.zeros(M, dtype=np.int64) for _ in range(S)]
        self.ovf = [np.zeros(d) for _ in range(S)]
        self.nov = [0] * S
        self.pods_on: List[Dict[int, set]] = [dict() for _ in range(S)]
        for s in range(S):
            for r, b in enumerate(self.assign[s].tolist()):
                if b >= 0:
                    self.load[s][b] += self.dem[s][r]
                    self.cnt[s][b] += 1
                    self.pods_on[s].setdefault(b, set()).add(r)
                else:
                    self.ovf[s] += self.dem[s][r]
                    self.nov[s] += 1
        self.type_count = np.bincount(self.node_type[self.alive], minlength=self.k).astype(np.int64)
        self.n_alive = int(self.alive.sum())
        self.fill = np.zeros((S, M))
        self.contrib = np.zeros((S, M))
        self.fsum = np.zeros(S)
        for b in np.flatnonzero(self.alive):
            self._refresh_one(int(b))
        self.touched = set()

    def _refresh_one(self, b: int) -> None:
        alive = bool(self.alive[b])
        price = float(self.prices[self.node_type[b]]) if alive else 0.0
        for s in range(self.S):
            f = float((self.load[s][b] / self.cap[b]).max()) if alive else 0.0
            new = price * f * f
            self.fsum[s] += new - self.contrib[s, b]
            self.contrib[s, b] = new
            self.fill[s, b] = f

    # ---- énergie -------------------------------------------------------- #
    def cost(self) -> float:
        return float(self.prices @ self.type_count)

    def violation(self) -> float:
        v = 0.0
        for s in range(self.S):
            if self.nov[s] == 0:
                continue
            ratios = np.divide(self.ovf[s], self.tot[s], out=np.zeros(self.d), where=self.tot[s] > 0)
            o = float(ratios.sum() / self.d)
            v = max(v, max(0.0, o - self.eps[s]) / (1.0 - self.eps[s]))
        return v

    def energy(self) -> Tuple[float, float]:
        """(E1, g) : E1 = C/C_ub si V = 0 sinon 1 + V ; g = 1 - moyenne_s Falkenauer_s."""
        cost = self.cost()
        v = self.violation()
        e1 = cost / self.c_ub if v <= 0.0 else 1.0 + v
        g = 1.0 - float(np.mean(self.fsum / cost)) if cost > 0 else 0.0
        return e1, g

    def balances(self) -> np.ndarray:
        """Bal_s par scénario : moyenne_d écart-type_b(u_{b,s}^d) sur les nœuds actifs du scénario s."""
        out = np.zeros(self.S)
        for s in range(self.S):
            idx = np.flatnonzero(self.cnt[s] > 0)
            if idx.size:
                u = self.load[s][idx] / self.cap[idx]
                out[s] = float(u.std(axis=0).mean())
        return out

    def balance_worst(self) -> float:
        """Bal pire cas : max_s Bal_s."""
        return float(self.balances().max()) if self.S else 0.0

    def energy2(self, c_star: float, eta: float) -> Tuple[float, float]:
        """(E2, gbar) de la phase 2.

        E2 (B1 éq. 33-34) = 1 + V2 si la contrainte de coût/service est violée, sinon Bal pire cas.
        gbar = Bal moyen sur les scénarios : niveau 2 lexicographique, utilisé SEULEMENT à E2 égal,
        pour que les scénarios qui ne sont pas le pire cas ne dérivent pas au hasard (le maximum est plat).
        """
        v2 = max(self.violation(), max(0.0, self.cost() - (1.0 + eta) * c_star) / c_star)
        bal = self.balances()
        gbar = float(bal.mean()) if self.S else 0.0
        return (1.0 + v2, gbar) if v2 > E_TOL else (float(bal.max()) if self.S else 0.0, gbar)

    # ---- primitives avec journal d'annulation --------------------------- #
    def begin(self) -> None:
        self.log, self.touched, self._rec = [], set(), True

    def _set_pod(self, s: int, r: int, nb: int) -> None:
        ob = int(self.assign[s][r])
        if ob == nb:
            return
        dm = self.dem[s][r]
        if ob >= 0:
            self.load[s][ob] -= dm
            self.cnt[s][ob] -= 1
            pods = self.pods_on[s][ob]
            pods.discard(r)
            if not pods:
                del self.pods_on[s][ob]
            self.touched.add(ob)
        else:
            self.ovf[s] -= dm
            self.nov[s] -= 1
            if self.nov[s] == 0:
                self.ovf[s][:] = 0.0
        if nb >= 0:
            self.load[s][nb] += dm
            self.cnt[s][nb] += 1
            self.pods_on[s].setdefault(nb, set()).add(r)
            self.touched.add(nb)
        else:
            self.ovf[s] += dm
            self.nov[s] += 1
        self.assign[s][r] = nb
        if self._rec:
            self.log.append(("pod", s, r, ob))

    def _open(self, j: int, slot: Optional[int] = None) -> int:
        if slot is None:
            slot = int(np.flatnonzero(~self.alive)[0])
        self.node_type[slot] = j
        self.alive[slot] = True
        self.cap[slot] = self.type_cap[j]
        self.type_count[j] += 1
        self.n_alive += 1
        self.touched.add(slot)
        if self._rec:
            self.log.append(("open", slot))
        return slot

    def _close(self, b: int) -> None:
        j = int(self.node_type[b])
        self.alive[b] = False
        self.node_type[b] = -1
        self.cap[b] = 0.0
        self.type_count[j] -= 1
        self.n_alive -= 1
        self.touched.add(b)
        if self._rec:
            self.log.append(("close", b, j))

    def _retype(self, b: int, j2: int) -> None:
        old = int(self.node_type[b])
        self.type_count[old] -= 1
        self.type_count[j2] += 1
        self.node_type[b] = j2
        self.cap[b] = self.type_cap[j2]
        self.touched.add(b)
        if self._rec:
            self.log.append(("retype", b, old))

    def _undo_log(self) -> None:
        self._rec = False
        for entry in reversed(self.log):
            if entry[0] == "pod":
                self._set_pod(entry[1], entry[2], entry[3])
            elif entry[0] == "open":
                self._close(entry[1])
            elif entry[0] == "close":
                self._open(entry[2], slot=entry[1])
            else:
                self._retype(entry[1], entry[2])
        self.log = []

    def _empty(self, b: int) -> bool:
        return all(self.cnt[s][b] == 0 for s in range(self.S))

    def cleanup(self) -> None:
        """Ferme les nœuds touchés qui sont vides dans tous les scénarios, puis rafraîchit les agrégats."""
        for b in sorted(self.touched):
            if self.alive[b] and self._empty(b):
                self._close(b)
        for b in sorted(self.touched):
            self._refresh_one(b)

    def cleanup_all(self) -> None:
        self._rec = False
        for b in np.flatnonzero(self.alive):
            if self._empty(int(b)):
                self._close(int(b))
        self.rebuild()

    def commit(self) -> None:
        self.log, self.touched, self._rec = [], set(), False

    def revert(self) -> None:
        self._undo_log()
        for b in sorted(self.touched):
            self._refresh_one(b)
        self.log, self.touched, self._rec = [], set(), False

    # ---- aides ---------------------------------------------------------- #
    def fits(self, s: int, r: int, b: int) -> bool:
        if not self.alive[b]:
            return False
        j = self.node_type[b]
        return bool(self.elig[s][r, j] and self.cnt[s][b] < self.max_pods[j]
                    and np.all(self.load[s][b] + self.dem[s][r] <= self.cap[b] + FLOAT_TOL))

    def openable_types(self, s: int, r: int) -> np.ndarray:
        if self.n_max_total is not None and self.n_alive >= self.n_max_total:
            return np.zeros(0, dtype=np.int64)
        mask = (self.elig[s][r] & np.all(self.dem[s][r] <= self.type_cap + FLOAT_TOL, axis=1)
                & (self.type_count < self.quota))
        return np.flatnonzero(mask)

    def place_somewhere(self, rng: np.random.Generator, s: int, r: int, exclude: int, p_open: float) -> bool:
        if rng.random() < p_open:
            types = self.openable_types(s, r)
            if types.size == 0:
                return False
            b = self._open(int(rng.choice(types)))
        else:
            cands = np.flatnonzero(self.alive)
            cands = cands[cands != exclude]
            if cands.size == 0:
                return False
            b = int(rng.choice(cands))
            if not self.fits(s, r, b):
                return False
        self._set_pod(s, r, b)
        return True

    # ---- mouvements ----------------------------------------------------- #
    def mv_relocate(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        s = int(rng.integers(self.S))
        occ = np.flatnonzero(self.cnt[s] > 0)
        if occ.size == 0:
            return False
        src = int(occ[np.argmin(self.fill[s][occ])]) if rng.random() < cfg.p_low else int(rng.choice(occ))
        r = int(rng.choice(sorted(self.pods_on[s][src])))
        return self.place_somewhere(rng, s, r, src, cfg.p_open)

    def mv_swap(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        s = int(rng.integers(self.S))
        occ = np.flatnonzero(self.cnt[s] > 0)
        if occ.size < 2:
            return False
        b1, b2 = (int(x) for x in rng.choice(occ, 2, replace=False))
        i = int(rng.choice(sorted(self.pods_on[s][b1])))
        j = int(rng.choice(sorted(self.pods_on[s][b2])))
        t1, t2 = self.node_type[b1], self.node_type[b2]
        if not (self.elig[s][i, t2] and self.elig[s][j, t1]):
            return False
        n1 = self.load[s][b1] - self.dem[s][i] + self.dem[s][j]
        n2 = self.load[s][b2] - self.dem[s][j] + self.dem[s][i]
        if np.any(n1 > self.cap[b1] + FLOAT_TOL) or np.any(n2 > self.cap[b2] + FLOAT_TOL):
            return False
        self._set_pod(s, i, b2)
        self._set_pod(s, j, b1)
        return True

    def mv_dissolve(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        alive = np.flatnonzero(self.alive)
        if alive.size < 2:
            return False
        if rng.random() < cfg.p_low:
            b = int(alive[np.argmin(self.fill[:, alive].sum(axis=0))])
        else:
            b = int(rng.choice(alive))
        safe_type = np.maximum(self.node_type, 0)
        for s in range(self.S):
            pods = sorted(self.pods_on[s].get(b, ()))
            pods.sort(key=lambda r: -float((self.dem[s][r] / self.cap[b]).max()))
            for r in pods:
                ok = (self.alive & self.elig[s][r, safe_type] & (self.cnt[s] < self.max_pods[safe_type])
                      & np.all(self.load[s] + self.dem[s][r] <= self.cap + FLOAT_TOL, axis=1))
                ok[b] = False
                if ok.any():
                    self._set_pod(s, r, int(ok.argmax()))
                elif self.eps[s] > 0:
                    self._set_pod(s, r, OVERFLOW)           # tolérance de service : le pod part en débordement
                else:
                    self._undo_log()
                    self.log, self.touched = [], set()
                    return False
        return True

    def mv_retype(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        alive = np.flatnonzero(self.alive)
        if alive.size == 0 or self.k < 2:
            return False
        b = int(rng.choice(alive))
        j2 = int(rng.integers(self.k))
        if j2 == self.node_type[b] or self.type_count[j2] >= self.quota[j2]:
            return False
        for s in range(self.S):
            if self.cnt[s][b] == 0:
                continue
            if np.any(self.load[s][b] > self.type_cap[j2] + FLOAT_TOL) or self.cnt[s][b] > self.max_pods[j2]:
                return False
            if not self.elig[s][sorted(self.pods_on[s][b]), j2].all():
                return False
        self._retype(b, j2)
        return True

    def mv_insert(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        scen = [s for s in range(self.S) if self.nov[s] > 0]
        if not scen:
            return self.mv_relocate(rng, cfg)
        s = int(rng.choice(scen))
        r = int(rng.choice(np.flatnonzero(self.assign[s] == OVERFLOW)))
        return self.place_somewhere(rng, s, r, -1, cfg.p_open)

    def mv_split(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        """Scinde un nœud b en deux : b devient de type j1 et un nouveau nœud de type j2 est ouvert.

        Les pods de b (tous scénarios) sont répartis par FFD entre les deux ; échec si l'un ne tient pas.
        """
        alive = np.flatnonzero(self.alive)
        if alive.size == 0:
            return False
        b = int(rng.choice(alive))
        if sum(len(self.pods_on[s].get(b, ())) for s in range(self.S)) < 2:
            return False
        cur = int(self.node_type[b])
        j1, j2 = int(rng.integers(self.k)), int(rng.integers(self.k))
        if self.n_max_total is not None and self.n_alive + 1 > self.n_max_total:
            return False
        tc = self.type_count.copy()
        tc[cur] -= 1
        tc[j1] += 1
        tc[j2] += 1
        if np.any(tc > self.quota):
            return False
        types = (j1, j2)
        caps = (self.type_cap[j1], self.type_cap[j2])
        moves: List[Tuple[int, int]] = []                     # (scénario, pod) à envoyer vers le nouveau nœud
        for s in range(self.S):
            pods = sorted(self.pods_on[s].get(b, ()))
            pods.sort(key=lambda r: -float((self.dem[s][r] / self.cap[b]).max()))
            loads = [np.zeros(self.d), np.zeros(self.d)]
            counts = [0, 0]
            for r in pods:
                for t in (0, 1):
                    if (self.elig[s][r, types[t]] and counts[t] < self.max_pods[types[t]]
                            and np.all(loads[t] + self.dem[s][r] <= caps[t] + FLOAT_TOL)):
                        loads[t] += self.dem[s][r]
                        counts[t] += 1
                        if t == 1:
                            moves.append((s, r))
                        break
                else:
                    return False
        nb = self._open(j2)
        for s, r in moves:
            self._set_pod(s, r, nb)
        if j1 != cur:
            self._retype(b, j1)
        return True

    def mv_merge(self, rng: np.random.Generator, cfg: SAConfig) -> bool:
        """Fusionne deux nœuds b1, b2 : tout va dans b1, retypé en un type j qui contient les deux charges."""
        alive = np.flatnonzero(self.alive)
        if alive.size < 2:
            return False
        b1, b2 = (int(x) for x in rng.choice(alive, 2, replace=False))
        comb = np.zeros(self.d)
        max_cnt = 0
        ok_type = np.ones(self.k, dtype=bool)
        for s in range(self.S):
            comb = np.maximum(comb, self.load[s][b1] + self.load[s][b2])
            max_cnt = max(max_cnt, int(self.cnt[s][b1] + self.cnt[s][b2]))
            pods = sorted(self.pods_on[s].get(b1, set()) | self.pods_on[s].get(b2, set()))
            if pods:
                ok_type &= self.elig[s][pods].all(axis=0)
        cur = int(self.node_type[b1])
        ok_type &= np.all(comb <= self.type_cap + FLOAT_TOL, axis=1) & (self.max_pods >= max_cnt)
        free = self.type_count < self.quota
        free[cur] = True                                       # b1 libère son propre emplacement
        cand = np.flatnonzero(ok_type & free)
        if cand.size == 0:
            return False
        j = int(rng.choice(cand))
        for s in range(self.S):
            for r in sorted(self.pods_on[s].get(b2, ())):
                self._set_pod(s, r, b1)
        if j != cur:
            self._retype(b1, j)
        return True

    def try_move(self, name: str, rng: np.random.Generator, cfg: SAConfig) -> bool:
        """Applique un mouvement ; retourne False (état inchangé) s'il est irréalisable."""
        self.begin()
        if not getattr(self, "mv_" + name)(rng, cfg):
            self.commit()
            return False
        self.cleanup()
        return True

    # ---- export et contrôle -------------------------------------------- #
    def snapshot(self) -> tuple:
        return (self.node_type.copy(), self.alive.copy(), [a.copy() for a in self.assign])

    def restore(self, snap: tuple) -> None:
        self.node_type, self.alive = snap[0].copy(), snap[1].copy()
        self.cap = np.zeros((self.M, self.d))
        idx = np.flatnonzero(self.alive)
        if idx.size:
            self.cap[idx] = self.type_cap[self.node_type[idx]]
        self.assign = [a.copy() for a in snap[2]]
        self.rebuild()

    def compact(self) -> Solution:
        idx = np.flatnonzero(self.alive)
        mapping = np.full(self.M, -1, dtype=np.int64)
        mapping[idx] = np.arange(idx.size)
        return Solution(self.node_type[idx], tuple(np.where(a >= 0, mapping[np.maximum(a, 0)], OVERFLOW)
                                                   for a in self.assign))

    def check_consistency(self) -> None:
        """Compare les agrégats incrémentaux à un recalcul exact (utilisé par les tests)."""
        for s in range(self.S):
            load = np.zeros((self.M, self.d))
            cnt = np.zeros(self.M, dtype=np.int64)
            for r, b in enumerate(self.assign[s].tolist()):
                if b >= 0:
                    assert self.alive[b], "pod sur un nœud fermé"
                    load[b] += self.dem[s][r]
                    cnt[b] += 1
            assert np.allclose(load, self.load[s], atol=1e-7), "charges incohérentes"
            assert np.array_equal(cnt, self.cnt[s]), "compteurs incohérents"
            assert self.nov[s] == int((self.assign[s] < 0).sum()), "compteur de débordement incohérent"
            for b, pods in self.pods_on[s].items():
                assert pods == {r for r, x in enumerate(self.assign[s].tolist()) if x == b}, "ensembles de pods incohérents"
            assert np.all(self.load[s][self.alive] <= self.cap[self.alive] + 1e-7), "capacité violée"
        assert np.array_equal(self.type_count, np.bincount(self.node_type[self.alive], minlength=self.k))
        assert self.n_alive == int(self.alive.sum())
        fsum = np.zeros(self.S)
        for b in np.flatnonzero(self.alive):
            for s in range(self.S):
                f = float((self.load[s][b] / self.cap[b]).max())
                fsum[s] += float(self.prices[self.node_type[b]]) * f * f
        assert np.allclose(fsum, self.fsum, atol=1e-7), "agrégat de Falkenauer incohérent"


# --------------------------------------------------------------------------- #
# Résultat et boucle principale
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class SAResult:
    solution: Solution
    evaluation: Evaluation
    initial_evaluation: Evaluation
    history: Tuple[Dict[str, float], ...]
    stats: Dict[str, object]
    config: SAConfig
    stop_reason: str
    elapsed_s: float
    temperatures: Dict[str, float]


def _acceptance_temperature(deltas: Sequence[float], p: float, reference: str = "mean") -> Optional[float]:
    """T tel que la montée de référence (moyenne ou minimum) soit acceptée avec la probabilité p."""
    if not deltas:
        return None
    ref = float(np.min(deltas)) if reference == "min" else float(np.mean(deltas))
    return ref / -math.log(p)


def _pick_move(rng: np.random.Generator, names: Sequence[str], probs: np.ndarray) -> str:
    return names[int(rng.choice(len(names), p=probs))]


def initial_solution(instance: Instance) -> Solution:
    """Solution de départ : FFD avec ouverture (B3b), flotte fusionnée par maximum type par type."""
    return merge_runs(instance, [open_assignment(instance, s, "ffd") for s in instance.scenarios])


def simulated_annealing(instance: Instance, config: SAConfig = SAConfig(),
                        init: Optional[Solution] = None) -> SAResult:
    """Recuit simulé (phase 1) : minimise C(X) sous V = 0 ; voir la docstring du module."""
    config.validate()
    start = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    init_sol = initial_solution(instance) if init is None else init
    validate_solution(instance, init_sol).raise_if_invalid()
    initial_eval = evaluate(instance, init_sol)
    try:
        c_ub = instance.cost_upper_bound()
    except DataValidationError:
        c_ub = float(instance.prices.max())
    st = _State(instance, init_sol.node_types, init_sol.assignments, c_ub)

    names = [n for n, _ in config.move_weights]
    probs = np.array([w for _, w in config.move_weights], dtype=float)
    probs /= probs.sum()
    stats: Dict[str, object] = {
        "attempted": {n: 0 for n in names}, "applied": {n: 0 for n in names}, "accepted": {n: 0 for n in names},
        "resyncs": 0, "calibration_fallback": False,
    }

    # ---- calibration des températures ---------------------------------- #
    e1, g = st.energy()
    up_e: List[float] = []
    up_g: List[float] = []
    for _ in range(config.calibration_samples):
        name = _pick_move(rng, names, probs)
        if st.try_move(name, rng, config):
            e1n, gn = st.energy()
            if abs(e1n - e1) > E_TOL:
                if e1n > e1:
                    up_e.append(e1n - e1)
            elif gn > g + 1e-12:
                up_g.append(gn - g)
            st.revert()
    t0 = _acceptance_temperature(up_e, config.initial_acceptance)
    t_end = _acceptance_temperature(up_e, config.final_acceptance, config.final_reference)
    tg0 = _acceptance_temperature(up_g, config.initial_acceptance)
    tg_end = _acceptance_temperature(up_g, config.final_acceptance, config.final_reference)
    if t0 is None or tg0 is None:
        stats["calibration_fallback"] = True                   # pas de montée observée : recherche quasi gloutonne
    t0, t_end = (t0, t_end) if t0 is not None else (1e-9, 1e-10)
    tg0, tg_end = (tg0, tg_end) if tg0 is not None else (1e-9, 1e-10)
    n_it = max(config.iterations, 1)
    alpha, alpha_g = (t_end / t0) ** (1.0 / n_it), (tg_end / tg0) ** (1.0 / n_it)
    temp, temp_g = t0, tg0
    temperatures = {"T0": t0, "T_end": t_end, "Tg0": tg0, "Tg_end": tg_end, "alpha": alpha, "alpha_g": alpha_g}

    # ---- boucle principale --------------------------------------------- #
    best_key = (e1, g)
    best_snap = st.snapshot()
    last_improve = 0
    history: List[Dict[str, float]] = []
    window_acc = window_n = 0
    stop = "iterations"
    it = 0
    for it in range(1, config.iterations + 1):
        name = _pick_move(rng, names, probs)
        stats["attempted"][name] += 1
        if st.try_move(name, rng, config):
            stats["applied"][name] += 1
            e1n, gn = st.energy()
            d_e = e1n - e1
            if abs(d_e) <= E_TOL:
                d_g = gn - g
                accept = d_g <= 0.0 or rng.random() < math.exp(-d_g / temp_g)
            else:
                accept = d_e < 0.0 or rng.random() < math.exp(-d_e / temp)
            window_n += 1
            if accept:
                window_acc += 1
                stats["accepted"][name] += 1
                st.commit()
                e1, g = e1n, gn
                if e1 < best_key[0] - E_TOL or (abs(e1 - best_key[0]) <= E_TOL and g < best_key[1] - 1e-12):
                    best_key, best_snap, last_improve = (e1, g), st.snapshot(), it
            else:
                st.revert()
        temp *= alpha
        temp_g *= alpha_g
        if it % config.resync_every == 0:
            st.rebuild()
            stats["resyncs"] += 1
            e1, g = st.energy()
        if it % config.log_every == 0:
            history.append({"iter": it, "T": temp, "E1": e1, "g": g, "cost": st.cost(),
                            "best_E1": best_key[0], "accept_rate": window_acc / window_n if window_n else 0.0})
            window_acc = window_n = 0
        if config.patience is not None and it - last_improve >= config.patience:
            stop = "patience"
            break
        if config.time_limit_s is not None and it % 100 == 0 and time.perf_counter() - start >= config.time_limit_s:
            stop = "time_limit"
            break
    stats["iterations_done"] = it if config.iterations else 0

    st.restore(best_snap)
    solution = st.compact()
    evaluation = evaluate(instance, solution)                  # validateur indépendant : jamais de faux « faisable »
    elapsed = time.perf_counter() - start
    return SAResult(solution, evaluation, initial_eval, tuple(history), stats, config, stop, elapsed, temperatures)


def multistart_annealing(instance: Instance, config: SAConfig = SAConfig(), n_runs: int = 5,
                         init: Optional[Solution] = None) -> Tuple[SAResult, List[SAResult]]:
    """n_runs recuits indépendants (graines config.seed, config.seed + 1, ...) ; garde le meilleur.

    Meilleur = clé lexicographique (V, C) de la phase 1. Les résultats individuels sont retournés
    pour l'analyse statistique (dispersion entre graines).
    """
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 1:
        raise DataValidationError("n_runs doit être un entier >= 1")
    results = [simulated_annealing(instance, replace(config, seed=config.seed + r), init) for r in range(n_runs)]
    best = min(results, key=lambda res: (res.evaluation.violation, res.evaluation.cost))
    return best, results


# --------------------------------------------------------------------------- #
# Phase 2 (B5) : équilibrer la charge à coût (quasi) constant
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Phase2Result:
    solution: Solution
    evaluation: Evaluation
    start_evaluation: Evaluation
    history: Tuple[Dict[str, float], ...]
    stats: Dict[str, object]
    eta: float
    c_star: float                    # C* final (ramené au meilleur coût vu)
    stop_reason: str                 # "iterations", "patience", "time_limit" ou "skipped"
    elapsed_s: float
    temperatures: Dict[str, float]


def balance_annealing(instance: Instance, start: Solution, config: SAConfig = SAConfig(), eta: float = 0.0,
                      iterations: Optional[int] = None) -> Phase2Result:
    """Phase 2 : minimise Bal pire cas sous C <= (1 + eta) C* et V = 0, en partant de `start`.

    Ignorée (stop_reason "skipped", solution inchangée) si `start` viole le service (V > 0) ou coûte 0.
    Le résultat n'est jamais moins équilibré que `start` (meilleur état mémorisé).
    """
    config.validate()
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(eta) or eta < 0:
        raise DataValidationError("eta doit être un réel fini >= 0")
    n_iter = config.iterations if iterations is None else iterations
    if n_iter < 0:
        raise DataValidationError("iterations doit être >= 0")
    start_time = time.perf_counter()
    validate_solution(instance, start).raise_if_invalid()
    start_eval = evaluate(instance, start)
    stats: Dict[str, object] = {"attempted": {}, "applied": {}, "accepted": {}, "resyncs": 0,
                                "calibration_fallback": False, "ratchets": 0, "bal_start": start_eval.worst_balance}
    if start_eval.violation > 0 or start_eval.cost <= 0:
        return Phase2Result(start, start_eval, start_eval, (), stats, float(eta), start_eval.cost, "skipped",
                            time.perf_counter() - start_time, {})
    rng = np.random.default_rng([config.seed, 2])                       # flux distinct de la phase 1
    st = _State(instance, start.node_types, start.assignments, 1.0)
    c_star = start_eval.cost
    names = [n for n, _ in config.move_weights]
    probs = np.array([w for _, w in config.move_weights], dtype=float)
    probs /= probs.sum()
    stats.update({"attempted": {n: 0 for n in names}, "applied": {n: 0 for n in names},
                  "accepted": {n: 0 for n in names}})

    e2, g2 = st.energy2(c_star, eta)
    ups: List[float] = []
    ups_g: List[float] = []
    for _ in range(config.calibration_samples):
        if st.try_move(_pick_move(rng, names, probs), rng, config):
            e2n, gn = st.energy2(c_star, eta)
            if abs(e2n - e2) > E_TOL:
                if e2 < e2n < 1.0:
                    ups.append(e2n - e2)
            elif e2n < 1.0 and gn > g2 + 1e-12:
                ups_g.append(gn - g2)
            st.revert()
    t0 = _acceptance_temperature(ups, config.initial_acceptance)
    t_end = _acceptance_temperature(ups, config.final_acceptance, config.final_reference)
    tg0 = _acceptance_temperature(ups_g, config.initial_acceptance)
    tg_end = _acceptance_temperature(ups_g, config.final_acceptance, config.final_reference)
    if t0 is None or tg0 is None:
        stats["calibration_fallback"] = True
    t0, t_end = (t0, t_end) if t0 is not None else (1e-9, 1e-10)
    tg0, tg_end = (tg0, tg_end) if tg0 is not None else (1e-9, 1e-10)
    n_den = max(n_iter, 1)
    alpha, alpha_g = (t_end / t0) ** (1.0 / n_den), (tg_end / tg0) ** (1.0 / n_den)
    temp, temp_g = t0, tg0
    temperatures = {"T0": t0, "T_end": t_end, "Tg0": tg0, "Tg_end": tg_end, "alpha": alpha, "alpha_g": alpha_g}

    best_e2, best_g, best_snap, last_improve = e2, g2, st.snapshot(), 0
    history: List[Dict[str, float]] = []
    window_acc = window_n = 0
    stop, it = "iterations", 0
    for it in range(1, n_iter + 1):
        name = _pick_move(rng, names, probs)
        stats["attempted"][name] += 1
        if st.try_move(name, rng, config):
            stats["applied"][name] += 1
            e2n, gn = st.energy2(c_star, eta)
            d = e2n - e2
            if abs(d) <= E_TOL:
                dg = gn - g2
                accept = dg <= 0.0 or rng.random() < math.exp(-dg / temp_g)
            else:
                accept = d < 0.0 or rng.random() < math.exp(-d / temp)
            window_n += 1
            if accept:
                window_acc += 1
                stats["accepted"][name] += 1
                st.commit()
                e2, g2 = e2n, gn
                cost = st.cost()
                if cost < c_star - 1e-12 * c_star and st.violation() <= 0.0:   # moins cher que C* : on resserre la contrainte
                    c_star = cost
                    e2, g2 = st.energy2(c_star, eta)
                    best_e2, best_g, best_snap, last_improve = e2, g2, st.snapshot(), it
                    stats["ratchets"] += 1
                elif e2 < best_e2 - 1e-15 or (abs(e2 - best_e2) <= 1e-15 and g2 < best_g - 1e-12):
                    best_e2, best_g, best_snap, last_improve = e2, g2, st.snapshot(), it
            else:
                st.revert()
        temp *= alpha
        temp_g *= alpha_g
        if it % config.resync_every == 0:
            st.rebuild()
            stats["resyncs"] += 1
            e2, g2 = st.energy2(c_star, eta)
        if it % config.log_every == 0:
            history.append({"iter": it, "T": temp, "E2": e2, "gbar": g2, "best_E2": best_e2, "cost": st.cost(),
                            "accept_rate": window_acc / window_n if window_n else 0.0})
            window_acc = window_n = 0
        if config.patience is not None and it - last_improve >= config.patience:
            stop = "patience"
            break
        if config.time_limit_s is not None and it % 100 == 0 and time.perf_counter() - start_time >= config.time_limit_s:
            stop = "time_limit"
            break
    stats["iterations_done"] = it if n_iter else 0
    st.restore(best_snap)
    solution = st.compact()
    evaluation = evaluate(instance, solution)
    return Phase2Result(solution, evaluation, start_eval, tuple(history), stats, float(eta), c_star, stop,
                        time.perf_counter() - start_time, temperatures)


@dataclass(frozen=True, eq=False)
class TwoPhaseResult:
    phase1: SAResult
    phase2: Phase2Result                       # meilleure exécution de la phase 2
    phase2_runs: Tuple[Phase2Result, ...] = ()  # toutes les exécutions (dispersion entre graines)

    @property
    def solution(self) -> Solution:
        return self.phase2.solution

    @property
    def evaluation(self) -> Evaluation:
        return self.phase2.evaluation


def two_phase_annealing(instance: Instance, config: SAConfig = SAConfig(), eta: float = 0.0,
                        phase2_iterations: Optional[int] = None, phase1_runs: int = 1,
                        phase2_runs: int = 1) -> TwoPhaseResult:
    """Phase 1 (coût, éventuellement multi-départ) puis phase 2 (équilibre à coût <= (1 + eta) C*).

    La phase 2 peut être répétée (`phase2_runs`, graines config.seed, +1, ...) depuis la même solution de
    phase 1 : le paysage de l'équilibre a des minima locaux pour relocate/swap. On garde la meilleure
    exécution selon (V, Bal pire cas, Bal moyen).
    """
    for label, v in (("phase1_runs", phase1_runs), ("phase2_runs", phase2_runs)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise DataValidationError(f"{label} doit être un entier >= 1")
    if phase1_runs == 1:
        p1 = simulated_annealing(instance, config)
    else:
        p1, _ = multistart_annealing(instance, config, phase1_runs)
    runs = tuple(balance_annealing(instance, p1.solution, replace(config, seed=config.seed + r), eta,
                                   phase2_iterations) for r in range(phase2_runs))
    best = min(runs, key=lambda r: (r.evaluation.violation, r.evaluation.worst_balance,
                                    float(np.mean([m.balance for m in r.evaluation.scenarios]))))
    return TwoPhaseResult(p1, best, runs)
