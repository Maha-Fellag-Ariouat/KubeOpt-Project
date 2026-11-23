"""Solveur EXACT minimal (petites instances) : référence indépendante pour valider le recuit.

Cherche la composition de coût minimal telle que chaque scénario puisse être placé avec un taux
de débordement O_s <= eps_s. Énumération des compositions par coût croissant + recherche en
profondeur de l'affectation (symétries entre nœuds identiques élaguées). Exponentiel : usage tests.
"""
import itertools

import numpy as np

TOL = 1e-9


def _scenario_feasible(inst, node_types, s):
    act = list(inst.active_dims)
    sc = inst.scenarios[s]
    dem = sc.demand[:, act]
    total = dem.sum(axis=0)
    cap = [inst.allocatable[j][act] for j in node_types]
    maxp = [inst.max_pods[j] for j in node_types]
    elig = inst.eligibility[sc.members]
    m, d, n = dem.shape[0], len(act), len(node_types)
    order = np.argsort(-(dem / np.maximum(total, 1e-12)).max(axis=1), kind="stable")
    loads = [np.zeros(d) for _ in range(n)]
    counts = [0] * n
    budget = sc.epsilon

    def rate(ovf):
        return float(np.divide(ovf, total, out=np.zeros(d), where=total > 0).sum() / d)

    def rec(pos, ovf):
        if rate(ovf) > budget + TOL:
            return False
        if pos == m:
            return True
        r = order[pos]
        seen = set()
        for b in range(n):
            key = (node_types[b], tuple(np.round(loads[b], 9)), counts[b])
            if key in seen:
                continue
            seen.add(key)
            if (elig[r, node_types[b]] and counts[b] < maxp[b]
                    and np.all(loads[b] + dem[r] <= cap[b] + TOL)):
                loads[b] += dem[r]; counts[b] += 1
                ok = rec(pos + 1, ovf)
                loads[b] -= dem[r]; counts[b] -= 1
                if ok:
                    return True
        return rec(pos + 1, ovf + dem[r])           # le pod déborde

    return rec(0, np.zeros(d))


def feasible_composition(inst, comp):
    nt = [j for j, c in enumerate(comp) for _ in range(int(c))]
    return all(_scenario_feasible(inst, nt, s) for s in range(len(inst.scenarios)))


def exact_min_cost(inst, max_nodes):
    """(coût minimal, composition) sur les compositions de <= max_nodes nœuds ; (inf, None) si aucune."""
    k = inst.k
    comps = [c for c in itertools.product(range(max_nodes + 1), repeat=k)
             if sum(c) <= max_nodes and all(c[j] <= inst.n_max_per_type[j] for j in range(k))]
    comps.sort(key=lambda c: (float(inst.prices @ np.array(c)), c))
    for c in comps:
        if feasible_composition(inst, c):
            return float(inst.prices @ np.array(c)), c
    return float("inf"), None


def _scenario_min_balance(inst, node_types, s):
    """Bal_s minimal sur TOUS les placements complets (eps ignoré : tous les pods placés) ; inf si aucun."""
    act = list(inst.active_dims)
    sc = inst.scenarios[s]
    dem = sc.demand[:, act]
    cap = [inst.allocatable[j][act] for j in node_types]
    maxp = [inst.max_pods[j] for j in node_types]
    elig = inst.eligibility[sc.members]
    m, n = dem.shape[0], len(node_types)
    loads = [np.zeros(len(act)) for _ in range(n)]
    counts = [0] * n
    best = [float("inf")]

    def rec(r):
        if r == m:
            used = [b for b in range(n) if counts[b] > 0]
            if used:
                u = np.array([loads[b] / cap[b] for b in used])
                best[0] = min(best[0], float(u.std(axis=0).mean()))
            else:
                best[0] = 0.0
            return
        seen = set()
        for b in range(n):
            key = (node_types[b], tuple(np.round(loads[b], 9)), counts[b])
            if key in seen:
                continue
            seen.add(key)
            if (elig[r, node_types[b]] and counts[b] < maxp[b] and np.all(loads[b] + dem[r] <= cap[b] + TOL)):
                loads[b] += dem[r]; counts[b] += 1
                rec(r + 1)
                loads[b] -= dem[r]; counts[b] -= 1

    rec(0)
    return best[0]


def exact_min_balance(inst, cost_limit, max_nodes):
    """Bal pire cas minimal sur les compositions de coût <= cost_limit (placements complets, eps = 0).

    Les scénarios étant indépendants une fois la flotte fixée : min_X max_s min_placement Bal_s.
    """
    k = inst.k
    best = float("inf")
    for c in itertools.product(range(max_nodes + 1), repeat=k):
        if sum(c) > max_nodes or any(c[j] > inst.n_max_per_type[j] for j in range(k)):
            continue
        if float(inst.prices @ np.array(c)) > cost_limit + 1e-9:
            continue
        nt = [j for j, cnt in enumerate(c) for _ in range(int(cnt))]
        vals = [_scenario_min_balance(inst, nt, s) for s in range(len(inst.scenarios))]
        if all(v < float("inf") for v in vals):
            best = min(best, max(vals))
    return best
