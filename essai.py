import sys
sys.stdout.reconfigure(encoding="utf-8")   # évite les caractères � sous Windows

from kubeopt import (paper_example, PAPER_FLEETS, node_types_from_composition,
                     place_all, validate_solution, evaluate)

inst = paper_example()                                   # exemple fictif de B1 §11
flotte = node_types_from_composition(PAPER_FLEETS["Xd"]) # X_d = 1 x T1 + 1 x T3
solution = place_all(inst, flotte, policy="ffd")
for scenario, affectation in zip(inst.scenarios, solution.assignments):
    print(f"Scénario {scenario.name:7s} -> nœud de chaque pod : {affectation.tolist()}  (-1 = débordement)")
print("Solution valide ?", validate_solution(inst, solution).valid)
ev = evaluate(inst, solution)
for m in ev.scenarios:
    print(f"[{m.scenario}] serveurs actifs = {m.active_nodes} | débordement O = {m.overflow_rate:.4f} "
          f"| fragmentation F = {m.fragmentation:.5f} | équilibre Bal = {m.balance:.4f}")
