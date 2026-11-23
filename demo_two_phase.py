"""Démonstration B5 : phase 1 (coût) puis phase 2 (équilibre à coût <= (1 + eta) C*). Données SYNTHÉTIQUES."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from kubeopt import GeneratorConfig, SAConfig, generate_instance, run_open_baseline, two_phase_annealing

inst = generate_instance(GeneratorConfig(seed=7, n_base_workloads=60, n_variable_states=2,
                                         cpu_median=1.2, heterogeneity=0.7, bandwidth_active=True))
print(f"Instance : {inst.name} | {inst.provenance}")
print("Objectif lexicographique : (V, coût C(X), équilibre Bal pire cas) -- aucun poids\n")

print(f"{'méthode':44s} {'coût $/h':>9s} {'nœuds':>6s} {'Bal pire cas':>13s} {'V':>6s}")
ref = run_open_baseline(inst, "first_fit_decreasing_open").evaluation
print(f"{'FFD avec ouverture (baseline)':44s} {ref.cost:9.4f} {ref.n_nodes:6d} {ref.worst_balance:13.4f} {ref.violation:6.3f}")

cfg = SAConfig(iterations=20000, seed=0)
for eta in (0.0, 0.05):
    res = two_phase_annealing(inst, cfg, eta=eta, phase2_iterations=6000, phase1_runs=3, phase2_runs=3)
    p1, p2 = res.phase1.evaluation, res.evaluation
    print(f"{'phase 1 (3 départs)':44s} {p1.cost:9.4f} {p1.n_nodes:6d} {p1.worst_balance:13.4f} {p1.violation:6.3f}")
    print(f"{f'phase 2, eta = {eta} (3 départs)':44s} {p2.cost:9.4f} {p2.n_nodes:6d} {p2.worst_balance:13.4f} {p2.violation:6.3f}"
          f"   [C* = {res.phase2.c_star:.4f}, plafond {(1 + eta) * res.phase2.c_star:.4f}]")
    if eta == 0.0:
        print("\nDétail par scénario (phase 2, eta = 0) :")
        for m in p2.scenarios:
            print(f"  {m.scenario:10s} nœuds actifs {m.active_nodes:2d} | utilisation CPU/RAM/BW des actifs "
                  f"{[round(float(x), 3) for x in m.active_utilization]} | Bal {m.balance:.4f}")
        print()
print("\nLecture : à eta = 0 on n'échange jamais de coût contre de l'équilibre ; eta > 0 autorise un surcoût borné.")
