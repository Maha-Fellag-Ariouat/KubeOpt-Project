"""Démonstration B4 : recuit simulé (phase 1) vs baselines, catalogue hétérogène (prix FICTIFS)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import statistics

from kubeopt import GeneratorConfig, SAConfig, generate_instance, run_all_open_baselines, simulated_annealing

inst = generate_instance(GeneratorConfig(seed=7, n_base_workloads=60, n_variable_states=2,
                                         cpu_median=1.2, heterogeneity=0.7, bandwidth_active=True))
print(f"Instance : {inst.name} | {inst.provenance}")
print("Objectif : minimiser le coût C(X) sous service garanti (eps = 0 : tous les conteneurs placés)\n")

print(f"{'méthode':34s} {'coût $/h':>9s} {'nœuds':>6s} {'V':>6s}")
base = run_all_open_baselines(inst)
for nom, res in base.items():
    ev = res.evaluation
    print(f"{nom:34s} {ev.cost:9.4f} {ev.n_nodes:6d} {ev.violation:6.3f}")

seeds = range(5)
runs = [simulated_annealing(inst, SAConfig(iterations=20000, seed=s)) for s in seeds]
for s, r in zip(seeds, runs):
    ev = r.evaluation
    print(f"{'recuit simulé (graine ' + str(s) + ')':34s} {ev.cost:9.4f} {ev.n_nodes:6d} {ev.violation:6.3f}   [{r.elapsed_s:.1f} s]")
costs = [r.evaluation.cost for r in runs]
print(f"\nRecuit sur {len(costs)} graines : moyenne {statistics.mean(costs):.4f} | min {min(costs):.4f} | max {max(costs):.4f} "
      f"| écart-type {statistics.pstdev(costs):.4f}")
ffd = base["first_fit_decreasing_open"].evaluation.cost
print(f"Gain moyen vs FFD avec ouverture : {100 * (1 - statistics.mean(costs) / ffd):.1f} %  (une instance, données synthétiques)")

best = min(runs, key=lambda r: r.evaluation.cost)
print("\nConvergence (graine la plus performante) :")
print(f"{'itération':>10s} {'T':>10s} {'coût courant':>13s} {'meilleur E1':>12s} {'acceptation':>12s}")
for h in best.history[:: max(1, len(best.history) // 8)]:
    print(f"{h['iter']:10d} {h['T']:10.5f} {h['cost']:13.4f} {h['best_E1']:12.5f} {h['accept_rate']:12.3f}")
