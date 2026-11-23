"""Démonstration B3b : baselines avec ouverture sur un catalogue hétérogène (prix FICTIFS)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from kubeopt import GeneratorConfig, generate_instance, run_all_open_baselines

# 18 types synthétiques (3 familles x 6 tailles), 60 conteneurs de base (+40 % au pic), 2 états variables
inst = generate_instance(GeneratorConfig(seed=7, n_base_workloads=60, n_variable_states=2,
                                         cpu_median=1.2, heterogeneity=0.7, bandwidth_active=True))
print(f"Instance : {inst.name} | {inst.provenance}")
print(f"Scénarios : " + ", ".join(f"{s.name} ({s.n_pods} conteneurs)" for s in inst.scenarios))

resultats = run_all_open_baselines(inst)
print(f"\n{'algorithme':28s} {'coût $/h':>9s} {'nœuds':>6s} {'composition (types utilisés)':s}")
for nom, res in resultats.items():
    ev = res.evaluation
    utilises = {inst.types[j].name: int(n) for j, n in enumerate(ev.composition) if n}
    print(f"{nom:28s} {ev.cost:9.4f} {ev.n_nodes:6d} {utilises}")

print(f"\n{'algorithme':28s} {'scénario':>10s} {'actifs':>7s} {'CPU act.':>9s} {'RAM act.':>9s} {'BW act.':>8s} {'Bal':>6s} {'débordés':>9s}")
for nom, res in resultats.items():
    for m in res.evaluation.scenarios:
        u = m.active_utilization
        print(f"{nom:28s} {m.scenario:>10s} {m.active_nodes:7d} {u[0]:9.3f} {u[1]:9.3f} {u[2]:8.3f} {m.balance:6.3f} {m.overflow_pods:9d}")
print("\nClé lexicographique (V, coût, Bal pire cas) :")
for nom, res in resultats.items():
    v, c, b = res.evaluation.objective_key()
    print(f"  {nom:28s} V={v:.3f}  C={c:.4f}  Bal={b:.3f}")
