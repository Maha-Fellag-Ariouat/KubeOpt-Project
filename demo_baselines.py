"""Démonstration B3 : RR, FF, FFD sur un cluster de m serveurs identiques (données SYNTHÉTIQUES)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from kubeopt import generate_cluster_instance, run_all_baselines, cluster_nodes, server_lower_bound

# 60 conteneurs, serveurs de 8 vCPU / 32 GiB, m = 60 serveurs disponibles (assez pour tout placer)
inst = generate_cluster_instance(seed=7, n_containers=60, m=60, server_vcpu=8, ram_per_vcpu=4.0,
                                 cpu_median=1.2, heterogeneity=0.7, n_variable_states=1)
nodes = cluster_nodes(inst)
resultats = run_all_baselines(inst, nodes)

for scenario in inst.scenarios:
    print(f"\n=== Scénario {scenario.name} : {scenario.n_pods} conteneurs, "
          f"plancher théorique = {server_lower_bound(inst, nodes, scenario)} serveurs ===")
    print(f"{'algorithme':24s} {'serveurs actifs':>16s} {'CPU actifs':>11s} {'RAM actifs':>11s} {'Bal':>7s} {'débordés':>9s}")
    for nom, res in resultats.items():
        m = res.evaluation.by_scenario(scenario.name)
        print(f"{nom:24s} {m.active_nodes:16d} {m.active_utilization[0]:11.3f} {m.active_utilization[1]:11.3f} "
              f"{m.balance:7.3f} {m.overflow_pods:9d}")

print("\nTemps de placement (les 3 scénarios cumulés) :")
for nom, res in resultats.items():
    print(f"  {nom:24s} {res.elapsed_s * 1000:8.2f} ms")
