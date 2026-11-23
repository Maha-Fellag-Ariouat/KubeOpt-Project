# KubeOpt — B2 (données, placement, métriques), B3/B3b (baselines), B4/B5 (recuit simulé, phases 1 et 2)

    pip install numpy pytest
    python -m pytest -q          # 430 tests
    python demo_baselines.py     # B3  : RR / FF / FFD sur m serveurs identiques
    python demo_open.py          # B3b : baselines avec ouverture, catalogue hétérogène (prix fictifs)
    python demo_sa.py            # B4  : recuit simulé (phase 1) vs baselines
    python demo_two_phase.py     # B5  : phase 1 (coût) + phase 2 (équilibre à coût <= (1+eta) C*)
    python essai.py              # démonstration B2 (exemple B1 §11)

Objectif (énoncé du Projet 3) : lexmin (V, C(X), Bal) — tout placer, minimiser le coût d'hébergement,
puis équilibrer la charge entre serveurs actifs ; eta = 0 par défaut (aucun poids). Avec des serveurs
identiques, coût = prix x serveurs actifs (B3).

Modules :
- `errors`, `models`, `units`, `generator` (exemple B1 §11, instances synthétiques, `generate_cluster_instance`) ;
- `solution`, `placement` (FF / FFD), `validator` (indépendant), `metrics` (O, U, F, σ, V, Bal, N_act), `evaluation` ;
- `baselines` : RR, FF, FFD sur cluster donné (adaptés de baselines.py du projet) ; B3b : FF/FFD avec ouverture du type
  éligible le moins cher, RR sur flotte mono-type dimensionnée par FFD, fusion par maximum type par type.

Toutes les données générées sont fictives/synthétiques.
- `annealing` : recuit simulé sur le placement (phase 1 : coût sous service garanti), énergie à 2 niveaux sans poids,
  7 mouvements, températures calibrées, multi-départ. `tests/exact_reference.py` : solveur exact pour petites instances.
  Phase 2 (`balance_annealing`, `two_phase_annealing`) : minimise Bal pire cas à coût <= (1+eta)·C*, eta = 0 par défaut.
