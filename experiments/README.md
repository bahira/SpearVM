# experiments/ — autotuner et bancs de mesure

Boucle ancrée : *générer → compiler → **vérifier** → chronométrer → garder*.
Rien n'est retenu sans mesure ; le journal complet et les conclusions sont dans
[`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md), les résultats bruts dans
`results/`.

| Fichier | Rôle |
| --- | --- |
| `harness.py` | compilation en cache, binding ctypes, vérification (NaN + référence float64), chronométrage alterné min-of-N |
| `gen_gemm.py` | générateur famille [P] : micro-noyau broadcast + packing (paramètres MR, NR, KC, MC, NC, prefetch, déroulage, OpenMP) |
| `gen_dotnt.py` | générateur famille [D] : dot-block NT sans packing |
| `sweep_gemm.py` | autotuner étagé famille [P] (490 variantes mesurées) |
| `sweep_dotnt.py` | autotuner étagé famille [D] (412 variantes) |
| `sweep_flags_threads.py` | options de compilation et montée en threads sur les champions |
| `sweep_shapes.py` | matrice de formes : legacy / pack / dot / numpy |
| `sweep_routing.py` | calibration de la règle d'aiguillage dans la `.so` livrée |
| `sweep_smalltile.py` | seuil en k de la branche « petite tuile » |
| `verify_port.py` | A/B du noyau porté contre le legacy et numpy (tables du README) |
| `verify_gelu_fused.py` | A/B du chemin GEMM+GELU |
| `bench_mlp.py` | impact bout-en-bout sur une itération d'entraînement |
| `smoke.py` | test rapide de 2 variantes (sanité du harnais) |

```bash
make spur_math/libspur_kernels.so      # depuis la racine
cd experiments
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_THREAD_TIMEOUT=1 \
  python verify_port.py st
```

`OPENBLAS_THREAD_TIMEOUT=1` doit être posé **avant l'import de numpy** : sinon
les threads OpenBLAS tournent à vide et affament OpenMP (§0 du journal).
