# Champions superspear dans SpearVM

Source : [github.com/bahira/superspear](https://github.com/bahira/superspear), registre `ledger/*.json`
(90 tâches). Pour chaque tâche : une forme « précise » (champion) et une forme « fast » (point de
fonctionnement rapide, validée au niveau 2).

## Comparatif (grille [-8, 8], référence exacte)

Script : `examples/compare_superspear_champions.py`.

| Fonction | Forme | L-inf | MSE |
|---|---|---|---|
| GELU | SpearVM `gelu` v1 (ancien fusionné) | 7.97e-2 | 4.07e-4 |
| GELU | SpearVM `gelu_quintic` | 1.74e-2 | 1.29e-4 |
| GELU | **SpearVM `gelu_erf` (= champion superspear exact)** | **2.05e-5** | **6.2e-11** |
| GELU | superspear fast `1.010719·relu(x) − 0.057684` | 1.21e-1 | 2.69e-3 |
| Sigmoid | SpearVM (0.5 + 0.5·tanh Padé[3/4]) | 7.81e-4 | 9.4e-8 |
| Sigmoid | superspear fast `0.619·x/(1.3696+|x|)+0.5` | 4.32e-2 | 4.58e-4 |
| SiLU | SpearVM (x·sigmoid Padé) | 1.29e-3 | 4.0e-7 |
| SiLU | superspear fast (relu) | 1.59e-1 | 8.53e-3 |
| Tanh | SpearVM Padé[3/4] | 1.56e-3 | 2.5e-7 |
| Tanh | superspear `tanh_sat` fast (clip) | 2.13e-1 | 8.19e-3 |

## Ce qui a été retenu

- **Champion GELU exact** : la loi exacte `0.5·x·(1+erf(x/√2))` est le champion précis de superspear.
  SpearVM l'implémente déjà (`gelu_erf`, rationnel erf_v2). Il est maintenant utilisé :
  - dans **l'épilogue du FFN fusionné** (`matmul_nt_gelu`, f32 et f64, AVX-512 et replis AVX2/scalaire) ;
  - dans **la dérivée** (`gelu_backward`, dérivée analytique exacte du rationnel, vérifiée à 1,4e-9 contre différences finies) ;
  - dans l'opération `gelu` du graphe (`spur_math.graph`).
  Effet mesuré : erreur du fusionné contre la loi exacte **0.08 → 2.2e-5** ; coût ≈ +5 à 7 % sur le GEMM fusionné.
- **Fast slots non retenus** : les fast slots superspear (relu, clip, rationnel simple) sont **moins précis**
  que les noyaux SpearVM en place, avec une erreur max 5 à 50 fois plus grande, pour un gain de nombre d'opérations.
  Leur intérêt est le coût d'opérations sur des backends scalaires, pas la qualité numérique ; ils ne
  remplacent donc aucun noyau SpearVM. Ils restent documentés ici comme points de fonctionnement « rapide » à part.

## Impact sur le benchmark encodeur (`docs/BENCH_ENCODER_E2E.md`)

| | avant (GELU v1 fusionnée) | après (GELU exacte fusionnée) |
|---|---|---|
| écart max vs PyTorch | 1.36e-1 | **1.0e-5** |
| cosinus vs PyTorch | 0.99730 | **1.000000** |
| batch 1 : SpearVM vs torch / ORT | ×0.89 / ×0.84 | ×0.79 / ×0.69 (bruit VM ±15 %) |
| batch 8 : SpearVM vs torch / ORT | ×0.61 / ×0.56 | ×0.81 / ×0.77 (bruit VM ±15 %) |

Le classement ne change pas : SpearVM reste plus lent que PyTorch et ONNX Runtime sur ce modèle,
mais la sortie est désormais numériquement équivalente à la référence.
