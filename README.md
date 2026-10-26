# SpearVM

[![CI](https://github.com/bahira/SpearVM/actions/workflows/ci.yml/badge.svg)](https://github.com/bahira/SpearVM/actions/workflows/ci.yml)

**Accélérateur mathématique SIMD** — noyaux transcendantaux vectorisés AVX2/AVX-512,
GELU exacte, matmul tuilé, couche FFN fusionnée avec rétropropagation, entraînement
de MLP, inférence par lots avec poids pré-packés, et mini-JIT x64 portable
Linux/macOS/Windows. Python/C. CPU seulement, pas de GPU.

```bash
pip install spur-math
```

> Windows : wheel binaire prêt. Linux/macOS : le sdist embarque les sources C ;
> au premier import, `spur_math` se compile tout seul si gcc/clang est présent
> (cache dans `spur_math/_native/`), sinon suivez le message d'aide.
> CPU requis : AVX2 + FMA (Haswell 2013+, Zen 2017+) — vérifié à l'import.
> AVX-512 est utilisé automatiquement quand il est présent (`SPUR_FORCE_AVX2=1` pour le désactiver).

## Sommaire

- [Cas d'usage concrets](#cas-dusage-concrets) (entraînement, diagnostic, classification, service par lots)
- [Entraînement de MLP](#entraînement-de-mlp-spur_mathnn)
- [Inférence par lots : `PackedNT` / `PackedMLP`](#inférence-par-lots-packednt--packedmlp)
- [Graphe de calcul et rétropropagation](#graphe-de-calcul-et-rétropropagation)
- [Précision](#précision)
- [Benchmarks](#benchmarks)
- [Build, tests, installation](#build-tests-installation)
- [Limitations](#limitations)

## Cas d'usage concrets

Quatre cas sur des données réelles embarquées dans scikit-learn (aucun téléchargement), modèles entraînés
depuis zéro avec `spur_math.nn`. Détails, tableaux complets et limites : [`docs/USECASES.md`](docs/USECASES.md).

```bash
cd examples/usecases && PYTHONPATH=../.. python digits_mlp_train.py
```

| Cas | Script | Résultat SpearVM | Référence (PyTorch CPU / autre) |
|---|---|---|---|
| Chiffres manuscrits : entraînement MLP 64→128→10 | `digits_mlp_train.py` | précision 0,9844 en **0,41 s** | même précision, 0,71 s |
| Service par lots du modèle digits | `digits_batch_serving.py` | **2,8 M échantillons/s** à lot 1024 ; plus rapide à partir du lot 32 | PyTorch 2,3 M/s ; numpy 0,3 M/s |
| Diagnostic breast cancer (CV 5 plis) | `breast_cancer_screening.py` | AUC 0,9920, entraînement 2,5 s | **régression logistique meilleure** (AUC 0,9955, 0,02 s) |
| Classification de vins, 3 cultivars (CV 5 plis) | `wine_quality_classify.py` | précision 0,9832, 1,4 s | même précision ; PyTorch mieux calibré, 5,7 s |

Conclusion honnête : SpearVM ne bat pas PyTorch en précision sur ces jeux, mais il entraîne et sert
des petits MLP plus vite (environ 1,7× à 4× pour l'entraînement). Sur données tabulaires, une régression logistique
suffit souvent.

## Entraînement de MLP (`spur_math.nn`)

```python
import numpy as np
from spur_math.nn import SpearMLP, Adam, train_step

net = SpearMLP([64, 128, 10], seed=0)      # GELU exacte sur les couches cachées
opt = Adam(net, lr=3e-3)
for xb, yb in batches:                      # X float32, y int64
    loss = train_step(net, opt, xb, yb)     # forward, backward exact, Adam
logits = net.predict_logits(X)
acc = net.accuracy(X_test, y_test)
```

- `SpearMLP.loss_and_grads` : entropie croisée softmax, rétropropagation exacte (`gelu_backward` + `matmul_nt`).
  Gradients vérifiés contre différences finies en f64 et contre PyTorch en f32 (`tests/test_nn_mlp.py`).
- `Adam` : tampon de paramètres plat (une dizaine d'opérations numpy par pas), écart 1e-7 avec l'Adam classique.
- Le forward d'entraînement garde la pré-activation (nécessaire au backward) : le FFN fusionné n'est pas utilisé à l'entraînement.

## Inférence par lots : `PackedNT` / `PackedMLP`

`PackedNT(W)` pré-empaquette les poids une fois ; chaque appel ne packe que l'entrée.

```python
from spur_math import PackedNT
from spur_math.nn import PackedMLP

P = PackedNT(W)                       # W (n, k) float32
C = P.matmul(A)                       # A . W^T
H = P.matmul_gelu(A, bias)            # gelu(A . W^T + bias), épilogue fusionné

serve = PackedMLP(net)                # MLP entraîné -> inférence rapide
logits = serve.predict_logits(X)      # écart 1e-5 vs net.predict_logits
```

Le packing de B représente environ 30 % du temps à M=128, d'où l'intérêt de pré-empaqueter les poids.
Hors AVX-512, `PackedNT` retombe sur `matmul_nt` / `matmul_nt_gelu` (mêmes résultats).

## Graphe de calcul et rétropropagation

```python
from spur_math.graph import Graph
g = Graph()
x = g.input("x")
h = g.gelu(g.matmul_nt(x, W1, bias=b1))   # fusionne -> matmul_nt_gelu (AVX-512)
y = g.matmul_nt(h, W2)
out = g.run({"x": X})[y]
```

Rétropropagation : `g.backward(out, feeds, dout, wrt=[...])` (gradients exacts, gradcheck en f64 dans
`tests/test_graph.py`). Le graphe recalcule le forward à chaque backward : pour une boucle d'entraînement,
utilisez `spur_math.nn`.

## Précision

Chaque noyau documente son erreur contre la loi exacte.

| Noyau | Erreur max | Notes |
|---|---|---|
| `gelu_erf` (**défaut du FFN fusionné et du backward**) | **2,05e-5** (MSE 8,3e-11) | Horner 5/6 + 1 division, rationnel `erf_v2` |
| `gelu_quintic` (smoothstep) | 0,0174 | 5 multiplications, 0 division |
| `gelu` (v1, legacy) | 0,079 | approximation grossière, conservée pour compatibilité |
| `erf`, `tanh` | ≤ 0,011 et ≤ 0,009 | sur [-2, 2] et [-3, 3] |

Le FFN fusionné et sa dérivée utilisent la GELU exacte (écart 2,2e-5 contre la loi exacte, dérivée
vérifiée à 1,4e-9 en f64). Les anciens chiffres « err max 0,0797 » datent de la GELU v1 et ne s'appliquent plus.
Comparatif des formes championnes : [`CHAMPIONS.md`](CHAMPIONS.md) et [`docs/CHAMPIONS_SUPERSPEAR.md`](docs/CHAMPIONS_SUPERSPEAR.md).

## Benchmarks

### Mesures actuelles (VM 2 cœurs, AVX-512 ; forcer AVX2 : `SPUR_FORCE_AVX2=1`)

| Cas | SpearVM AVX-512 | SpearVM AVX2 | numpy (OpenBLAS) |
|---|---|---|---|
| f32 `matmul_nt` 512³ | 1,8 ms (147 GF) | 3,2 ms (85 GF) | 1,3 ms |
| f32 `matmul_nt` 1024×768×3072 | 31,2 ms (155 GF) | 62,3 ms (78 GF) | 19,1 ms |
| f32 `matmul_nt_gelu` fusionné 1024×768×3072 | 28,6 ms | 177,9 ms | — |
| f64 `matmul_nt` 1024³ | 27,4 ms (78 GF) | 66,1 ms (33 GF) | 16,5 ms |
| f64 `matmul_nt_gelu` fusionné 1024×768×3072 | 67 ms | 131 ms | — |

Lecture honnête : en GEMM pur, SpearVM reste environ 1,3 à 1,6× plus lent qu'OpenBLAS. Son avantage est le
pipeline fusionné (GELU appliquée à l'écriture, sans buffer intermédiaire) et l'absence de dépendance BLAS.

### Encodeur Transformer end-to-end

[`docs/BENCH_ENCODER_E2E.md`](docs/BENCH_ENCODER_E2E.md) : encodeur 12 couches, d=384, FFN 1536, séquence 128,
contre PyTorch CPU et ONNX Runtime (attention, LayerNorm et softmax restent sur PyTorch).
Avec la GELU exacte, batch 1 : PyTorch 36,4 ms, ONNX Runtime 31,5 ms, SpearVM 45,9 ms (≈0,8× PyTorch).
**SpearVM est plus lent sur ce modèle**, et l'écart de sortie est de 1e-5 (cosinus 1,000000).

### Éléments (4M éléments, OpenMP) — historique v0.4

| Noyau | Natif libm | AVX2 SPEAR | Speedup |
|---|---|---|---|
| GELU | 120 ms | 8 ms | ×15,0 |
| ERF | 306 ms | 9 ms | ×34,0 |
| TANH | 234 ms | 9 ms | ×26,0 |

Ces chiffres datent de la version 0.4 (GELU approchée). Les noyaux exacts sont plus lents.

### Matmul C (`examples/bench_mm.c`) — historique

| n | naïf ikj | AVX2 dot | AVX4 bloc + OMP |
|---|---|---|---|
| 256 | 23 ms (1,5 GF) | ×1,02 | ×1,61 |
| 512 | 198 ms (1,4 GF) | ×0,37 | ×2,56 (3,5 GF) |

Les timings de la VM sont bruités (±15 % à ±2× selon les runs).

## Build, tests, installation

```bash
make                 # bibliothèque native + tests C
make test            # tests C puis pytest (PYTHONPATH=. python -m pytest tests/ -q)
python -m pytest tests/ -q
```

Dépendances des tests : `numpy`, `pytest`. Les cas d'usage et les comparaisons utilisent aussi
`scikit-learn`, `torch` (CPU) et `onnxruntime` (benchmark encodeur) ; ils ne sont pas requis pour le paquet.

Le test de fumée des cas d'usage (`tests/test_usecases_smoke.py`) exécute chaque script avec peu d'époques
sans écrire les résultats commités.

```bash
gcc -O3 -mavx2 -mfma -fopenmp src/spur_kernels.c -shared -o spur_kernels.dll   # manuel
python build_package.py --install                                              # depuis les sources
```

Installation depuis PyPI (`pip install spur-math`, version 0.5.3) : le wheel embarque un binaire Windows x64 ;
sur Linux/macOS, la bibliothèque se compile au premier import (ou avec la commande gcc ci-dessus).
Le JIT est portable (`src/spur_posix.c` : mmap/mprotect, ABI System V).

## Architecture

```
src/spur_kernels.c    noyaux AVX2/AVX-512 vectorisés + OpenMP
src/spur_posix.c      JIT x64 POSIX
include/spur.h        API C publique
spur_math/__init__.py API Python (matmul, GELU, PackedNT)
spur_math/nn.py       MLP, Adam, PackedMLP
spur_math/graph.py    graphe de calcul et rétropropagation
examples/             benchmarks, comparaisons, cas d'usage (examples/usecases/)
tests/                tests de correction (pytest + tests C)
docs/                 notes de benchmark et de précision
```

## Feuille de route

| Item | Statut | Déclencheur |
|---|---|---|
| GELU exacte dans le FFN fusionné f32/f64 | **fait** | — |
| Dispatch AVX-512 (f32 et f64 matmul, FFN) | **fait** | `SPUR_FORCE_AVX2=1` pour forcer AVX2 |
| `PackedNT` / `PackedMLP` pour l'inférence | **fait** | — |
| FFN fusionné dans le forward d'entraînement | à faire | nécessite de stocker la pré-activation autrement |
| bf16 / fp16 | différé | pas de `avx512_bf16` sur la machine de test |
| Wheels manylinux pré-compilés | job CI `linux-wheel` (artifact à chaque run) | publication au tag si besoin |

## Limitations

- Le paquet PyPI embarque un binaire Windows x64 ; Linux/macOS compilent à l'installation (gcc/clang requis).
- CPU x86-64 seulement (AVX2 + FMA requis). Pas de support GPU.
- Plus lent que OpenBLAS en GEMM pur, et plus lent que PyTorch sur l'encodeur Transformer testé.
- Précision : GELU exacte (2e-5) dans le FFN fusionné ; les autres noyaux « rapides » ont une erreur de 1e-3 à 1e-2.
- Les jeux de données des cas d'usage sont petits (178 à 1797 exemples) ; les conclusions sur la précision valent pour ces données.

## Licence

MIT
