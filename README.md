# SpearVM

[![CI](https://github.com/bahira/SpearVM/actions/workflows/ci.yml/badge.svg)](https://github.com/bahira/SpearVM/actions/workflows/ci.yml)

**Accélérateur mathématique SIMD** — noyaux transcendantals vectorisés AVX2,
précision certifiée, matmul tuilé, couche FFN fusionnée avec backprop,
mini-JIT x64 portable Linux/macOS/Windows pour programmes élément-par-élément. Python/C.

```bash
pip install spur-math
```

> Windows : wheel binaire prêt. Linux/macOS : le sdist embarque les sources C ;
> au premier import, `spur_math` se compile tout seul si gcc/clang est
> présent (cache dans `spur_math/_native/`), sinon suivez le message d'aide.
> CPU requis : AVX2 + FMA (Haswell 2013+, Zen 2017+) — vérifié à l'import.

## Speedups mesurés (4M éléments, OpenMP multi-cœur)

| Noyau | Natif libm | AVX2 SPEAR | **Speedup** |
|---|---|---|---|
| GELU | 120 ms | 8 ms | **×15.0** |
| ERF | 306 ms | 9 ms | **×34.0** |
| TANH | 234 ms | 9 ms | **×26.0** |

## v0.4 — fp32 ×2, backprop complète, map 4 entrées

- **float32 natif** : `matmul_nt`, `matmul_nt_gelu`, `gelu`, `gelu_backward`
  acceptent float32 (8 lanes/vector) → **×2.00 vs f64** mesuré
  (13.5 → 26.8 GFLOPS @1024³). Dispatch automatique sur le dtype numpy.
- **Backward erf/tanh/sigmoid** : gradients exacts des approximations
  rationnelles, gradcheck ~1.6e-09. Training complet possible :
  `erf_backward`, `tanh_backward`, `sigmoid_backward`.
- **`matmul_backward(dY,A,B)`** : gradients (dA,dB) du matmul NT —
  gradcheck 1e-10 ; la backprop d'une couche complète tient en
  `gelu_backward` + `matmul_backward` + biais.
- **Map kernel 4 entrées** : `out[i] = F(a[i], b[i], c[i], d[i])` via
  tableau de pointeurs ; `spur_free(handle)` + slots réutilisables ;
  builds sérialisés (SRWLOCK) — exec réentrant.

### Feuille de route

| Item | Statut | Déclencheur |
|---|---|---|
| bf16/fp16 | différé | aucun compute natif bf16 en AVX2 ; fp16 = upconvert partout — à faire sur besoin mémoire exprimé |
| Dispatch AVX-512 | différé | aucun CPU de test dispo (runners CI = AVX2 max) |
| Wheels manylinux pré-compilés | **job CI `linux-wheel`** (auditwheel, artifact à chaque run) | publication automatique au tag si besoin |

## Matmul (bench_mm)

`examples/bench_mm.c` — C = A·B double, validation vs naif incluse.

| n | naif ikj | dot scalaire | AVX2 dot | AVX4 bloc + OMP |
|---|---|---|---|---|
| 128 | 2.0 ms (2.1 GF) | ×0.12 | ×0.45 | ×0.73 (série) |
| 256 | 23 ms (1.5 GF) | ×0.24 | ×1.02 | **×1.61** |
| 512 | 198 ms (1.4 GF) | ×0.31 | ×0.37 | **×2.56** (3.5 GFLOPS) |

- err ≤ 1.6e-14 sur toutes les variantes
- blocage registres 4 lignes : chaque ligne de Bᵀ sert 4 rangs de A, 4 chaînes FMA indépendantes
- OpenMP activé seulement si n ≥ 192 ; lancer avec `OMP_WAIT_POLICY=ACTIVE`
- timings VM bruités (±2× entre runs), gains observés jusqu'à ×12 à n=256 machine au repos

## Matmul API (v0.2+)

`sm.matmul_nt(A, B)` — C = A·Bᵀ, convention BLAS NT, double précision.

- **Tuillage cache** KC×NC (512 Ko/tuile en L2) + blocage registres 4 lignes,
  4 chaînes FMA indépendantes, OpenMP
- err ≤ 1.4e-13 vs numpy sur toutes tailles (512→2048), y compris queues m%4
- `matmul_nt_gelu` : activation fusionnée — la gelu est non linéaire donc k
  n'est pas coupé, seulement le blocage colonnes

| Cas | SpearVM | numpy BLAS |
|---|---|---|
| carré 512 | 24.5 GFLOPS | 48.2 GFLOPS (×0.51) |
| carré 1024 | 13.9 GFLOPS | 18.0 GFLOPS (×0.77) |
| carré 2048 | 10.3 GFLOPS | 23.5 GFLOPS (×0.44) |
| **FFN 1024×768×3072 fusionné gelu** | **353 ms** | **858 ms (×2.43)** |

Lecture honnête : en GEMM carré pur, OpenBLAS/MKL reste devant (packing AVX,
microkernels plus larges). Là où SpearVM gagne, c'est le **pipeline fusionné**
— un passage au lieu de deux, zéro buffer intermédiaire — et l'intégration
NT sans copies.

## Pipeline NN end-to-end (bench_nn)

`Y = gelu(X · Wᵀ)` — pattern FFN transformer. M=1024, K=768, N=3072 (4.8 GFLOP).

| Implémentation | Total | Détail |
|---|---|---|
| **SpearVM** (`matmul_nt_gelu`, tuilé) | **353 ms (×2.43 vs numpy)** | 13.7 GFLOPS équivalents, err 0.0797 |

- err max vs référence exacte : **0.0797** (= contrat datasheet gelu ≤0.079)
- la forme linéarisée sature proprement hors domaine : err bornée ~0.002·|x| jusqu'à ±25
- validation : `bin/bench_nn.exe` puis `python examples/check_nn.py`

```bash
gcc -O3 -mavx2 -mfma -fopenmp examples/bench_nn.c src/spur_kernels.c -o bench_nn -lm
OMP_WAIT_POLICY=ACTIVE ./bench_nn && python examples/check_nn.py
```

## Précision (datasheet)

Chaque noyau documente son erreur max vs IEEE :
- gelu : ≤ 0.079 sur [-2, 2]
- erf : ≤ 0.011 sur [-2, 2]
- tanh : ≤ 0.009 sur [-3, 3]
- lse2 : hard-max (écart ≤ ln 2)

### GELU — trois variantes au choix

| Variante | L-inf | MSE | Opérations | Binding Python |
|---|---|---|---|---|
| `gelu` (v1, legacy) | 0.079 | 7.9e-4 | 3 mul + clamp | `gelu(x)` |
| `gelu_quintic` (v2 smoothstep) | 0.0174 | 1.35e-4 | 5 mul, 0 div | `gelu_quintic(x)` |
| `gelu_erf` (haute précision) | **2.05e-5** | **8.3e-11** | Horner 5/6 + 1 div | `gelu_erf(x)` |

Les trois sont **100 % ALU** (aucune transcendance `exp`/`erf`). Le `gelu_erf` réutilise le
rationnel `erf_v2` certifié : `GELU = 0.5·x·(1+erf(x/√2))`, avec clamp à 3.5 (saturation
`GELU→x` pour `x>~4.95`, correct asymptotiquement).

**Compromis débit vs précision** (batch AVX2, min-of-N, machine 2 cœurs partagés) :
le surcoût arithmétique de `gelu_erf` (1 division + Horner 5/6 en plus) est **masqué à
l'échelle batch par la bande passante mémoire** — mesuré entre −17 % et +21 % selon le run,
moyenne ≈ 0 (dans le bruit de la VM cloud). À taille cache, le surcoût médian est ~20 %.
Règle pratique : **quintic quand le nombre d'ops compte (kernel fuse), `erf` quand la
précision prime (training, backprop)** — le gain de précision est ×850 pour un coût arithmétique marginal.

> 📊 **Voir [`CHAMPIONS.md`](CHAMPIONS.md)** pour le répertoire complet des formes
> championnes (GELU / Tanh / Sigmoid / SiLU), leurs coefficients et le compromis
> speed vs accuracy.

## Build

```bash
make              # DLL + tests
gcc -O3 -mavx2 -mfma -fopenmp src/spur_kernels.c -shared -o spur_kernels.dll
```

## Installation Python

```bash
pip install spur-math          # v0.5.3 sur PyPI
```

Ou depuis les sources (compile la DLL puis installe) :

```bash
python build_package.py --install
```

> Note : le wheel actuel embarque un binaire Windows x64. Sur Linux/macOS,
> compilez le `.so` vous-meme avec la ligne gcc ci-dessus (le package affiche
> la commande exacte si la lib manque). Le JIT est désormais portable
> (voir `src/spur_posix.c` pour le portage POSIX : mmap/mprotect, System V ABI).

## Training (backprop)

```python
import spur_math as sm

# forward fusionne
Y = sm.matmul_nt_gelu(X, W)            # Y = gelu(X . W^T), un seul passage

# backward — les gradients passent par le meme matmul
T  = sm.matmul_nt(X, W)
dY = dLdY * sm.gelu_backward(np.ones_like(T), T)   # ou directement gelu_backward(dLdY, T)
dW = sm.matmul_nt(dY.T, X.T)
dX = sm.matmul_nt(dY, W.T)
```

Gradients verifies par differences finies (err ~1e-10) et SGD convergeant
(loss /87 en 300 pas, `tests/test_train.py`).

## Usage Python

```python
import ctypes
lib = ctypes.CDLL("spur_kernels.dll")
# voir examples/bench_kernels.c pour l'API C complète
```

## Architecture

```
src/spur_kernels.c    Kernels AVX2 vectorisés + OpenMP
include/spur.h        API publique
examples/             Benchmarks et démos
tests/                Tests de correction
```

## Limitations

- Windows x64 + Linux/macOS x64 (JIT POSIX porté)
- Précision approximative (±0.01–0.09 selon noyau)
- Pas de support GPU

## Licence

MIT
