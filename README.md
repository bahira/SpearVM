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

Depuis v0.6, deux micro-noyaux **autotunés** (902 variantes générées,
vérifiées et chronométrées — voir [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md))
coexistent derrière la même API, avec un aiguillage calibré par la mesure :

- **[P] packé** — micro-noyau *broadcast* façon BLIS (f64 4×12, f32 4×24) :
  A et B recopiés en panneaux contigus, zéro réduction horizontale.
- **[D] dot-block** — **sans packing** (3×4) : en convention NT, k est contigu
  des deux côtés, donc rien à copier. Gagnant dès qu'une dimension est étroite.
- aiguillage `legacy / dot / pack` choisi sous contrainte **« aucune
  régression »** ; `MC` s'adapte au nombre de threads ; `SPUR_MM_LEGACY=1`
  restaure l'ancien noyau.
- err ≤ 1.5e-15 (f64) / 8.3e-07 (f32) sur 324 formes par précision
  (`tests/test_matmul_v2.py`), y compris toutes les queues m%MR, n%NR, k%KC.

Mesuré sur 2 vCPU AVX2 (1 thread, numpy chronométré séparément,
`experiments/verify_port.py`) :

| Cas (f64) | v0.5 | **v0.6** | numpy BLAS |
|---|---|---|---|
| carré 512 | 20.8 GF | **43.3 GF** (×2.09) | 58.9 GF |
| carré 1024 | 18.2 GF | **41.4 GF** (×2.28) | 63.5 GF |
| 512×64×512 (k court) | 21.1 GF | **45.0 GF** (×2.13) | 43.1 GF (**×1.04**) |
| 512×16×512 (k très court) | 13.0 GF | **38.4 GF** (×2.95) | 25.8 GF (**×1.49**) |
| 2048×128×128 | 24.8 GF | **43.2 GF** (×1.74) | 41.6 GF (**×1.04**) |

Gain médian sur 20 formes : **×1.78 (f64)** et **×1.50 (f32)** en mono-thread,
×1.43 / ×1.30 sur 2 threads, **sans aucune régression** (pire cas ×1.00).
`matmul_nt_gelu` gagne **×2.23 (f64)** / **×2.34 (f32)** de médiane : la
fusion de la gelu dans la boucle j interdisait de couper k, donc interdisait le
micro-noyau packé ; un épilogue vectorisé coûte bien moins cher.
Bout-en-bout, une itération d'entraînement MLP 784-256-128-10 passe de
14.3 ms à **7.5 ms** (f64, `experiments/bench_mlp.py`).

Lecture honnête : en GEMM carré pur, OpenBLAS/MKL reste devant (0.65–0.73× ici,
contre 0.36× avant cette campagne). Là où SpearVM passe devant, c'est sur les
formes que BLAS amortit mal — **k court, dimension étroite** — et sur le
**pipeline fusionné** NT sans copies.

## exp, softmax et attention (v0.6+)

Le dépôt n'avait ni `exp` ni `softmax` — les deux briques que réclame
l'attention. Elles ont été construites et mesurées ([`docs/TRANSCEND.md`](docs/TRANSCEND.md)) :

```python
sm.exp(x)                                   # 1.69 ulp (f32), 2.12 ulp (f64)
sm.softmax(x, lengths=None)                 # par ligne ; lengths = masque causal
sm.attention_tile(q, k, v, lengths=None)    # softmax(Q·Kᵀ/√d) · V, une tête
sm.attention_mha(q, k, v, lengths=None)     # multi-têtes / GQA, une descente C
cache = sm.KVCache(k, v, dtype="bf16")      # packé une fois, réutilisé
cache.attend(q, lengths=None)
w8 = sm.QuantizedWeight(w, dtype="i8")      # poids ÷4 — pour le décodage
w8.matmul(x)
```

Bloc de décodeur complet (attention + FFN + résiduels), 2 vCPU : prefill
**×2.8 à ×9.65** vs numpy (jusqu'à 17 506 tok/s) ; décodage **×2.4 à ×9.7** avec
poids int8, jusqu'à **3 439 tok/s** en mono-flux.

| Noyau | Précision | Gain vs numpy |
|---|---|---|
| `exp` f32 | 1.69 ulp (libm : 1.66) | ×1.2 – ×1.9, jusqu'à 3.0 G elem/s |
| `softmax` | ≈ numpy f32 | ×1.9 – ×6.2 |
| `softmax` causal (longueurs) | idem | **×11.8** |
| `attention_tile` | 6.6e-07 relatif | **×2.77 médian**, jusqu'à 123 GFLOPS |
| `attention_mha` (multi-têtes, GQA) | idem | **×15.6 médian**, ×2.7 vs boucle par tête |
| `KVCache` (packé, réutilisé) | idem | ×4.4 de plus — 68.8 GFLOPS à tq=4 |
| `KVCache(dtype="bf16")` | 2e-03 relatif | empreinte ÷2 ; plus rapide au-delà de 17 Mo |
| `QuantizedWeight(dtype="i8")` | 1e-02 sur le bloc | **×5.3 – ×5.9 en GEMV**, poids ÷4 |

Le masque causal est porté par un vecteur de **longueurs** : rien à
matérialiser, et le noyau ne parcourt que les entrées valides. Le schéma
« online » de flash-attention a été implémenté puis rejeté — perdant d'un
facteur 1.2 à 4 sur CPU, où la ligne relue tient en L1.

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

## Simulation Lab — 6 cas d'usage Three.js (`web/`)

Démos temps réel **prêtes pour la production** où la physique est calculée par
les noyaux SIMD et le rendu par le GPU : serveur FastAPI + WebSocket binaire,
client Vite/TypeScript/Three.js.

| Cas d'usage | Noyaux SpearVM | Rendu |
|---|---|---|
| **Champ de flux neuronal** | `matmul_nt_gelu` ×2 couches sur une grille 3D → potentiel vecteur, `rot`, `tanh` | 16 k→262 k particules advectées en ping-pong GPU dans une texture 3D |
| **Champ implicite neuronal** | MLP par voxel (`matmul_nt_gelu`, k=14) → distance signée 1-lipschitzienne | sphere tracing GPU d'une texture 3D — aucune géométrie transmise |
| **Membrane non linéaire** | équation des ondes + saturation `tanh` par sous-pas | maillage déplacé depuis une texture R32F, 1 vertex = 1 cellule |
| **Entraînement live** | `matmul_nt` + `gelu`, `gelu_backward` + `matmul_backward`, Adam | surface prédite colorée par l'erreur + cible filaire + courbe de perte |
| **Attention Lab** | bloc de décodeur réel : `attention_mha` + `KVCache` + `QuantizedWeight` | carte d'attention (têtes × positions) en relief, débit qui bouge avec le format |
| **Kernel Lab** | banc d'essai : débit vs numpy, erreur vs IEEE, GFLOPS, gradcheck | barres 3D et courbes d'erreur log |

Gain apporté par les noyaux v2, mesuré frame par frame sur ces scènes
(`experiments/bench_sims.py`, 2 threads) : champ implicite 48³ **×3.00**
(49.9 → 16.7 ms), champ de flux 24³ **×2.84**, entraînement live ×1.52,
membrane ×1.0 (aucun matmul — rien à gagner, et c'est mesuré aussi).

```bash
make web-install     # venv + noyaux + npm install
make web-serve       # serveur de calcul  :8000
make web-dev         # front (proxy /api et /ws)  :5173
make web-test        # 40 tests serveur + typecheck client
```

Dégradation propre : noyaux AVX2 → repli numpy exact → moteur JavaScript local
si le serveur est injoignable ; le bandeau affiche toujours le mode réel.
Détails, protocole binaire et déploiement Docker : [`web/README.md`](web/README.md).

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
web/server/           Serveur de simulation FastAPI (WebSocket binaire + REST)
web/client/           Client Three.js (Vite + TypeScript), 4 cas d'usage
```

## Limitations

- Windows x64 + Linux/macOS x64 (JIT POSIX porté)
- Précision approximative (±0.01–0.09 selon noyau)
- Pas de support GPU

## Licence

MIT
