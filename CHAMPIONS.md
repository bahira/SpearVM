# 🏆 SPEAR — Championnes accuracy & speed

Répertoire des meilleures formes fermées (100 % ALU, aucune transcendance)
découvertes dans l'écosystème SPEAR (SpearVM + spear-fable + superspear),
avec leurs métriques mesurées et les lieux d'intégration.

**Convention** : `L-inf` = erreur maximale absolue vs référence IEEE, mesurée
sur grille dense indépendante ; `MSE` = erreur quadratique moyenne.

---

## POIDS QUANTIFIES — bf16 et int8 par ligne (SpearVM)

A m petit, un GEMM ne fait que 2.m flops par poids lu : le temps est decide par
les octets, pas par les operations.

| Format | Octets/poids | GEMV (m=1) | Erreur sur un bloc de decodeur |
|---|---|---|---|
| f32 | 4 | reference (~15 Go/s, plafond memoire) | 1e-6 |
| **bf16** | 2 | **x2.5 – x3.5** | 0.2 % |
| **int8** (echelle par ligne) | 1 | **x5.3 – x5.9** | 0.9 – 1.4 % |

A m >= 16 le GEMM redevient limite par le calcul et int8 perd (x0.93) :
outil de **decodage**, pas de prefill.

**C** : `spur_quant_i8_rows`, `spur_gemm_nt_i8b_f32`, `spur_gemm_nt_bf16w_f32`.
**Python** : `spur_math.QuantizedWeight(w, dtype="i8"|"bf16")`.

---

## CACHE KV — packe une fois, bf16 optionnel (SpearVM)

| Metrique | Valeur mesuree |
|---|---|
| Packing amorti | **x4.40 median** (x2.35 a x4.71) |
| Petites tuiles (tq=4) | 5.0 -> **68.8 GFLOPS** (x13.8) |
| Decodage tq=1, tk=4096 | 34.5 GFLOPS, x8.3 vs numpy |
| bf16 | empreinte /2 toujours ; vitesse x1.08-x1.41 au-dela de 17 Mo |

**Python** : `spur_math.KVCache(k, v, dtype="f32"|"bf16")`, `.attend(q, lengths=)`.

---

## EXP — minimax Remez + reduction d'argument (SpearVM)

`exp(x)`, reference IEEE. Forme : `x = k·ln2 + r` (ln2 scinde hi/lo, `k·hi`
exact), polynome minimax **en erreur relative** sur `r ∈ [−ln2/2, ln2/2]`,
`2^k` reconstruit par ecriture du champ d'exposant.

| Variante | Degre | L-inf relatif | ulp | Debit | vs numpy/libm |
|---|---|---|---|---|---|
| **`exp` f32** | 5 | **2.02e-7** | **1.69** | 3.0 G elem/s | x1.2 – x1.9 |
| **`exp` f64** | 10 | **4.72e-16** | 2.12 | 1.0 G elem/s | x0.9 – x1.9 |

libm f32 mesure 1.66 ulp : on egale sa precision en allant plus vite. En f64
libm est correctement arrondie (0 ulp) et nous ne le sommes pas — 2 ulp assumes.
Le degre est choisi par mesure : au-dela de 5 (f32) et 10 (f64), le format
limite, pas le polynome.

**C** : `spur_batch_exp[_f32]`, `spur_exp8_ps` / `spur_exp4_pd` (inline).
**Python** : `spur_math.exp(x)`. **Recherche** : `experiments/transcend/fit_exp.py`.

---

## SOFTMAX — trois passes + longueurs causales (SpearVM)

| Variante | Forme | Gain vs numpy | Note |
|---|---|---|---|
| **`softmax_rows`** | max, exp+somme, normalisation | **x1.9 – x6.2** | 37 GB/s |
| schema "online" | rescale incremental (flash-attention) | x0.5 – x0.8 | **perdant sur CPU** |
| **`softmax_rows` + `len[]`** | causal sans masque materialise | **x11.8** | ne lit que les entrees valides |

**C** : `spur_softmax_rows[_f32](X, Y, rows, cols, len)`.
**Python** : `spur_math.softmax(x, lengths=None)`.

---

## ATTENTION TILE — GEMM NT + softmax fusionne (SpearVM)

`O = softmax(scale·Q·Kᵀ, causal)·V`. Echelle absorbee par l'exponentielle,
masque porte par un vecteur de longueurs, les deux produits en convention NT.

| Metrique | Valeur mesuree |
|---|---|
| Gain vs numpy | **x2.77 median** (x1.19 a x3.57) |
| Debit | jusqu'a **122.9 GFLOPS** f32 |
| Erreur relative | <= 6.6e-7 |

**C** : `spur_attention_tile_f32`. **Python** : `spur_math.attention_tile(q,k,v)`.

---

## GELU — 3 variantes (SpearVM)

`GELU(x) = 0.5·x·(1+erf(x/√2))`, référence exacte.

| Variante | Forme | L-inf | MSE | Ops | Utiliser pour |
|---|---|---|---|---|---|
| `gelu` v1 | `0.997729·(x·clip(0.306923x+0.501,0,1.002)) − 0.004004` | 7.97e-2 | 5.3e-4 | 3 mul + clamp | héritage (paper 1) |
| **`gelu_quintic`** v2 | `t=clip(0.200055340257x+0.5,0,1)` ; `x·t³(6t²−15t+10) − 0.01104961` | 1.74e-2 | 1.3e-4 | 5 mul, **0 div** | **le + rapide**, kernel fuse |
| **`gelu_erf`** | `0.5·x·(1+erf_v2(x/√2))`, erf_v2 = rationnel `x·P(y)/D(y)` | **2.05e-5** | **8.3e-11** | Horner 5/6 + 1 div | **le + précis** (training, backprop) |

- `gelu_quintic` : forme smoothstep quintique certifiée — L-inf bornée sur R
  (queue gauche sature à `−0.01104961`, droite suit `x`).
- `gelu_erf` : réutilise l'`erf_v2` de SpearVM (max_err 2.3e-5) — **850× plus
  précis que le quintique** pour un coût arithmétique marginal (division +
  Horner, masqué à l'échelle batch par la bande passante mémoire).

**Bindings Python** : `spur_math.gelu(x)` / `spur_math.gelu_quintic(x)` /
`spur_math.gelu_erf(x)`.
**C** : `spur_batch_gelu[_quintic|_erf]`, `spur_k_gelu[_quintic|_erf]`.

---

## TANH — champion Pade [3/4] (spear-fable)

`tanh(x)` référence exacte. Forme : `(a·y + b·y³)/(1 + c·y² + d·y⁴)`,
`y = clamp(x, −4, 4)`.

| Coefficients | Valeur |
|---|---|
| `a` | 0.994894946 |
| `b` | 0.076611228 |
| `c` | 0.402171314 |
| `d` | 0.005670342 |

| Variante | Forme | L-inf (sur [-5,5]) | Gain |
|---|---|---|---|
| polynôme historique | `x/(1+0.856x²+0.037x⁴)` | 8.9e-1 | — |
| SpearVM [3/2] | `cn·(y+c3y³)/(b0+b2y²)`, clamp ±3 | 8.5e-3 | ×105 vs OLD |
| **`Pade[3/4]`** | ci-dessus, clamp ±4 | **1.56e-3** | **×5.4 vs SpearVM, ×571 vs OLD** |

Le Pade[3/4] est le champion **sur les deux critères** : il n'a qu'**une
division**, et son fit minimax direct sur les erreurs gauss/lorentz composées
le rend plus précis que la forme SpearVM [3/2] sans sacrifier la vitesse.

**C** : `fast_tanh_avx` / `fast_tanh_scalar` dans `kernels/spear_avx_emb.c`.
**GPU** : `TanhALU` dans `examples/train_gpu_pipeline.py`.

---

## Fonctions dérivées (du Pade[3/4], testées)

| Fonction | Construction | L-inf | Gain vs avant |
|---|---|---|---|
| **SigmoidALU** | `0.5 + 0.5·tanh(x/2)` | 7.8e-4 | ×5.4 |
| **SiLUALU** (Swish) | `x·sigmoid(x)` | 9.3e-4 | ×27 |
| **gauss** (kernel AVX) | `tanh(0.6x)` | 1.56e-3 | ×571 |
| **lorentz** (kernel AVX) | `1/√(1−0.8·tanh(0.5x)²)` | 3.6e-3 | ×347 |

---

## Compromis speed vs accuracy (règle pratique)

| Activation | **Meilleur accuracy** | **Meilleur speed** | Choix |
|---|---|---|---|
| **GELU** | `gelu_erf` (2.05e-5) | `gelu_quintic` (0 div, 5 mul) | training/backprop → erf ; kernel fuse → quintic |
| **Tanh / Sigmoid / SiLU** | **Pade[3/4]** (1.56e-3) | **Pade[3/4]** (1 div) | Pade[3/4] domine les deux |

> Mesure throughput (batch AVX2, min-of-N, VM 2 cœurs partagés) : le surcoût
> de `gelu_erf` vs `gelu_quintic` est dans le bruit (−17 % à +21 % selon le
> run, moyenne ≈ 0) à l'échelle batch ; à taille cache, surcoût médian ~20 %.
> Le gain de précision ×850 est donc quasi-gratuit en débit best-case.