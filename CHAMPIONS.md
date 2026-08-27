# 🏆 SPEAR — Championnes accuracy & speed

Répertoire des meilleures formes fermées (100 % ALU, aucune transcendance)
découvertes dans l'écosystème SPEAR (SpearVM + spear-fable + superspear),
avec leurs métriques mesurées et les lieux d'intégration.

**Convention** : `L-inf` = erreur maximale absolue vs référence IEEE, mesurée
sur grille dense indépendante ; `MSE` = erreur quadratique moyenne.

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