# TRANSCEND — exp minimax, softmax, tuile d'attention

Troisième campagne du dépôt, après les GEMM (`EXPERIMENTS.md`) et l'audit
d'architecture (`ATTENTION_AUDIT.md`). Point de départ : l'audit a montré que
l'attention est le vrai client des noyaux — or SpearVM n'avait **ni `exp` ni
`softmax`**. Trois briques ont été construites et mesurées, chacune s'appuyant
sur la précédente.

Reproduction :

```bash
cd experiments/transcend
python fit_exp.py            # recherche minimax (Remez, erreur relative)
OMP_NUM_THREADS=2 python bench_softmax.py
cd ../attention && OMP_NUM_THREADS=2 python bench_tile.py
```

---

## 1. `exp` : recherche minimax plutôt que coefficients recopiés

Réduction d'argument `x = k·ln2 + r`, `|r| ≤ ln2/2`, puis polynôme sur `r`.
`ln2` est scindé en (hi, lo) avec `hi` à 11 bits de mantisse : `k·hi` est alors
exact et `x − k·hi` l'est aussi (Sterbenz). `2^k` est reconstruit par écriture
directe du champ d'exposant IEEE.

Le polynôme n'est pas recopié d'une bibliothèque : il est cherché par
**itérations de Remez en erreur relative** sur `[−ln2/2, ln2/2]`
(`experiments/transcend/fit_exp.py`), et le degré est choisi par la mesure.

| degré | minimax théorique | L∞ relatif f32 | L∞ relatif f64 |
| --- | --- | --- | --- |
| 3 | 7.48e-05 | 7.48e-05 | 7.48e-05 |
| 4 | 2.59e-06 | 2.69e-06 | 2.59e-06 |
| **5** | 7.49e-08 | **2.22e-07** | 7.49e-08 |
| 7 | 4.02e-11 | 9.20e-08 (plancher f32) | 4.02e-11 |
| **10** | 4.72e-16 | — | **6.28e-16** |
| 11 | 3.11e-16 | — | 3.37e-16 |

Au-delà du degré 5 en f32 et 10 en f64, le degré n'achète plus de précision :
c'est le format qui limite. **Retenus : degré 5 (f32), degré 10 (f64).**

Précision et débit du noyau livré, mesurés sur 500 001 points :

| | L∞ relatif | en ulp | numpy/libm | débit | gain |
| --- | --- | --- | --- | --- | --- |
| f32 | 2.02e-07 | **1.69 ulp** | 1.66 ulp | jusqu'à 3.0 G elem/s | ×1.2 – ×1.9 |
| f64 | 4.72e-16 | 2.12 ulp | 0 ulp (arrondi correct) | ~1.0 G elem/s | ×0.9 – ×1.9 |

En f32 on égale la précision de libm en allant plus vite ; en f64 libm est
correctement arrondie et nous ne le sommes pas — c'est le prix assumé de
2 ulp pour un facteur ~1.9 sur les grands tableaux.

### Un faux plancher, et comment il a été démasqué

La première mesure montrait un plancher f32 à **3.9e-06 quel que soit le
degré** — de quoi conclure « inutile d'aller au-delà du degré 4 ». C'était un
artefact du protocole : la référence était calculée sur `x` en float64 alors
que le noyau reçoit `x` arrondi en float32. On mesurait donc `|x|·ε₃₂`
(4e-6 à x = −80, et l'erreur croissait bien linéairement en |x|, ce qui a mis
la puce à l'oreille), pas l'algorithme. En arrondissant l'entrée **avant** de
calculer la référence, le vrai comportement apparaît : le degré 5 atteint 2 ulp.

## 2. `softmax` : trois passes battent le schéma « online »

Le softmax par ligne se décompose en max → exp+somme → normalisation. Deux
stratégies ont été implémentées et mises en concurrence :

* **3 passes** : le schéma naïf ;
* **online** (façon flash-attention) : max et somme en une seule lecture, avec
  rescale incrémental de la somme quand le maximum change.

| forme | 3 passes | online | numpy | gain | débit |
| --- | --- | --- | --- | --- | --- |
| 64×128 | **0.009 ms** | 0.021 | 0.038 | ×4.14 | 11 GB/s |
| 256×512 | **0.041** | 0.082 | 0.258 | ×6.23 | 38 GB/s |
| 1024×1024 | **0.337** | 1.381 | 1.972 | ×5.86 | 37 GB/s |
| 4096×512 | **2.694** | 3.386 | 5.183 | ×1.92 | 9 GB/s |
| 256×8192 | **0.685** | 1.284 | 3.348 | ×4.88 | 37 GB/s |

**Le schéma online perd systématiquement, d'un facteur 1.2 à 4.** Sur CPU, la
ligne relue tient en L1 : la relecture coûte moins cher que les `exp`
supplémentaires du rescale incrémental. L'intuition venue du GPU (où le rescale
évite un aller-retour en HBM) ne se transporte pas. Mesuré, donc abandonné.

Précision : l'erreur absolue du noyau (2e-07 à 7e-07) est du même ordre que
celle de **numpy en float32** (2.2e-07 à 3.4e-07), sauf sur les lignes très
longues (256×8192 : 3.8e-06 contre 2.2e-07) où numpy gagne grâce à sa sommation
par paires. Le noyau utilise 8 accumulateurs SIMD ; c'est la limite connue de
cette forme.

### Le softmax causal : la forme dont l'attention a réellement besoin

`spur_softmax_rows_f32(X, Y, rows, cols, len)` où `len[i]` est le nombre
d'entrées valides de la ligne `i` (le reste est mis à zéro). Aucun masque `-inf`
à matérialiser, aucune lecture de masque, et surtout **le noyau ne parcourt que
les entrées valides**.

Sur un masque triangulaire 2048×1024, contre la version numpy vectorisée
(matrice complète masquée à `-inf`) : **0.71 ms contre 8.44 ms, soit ×11.8**.

## 3. La tuile d'attention

`spur_attention_tile_f32` enchaîne les trois briques :

```
scores = Q · Kᵀ            (GEMM NT — noyau v2)
poids  = softmax(scale·scores, longueurs)   (échelle absorbée : exp(a(x−m)))
O      = poids · V         (GEMM NT, V fourni transposé)
```

Deux détails qui suppriment du travail plutôt que de l'accélérer : l'échelle
`1/√d` est absorbée dans l'exponentielle du softmax (pas de passe
supplémentaire sur la matrice de scores), et le masque causal est porté par le
vecteur de longueurs (rien à écrire, rien à lire).

| tq | tk | d | spur | numpy | gain | GFLOPS | erreur rel. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 512 | 64 | 0.045 ms | 0.054 | ×1.19 | 11.6 | 1.4e-07 |
| 16 | 512 | 64 | 0.048 | 0.132 | ×2.77 | 44.0 | 1.9e-07 |
| 64 | 1024 | 64 | 0.178 | 0.561 | ×3.15 | 94.2 | 3.7e-07 |
| 64 | 2048 | 128 | 0.824 | 1.341 | ×1.63 | 81.5 | 5.2e-07 |
| 256 | 2048 | 128 | 2.185 | 3.766 | ×1.72 | 122.9 | 6.6e-07 |
| 512 | 512 | 64 | 0.566 | 2.018 | ×3.57 | 118.6 | 8.4e-08 |

**Gain médian ×2.77**, jusqu'à **123 GFLOPS** en float32 (le plafond mesuré de
cette machine est ~125 GF/cœur, ~250 GF sur 2 threads), erreur relative
maximale 6.6e-07 — conforme au float32.

## 4. API et couverture

```python
import spur_math as sm
sm.exp(x, out=None)                         # 1.69 ulp (f32), 2.12 ulp (f64)
sm.softmax(x, lengths=None, out=None)       # par ligne, longueurs causales
sm.attention_tile(q, k, v, scale=None, lengths=None)
```

`tests/test_softmax.py` (14 tests) couvre : contrat en ulp sur `exp`, cas
limites (`-1e4`, `-800`, saturations `±1e30`), lignes de longueur nulle,
invariance par translation, sommes à 1, tuile d'attention contre référence
float64, masque causal (la première requête ne voit qu'une clé : sa sortie doit
être exactement `v[0]`), et validation des arguments. Suite complète du dépôt :
**67 tests**.

## 5. Ce qui n'a pas marché

* **Le softmax « online »** : −20 % à −75 % contre les trois passes (§2).
  L'argument flash-attention est un argument de hiérarchie mémoire GPU.
* **Monter le degré du polynôme** au-delà de 5 (f32) / 10 (f64) : aucune
  précision gagnée, uniquement des FMA en plus.
* **Mesurer la précision contre une référence float64 sur des entrées float64**
  quand le noyau, lui, reçoit du float32 : produit un faux plancher de 4e-06
  qui masque complètement le comportement du polynôme (§1).

---

*Mesures : `experiments/results/exp_fit.json`, `softmax_bench.csv/.json`,
`attention_tile.csv/.json`, journaux `softmax_log.txt`, `attention_tile_log.txt`.*
