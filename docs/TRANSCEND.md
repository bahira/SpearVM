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


## 4. Attention multi-têtes : supprimer le coût par appel

La tuile du §3 atteint 123 GFLOPS sur les grandes formes mais **5 GFLOPS à
tq = 4** — la taille de micro-bloc de QSA. Ce n'est pas le calcul qui limite,
c'est ce qui l'entoure : un appel ctypes par tête, une allocation de scratch par
appel, et un packing de K/V refait à chaque fois.

Quatre corrections, chacune mesurée séparément :

**(a) Tout faire en C.** `spur_attention_mha_f32` boucle sur les têtes à
l'intérieur du noyau, avec parallélisme OpenMP sur les groupes GQA et des
tampons par thread. Plus rien ne repasse par Python entre les têtes.

**(b) Empiler les têtes d'un groupe GQA.** Les `rep = H_q/H_kv` têtes de
requêtes d'un groupe partagent la même tête de clés : au lieu de `rep` GEMM de
`tq` lignes, on en fait **un seul de `rep·tq` lignes**. À tq = 4 et rep = 4, m
passe de 4 à 16 — d'un régime dominé par les coûts fixes à un régime de calcul.

**(c) Choisir le noyau localement.** Le dispatcher global ignore qu'il s'agit
d'une tuile d'attention ; ici la forme est connue, et les deux GEMM ont des
régimes **opposés** :

| GEMM | k | n | noyau gagnant |
| --- | --- | --- | --- |
| scores = Q·Kᵀ | d (court) | tk (long) | `dot` si m < 16, sinon `legacy` |
| PV = W·Vᵀ | tk (long) | d (court) | `dot` presque toujours (jusqu'à ×4.2) |

Le noyau `dot` accumule le long de k avec une seule réduction horizontale par
case : il gagne quand k est profond. Le `legacy` bloque 8 lignes : m multiple
de 8 est sa zone de force — la règle l'y laisse. Recherche sur 62 formes de
tuiles : **moyenne ×1.28, pire cas ×1.00** (aucune régression). La règle
d'aiguillage *globale* n'est pas touchée : elle est déjà Pareto-optimale sous
la même contrainte, aucune variante testée ne la bat sans régresser quelque
part (279 formes f32).

**(d) Amortir le packing : le cache KV.** Le packing est en `O(tk·d·H_kv)`
alors que le calcul n'est qu'en `O(tq·tk·d·H_q)` : à tq petit, il domine. Or
les mêmes clés servent à des dizaines de tuiles successives.

```python
cache = sm.KVCache(k, v)            # packé une fois
out = cache.attend(q, lengths=L)    # réutilisé à chaque pas
```

Résultat cumulé (2 threads, f32) :

| | GFLOPS à tq=4 | |
| --- | --- | --- |
| boucle par tête (état précédent) | 5.0 | |
| + une descente C multi-têtes | 23.8 | ×4.8 |
| + cache KV amorti | **68.8** | **×13.8** |

| | médiane | min | max |
| --- | --- | --- | --- |
| multi-têtes vs boucle par tête | ×2.66 | ×1.36 | ×5.17 |
| multi-têtes vs numpy | ×15.63 | ×5.44 | ×21.84 |
| cache KV amorti vs appel complet | ×4.40 | ×2.35 | ×4.71 |

Le décodage token par token (tq = 1, tk = 4096) tourne à 34.5 GFLOPS, contre
2.0 pour numpy. Le chemin packé est **bit-à-bit identique** au chemin complet
(vérifié par test).

## 5. Cache KV en bf16 : la mémoire toujours, la vitesse parfois

Le bf16 (8 bits de mantisse) est la troncature naturelle du float32 : conversion
et reconversion sont des décalages de 16 bits. Le GEMM associé garde A en
float32 (requêtes et poids, peu volumineux) et lit B en bf16 — c'est là que
passe la bande passante.

| cache f32 | cache bf16 | ms f32 | ms bf16 | gain | erreur rel. |
| --- | --- | --- | --- | --- | --- |
| 1.05 Mo | 0.52 | 0.124 | 0.169 | **×0.73** | 1.3e-03 |
| 4.19 Mo | 2.10 | 0.482 | 0.589 | ×0.82 | 1.6e-03 |
| 8.39 Mo | 4.19 | 1.032 | 1.174 | ×0.88 | 1.6e-03 |
| 16.78 Mo | 8.39 | 2.683 | 2.494 | ×1.08 | 1.3e-03 |
| 16.78 Mo (d=128) | 8.39 | 1.948 | 1.382 | **×1.41** | 1.4e-03 |
| 67.11 Mo | 33.55 | 8.020 | 7.443 | ×1.08 | 2.0e-03 |

**L'empreinte est divisée par deux dans tous les cas ; la vitesse ne gagne
qu'au-delà de ~17 Mo de cache** — en dessous, la conversion coûte plus que la
bande passante économisée. C'est exactement le seuil où le cache KV cesse de
tenir dans les caches de la machine.

Le choix reste **explicite** (`dtype="bf16"`) et non automatique : basculer
silencieusement d'une erreur de 1e-7 à 2e-3 selon la taille du contexte serait
un piège pour l'appelant.

### Une attribution qu'il a fallu corriger

La première mesure donnait bf16 ×1.95 à d = 128. En isolant les GEMM, il est
apparu que le chemin bf16 utilisait le noyau `dot` alors que le chemin f32
partait sur `legacy` : **la moitié du « gain bf16 » était un effet de choix de
noyau**. C'est cette mesure qui a produit la règle (c) ci-dessus ; une fois les
deux chemins à armes égales, le gain bf16 réel tombe à ×1.08–1.41.

## 6. Bout-en-bout : un bloc de décodeur complet

Les briques assemblées en un bloc transformeur standard
(`experiments/attention/bench_block.py`) :

```
h = x + Wo · Attention(RMSNorm(x))            # attention_mha / KVCache
y = h + W2 · GELU(W1 · RMSNorm(h))            # matmul_nt_gelu + matmul_nt
```

**Prefill** (tout le contexte d'un coup, masque causal) :

| N | d_model | H_q/H_kv | d_ff | SpearVM | numpy | gain | tokens/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 512 | 8/2 | 2048 | 14.6 ms | 41.2 | ×2.82 | 17 506 |
| 512 | 512 | 8/2 | 2048 | 32.9 ms | 144.6 | ×4.39 | 15 546 |
| 1024 | 768 | 12/4 | 3072 | 129.0 ms | 781.1 | ×6.05 | 7 936 |
| 2048 | 768 | 12/4 | 3072 | **327.7 ms** | 3160.7 | **×9.65** | 6 250 |

Le gain croît avec N : c'est la part quadratique de l'attention qui domine, et
c'est exactement là que le softmax causal par longueurs et la tuile fusionnée
paient.

**Décodage** (un token contre un cache KV de tk) :

| tk | d_model | H_q/H_kv | SpearVM | numpy | gain | tokens/s |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 8/2 | 1.28 ms | 1.14 | **×0.89** | 783 |
| 2048 | 512 | 8/2 | 1.40 ms | 2.41 | ×1.72 | 715 |
| 4096 | 768 | 12/4 | 3.24 ms | 7.66 | ×2.37 | 309 |
| 8192 | 768 | 12/4 | 4.04 ms | 14.91 | ×3.69 | 248 |

Écart maximal avec la référence numpy : **1.1e-06** (les deux chemins partagent
la même GELU SPEAR ; le reste vient de l'ordre des sommations en float32).

La ligne à ×0.89 est publiée telle quelle : à contexte court, le bloc est
dominé par les GEMV du FFN (m = 1), purement limités par la bande passante —
il n'y a rien à y gagner, et nous payons quelques appels de plus. Le gain
n'apparaît qu'à partir de tk ≈ 2048.

## 7. Poids quantifiés : briser le mur de bande passante du décodage

Le §6 laissait une ligne à ×0.89 : à contexte court, le bloc est dominé par les
GEMV du FFN. Diagnostic mesuré : à m petit, un GEMM ne fait que **2·m flops par
poids lu**. Le temps n'est donc pas décidé par le nombre d'opérations mais par
le **nombre d'octets par poids**. Le seul levier est le format.

| format | octets/poids | échelle |
| --- | --- | --- |
| f32 | 4 | — |
| bf16 | 2 | aucune (le format porte l'exposant) |
| int8 | 1 | une **par ligne de sortie** (par neurone) |

Les activations restent en float32 : elles sont peu volumineuses et c'est sur
elles que se joue la précision du produit.

**GEMM à m petit** (`experiments/attention/bench_quant.py`) :

| m | k | n | f32 | bf16 | int8 | ×bf16 | ×int8 | Go/s f32 | Go/s int8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 768 | 3072 | 0.631 ms | 0.214 | 0.120 | ×2.95 | **×5.26** | 15.0 | 19.7 |
| 1 | 3072 | 768 | 0.613 ms | 0.173 | 0.104 | ×3.54 | **×5.87** | 15.4 | 22.6 |
| 4 | 3072 | 768 | 0.793 ms | 0.468 | 0.271 | ×1.69 | ×2.92 | 11.9 | 8.7 |
| 16 | 768 | 3072 | 1.499 ms | 1.042 | 1.607 | ×1.44 | **×0.93** | 6.3 | 1.5 |

Les colonnes de débit démontrent le mécanisme : à m = 1 le chemin f32 lit à
~15 Go/s — le plafond mémoire de la machine — et int8 lit **quatre fois moins
d'octets**. À partir de m ≈ 16 le GEMM redevient limité par le calcul, le
format cesse d'aider, et le noyau int8 (bloc 3×4 simple) perd contre les noyaux
f32 optimisés. **La quantification des poids est un outil de décodage, pas de
prefill** — c'est écrit dans la docstring de l'API.

**Bloc de décodeur complet, un token contre un cache KV** :

| tk | d_model | numpy | f32 | bf16 | int8 | ×numpy (int8) | poids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 0.692 ms | 0.901 | 0.406 | **0.291** | **×2.38** | 2.6 Mo |
| 2048 | 512 | 1.678 ms | 1.015 | 0.488 | **0.391** | ×4.30 | 2.6 Mo |
| 512 | 768 | 1.209 ms | 2.150 | 0.769 | **0.496** | ×2.44 | 5.9 Mo |
| 4096 | 768 | 6.390 ms | 2.426 | 1.192 | **0.826** | ×7.74 | 5.9 Mo |
| 8192 | 768 | 13.262 ms | 3.238 | 1.790 | **1.370** | **×9.68** | 5.9 Mo |

La ligne qui était à ×0.89 au §6 est maintenant à **×2.38**. Médiane ×2.94
contre le chemin f32 de SpearVM lui-même, jusqu'à **3 439 tokens/s** en
décodage mono-flux sur 2 vCPU, avec des poids quatre fois plus petits
(23.6 Mo → 5.9 Mo).

Coût : l'erreur relative en sortie de bloc passe de 1e-06 (f32) à **0.9–1.4 %**
en int8, et ~0.2 % en bf16. C'est le prix d'une quantification par ligne sans
étalonnage ; le publier est la moitié du résultat.

```python
w8 = sm.QuantizedWeight(w, dtype="i8")   # ou "bf16"
y = w8.matmul(x)                         # y = x · wᵀ
w8.dequantize()                          # ce que le noyau utilise vraiment
```

### Décodage par lots : où la quantification cesse de servir

Quand B séquences décodent ensemble, les poids sont lus une fois pour B tokens.
La quantification ne sert que tant qu'on est limité par la mémoire — le point de
bascule se mesure (`bench_batch.py`, contexte 4096, d_model 768) :

| B | f32 | bf16 | int8 | tok/s int8 | gain int8/f32 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.81 ms | 1.25 | **0.82** | 1 214 | ×3.41 |
| 4 | 4.16 ms | 2.84 | **1.90** | 2 109 | ×2.19 |
| 16 | 8.26 ms | 6.33 | **5.91** | 2 705 | ×1.40 |
| 64 | 18.00 ms | 18.00 | 17.95 | 3 566 | ×1.00 |

Deux phénomènes distincts, séparés par la mesure composant par composant :

* **au niveau du GEMM**, le croisement annoncé a bien lieu : à B = 64 la couche
  W1 met 3.24 ms en f32 contre 3.69 ms en int8 (×0.88) — le GEMM est redevenu
  limité par le calcul et le noyau int8, plus simple, perd ;
* **au niveau du bloc**, les trois formats convergent pour une *autre* raison :
  l'attention contre le cache de 4096 tokens pèse alors 7.3 ms sur ~18 ms et
  elle est identique dans les trois variantes. Ce n'est pas le f32 qui rattrape,
  c'est le poste « poids » qui cesse d'être celui qui décide.

Débit maximal mesuré sur 2 vCPU pour un bloc de décodeur : **3 566 tokens/s**
à B = 64, contre 356 tokens/s à B = 1 en f32 — le lot vaut ×10, la
quantification ×3.4, et les deux ne se cumulent pas.

## 8. API et couverture

```python
import spur_math as sm
sm.exp(x, out=None)                         # 1.69 ulp (f32), 2.12 ulp (f64)
sm.softmax(x, lengths=None, out=None)       # par ligne, longueurs causales
sm.attention_tile(q, k, v, scale=None, lengths=None)      # une tete
sm.attention_mha(q, k, v, scale=None, lengths=None)       # multi-tetes, GQA
cache = sm.KVCache(k, v, dtype="f32")       # ou "bf16" : empreinte / 2
cache.attend(q, lengths=None)
w8 = sm.QuantizedWeight(w, dtype="i8")      # ou "bf16" : poids / 4 ou / 2
w8.matmul(x)                                # pour le decodage (m petit)
```

`tests/test_softmax.py` (29 tests) couvre : contrat en ulp sur `exp`, cas
limites (`-1e4`, `-800`, saturations `±1e30`), lignes de longueur nulle,
invariance par translation, sommes à 1, tuile d'attention contre référence
float64, masque causal (la première requête ne voit qu'une clé : sa sortie doit
être exactement `v[0]`), équivalence multi-têtes/tête-par-tête, égalité
**bit-à-bit** du cache packé, bornes du bf16, et pour les poids quantifiés :
empreinte réduite du bon facteur, cohérence avec la déquantification, et le fait
qu'une ligne de très faible amplitude ne soit **pas écrasée** par les autres
(vérification que l'échelle est bien par ligne). Suite complète du dépôt :
**82 tests**.

## 9. Ce qui n'a pas marché

* **Le softmax « online »** : −20 % à −75 % contre les trois passes (§2).
  L'argument flash-attention est un argument de hiérarchie mémoire GPU.
* **Monter le degré du polynôme** au-delà de 5 (f32) / 10 (f64) : aucune
  précision gagnée, uniquement des FMA en plus.
* **Mesurer la précision contre une référence float64 sur des entrées float64**
  quand le noyau, lui, reçoit du float32 : produit un faux plancher de 4e-06
  qui masque complètement le comportement du polynôme (§1).
* **Attribuer un gain à bf16 sans vérifier que les deux chemins utilisent le
  même noyau** : la moitié du ×1.95 initial venait du choix de kernel (§7).
* **Utiliser des poids quantifiés en prefill** : à m ≥ 16 le GEMM redevient
  limité par le calcul et le noyau int8 perd (×0.93). Le format ne sert qu'à
  m petit.
* **Toucher à la règle d'aiguillage globale** pour capter les gains du régime
  m < 64 : aucune variante ne le fait sans régresser ailleurs, sur les 279
  formes f32 mesurées. La connaissance de forme est donc restée locale à
  l'attention.

---

*Mesures : `experiments/results/exp_fit.json`, `softmax_bench.csv/.json`,
`attention_tile.csv/.json`, journaux `softmax_log.txt`, `attention_tile_log.txt`.*
