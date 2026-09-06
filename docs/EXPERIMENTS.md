# EXPERIMENTS — campagne d'optimisation des noyaux GEMM (boucle ancree)

Toutes les valeurs de ce document sont **mesurees dans ce dépôt**, par les
scripts de `experiments/`, sur la machine de développement décrite plus bas.
Chaque chiffre renvoie au fichier de résultats qui le produit. Aucun chiffre
n'est repris d'une datasheet ou d'une estimation.

## 0. Protocole

| Élément | Choix | Pourquoi |
| --- | --- | --- |
| Machine | 2 vCPU x86-64, AVX2 + FMA, pas d'AVX-512 | `lscpu`, `spur_cpu_ok()` |
| Compilation | `gcc -O3 -mavx2 -mfma -fno-math-errno` | mêmes options que le paquet livré |
| Estimateur | min-of-N après warmup, blocs alternés entre variantes | le min est le moins pollué par l'ordonnanceur ; l'alternance empêche une variante de monopoliser une fenêtre de contention |
| Répétitions | 4 blocs (grandes formes) → 200 blocs (< 5 MFLOP) | une forme de 14 µs demande beaucoup plus de répétitions pour sortir du bruit |
| Threads | `OMP_NUM_THREADS` **et** `OPENBLAS_NUM_THREADS` réglés à la même valeur | sinon la comparaison avec numpy est truquée |
| Correction | chaque variante est vérifiée avant d'être chronométrée | `harness.verify` : C pré-rempli de NaN, 6 formes non alignées, référence float64 ; une variante fausse est **rejetée**, jamais classée |
| Score de tri | moyenne géométrique des GFLOPS sur 3 formes | évite de sur-optimiser une seule taille |

Piège rencontré et corrigé : mélanger numpy dans la même « course » que les
noyaux C fausse les formes de quelques dizaines de µs (ses threads et ses
allocations polluent le voisinage). Depuis, numpy est chronométré séparément —
c'est ce qui explique quelques écarts entre `results/shapes.csv` (protocole
initial) et `results/port_st.csv` (protocole final).

Volume : **902 variantes** générées, compilées, vérifiées et chronométrées
pour la recherche (`gemm_sweep.csv` 490 + `dotnt_sweep.csv` 412), puis ~450
mesures supplémentaires pour la calibration et la validation du portage.

## 1. Point de départ

Le noyau livré (`spur_matmul_nt`, conservé sous `spur_matmul_nt_legacy`) est un
noyau « produit scalaire » : 4 lignes de A × 1 ligne de B, une réduction
horizontale (`hsum`) par case de C.

| f64, mono-thread | 256³ | 512³ | 1024×768×1024 |
| --- | --- | --- | --- |
| SpearVM (legacy) | 21.7 GF | 19.9 GF | 20.5 GF |
| numpy / OpenBLAS | 53.0 GF | 55.3 GF | 62.8 GF |

Soit **0.36× BLAS par cœur**. Repère matériel : AVX2+FMA = 16 flops/cycle/cœur
en f64 ; numpy à 62.8 GF implique ~3.9 GHz, donc un plafond ≈ 62 GF/cœur en f64
et ≈ 125 GF/cœur en f32. Le noyau legacy est donc à ~1/3 du plafond.

Source : `results/gemm_summary.json`.

## 2. Deux familles de noyaux, deux générateurs

`experiments/gen_gemm.py` et `experiments/gen_dotnt.py` émettent du C paramétré
que l'autotuner compile à la volée (cache sha1 dans `experiments/build/`).

**[P] « pack »** — micro-noyau *broadcast* façon BLIS. A et B sont recopiés en
panneaux contigus (le packing de B transpose la convention NT), la maille
interne diffuse MR scalaires de A sur NR/VL vecteurs de B et accumule
MR×NR/VL registres : **zéro réduction horizontale**, chaque octet chargé sert
MR fois.

**[D] « dot-block »** — **zéro packing**. En convention NT (`C = A·Bᵀ`), k est
contigu des deux côtés : on peut donc bloquer MR lignes de A × NR lignes de B
directement en mémoire, avec MR×NR accumulateurs et une seule `hsum` par case
en fin de boucle k. MR+NR chargements alimentent MR×NR FMA, et aucun octet
n'est copié.

## 3. Recherche étagée, famille [P] (490 variantes)

`experiments/sweep_gemm.py`, descente par coordonnées en 4 étages
(forme → déroulage/prefetch → tuiles → raffinement).

**Étage 1 — forme MR×NR** (f64, moyenne géométrique) :

| forme | 2×4 | 4×4 | 6×4 | 4×8 | **6×8** | 3×12 | **4×12** | 2×16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GF | 12.4 | 21.6 | 25.4 | 34.6 | **37.3** | 35.4 | **37.3** | 30.5 |

Déjà ×1.8 sur le noyau livré, uniquement en changeant la forme du micro-noyau.

**Étage 2 — dérouler la boucle k *dégrade*** :

| variante | u1 | u2 | u4 |
| --- | --- | --- | --- |
| 6×8 | **37.3** | 32.6 | 27.4 |
| 4×12 | **37.3** | 29.9 | 28.3 |

gcc ordonnance déjà correctement la boucle d'intrinsics ; le déroulage manuel
provoque des débordements de registres. Le prefetch de C est neutre à
légèrement positif (retenu : distance 4 lignes).

**Champions** (`results/gemm_sweep.csv`) :

* f64 : `MR=4 NR=12 KC=576 MC=64 NC=2048 pf4` → **41.8 GF** (×2.02)
* f32 : `MR=4 NR=24 KC=384 MC=192 NC=2048 pf4` → **83.8 GF** (×1.43)

## 4. Recherche étagée, famille [D] (412 variantes)

`experiments/sweep_dotnt.py`. La contrainte de registres
(`MR*NR + MR + 1 ≤ 16`) plafonne la forme à 3×4.

* f64 : `MR=3 NR=4 KC=1024 NC=96 pf1` → **38.7 GF**
* f32 : `MR=3 NR=4 KC=1024 NC=256 pf2` → **75.9 GF**

Légèrement en dessous de [P] sur les grandes formes carrées… mais sans aucune
copie, donc largement devant dès qu'une dimension est petite (§6).

## 5. Options de compilation et threads (`results/flags_threads.json`)

| jeu d'options | f64 1024×768×1024 | f32 1024×768×1024 |
| --- | --- | --- |
| base (`-O3 -mavx2 -mfma`) | 41.4 | 79.5 |
| `-funroll-loops` | 30.1 | 67.0 |
| `-march=native` | 40.7 | 82.3 |
| `-march=native -fipa-cp-clone` | 41.0 | 82.5 |

Conclusion : `-funroll-loops` coûte 25 % (même cause qu'au §3), `-march=native`
rapporte ~2 % — pas assez pour sacrifier la portabilité du binaire livré. **Les
options de compilation restent inchangées.**

OpenMP sur 2 threads, en revanche, passe à l'échelle — à condition que MC soit
assez petit pour donner au moins un bloc par thread :

| f32, 2 threads | 256³ | 512³ | 1024×768×1024 |
| --- | --- | --- | --- |
| MC=192 (champion mono-thread) | 90.8 | 100.8 | 136.2 |
| MC=128 | 123.6 | 154.3 | **169.3** |

D'où le `MC` adaptatif du portage : `MC_eff = min(MC_max, ⌈m/nthreads⌉)`
arrondi à MR.

## 6. Aiguillage : quel noyau, pour quelle forme ?

Aucun des deux noyaux ne gagne partout. Mesure croisée sur 149 formes × 2
précisions, **dans la .so livrée** (`experiments/sweep_routing.py` →
`results/routing.csv`), les trois noyaux étant exportés pour l'A/B.

Faits saillants (f64, rapport au noyau legacy) :

| forme | [P] pack | [D] dot |
| --- | --- | --- |
| 1×512×512 | 0.30 | 1.04 |
| 16×512×512 | 0.88 | 1.19 |
| 512×256×48 | 1.56 | 1.39 |
| 2048×128×128 | 2.02 | 1.72 |

Le packing coûte O(mk + nk) copies pour 2mnk flops : il n'est amorti que si
`m` **et** `n` sont assez grands. Une règle « toujours packer » perdrait
jusqu'à ×0.09 sur les formes étroites.

La règle a donc été **cherchée** (`(min_m, min_n, dot_lo, min_k_tile)`) sous
contrainte explicite « aucune régression » : on maximise le gain moyen sur les
formes de vraie taille (`m·k·n ≥ 2²³`) **sous** pire-cas ≥ 1.00 du legacy.

```
si  m < min_m ou n < min_n            -> legacy   (trop étroit pour les deux nouveaux)
sinon si m ≤ 64 et n ≤ 64             -> dot si k ≥ min_k_tile, sinon legacy
sinon si min(m, n) ≤ dot_lo           -> dot      (une dimension étroite : ne pas packer)
sinon                                 -> pack
```

| | min_m | min_n | dot_lo | min_k_tile |
| --- | --- | --- | --- | --- |
| f64 | 24 | 32 | 32 | 96 |
| f32 | 64 | 48 | 32 | 512 |

Les seuils diffèrent parce que le micro-noyau f32 est deux fois plus large
(NR=24 contre 12) : il lui faut plus de colonnes pour amortir le packing. Le
garde-fou `min_k_tile` vient d'une mesure dédiée du plan (petite tuile × k)
(`results/smalltile.csv`) : sous ces profondeurs, les MR×NR réductions finales
ne sont pas amorties.

## 7. Résultat du portage (`src/spur_kernels.c`)

Les champions sont portés derrière l'API existante ; l'ancien noyau reste
exporté (`spur_matmul_nt_legacy`, `spur_matmul_nt_f32_legacy`) et reste
activable en production par `SPUR_MM_LEGACY=1`.

Mesure A/B des deux noyaux **dans la .so livrée**, 20 formes, numpy chronométré
séparément (`experiments/verify_port.py` → `results/port_st.csv`, `port_mt.csv`) :

| | médiane | min | max | erreur max |
| --- | --- | --- | --- | --- |
| f64, 1 thread | **×1.78** | ×1.00 | ×3.39 | 1.5e-15 |
| f32, 1 thread | **×1.50** | ×1.00 | ×4.58 | 8.3e-07 |
| f64, 2 threads | ×1.43 | ×1.00 | ×3.21 | 1.5e-15 |
| f32, 2 threads | ×1.30 | ×0.92 | ×4.51 | 8.3e-07 |

Extraits (f64, 1 thread, GFLOPS) :

| forme | legacy | v2 | numpy | v2/legacy | v2/numpy |
| --- | --- | --- | --- | --- | --- |
| 512×512×512 | 20.8 | 43.3 | 58.9 | ×2.09 | 0.73 |
| 1024×1024×1024 | 18.2 | 41.4 | 63.5 | ×2.28 | 0.65 |
| 512×64×512 | 21.1 | 45.0 | 43.1 | ×2.13 | **1.04** |
| 512×16×512 | 13.0 | 38.4 | 25.8 | ×2.95 | **1.49** |
| 2048×128×128 | 24.8 | 43.2 | 41.6 | ×1.74 | **1.04** |

Le rapport à OpenBLAS passe de 0.36 à **0.65–0.73** sur les grandes formes
carrées, et le noyau **dépasse OpenBLAS** sur les formes à k court ou à
dimension étroite (là où BLAS paie son packing).

## 8. GEMM + GELU : la fusion coûtait plus qu'elle ne rapportait

`spur_matmul_nt_gelu` fusionnait la non-linéarité dans la boucle j. Cette
fusion **interdit de couper k**, donc interdit le micro-noyau packé. La v2 fait
le GEMM autotune puis un épilogue biais+GELU vectorisé (une relecture de C,
purement mémoire) — `experiments/verify_gelu_fused.py` :

| forme | f64 legacy | f64 v2 | gain | f32 legacy | f32 v2 | gain |
| --- | --- | --- | --- | --- | --- | --- |
| 256×256×256 | 17.2 | 41.2 | ×2.39 | 27.9 | 81.1 | ×2.91 |
| 1024×768×1024 | 20.9 | 43.2 | ×2.06 | 40.3 | 87.7 | ×2.18 |
| 512×64×512 | 8.9 | 40.2 | ×4.52 | 11.7 | 80.4 | ×6.85 |

Médiane **×2.23 (f64)** et **×2.34 (f32)** en mono-thread, écart numérique
entre les deux chemins ≤ 1.9e-15 (f64) / 8.7e-07 (f32).

## 9. Bout-en-bout : une itération d'entraînement

`experiments/bench_mlp.py` — MLP 784-256-128-10, batch 128, avant + arrière
complets via l'API publique, deux processus (`SPUR_MM_LEGACY=1` vs v2) :

| | legacy | v2 | gain |
| --- | --- | --- | --- |
| f64, 1 thread | 14.31 ms | **7.50 ms** | ×1.91 |
| f32, 1 thread | 5.93 ms | **4.05 ms** | ×1.47 |
| f64, 2 threads | 11.71 ms | **6.89 ms** | ×1.70 |
| f32, 2 threads | 4.54 ms | **3.44 ms** | ×1.32 |

## 10. Correction

* `tests/test_matmul_v2.py` : 324 formes par précision (dimensions autour des
  seuils d'aiguillage 1…129 et profondeurs 1…1153 qui traversent les blocages
  KC), C pré-rempli de NaN pour détecter toute case non écrite, comparaison au
  legacy, chaîne `matmul_backward`, et les deux variantes GELU (avec/sans biais).
* Suite complète : `47 passed, 1 skipped`.
* Erreur relative maximale observée sur l'ensemble des mesures : **1.5e-15**
  (f64) et **8.3e-07** (f32) — cohérent avec l'accumulation en précision native.

## 11. Ce qui n'a pas marché (et ne sera pas retenté)

* **Dérouler la boucle k** du micro-noyau : −12 % à −26 % (§3).
* **`-funroll-loops`**, **`-fassociative-math` & co.** : −25 % et −20 % (§5).
* **`-march=native`** : +2 %, insuffisant face à la perte de portabilité.
* **Hisser les NR vecteurs de B** dans la famille [D] (`hoist_b`) : neutre à
  négatif — le compilateur le fait déjà quand c'est rentable.
* **Une règle d'aiguillage unique pour les deux précisions** : impossible sans
  régression, les micro-noyaux n'ont pas la même largeur (§6).
* **Mélanger numpy et les noyaux C dans la même course de mesure** : produit des
  écarts fantômes de ±25 % sur les formes courtes (§0).

## 12. Reproduire

```bash
make spur_math/libspur_kernels.so
cd experiments

# recherche (long : ~7 min par famille sur 2 vCPU)
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_THREAD_TIMEOUT=1 python sweep_gemm.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_THREAD_TIMEOUT=1 python sweep_dotnt.py

# calibration de l'aiguillage et validation du portage
python sweep_routing.py && python sweep_smalltile.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python verify_port.py st
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python verify_port.py mt
python verify_gelu_fused.py st && python bench_mlp.py
```

`OPENBLAS_THREAD_TIMEOUT=1` est nécessaire **avant l'import de numpy** : sans
lui, les threads OpenBLAS continuent de tourner à vide et affament OpenMP, ce
qui faisait apparaître le noyau SpearVM 3× plus lent qu'il ne l'est.

Les résultats bruts (CSV/JSON + journaux) sont versionnés dans
`experiments/results/`.
