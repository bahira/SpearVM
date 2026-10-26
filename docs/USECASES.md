# Cas d'usage concrets (SpearVM, CPU sans GPU)

Quatre cas sur des données réelles embarquées dans scikit-learn (aucun téléchargement).
Les modèles sont entraînés depuis zéro avec `spur_math.nn` (MLP, GELU exacte, Adam).
Machine : VM 2 cœurs, AVX-512, `OMP_NUM_THREADS=2`. Le bruit de la VM est d'environ ±15 %.

Reproduction : `cd examples/usecases && PYTHONPATH=../.. python <script>.py`
(résultats JSON dans `docs/bench/usecases/`). Test de fumée : `tests/test_usecases_smoke.py`.

## 1. Reconnaissance de chiffres : entraînement (`digits_mlp_train.py`)

Données : 1797 images 8×8 (sklearn `load_digits`), split 75/25 stratifié. MLP 64→128→10, 40 époques.

| Moteur | Précision test | Temps d'entraînement |
|---|---|---|
| SpearVM | 0,9844 | **0,41 s** |
| PyTorch CPU | 0,9844 | 0,71 s |

SpearVM est environ 1,7× plus rapide, avec la même précision. Le petit MLP est dominé par
l'overhead par opération de PyTorch ; SpearVM fait un appel natif par couche. Écart final des poids
3,2e-4 (arrondis fp32).

## 2. Aide au diagnostic clinique : tumeur maligne / bénigne (`breast_cancer_screening.py`)

Données : 569 patientes, 30 mesures cellulaires (sklearn `load_breast_cancer`), validation croisée 5 plis stratifiée.

| Modèle | Précision | AUC ROC | Rappel malin | Temps total (5 plis) |
|---|---|---|---|---|
| Régression logistique (baseline) | **0,9789** | **0,9955** | **0,9915** | **0,02 s** |
| MLP SpearVM 30→32→2 | 0,9701 | 0,9920 | 0,9775 | 2,54 s |
| MLP PyTorch (même architecture) | 0,9684 | 0,9936 | 0,9776 | 9,07 s |

Conclusion honnête : **sur ce jeu tabulaire, un MLP n'apporte pas de gain** ; la régression logistique
est meilleure et environ 100× plus rapide. SpearVM entraîne le même MLP 3,6× plus vite que PyTorch.
Ce cas sert donc surtout de garde-fou : utiliser le réseau seulement si le problème le justifie.

## 3. Classification de vins par cultivar (`wine_quality_classify.py`)

Données : 178 vins, 13 mesures chimiques, 3 cultivars (sklearn `load_wine`), validation croisée 5 plis stratifiée.
MLP 13→32→3, 150 époques.

| Modèle | Précision | F1 macro | Log-loss | Temps total (5 plis) |
|---|---|---|---|---|
| SpearVM | 0,9832 | 0,9826 | 0,0591 | **1,42 s** |
| PyTorch (même architecture) | 0,9832 | 0,9826 | **0,0511** | 5,75 s |
| Régression logistique | 0,9832 | 0,9826 | 0,0627 | 0,02 s |

Les trois modèles ont la même précision. PyTorch est le mieux calibré (log-loss le plus bas), SpearVM
entraîne 4× plus vite. La régression logistique reste la plus simple et la plus rapide à entraîner.

## 4. Service d'inférence par lots (`digits_batch_serving.py`)

Le modèle digits (cas 1) est servi avec `spur_math.PackedMLP` (poids pré-packés, GEMM + biais + GELU exacte fusionnés),
PyTorch CPU et numpy/BLAS. Écart des logits vs SpearVM : 9,4e-6 (SpearVM), 1,3e-4 (PyTorch), 1,2e-4 (numpy).

Débit (échantillons/s, latence p50 ; voir le JSON pour p90) :

| Lot | SpearVM | PyTorch CPU | numpy |
|---|---|---|---|
| 1 | 57 400 (17,4 µs) | 25 700 (38,9 µs) | 66 600 (15,0 µs) |
| 32 | **1 079 800** (29,6 µs) | 541 800 (59,1 µs) | 406 600 (78,7 µs) |
| 256 | **2 545 400** (100,6 µs) | 1 833 600 (139,6 µs) | 378 700 (676,0 µs) |
| 1024 | **2 819 800** (363,1 µs) | 2 278 500 (449,4 µs) | 308 200 (3322 µs) |

SpearVM est le plus rapide à partir du lot 32 (environ 2× PyTorch à lot 32, 1,2× à lot 1024). À lot 1, numpy est au même niveau.

### Optimisation trouvée par ce cas

La première version de `PackedMLP.predict_logits` appelait `matmul` puis `gelu_erf` en numpy. À lot 1024,
elle prenait **2,13 ms**, alors que les noyaux ne prennent que 0,16 ms. Le passage à `matmul_gelu` (épilogue fusionné)
ramène le temps à **0,36 ms** (×6), sans changer la précision. Un test de non-régression est dans
`tests/test_nn_mlp.py`.

### Optimisation de l'entraînement

Un pas d'entraînement (batch 64, MLP 64→128→10) dure environ 475 µs. Le profil montre : forward ~90 µs,
backward ~250 µs, Adam ~110 µs. Adam utilise maintenant un tampon plat unique (une dizaine d'opérations numpy
par pas au lieu d'environ cinquante) : écart 1e-7 avec l'ancienne version après 50 pas, gain d'environ 5 % sur le pas complet.
Le reste du coût est dans les petits appels natifs du forward et du backward.

## Limites connues

- Le forward d'entraînement (`SpearMLP`) garde la pré-activation `T` (nécessaire au backward) et n'utilise pas
  le FFN fusionné : le gain à l'entraînement reste limité sur de petits réseaux.
- Les mesures sont faites sur 2 cœurs. Sur une machine plus grosse, PyTorch et BLAS prendraient l'avantage sur les gros lots.
- Les jeux de données sont petits (178 à 1797 exemples) : les conclusions sur la précision valent pour ces données.
