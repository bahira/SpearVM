# Cas d'usage concrets (SpearVM, CPU sans GPU)

Trois cas sur des données réelles embarquées dans scikit-learn (aucun téléchargement).
Les modèles sont entraînés depuis zéro avec `spur_math.nn` (MLP, GELU exacte, Adam).
Machine : VM 2 cœurs, AVX-512, `OMP_NUM_THREADS=2`. Le bruit de la VM est d'environ ±15 %.

Reproduction : `cd examples/usecases && PYTHONPATH=../.. python <script>.py`
(résultats JSON dans `docs/bench/usecases/`).

## 1. Reconnaissance de chiffres : entraînement (`digits_mlp_train.py`)

Données : 1797 images 8×8 (sklearn `load_digits`), split 75/25 stratifié. MLP 64→128→10.

| Moteur | Précision test | Temps d'entraînement (40 époques) |
|---|---|---|
| SpearVM | 0,9844 | **0,43 à 0,48 s** |
| PyTorch CPU | 0,9844 | 0,71 à 0,76 s |

SpearVM est environ 1,6× plus rapide, avec la même précision. Le petit MLP est dominé par
l'overhead par opération de PyTorch ; SpearVM fait un appel natif par couche. Écart final des poids
3,2e-4 (arrondis fp32), la précision est identique.

## 2. Aide au diagnostic clinique : tumeur maligne / bénigne (`breast_cancer_screening.py`)

Données : 569 patientes, 30 mesures cellulaires (sklearn `load_breast_cancer`), validation croisée 5 plis stratifiée.

| Modèle | Précision | AUC ROC | Rappel malin | Temps total (5 plis) |
|---|---|---|---|---|
| Régression logistique (baseline) | **0,9789** | **0,9955** | **0,9915** | **0,02 s** |
| MLP SpearVM 30→32→2 | 0,9701 | 0,9920 | 0,9775 | 2,98 s |
| MLP PyTorch (même architecture) | 0,9684 | 0,9936 | 0,9776 | 8,98 s |

Conclusion honnête : **sur ce jeu tabulaire, un MLP n'apporte pas de gain** ; la régression logistique
est meilleure et 150× plus rapide. SpearVM entraîne le même MLP 3× plus vite que PyTorch.
Ce cas sert donc surtout de garde-fou : utiliser le réseau seulement si le problème le justifie.

## 3. Service d'inférence par lots (`digits_batch_serving.py`)

Le modèle digits (cas 1) est servi avec `spur_math.PackedMLP` (poids pré-packés, GEMM + biais + GELU exacte fusionnés),
PyTorch CPU et numpy/BLAS. Précision des logits vs SpearVM : 8,7e-6 (SpearVM), 1,3e-4 (PyTorch), 1,2e-4 (numpy).

Débit (échantillons/s, latence p50 ; voir le JSON pour p90) :

| Lot | SpearVM | PyTorch CPU | numpy |
|---|---|---|---|
| 1 | 57 900 (17,3 µs) | 26 200 (38,1 µs) | 66 400 (15,1 µs) |
| 32 | **1 081 800** (29,6 µs) | 540 400 (59,2 µs) | 389 600 (82,1 µs) |
| 256 | **2 615 300** (97,9 µs) | 1 897 900 (134,9 µs) | 355 800 (719,6 µs) |
| 1024 | **3 050 700** (335,7 µs) | 2 403 800 (426,0 µs) | 404 000 (2534 µs) |

SpearVM est le plus rapide à partir du lot 32 (environ 2× PyTorch à lot 32, 1,3× à lot 1024). À lot 1, numpy est au même niveau.

### Optimisation trouvée par ce cas

La première version de `PackedMLP.predict_logits` appelait `matmul` puis `gelu_erf` en numpy. À lot 1024,
elle prenait **2,13 ms**, alors que les noyaux ne prennent que 0,16 ms. Le passage à `matmul_gelu` (épilogue fusionné)
ramène le temps à **0,34 ms** (×6,3), sans changer la précision. Un test de non-régression est dans
`tests/test_nn_mlp.py`.

## Limites connues

- Le forward d'entraînement (`SpearMLP`) garde la pré-activation `T` et n'utilise pas encore le FFN fusionné
  (le gain d'entraînement n'est pas encore exploité au maximum). Le packing `PackedNT` ne sert qu'à l'inférence.
- Les mesures sont faites sur 2 cœurs. Sur une machine plus grosse, PyTorch et BLAS prendraient l'avantage sur les gros lots.
