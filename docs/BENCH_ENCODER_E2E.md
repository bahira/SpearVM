# Benchmark end-to-end : encodeur Transformer sur CPU (SpearVM vs PyTorch vs ONNX Runtime)

Script : `examples/bench_encoder_e2e.py` — résultats bruts : `docs/bench/encoder_e2e_cpu2.json`.

## Modèle et protocole
- Architecture encodeur BERT / BGE-small : **12 couches, d=384, 12 têtes, FFN 1536 GELU, seq 128**, pre-LN.
- **Poids aléatoires (seed fixe)** : Hugging Face est inaccessible depuis le sandbox. Le coût de calcul
  est celui d'un vrai checkpoint de même forme ; la *qualité* des sorties n'est pas évaluée.
- Embeddings et tokenizer exclus : entrée = tenseur (B,T,384) identique pour tous les chemins.
- Machine : VM 2 cœurs, AVX-512 présent, 3 Go RAM. Threads = 2 pour tous les chemins.
- Mesure : médiane de 20 exécutions après 3 de chauffe (p90 reporté dans le JSON).
- Chemins comparés :
  1. **PyTorch CPU** (torch 2.14, GELU exacte) — référence.
  2. **ONNX Runtime CPU** (onnxruntime 1.31) sur le graphe exporté depuis le même modèle.
  3. **Hybride SpearVM** : attention / LayerNorm / softmax en PyTorch ; FFN (matmul + GELU fusionné + matmul)
     via `spur_math.PackedNT` (poids pré-packés une fois).

## Résultats (médiane, ms)

| Batch | PyTorch CPU | ONNX Runtime | Hybride SpearVM | SpearVM vs torch | SpearVM vs ORT |
|---|---|---|---|---|---|
| 1 | 48.1 | 45.4 | 54.2 | ×0.89 | ×0.84 |
| 8 | 253.0 | 234.3 | 415.7 | ×0.61 | ×0.56 |

Les temps PyTorch varient de ~39 à ~48 ms selon les exécutions (VM partagée) ; l'ordre des résultats est stable.

## Décomposition (couche FFN seule, 1 couche, médiane)

| M (lignes) | PyTorch FFN | SpearVM noyaux seuls | + conversions torch↔numpy |
|---|---|---|---|
| 128 | 1.43 ms | 1.71 ms (×0.83) | 1.76 ms |
| 1024 | 10.09 ms | 13.05 ms (×0.77) | 15.24 ms |

Lecture : le noyau GEMM est ~20 à 25 % plus lent que MKL/oneDNN *même avec la fusion*. Les conversions
coûtent surtout à grand batch.

## Précision
- Hybride avec GELU SpearVM (approximation polynomiale) : **max|écart| = 0.136, cos = 0.99730** vs PyTorch.
- Même hybride avec GELU exacte appliquée sur la sortie du matmul : **max|écart| = 1.8e-7, cos = 1.000000**.
- Conclusion : l'écart vient **entièrement** de l'approximation GELU du noyau fusionné, pas du matmul.

## Mise à jour : GELU exacte dans le fusionné (champion superspear)

Les résultats ci-dessus utilisent la GELU v1 approximative dans le FFN fusionné. Après passage à la
GELU exacte (`gelu_erf`) : écart max 1.0e-5, **cos 1.000000** ; SpearVM vs torch ×0.79 (batch 1) / ×0.81 (batch 8),
vs ORT ×0.69 / ×0.77. Données : `docs/bench/encoder_e2e_cpu2_exactgelu.json`. Voir `docs/CHAMPIONS_SUPERSPEAR.md`.

## Verdict
Sur ce modèle et cette machine, **SpearVM ne bat pas PyTorch ni ONNX Runtime** : il est plus lent
(×0.56 à ×0.89) et moins précis avec le GELU fusionné actuel. Le pré-packing des poids a gagné
~29 % sur le noyau FFN, mais ne suffit pas.

Pistes qui pourraient changer le résultat (non faites) :
- un épilogue GELU précis (`gelu_erf`, erreur 2e-5) dans le noyau fusionné, pour supprimer l'écart de précision ;
- un GEMM qui atteint au moins MKL/oneDNN sur M=128–1024 (tuning de tuiles, packing de A plus fin) ;
- intégrer le FFN *et* les projections d'attention, pour réduire le nombre de passages torch↔numpy.

Pour la démonstration technique, le résultat est exploitable : fusion + pré-packing + repli automatique
fonctionnent et sont vérifiés. Pour la vente, il faut d'abord les pistes ci-dessus.
