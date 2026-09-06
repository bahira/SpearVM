# transcend/ — recherche d'approximations et noyaux exp/softmax

| Fichier | Rôle |
| --- | --- |
| `fit_exp.py` | recherche minimax de `exp` par itérations de Remez **en erreur relative**, balayage du degré, choix par mesure |
| `gen_kernels.py` | génération du C : `exp` AVX2 (f32/f64) + trois stratégies de softmax (3 passes, online, causal par longueurs) |
| `bench_softmax.py` | vérification (en ulp, contre numpy f32 **et** la référence f64) puis débit |

```bash
python fit_exp.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python bench_softmax.py
```

Rapport : [`../../docs/TRANSCEND.md`](../../docs/TRANSCEND.md).
Le C généré est conservé dans `../results/spur_softmax_generated.c` ; la version
portée dans le noyau livré est dans `src/spur_kernels.c`.

Deux pièges de mesure documentés dans le rapport, tous deux rencontrés ici :
arrondir l'entrée à la précision cible **avant** de calculer la référence, et se
souvenir que `harness.race` renvoie des millisecondes.
