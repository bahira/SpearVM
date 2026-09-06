# attention/ — audit d'architecture, index hiérarchique, noyaux

| Fichier | Rôle |
| --- | --- |
| `modules.py` | les modules du mémoire audité, codés **tels qu'écrits** (les deux lectures de la règle Delta, QSA/IndexPool, Stiefel, Muon, ReplaySSM, W8A8) |
| `audit.py` | un test par affirmation chiffrée → TENU / NUANCE / LACUNE / FAUX |
| `hierarchical.py` | l'index hiérarchique manquant (arbre de résumés + faisceau causal) |
| `bench_hier.py` | coût (comptage exact) et fidélité (masse d'attention) du routage |
| `bench_tile.py` | tuile d'attention C contre numpy |
| `bench_mha.py` | multi-têtes contre boucle par tête, et régime décodage avec `KVCache` |
| `bench_bf16.py` | effet de la règle de noyau, et cache KV bf16 (mémoire / vitesse / précision) |
| `bench_qsa_e2e.py` | QSA complet : listing d'origine contre version corrigée + noyaux |

```bash
cd experiments/attention
OMP_NUM_THREADS=2 python audit.py         # les 25 verdicts
OMP_NUM_THREADS=2 python bench_hier.py    # routage N^1.30 contre N^2.00
OMP_NUM_THREADS=2 python bench_mha.py     # x13.8 sur les petites tuiles
OMP_NUM_THREADS=2 python bench_qsa_e2e.py # x20.4 bout en bout
```

Rapports : [`ATTENTION_AUDIT.md`](../../docs/ATTENTION_AUDIT.md) et
[`TRANSCEND.md`](../../docs/TRANSCEND.md).
