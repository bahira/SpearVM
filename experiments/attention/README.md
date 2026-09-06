# attention/ — audit mesuré d'un mémoire d'architecture

Réimplémentation et vérification de « Théorie unifiée des architectures
d'attention hybrides et variétés non-euclidiennes à contexte ultra-long »
(R. Abdel-Aal, 27 août 2026).

| Fichier | Rôle |
| --- | --- |
| `modules.py` | les modules du mémoire codés **tels qu'écrits** (DeltaNet dans ses deux lectures, QSA/IndexPool, projection de Stiefel, Muon cubique et quintique, ReplaySSM, W8A8, hachage N-grammes) + les variantes corrigées |
| `audit.py` | un test par affirmation chiffrée → verdict TENU / NUANCE / LACUNE / FAUX |

```bash
cd experiments/attention && OMP_NUM_THREADS=2 python audit.py
```

Sorties : `../results/attention_audit.json` et `../results/attention_audit_log.txt`.
Rapport rédigé : [`../../docs/ATTENTION_AUDIT.md`](../../docs/ATTENTION_AUDIT.md).

Principe : on ne corrige rien en silence. Quand le texte et le listing du
mémoire divergent (c'est le cas pour la règle Delta), les deux sont implémentés
et mesurés côte à côte.
