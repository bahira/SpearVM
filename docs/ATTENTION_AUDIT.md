# Audit mesuré — « Théorie unifiée des architectures d'attention hybrides »

Document audité : R. Abdel-Aal, *Théorie unifiée des architectures d'attention
hybrides et variétés non-euclidiennes à contexte ultra-long (1M tokens)*,
27 août 2026.

Méthode : chaque affirmation chiffrée du mémoire a été réimplémentée **telle
qu'écrite** (`experiments/attention/modules.py`) puis mesurée
(`experiments/attention/audit.py` → `experiments/results/attention_audit.json`).
Aucun verdict n'est une opinion : chacun renvoie à une exécution reproductible
sur la machine de ce dépôt (2 vCPU AVX2).

```bash
cd experiments/attention && OMP_NUM_THREADS=2 python audit.py
```

**Synthèse : 25 affirmations testées — 11 fausses, 8 à nuancer, 2 lacunes, 4 tenues.**

Le mémoire contient de vraies bonnes idées (le ratio hybride 3:1, l'état
récurrent O(1), le routage par micro-blocs, la contrainte orthogonale sur les
mélanges résiduels) — ce sont les briques standard de la littérature 2024-2025
et elles fonctionnent. Ce qui ne tient pas, c'est l'appareil de preuve : les
chiffres annoncés comme « prouvés numériquement » ne sont pas des mesures, et
trois théorèmes démontrent autre chose que ce qu'ils énoncent.

---

## 1. Le problème central : les benchmarks à 1M tokens n'ont pas pu être exécutés

Le résumé affirme que « les benchmarks numériques NumPy conduits dans ce travail
prouvent formellement une réduction de 625× […] et une accélération de 72.3× ».
J'ai chronométré le listing fourni et extrapolé sa loi d'échelle mesurée :

| Module (listing du mémoire) | coût mesuré | extrapolation à N = 1 M |
| --- | --- | --- |
| QSA / IndexPool | N^1.92 (256 → 2048) | **78 h pour une seule couche, une seule passe** |
| Gated DeltaNet | 1 396 µs/token (linéaire ✓) | **14 h pour ses 36 couches** |

Plus de 100 heures pour une passe avant. Les lignes 262 k et 1 M du tableau 4.1
sont donc des **sorties de modèle analytique**, pas des mesures. Un modèle
analytique est parfaitement légitime — il faut simplement l'annoncer comme tel.
Présenté comme un benchmark exécuté, il invalide la section « validation ».

Corollaire mesuré et plus gênant : le coût du routage QSA croît en **N^1.92**.
Chaque micro-bloc score *tous* les blocs passés (N/4 produits scalaires) avant
d'en retenir 512. Borner le budget d'attention ne borne pas le coût de la
sélection : l'architecture reste quadratique. Il faudrait un index hiérarchique,
absent du mémoire.

## 2. Verdicts

| # | Affirmation | Annoncé | Mesuré | Verdict |
| --- | --- | --- | --- | --- |
| 1 | KV cache dense à 1M | 457.7 GB (§1) **et** 468.75 GB (§4.1) | 491.5 GB = 457.8 GiB | **FAUX** |
| 2 | KV cache hybride à 1M | 750.0 MB | 849.8 MB avec ses propres formules | **FAUX** |
| 3 | Facteur de réduction KV | 625.00× | 578.4× | **FAUX** |
| 4 | Budget KV en O(1) | O(1) + O(k·B·d) | l'index compressé est en O(N) | NUANCE |
| 5 | FLOPs denses à 1M | 582.40 PFLOPs | 491.5 PFLOPs (attention seule) | NUANCE |
| 6 | Accélération 72.26× | mesurée | non exécutable ici | LACUNE |
| 7 | κ(produit des mélanges projetés) | 1.0000000000000056 | 1.0000000000000058 | **TENU** |
| 8 | Propagation §1.2 avec F = 0 | isométrie | ‖x_L‖/‖x_0‖ = 5.96e-08 | **FAUX** |
| 9 | κ(J) global = 1 ⇒ pas d'évanouissement | 1.000 | 1.6e+18 jacobienne complète | **FAUX** |
| 10 | « variété de Stiefel St(M,M) » | — | c'est le groupe orthogonal O(M) | NUANCE |
| 11 | Équation 2.1 ≡ listing §3 | même règle Delta | opérateurs différents | **FAUX** |
| 12 | BIBO-stable ∀ g∈[0,1[, β∈]0,2[ | toujours | vrai pour le code, faux pour l'équation | NUANCE |
| 13 | Certificat de Lyapunov | ΔV ≤ 0 | borne non homogène, ne démontre pas la conclusion | NUANCE |
| 14 | ReplaySSM \|Δ\|∞ | 0.0 | 0.0 | **TENU** |
| 15 | Compression ReplaySSM 128× | propriété du modèle | choix du pas de checkpoint | NUANCE |
| 16 | W8A8 SQNR | 39.14 dB | 35.42 dB per-tensor, 39.10 dB per-canal | NUANCE |
| 17 | Muon : orthogonalisation (6 iters) | orthogonalisation | défaut 0.521 ; il en faut ≈ 15 | **FAUX** |
| 18 | Newton-Schulz « ordre 3/5 » | — | seul l'ordre 3 est implémenté ; le quintique plafonne à 0.33 | NUANCE |
| 19 | Causalité du listing QSA | modèle autorégressif | fuite de 3 tokens futurs par position | **FAUX** |
| 20 | Fidélité du routage IndexPool | non mesurée | 59.5 % de la masse d'attention capturée en moyenne, 24.8 % au pire | LACUNE |
| 21 | Débit table N-grammes | 11 372 tokens/s | 1.9 M tokens/s en RAM | **FAUX** |
| 22 | Table N-grammes : 51 B / 25.6 GB / 0 FLOP | — | 51.2 B / 25.6 GB / 0 FLOP | **TENU** |
| 23 | Amdahl p=0.88, s=24.5 | 6.41× | 6.414× | **TENU** |
| 24 | Routage QSA linéaire | budget borné | N^1.92 mesuré | **FAUX** |
| 25 | Benchmarks 1M exécutés | « prouvent formellement » | > 100 h/passe avec le listing | **FAUX** |

## 3. Les trois erreurs qui comptent

### 3.1 Le théorème mHC démontre une propriété de `H`, pas du réseau

La projection Π(H) = W_U W_Vᵀ est correcte, et le produit de 48 matrices
orthogonales a bien κ = 1 — je le mesure à 1.0000000000000058. Mais c'est un
résultat **trivial** (un produit d'orthogonales est orthogonal), et surtout ce
n'est pas la jacobienne du réseau. La propagation écrite en §1.2 est

  x_{l+1} = (1/√2)·( Π(H_l)x_l + F(Π(H_l)x_l) )

dont la jacobienne de couche est (1/√2)(I + F′)Π(H_l). Avec un bloc résiduel
générique, je mesure **κ = 1.6 × 10¹⁸** sur 48 couches. La contrainte de Stiefel
ne contraint que le mélange linéaire ; elle ne dit rien du terme F, qui est
justement là où les gradients explosent.

Pire, l'équation se contredit elle-même : avec F = 0, le facteur 1/√2 contracte
le signal d'un facteur 2 par couche. Mesure sur 48 couches :
**‖x_L‖/‖x_0‖ = 5.96 × 10⁻⁸**, exactement 2^(−24). L'équation censée prouver
l'absence d'évanouissement *produit* un évanouissement.

### 3.2 L'équation DeltaNet et son code n'ont pas le même domaine de stabilité

Le texte (§2.1) définit l'erreur `e_t = v_t − S_{t−1} k̃_t` avec l'état **non
amorti**, le listing applique la porte `g` **avant** de calculer l'erreur. Ce
détail change l'opérateur homogène :

| | opérateur | rayon spectral | stable si |
| --- | --- | --- | --- |
| listing §3 | g(I − β k̃k̃ᵀ) | g·max(1, \|1−β\|) = g | ✅ tout β ∈ ]0,2[ |
| équation §2.1 | gI − β k̃k̃ᵀ | max(\|g\|, \|g−β\|) | β < 1 + g |

Mesure directe (impulsion unique puis entrée nulle, 600 pas, g = 0.9, β = 1.95) :
le listing converge vers 0, l'équation diverge à **‖S‖ = 9.6 × 10¹²**. La
conclusion « BIBO-stable » est donc vraie pour le code livré et fausse pour
l'équation qui est censée la démontrer.

### 3.3 Le facteur 625× a été posé avant d'être calculé

468.75 GB / 750 MB = 625.0000 **exactement**. Or 468.75 GB ne correspond ni à la
valeur du §1 (457.7, qui est en GiB), ni au calcul direct (491.5 GB avec d=2560 ;
468.75 impliquerait d = 2441). Et en recalculant l'empreinte hybride avec les
formules du mémoire lui-même, on trouve 849.8 MB, dominée à 90 % par l'index
compressé (768 MB) que le texte présente comme un détail. Le vrai facteur est
**578×** — ce qui reste excellent, et n'avait pas besoin d'être arrondi.

## 4. Ce que le mémoire ne mesure jamais et qui décide de tout

Le document chiffre la mémoire économisée ; il ne chiffre jamais **ce qui est
perdu**. J'ai donc mesuré la fidélité du routage IndexPool : fraction de la masse
d'attention exacte capturée par les blocs sélectionnés (N = 1024, budget 256
tokens, par tête) :

* **59.5 %** en moyenne, **24.8 %** au pire ;
* écart maximal à l'attention dense causale : 8.3 %.

C'est le seul arbitrage qui compte pour une architecture d'attention creuse, et
il est absent. Un facteur 625× sur le KV cache ne veut rien dire sans la perte de
qualité associée.

## 5. Ce qui tient, et ce qu'il faudrait écrire à la place

Tiennent sans réserve : la projection orthogonale par SVD, l'exactitude
bit-à-bit du ReplaySSM (c'est du déterminisme, pas de la précision — le rejeu
d'une même séquence d'opérations flottantes *doit* redonner le même résultat),
l'arithmétique de la table N-grammes, et le calcul d'Amdahl.

Une version défendable du même travail dirait :

1. « modèle analytique de budget mémoire/calcul » au lieu de « benchmarks
   prouvent formellement » — et publierait le script qui produit le tableau ;
2. « le mélange résiduel est isométrique par construction » au lieu de
   « κ(J_global) = 1 élimine formellement l'évanouissement du gradient » ;
3. « stable pour β < 1 + g » (ou : appliquer la porte avant l'erreur, comme le
   code) au lieu de « pour tout β ∈ ]0,2[ » ;
4. « 578× de réduction du KV cache, dominée par l'index O(N) qu'il reste à
   hiérarchiser » au lieu de « 625× » ;
5. une courbe qualité/budget du routage IndexPool, qui manque entièrement.

Aucun de ces cinq changements n'affaiblit l'architecture proposée. Ils la
rendent vérifiable — ce qui, dans un document qui revendique une doctrine
« GROUNDED », est le seul critère qui compte.

---

*Toutes les valeurs de ce rapport proviennent de `experiments/attention/audit.py`.
Journal complet : `experiments/results/attention_audit_log.txt`, données
machine : `experiments/results/attention_audit.json`.*
