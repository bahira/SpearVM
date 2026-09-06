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
Le §6 propose l'index hiérarchique manquant et le §7 mesure l'architecture
complète une fois corrigée : **×10.6** sur le listing d'origine, à qualité
identique.

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
| QSA / IndexPool | N^1.15 (4 k → 16 k, régime où le budget mord) | **0.91 h par couche**, ×12 couches |
| Gated DeltaNet | 1 385 µs/token (linéaire ✓) | **13.3 h pour ses 36 couches** |

Soit **≈ 24 h pour une seule passe avant** à 1 M tokens. Les lignes 262 k et 1 M
du tableau 4.1 sont donc des **sorties de modèle analytique**, pas des mesures.
Un modèle analytique est parfaitement légitime — il faut simplement l'annoncer
comme tel. Présenté comme un benchmark exécuté, il invalide la section
« validation ».

> **Correction d'une erreur de cet audit.** La première version de ce rapport
> annonçait « 78 h pour une seule couche », mesuré à N ≤ 2048 avec un budget de
> 2048 tokens. Dans ce régime le budget **ne mord pas** (N/4 ≤ budget/4) :
> l'attention dite creuse était en fait dense, ce qui gonflait l'exposant à
> N^1.92 et le temps extrapolé d'un facteur ~80. La mesure corrigée n'utilise
> que les tailles où le budget est réellement contraignant. La conclusion tient
> — 24 h par passe reste incompatible avec « benchmarks conduits » — mais le
> chiffre initial était faux, et le signaler fait partie du même contrat que
> celui exigé du mémoire.

Corollaire mesuré et plus gênant, celui-là robuste : le coût du routage QSA est
**quadratique en nombre d'opérations** — N^2.00 exactement, compté et non
chronométré (`results/hier_routing.csv`). Chaque micro-bloc score *tous* les
blocs passés (N/4 produits scalaires) avant d'en retenir 512. Borner le budget
d'attention ne borne pas le coût de la sélection. À 1 M tokens cela représente
3.1 × 10¹⁰ produits scalaires de routage contre 1.2 × 10¹¹ pour l'attention
elle-même : le routage cesse d'être un détail. En temps de paroi jusqu'à 16 k
l'exposant n'est que N^1.15, car le surcoût Python par bloc masque encore le
terme quadratique — raison de plus pour compter les opérations plutôt que de
chronométrer.

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
| 20 | Fidélité du routage IndexPool | non mesurée | 6.3 % de la masse captée à N=8192 (budget 512) | LACUNE |
| 21 | Débit table N-grammes | 11 372 tokens/s | 1.9 M tokens/s en RAM | **FAUX** |
| 22 | Table N-grammes : 51 B / 25.6 GB / 0 FLOP | — | 51.2 B / 25.6 GB / 0 FLOP | **TENU** |
| 23 | Amdahl p=0.88, s=24.5 | 6.41× | 6.414× | **TENU** |
| 24 | Routage QSA linéaire | budget borné | N^2.00 en nombre d'opérations | **FAUX** |
| 25 | Benchmarks 1M exécutés | « prouvent formellement » | ≈ 24 h/passe avec le listing | **FAUX** |

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

* sur les **premiers** blocs (qui tiennent dans le budget) : 84.6 % — métrique
  trompeuse, ils sélectionnent quasiment tout ce qui existe ;
* sur les **derniers** blocs, là où le budget mord réellement : **26.2 %** à
  N = 2048, **12.8 %** à N = 4096, **6.3 %** à N = 8192 — la masse captée
  s'effondre mécaniquement quand le budget devient une fraction décroissante du
  contexte ;
* écart maximal à l'attention dense causale : 8.3 % à N = 1024.

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


## 6. Contre-proposition mesurée : l'index hiérarchique manquant

L'audit reproche au mémoire que borner le budget d'attention ne borne pas le
coût du routage. Reproche facile ; voici l'implémentation qui le corrige et sa
mesure (`experiments/attention/hierarchical.py`, banc `bench_hier.py`).

**Principe.** Un arbre de résumés construit par pooling successif au-dessus des
blocs IndexPool du mémoire (facteur `f` par niveau), parcouru en faisceau du
grossier vers le fin, avec élagage causal à chaque niveau. Coût attendu
`O((N/B) · beam · f · log N)` au lieu de `O((N/B)²)`.

**Première tentative : échec, et le pourquoi est instructif.** Avec un faisceau
égal au budget (512 blocs, le réflexe naturel), le développement `512 × 8 = 4096`
candidats couvre l'index entier tant que `N < 16 k` : aucun élagage, et un
surcoût de 15 %. Mesuré avant d'être corrigé — le faisceau minimal viable est
`⌈k/f⌉`, pas `k`.

**Coût après correction** (comptage exact des produits scalaires d'index,
budget 512 tokens, f = 8) :

| N | routage plat | routage hiérarchique | gain | plat / requête | hiér. / requête |
| --- | --- | --- | --- | --- | --- |
| 1 024 | 32 896 | 31 247 | 1.05× | 128 | 122 |
| 4 096 | 524 800 | 212 658 | 2.47× | 512 | 208 |
| 16 384 | 8 390 656 | 1 233 096 | 6.80× | 2 048 | 301 |
| 32 768 | 33 558 528 | 2 869 031 | **11.70×** | 4 096 | 350 |

Exposants empiriques : **N^2.00 pour le routage plat, N^1.30 pour le
hiérarchique**. Le coût par requête du routeur plat suit exactement N (2 048 à
N = 16 k) ; celui du hiérarchique croît en log (122 → 350 pour un facteur 32 sur
N). Extrapolé à 1 M tokens : `3.11e10` contre `2.64e8` produits scalaires, soit
**118×**. Autrement dit le routage passe de **5.7× le coût de l'attention** à
**0.048×** — il cesse d'être le goulot, ce qui était tout l'enjeu.

**Ce que ça coûte en qualité** (N = 2 048, budget 512 tokens, mesure par tête) :

| routeur | ops | accord avec le top-k exact | masse d'attention captée |
| --- | --- | --- | --- |
| plat (exhaustif) | 131 328 | 100 % | 84.6 % (min 49.8 %) |
| hiér. f=8, faisceau minimal | 81 844 | 67.6 % | **83.9 %** (min 49.2 %) |
| hiér. f=8, faisceau 32 | 174 114 | 90.4 % | 84.6 % (min 49.8 %) |
| hiér. f=16, faisceau minimal | 68 437 | 64.2 % | 83.0 % (min 46.9 %) |
| hiér. f=8, résumé par max | 81 919 | 64.0 % | 83.8 % (min 49.0 %) |

Le résultat important est l'écart entre les deux dernières colonnes : le
faisceau minimal ne retrouve que **67.6 %** des blocs du top-k exact, mais
capte **83.9 %** de la masse d'attention contre 84.6 % — soit **0.7 point de
perte**. Les blocs qu'il rate ne pèsent presque rien. C'est exactement le genre
d'arbitrage que le mémoire aurait dû publier : l'identité du top-k n'est pas la
bonne métrique, la masse l'est.

Le résumé par maximum ne bat pas la moyenne (83.8 % contre 83.9 % pour un coût
identique) : l'intuition « le max borne mieux le score du sous-arbre » ne se
vérifie pas ici. Mesuré, donc abandonné.

**Vérifications** (`bench_hier.py` §4) : causalité stricte respectée à chaque
position, budget jamais dépassé, et — test le plus utile — avec un budget égal
au contexte entier le routeur hiérarchique redonne l'attention dense causale à
**4.05e-08** près. Le chemin rapide et le chemin exact coïncident donc bien.

Reste une limite honnête : ces mesures s'arrêtent à N = 32 768, et le facteur
118× à 1 M est une extrapolation de lois d'échelle mesurées, pas une exécution.
C'est précisément la distinction que ce rapport reproche au mémoire de ne pas
faire — elle vaut aussi pour lui.


## 7. Bout-en-bout : l'architecture du mémoire, écrite correctement

Dernière étape de la boucle : reprendre QSA avec les correctifs de l'audit
(masque causal intra-bloc, routage hiérarchique) et les noyaux du dépôt
(`spur_math.attention_tile` : GEMM NT + softmax masqué en C, voir
[`TRANSCEND.md`](TRANSCEND.md)), puis mesurer — `experiments/attention/bench_qsa_e2e.py`.

| N = 8192, budget 512 | temps | débit | gain | masse captée |
| --- | --- | --- | --- | --- |
| listing du mémoire | 3 908 ms | 2 096 tok/s | ×1.00 | 6.3 % |
| corrigé, attention numpy | 4 082 ms | 2 007 tok/s | ×0.96 | 6.3 % |
| corrigé + noyaux SpearVM | 2 715 ms | 3 018 tok/s | ×1.44 | 6.3 % |
| + routage partagé par 4 blocs | 882 ms | 9 288 tok/s | ×4.43 | 6.3 % |
| **+ routage partagé par 16 blocs** | **368 ms** | **22 251 tok/s** | **×10.62** | **6.3 %** |

Deux enseignements, tous deux mesurés :

**Le routage hiérarchique ne coûte rien en qualité.** La masse d'attention
captée est identique à celle du routage exhaustif (6.3 % contre 6.3 % ; 12.7 %
contre 12.8 % à N = 4096). Ce qui limite la qualité, ce n'est pas le routeur,
c'est le **budget** : à N = 8192 avec 512 tokens, le modèle ne voit que 6 % de
la masse d'attention réelle. Le mémoire ne mesure jamais cette quantité, et
c'est pourtant elle qui décide si l'architecture est utilisable.

*(Cette mesure a d'abord été faite sur les 128 premiers blocs et donnait
« 100 % » : les premiers blocs d'une séquence tiennent entièrement dans le
budget, la métrique y est vide de sens. Elle est désormais prise sur les
derniers blocs.)*

**Partager une décision de routage entre blocs de requêtes voisins est
gratuit.** Seize micro-blocs consécutifs (64 tokens) attendent presque le même
contexte : router une fois pour les seize ne change pas la masse captée
(6.3 %) et fait passer la tuile de 4 à 64 lignes — le régime où le noyau C
atteint 94 à 123 GFLOPS au lieu de 12. D'où le ×10.6 total.

Aucune extrapolation à 1 M n'est donnée ici : le débit *monte* avec N sur ces
tailles (le surcoût Python par tuile s'amortit), un ajustement en loi de
puissance donnerait un exposant < 1, ce qui n'a pas de sens. La loi d'échelle
rigoureuse reste celle du comptage d'opérations (§6), pas celle du temps de
paroi à ces tailles.

---

*Toutes les valeurs de ce rapport proviennent de `experiments/attention/audit.py`
et `experiments/attention/bench_hier.py`.
Journaux : `experiments/results/attention_audit_log.txt`,
`experiments/results/hier_routing_log.txt`. Données machine :
`attention_audit.json`, `hier_routing.csv` / `.json`.*
