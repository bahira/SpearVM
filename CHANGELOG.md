# Changelog

## non publie
- **`exp` AVX2 minimax : 1.69 ulp en f32 (libm : 1.66), x1.2 a x1.9** — polynome
  cherche par iterations de Remez en erreur relative, degre choisi par mesure
  (5 en f32, 10 en f64) ; reduction d'argument ln2 scindee hi/lo, 2^k par champ
  d'exposant. `spur_batch_exp[_f32]`, `spur_math.exp`.
- **`softmax` par ligne : x1.9 a x6.2 vs numpy, et x11.8 en causal** — le noyau
  prend un vecteur de **longueurs** au lieu d'un masque -inf materialise, et ne
  parcourt que les entrees valides. Le schema "online" facon flash-attention a
  ete implemente puis abandonne : perdant d'un facteur 1.2 a 4 sur CPU (la
  ligne relue tient en L1). `spur_softmax_rows[_f32]`, `spur_math.softmax`.
- **`attention_tile` : x2.77 median vs numpy, jusqu'a 123 GFLOPS f32** — GEMM NT
  -> softmax masque -> GEMM NT, echelle 1/sqrt(d) absorbee par l'exponentielle
  (aucune passe de plus sur les scores). `spur_attention_tile_f32`,
  `spur_math.attention_tile`. Journal : `docs/TRANSCEND.md`.
- **`web/` — 5e cas d'usage : champ implicite neuronal (SDF)** — un MLP evalue
  par voxel sculpte une surface signee ; le navigateur la ray-marche (sphere
  tracing dans une texture 3D demi-flottante, normales par differences
  centrees, AO). Le serveur garantit et publie la borne de **Lipschitz** que le
  rendu exige. Repli JavaScript inclus, 10 tests dedies (dont un rejeu numpy du
  sphere tracing), rendu de controle sans navigateur (`experiments/preview_sdf.py`).
  Gain des noyaux v2 sur ce cas : **49.9 ms -> 16.7 ms par frame (x3.00)**.
- **`matmul_nt` / `matmul_nt_gelu` acceptent `out=`** et allouent leur sortie
  avec `np.empty` au lieu de `np.zeros` (les noyaux ecrivent toute la matrice,
  verifie par pre-remplissage NaN). Sur une couche (64000, 64) l'allocation
  coutait plus cher que le GEMM : le champ implicite passe de 56 ms a 9 ms par
  frame avec, en plus, une evaluation par paquets de lignes (cache L2).
- **perf `matmul_nt` : x1.78 (f64) / x1.50 (f32) de mediane, sans regression** —
  campagne d'autotuning de 902 variantes compilees/verifiees/chronometrees
  (`experiments/`, journal complet dans `docs/EXPERIMENTS.md`). Deux nouveaux
  micro-noyaux derriere l'API existante : [P] broadcast + packing facon BLIS
  (f64 4x12 KC576, f32 4x24 KC384) et [D] dot-block **sans packing** (3x4) pour
  les formes etroites, avec un aiguillage calibre sous contrainte « aucune
  regression » (pire cas x1.00 sur 20 formes). MC s'adapte au nombre de threads
  (>= 1 bloc par thread). `SPUR_MM_LEGACY=1` restaure l'ancien noyau, toujours
  exporte sous `spur_matmul_nt_legacy` / `spur_matmul_nt_f32_legacy`.
  Rapport a OpenBLAS : 0.36x -> 0.65-0.73x en carre, et **>1** des que k est
  court ou une dimension etroite (512x16x512 : x1.49 vs numpy).
- **perf `matmul_nt_gelu` : x2.23 (f64) / x2.34 (f32) de mediane** — la fusion
  de la gelu dans la boucle j interdisait de couper k, donc interdisait le
  micro-noyau packe ; remplacee par GEMM autotune + epilogue biais+gelu
  vectorise (ecart <= 1.9e-15 / 8.7e-07 avec l'ancien chemin).
- **bout-en-bout** : iteration d'entrainement MLP 784-256-128-10 (batch 128)
  14.3 ms -> 7.5 ms en f64, 5.9 ms -> 4.1 ms en f32 (`experiments/bench_mlp.py`).
- `tests/test_matmul_v2.py` : 324 formes par precision autour des seuils
  d'aiguillage et des blocages KC, C pre-rempli de NaN (detecte toute case non
  ecrite), equivalence avec l'ancien noyau, variantes gelu avec/sans biais.
- **fix f32 `matmul_nt` : queue `k%8` ignoree** pour les blocs de 8 lignes —
  resultats faux des que `k` n'etait pas multiple de 8 et `m >= 8`
  (ex. k=33 : erreur absolue ~3). Queue scalaire ajoutee, valide de k=1 a 257.
- **perf f32 `matmul_nt_gelu` : x2.3 a x2.7** — la ligne de B etait rechargee
  8 fois par bloc ; blocage registres 8 lignes (1 chargement de B, 8 chaines
  FMA) comme dans `matmul_nt_f32`. Sortie bit-a-bit identique.
  (1024x768x1024 : 48.7 ms -> 18.0 ms)
- **`web/` — SpearVM Simulation Lab** : 4 cas d'usage Three.js prets pour la
  production adosses aux noyaux (champ de flux neuronal, membrane non lineaire,
  entrainement live avec backprop, banc d'essai des noyaux). Serveur FastAPI +
  WebSocket binaire (int16 quantifie), client Vite/TypeScript/Three.js,
  repli numpy puis repli JavaScript, Dockerfile, 40 tests, workflow CI dedie.

## 0.5.1
- `dependencies=["numpy"]` declaree (manquait depuis 0.1.0)
- job CI `linux-wheel` : manylinux wheel + auditwheel + smoke test, artifact par run

## 0.5.0
- `matmul_backward(dY,A,B)` : gradients dA/dB du matmul NT (gradcheck <1e-8)
- roadmap README : bf16 et AVX-512 differes avec declencheurs explicites

## 0.4.2
- sdist embarque les sources C ; auto-compilation au premier import sous Linux/macOS x64 (cache `_native/`)
- Makefile reecrit ; badge CI

## 0.4.1
- `add_dll_directory` pour les runtimes MinGW a cote de la DLL

## 0.4.0
- **float32 natif** : matmul_nt(+gelu), gelu, gelu_backward — dispatch dtype auto, ×2 vs f64
- backward erf/tanh/sigmoid (gradcheck ~1.6e-9)
- map kernel 4 entrees (tableau de pointeurs), `spur_free`, SRWLOCK builds, MAXK 32
- wheel platform-tag win_amd64 ; CI ubuntu+windows complete

## 0.3.0
- biais optionnel dans matmul_nt_gelu ; check CPU anti-SIGILL ; JIT dans le package

## 0.2.x
- matmul NT exporte + variante gelu fusionnee ; gelu_backward ; tuilage cache KCxNC

## 0.1.0
- noyaux batch gelu/erf/tanh/lse2 AVX2 certifies ; bindings ctypes ; packaging initial
