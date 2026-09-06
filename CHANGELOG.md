# Changelog

## non publie
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
