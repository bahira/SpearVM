# Changelog

## Unreleased
- **Cas d'usage concrets** (`examples/usecases/`, `docs/USECASES.md`) sur données réelles sklearn, sans téléchargement :
  1) entraînement MLP digits : SpearVM 0,43 s contre PyTorch 0,73 s, même précision (0,9844) ;
  2) diagnostic breast cancer en validation croisée : la régression logistique gagne, le MLP n'apporte rien (documenté) ;
  3) service par lots digits : SpearVM plus rapide que PyTorch à partir du lot 32 (jusqu'à 2,8 M échantillons/s à lot 1024) ;
  4) classification de vins (3 cultivars) : même précision que PyTorch, entraînement 4× plus rapide ; PyTorch mieux calibré.
  Test de fumée des cas d'usage : `tests/test_usecases_smoke.py`.
- **Adam** : tampon de paramètres plat (une dizaine d'opérations numpy par pas au lieu d'environ cinquante). Écart 1e-7.
- **Correctif performance** : `PackedMLP.predict_logits` utilise `matmul_gelu` (épilogue fusionné) au lieu de numpy + `gelu_erf`.
  Lot 1024 : 2,13 ms -> 0,34 ms, même précision. Test de non-régression ajouté.
- **GELU exacte dans le FFN fusionné (champion superspear)** : l'épilogue de `matmul_nt_gelu` (f32/f64,
  AVX-512 et replis) utilise maintenant `erf_v2` (= `gelu_erf`) au lieu de la GELU v1 approximative.
  Erreur contre la loi exacte : 0.08 -> 2.2e-5. Surcoût ~5-7 % sur le GEMM fusionné.
- **Backward cohérent** : `gelu_backward` (f32/f64) est la dérivée analytique exacte de `gelu_erf`
  (écart 1.4e-9 vs différences finies). Avant, il dérivait l'approximation v1 : gradient faux.
- `spur_math.graph` : l'opération `gelu` utilise `gelu_erf`.
- Comparatif des champions superspear : `docs/CHAMPIONS_SUPERSPEAR.md`, `examples/compare_superspear_champions.py`.
  Les fast slots superspear (relu/clip) sont moins précis que les noyaux SpearVM : non retenus comme défaut.
- **Poids pré-packés** : `spur_math.PackedNT(W)` (f32) : packing de B une fois, puis `matmul` / `matmul_gelu`.
  FFN fusionné : 1.26 -> 0.89 ms à M=128. Repli automatique hors AVX-512.
- **Benchmark end-to-end** (`examples/bench_encoder_e2e.py`, `docs/BENCH_ENCODER_E2E.md`) : encodeur 12 couches
  d=384 vs PyTorch CPU et ONNX Runtime. Résultat : SpearVM plus lent (×0.56 à ×0.89) ; écart de précision
  dû entièrement à l'approximation GELU (cos 0.9973, exacte : 1.0).
- **Build** : `make all` et `make test` fonctionnent sous Linux (le JIT utilisait `src/spur.c`
  Win64 et ne compilait pas ; on prend maintenant `src/spur_posix.c` -> `bin/spur.so`).
  `tests/test_encoding.c` : encodages verifies partout, smoke d'execution Win64 seulement.
  `make clean` ne supprime plus le fichier suivi `map_dbg.bin`.
- **GEMM f32/f64 AVX-512 en micro-noyau 8x32 / 8x16** avec packing de B et de A ; f32 ~146 GFLOPS
  (AVX2 : ~85), f64 ~97 GFLOPS (AVX2 : ~41) sur 512^3, VM 2 coeurs.
- **GELU fusionné f64 AVX-512** (`matmul_nt_gelu`, 8x16) : 67 ms vs 131 ms en AVX2 sur 1024×768×3072.
- **Graphe** : rétropropagation `Graph.backward` (matmul NT + biais + gelu), gradcheck
  par differences finies en f64 et entrainement SGD reel (test).
- **fix** `matmul_nt` float32 : résultats faux quand `k % 8 != 0` (le reste de la
  boucle k était ignoré dans le bloc 8 lignes ; erreur ~8 sur `N(0,1)`). Test de régression
  `tests/test_matmul_f32_tail.py`.
- **AVX-512** : micro-noyau GEMM f32 (8×16, B packé par panneaux), dispatch à l'exécution.
  ~100 GFLOPS vs ~80 en AVX2 sur 512³ (VM 2 cœurs). Forcer AVX2 : `SPUR_FORCE_AVX2=1`.
- **AVX-512 fusionné** : `matmul_nt_gelu` f32 (k non tuilé, biais + gelu à l'écriture) :
  42.9 ms vs 178.6 ms en AVX2 sur 1024×768×3072 (×4.2).
- **Runtime de graphe** `spur_math.graph` : `Graph`, passe de fusion `matmul_nt -> gelu`
  vers le noyau fusionné (seulement si le matmul n'a qu'un consommateur).

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
