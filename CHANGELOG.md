# Changelog

## Unreleased
- **Build** : `make all` et `make test` fonctionnent sous Linux (le JIT utilisait `src/spur.c`
  Win64 et ne compilait pas ; on prend maintenant `src/spur_posix.c` -> `bin/spur.so`).
  `tests/test_encoding.c` : encodages verifies partout, smoke d'execution Win64 seulement.
  `make clean` ne supprime plus le fichier suivi `map_dbg.bin`.
- **GEMM f32/f64 AVX-512 en micro-noyau 8x32 / 8x16** avec packing de B et de A ; f32 ~146 GFLOPS
  (AVX2 : ~85), f64 ~97 GFLOPS (AVX2 : ~41) sur 512^3, VM 2 coeurs.
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
