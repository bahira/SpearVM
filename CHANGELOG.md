# Changelog

## non publie
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
