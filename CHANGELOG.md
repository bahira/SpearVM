# Changelog

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
