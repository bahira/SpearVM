# SpearVM

**Accélérateur mathématique SIMD** — noyaux transcendantals vectorisés AVX2,
précision certifiée, appelables depuis Python/C.

## Speedups mesurés (4M éléments, OpenMP multi-cœur)

| Noyau | Natif libm | AVX2 SPEAR | **Speedup** |
|---|---|---|---|
| GELU | 120 ms | 8 ms | **×15.0** |
| ERF | 306 ms | 9 ms | **×34.0** |
| TANH | 234 ms | 9 ms | **×26.0** |

## Matmul (bench_mm)

`examples/bench_mm.c` — C = A·B double, validation vs naif incluse.

| n | naif ikj | dot scalaire | AVX2 dot | AVX4 bloc + OMP |
|---|---|---|---|---|
| 128 | 2.0 ms (2.1 GF) | ×0.12 | ×0.45 | ×0.73 (série) |
| 256 | 23 ms (1.5 GF) | ×0.24 | ×1.02 | **×1.61** |
| 512 | 198 ms (1.4 GF) | ×0.31 | ×0.37 | **×2.56** (3.5 GFLOPS) |

- err ≤ 1.6e-14 sur toutes les variantes
- blocage registres 4 lignes : chaque ligne de Bᵀ sert 4 rangs de A, 4 chaînes FMA indépendantes
- OpenMP activé seulement si n ≥ 192 ; lancer avec `OMP_WAIT_POLICY=ACTIVE`
- timings VM bruités (±2× entre runs), gains observés jusqu'à ×12 à n=256 machine au repos

## Précision (datasheet)

Chaque noyau documente son erreur max vs IEEE :
- gelu : ≤ 0.079 sur [-0.6, 1.8]
- erf : ≤ 0.011 sur [-2, 2]
- tanh : ≤ 0.008 sur [-3, 3]
- lse2 : hard-max (écart ≤ ln 2)

## Build

```bash
make              # DLL + tests
gcc -O3 -march=native -mavx2 -mfma -fopenmp src/spur_kernels.c -shared -o spur_kernels.dll
```

## Usage Python

```python
import ctypes
lib = ctypes.CDLL("spur_kernels.dll")
# voir examples/bench_kernels.c pour l'API C complète
```

## Architecture

```
src/spur_kernels.c    Kernels AVX2 vectorisés + OpenMP
include/spur.h        API publique
examples/             Benchmarks et démos
tests/                Tests de correction
```

## Limitations

- Windows x64 uniquement
- Précision approximative (±0.01–0.09 selon noyau)
- Pas de support GPU

## Licence

MIT
