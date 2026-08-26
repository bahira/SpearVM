# SpearVM

VM registres x64 avec **coprocesseur math approximatif** : les kernels sont
compilés en code machine natif à la volée (JIT), et les opérations
transcendantales (`gelu`, `erf`, `tanh`, `lse2`) sont remplacées par des
**noyaux polynomiaux certifiés SPEAR** vectorisables SIMD.

## Résultats mesurés (bench autonome, workload 500k iters)

| Scénario | Checksum | Temps |
|---|---|---|
| Programme natif C (libm IEEE) | `+3632654.237871` | 0.12 ms* |
| Interpréteur SPUR (maths exactes) | `+3632654.237871` — **bit-exact natif** | 1.80 ms |
| Interpréteur + coprocesseur SPEAR | `+2064999.853815` | 1.00 ms |
| **Gain coprocesseur SPEAR vs exact** | | **×1.80** |

\* Horloge virtualisée — voir bench_turbo.c pour les timings AVX2 réels.

### Benchmark gelu AVX2 pur (4M éléments, horloge hôte)

| | Temps | Débit |
|---|---|---|
| gelu scalaire | 3.20 ms | 1.31 Gelem/s |
| **gelu AVX2 vectorisé** | **0.25 ms** | **16.78 Gelem/s** |

### Contrat de précision (datasheet, vérifié au lancement)

| Noyau | Err max vs IEEE | Domaine garanti |
|---|---|---|
| gelu  | ≤ 0.079 | y ∈ [-2, 2] |
| erf   | ≤ 0.011 | x ∈ [-2, 2] |
| tanh  | ≤ 0.008 | x ∈ [-3, 3] |
| lse2  | ≤ ln 2  | toujours (hard-max) |

## Build & Run

```
make                # compile DLL + tests
make bench          # bench autonome natif-vs-JIT (recommandé)
```

Ou manuellement :
```
gcc -O3 -I include examples/bench_native_vs_jit.c -o bench -lm
./bench
```

## Architecture

```
include/spur.h       API publique (opcodes, struct SpurIns, prototypes)
src/spur.c           noyaux scalaires + émetteur x64 + API complète
examples/
  bench_native_vs_jit.c   bench autonome auto-validant ← POINT D'ENTRÉE
  test_spur.c             validation kernels boucle + map via DLL
experimental/        map element-wise multi-entrées (en cours de débogage)
examples/python/     bindings ctypes
```

L'émetteur génère du x64 SSE2 : prologue/épilogue conformes Win64 ABI,
vregs dans `xmm6..13`, accumulateur `xmm14`, zéro `xmm15`. Les opérations
transcendantales sont émises inline (séquences mulsd/addsd/maxsd/minsd/divsd
sur constantes du pool) — zéro appel libm.

### Ajouter un nouveau noyau copro

1. Trouver une forme polynomiale/rationnelle bornée (pipeline SPEAR : GP
   contraint sans transcendantales, ou identité exacte via tanh comme
   `sigmoid(x)=½(1+tanh(x/2))`).
2. Certifier : err max vs IEEE sur le domaine, documenter dans le tableau.
3. Ajouter un case dans `emit_op()`.
4. Ajouter le opcode dans l'enum `spur.h`.

## Limitations (honnêtes)

- Windows x64 uniquement (émetteur Win64 ABI).
- Kernel map element-wise : en débogage actif (voir experimental/).
- Précision approximative : NE PAS utiliser pour du financier/légal.
- La VM est plus lente que le natif pour du code non-transcendental.

## Origine des noyaux

Noyaux découverts par GP contraint (pipeline SPEAR) ou dérivés d'identités
exactes sur des noyaux certifiés. Chaque noyau documente son erreur max
mesurée — c'est le contrat, pas une promesse.
