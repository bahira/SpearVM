# SpearVM

VM registres x64 avec **coprocesseur math approximatif** : les kernels sont
compilés en code machine natif à la volée (JIT), et les opérations
transcendantales (`gelu`, `erf`, `tanh`, `lse2`, `sigmoid`) sont remplacées par
des **noyaux polynomiaux certifiés SPEAR** ~5× moins chers que libm.

## Speedups réels mesurés (horloge hôte, workload 500k iters dense en
transcendantales, même trajectoire FP)

| Scénario | Temps mur | vs natif |
|---|---|---|
| Programme natif C (libm IEEE) | 1673–2794 ms | ×1 |
| Interpréteur SPUR (maths exactes libm) | 1673–2203 ms | ×0.75–1 |
| Interpréteur + coprocesseur SPEAR | 549–621 ms | **×3.0–4.5** |

Le gain vient exclusivement du coprocesseur approximatif : chaque
transcendantale coûte ~5 ops SSE au lieu d'un appel libm (~15-40 cycles).

### Contrat de précision (datasheet, vérifié au lancement)

| Noyau | Err max vs IEEE | Domaine garanti |
|---|---|---|
| gelu  | ≤ 0.090 | y ∈ [-2, 2] |
| erf   | ≤ 0.012 | x ∈ [-2, 2] |
| tanh  | ≤ 0.012 | x ∈ [-3, 3] |
| lse2  | ≤ ln 2  | toujours (hard-max) |

C'est le compromis GPU/NPU : on troque la précision bit-à-bit contre de la
vitesse, là où le budget d'erreur est acceptable (inférence ML, audio,
simulation Monte-Carlo).

## Build

```
make          # bin/spur.dll + bin/test_spur.exe
make bench    # bench natif vs JIT (autonome, sans DLL)
```

## Usage C

```c
#include "spur.h"

SpurIns prog[] = {
    {MOVI,0,0,7,500000.0},   /* compteur */
    {MOVI,0,0,0,-1.0},       /* x = -1   */
    {ADDI,0,0,0,0.000008},   /* x += pas */
    {GELU,0,0,1,0.6},        /* g = gelu(0.6x) */
    {SIGMOID,1,2,2,0},       /* e = sigmoid(x) */
    ...
    {ACC,3,0,0,0},           /* acc += v3 */
    {SUBI,7,7,7,1.0},
    {BNZ,7,0,0,2},           /* boucle vers pc=2 */
};
int h = spur_jit_build(prog, sizeof(prog)/sizeof(prog[0]));
double acc = spur_exec(h);
```

## Usage Python (ctypes)

```python
import spur   # examples/python/spur.py — voir ce fichier pour la syntaxe
```

## Architecture

```
include/spur.h      API publique (opcodes, struct SpurIns, prototypes)
src/spur.c          noyaux scalaires + émetteur x64 + API
examples/           démos C et Python
experimental/       map element-wise multi-entrées (en cours de débogage)
```

L'émetteur génère du x64 SSE2 : prologue/épilogue conformes Win64 (shadow
space 32, pile alignée 16), vregs dans `xmm6..13`, accumulateur `xmm14`,
zéro `xmm15`. Les opérations transcendantales sont émises inline (séquences
mulsd/addsd/maxsd/minsd/divsd sur constantes du pool) — zéro appel libm.

### Ajouter un nouveau noyau copro

1. Trouver une forme polynomiale/rationnelle bornée (pipeline SPEAR : GP
   contraint sans transcendantales, ou identité exacte via tanh comme
   `sigmoid(x)=½(1+tanh(x/2))`).
2. Certifier : err max vs IEEE sur le domaine, documenter dans le tableau
   datasheet ci-dessus.
3. Ajouter un case dans `emit_op()` qui émet la séquence SSE (constantes
   via `const_slot()`, calculs via `e_sd()`).
4. Ajouter le opcode dans l'enum `spur.h`.

## Limitations (honnêtes)

- Windows x64 uniquement (émetteur Win64 ABI).
- Kernels map element-wise : en cours de débogage (voir experimental/).
- Précision approximative : NE PAS utiliser pour du financier/légal.
- La VM est plus lente que le natif pour du code non-transcendental.

## Origine des noyaux

Noyaux découverts par GP contraint (SPEAR, arXiv:2510.21861 pipeline) ou
dérivés d'identités exactes sur des noyaux certifiés. Chaque noyau documente
son erreur max mesurée — c'est le contrat, pas une promesse.
