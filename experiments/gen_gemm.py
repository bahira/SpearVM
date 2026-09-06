"""Generateur de micro-noyaux GEMM NT parametres (AVX2/FMA).

Le noyau actuel de SpearVM est de type "produit scalaire" : pour chaque C[i,j]
il accumule un vecteur puis fait une **reduction horizontale** (hsum). Cout :
1 hsum (~5 instructions, latence ~10 cycles) pour k/4 FMA, et la ligne de B est
relue pour chaque bloc de lignes de A.

L'alternative standard (BLIS/OpenBLAS) est le micro-noyau **broadcast** :
  * A et B sont d'abord *packes* en panneaux contigus (Ap: kc x MR, Bp: kc x NR),
  * la maille interne charge NR/vec vecteurs de B, diffuse MR scalaires de A et
    accumule MR x NR/vec registres C -> **zero reduction horizontale**, chaque
    FMA sert MR fois la meme donnee B.

Ce module emet le code C d'une variante (MR, NR, KC, MC, NC, dtype, prefetch,
deroulage) pour que l'autotuner puisse en compiler et en mesurer des centaines.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    dtype: str = "f64"      # "f64" | "f32"
    mr: int = 4             # lignes du micro-noyau
    nr: int = 12            # colonnes du micro-noyau (multiple de la largeur SIMD)
    kc: int = 256           # profondeur de tuile (panneaux en L1/L2)
    mc: int = 256           # hauteur de tuile A
    nc: int = 1024          # largeur de tuile B
    prefetch: int = 0       # distance de prefetch sur C (0 = aucun)
    unroll: int = 1         # deroulage de la boucle k du micro-noyau
    omp: bool = False       # variante multi-thread

    @property
    def lanes(self) -> int:
        return 4 if self.dtype == "f64" else 8

    @property
    def name(self) -> str:
        return (f"gemm_{self.dtype}_mr{self.mr}_nr{self.nr}_kc{self.kc}"
                f"_mc{self.mc}_nc{self.nc}_pf{self.prefetch}_u{self.unroll}"
                f"{'_omp' if self.omp else ''}")

    def valid(self) -> bool:
        if self.nr % self.lanes:
            return False
        vregs = self.mr * (self.nr // self.lanes) + (self.nr // self.lanes) + 1
        return vregs <= 16 and self.mr >= 1 and self.nr >= self.lanes


TEMPLATE = r"""
/* genere par experiments/gen_gemm.py — variante {name} */
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
{omp_include}

#define MR {mr}
#define NR {nr}
#define KC {kc}
#define MC {mc}
#define NC {nc}
#define NV ({nr} / {lanes})     /* vecteurs par ligne du micro-noyau */
/* tailles de panneaux arrondies au multiple du micro-noyau : le packing ecrit
   toujours des blocs pleins MR/NR (zero-padding des bords).                  */
#define MCP (((MC) + MR - 1) / MR * MR)
#define NCP (((NC) + NR - 1) / NR * NR)

typedef {ctype} T;
typedef {vtype} V;
#define VLOAD  {vload}
#define VSTORE {vstore}
#define VADD   {vadd}
#define VZERO  {vzero}
#define VFMA   {vfma}
#define VBCAST {vbcast}

/* --- packing ------------------------------------------------------------
   A (m,k) row-major        -> Ap : panneaux [MR x kc] (q majeur, r mineur)
   B (n,k) row-major (NT !) -> Bp : panneaux [NR x kc] (q majeur, j mineur)
   Le packing de B transpose donc B en (k,n) par blocs : c'est lui qui rend le
   micro-noyau broadcast possible sur la convention NT.                      */
static void pack_a(const T *restrict A, long long lda, T *restrict Ap,
                   long long mc, long long kc) {{
    for (long long i = 0; i < mc; i += MR) {{
        const long long rows = (mc - i < MR) ? (mc - i) : MR;
        T *dst = Ap + i * kc;
        for (long long q = 0; q < kc; q++) {{
            for (long long r = 0; r < rows; r++) dst[q * MR + r] = A[(i + r) * lda + q];
            for (long long r = rows; r < MR; r++) dst[q * MR + r] = (T)0;
        }}
    }}
}}

static void pack_b(const T *restrict B, long long ldb, T *restrict Bp,
                   long long nc, long long kc) {{
    for (long long j = 0; j < nc; j += NR) {{
        const long long cols = (nc - j < NR) ? (nc - j) : NR;
        T *dst = Bp + j * kc;
        for (long long q = 0; q < kc; q++) {{
            for (long long c = 0; c < cols; c++) dst[q * NR + c] = B[(j + c) * ldb + q];
            for (long long c = cols; c < NR; c++) dst[q * NR + c] = (T)0;
        }}
    }}
}}

/* --- micro-noyau : C[MR x NR] += Ap[MR x kc] . Bp[kc x NR] --------------- */
static inline void micro(long long kc, const T *restrict Ap, const T *restrict Bp,
                         T *restrict C, long long ldc, int accumulate) {{
    V c[MR][NV];
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NV; j++) c[i][j] = VZERO();

    long long q = 0;
{kloop}
    for (; q < kc; q++) {{
        V b[NV];
        for (int j = 0; j < NV; j++) b[j] = VLOAD(Bp + q * NR + j * {lanes});
        for (int i = 0; i < MR; i++) {{
            V a = VBCAST(Ap + q * MR + i);
            for (int j = 0; j < NV; j++) c[i][j] = VFMA(a, b[j], c[i][j]);
        }}
    }}

    if (accumulate) {{
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < NV; j++)
                VSTORE(C + i * ldc + j * {lanes},
                       VADD(VLOAD(C + i * ldc + j * {lanes}), c[i][j]));
    }} else {{
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < NV; j++) VSTORE(C + i * ldc + j * {lanes}, c[i][j]);
    }}
}}

/* --- pilote : C = A . B^T ------------------------------------------------ */
void {name}(const T *A, const T *B, T *C, long long m, long long k, long long n) {{
    T *Bp = NULL, *Ap0 = NULL;
    if (posix_memalign((void **)&Bp, 64, sizeof(T) * (size_t)KC * NCP)) return;
    if (posix_memalign((void **)&Ap0, 64, sizeof(T) * (size_t)KC * MCP)) {{ free(Bp); return; }}

    for (long long jc = 0; jc < n; jc += NC) {{
        const long long nc = (n - jc < NC) ? (n - jc) : NC;
        for (long long pc = 0; pc < k; pc += KC) {{
            const long long kc = (k - pc < KC) ? (k - pc) : KC;
            const int accumulate = (pc != 0);
            pack_b(B + jc * k + pc, k, Bp, nc, kc);
{body}
        }}
    }}
    free(Ap0);
    free(Bp);
}}
"""

BODY_ST = r"""            for (long long ic = 0; ic < m; ic += MC) {{
                const long long mc = (m - ic < MC) ? (m - ic) : MC;
                T *Ap = Ap0;
                pack_a(A + ic * k + pc, k, Ap, mc, kc);
                for (long long jr = 0; jr < nc; jr += NR) {{
                    const long long cols = (nc - jr < NR) ? (nc - jr) : NR;
                    for (long long ir = 0; ir < mc; ir += MR) {{
                        const long long rows = (mc - ir < MR) ? (mc - ir) : MR;
                        T *cptr = C + (ic + ir) * n + jc + jr;
{prefetch}
                        if (rows == MR && cols == NR) {{
                            micro(kc, Ap + ir * kc, Bp + jr * kc, cptr, n, accumulate);
                        }} else {{
                            T tmp[MR * NR] __attribute__((aligned(64)));
                            micro(kc, Ap + ir * kc, Bp + jr * kc, tmp, NR, 0);
                            for (long long r = 0; r < rows; r++)
                                for (long long c = 0; c < cols; c++)
                                    cptr[r * n + c] = accumulate ? cptr[r * n + c] + tmp[r * NR + c]
                                                                 : tmp[r * NR + c];
                        }}
                    }}
                }}
            }}"""

BODY_MT = r"""            #pragma omp parallel
            {{
                T *Ap = NULL;
                if (posix_memalign((void **)&Ap, 64, sizeof(T) * (size_t)KC * MCP) == 0) {{
                #pragma omp for schedule(static)
                for (long long ic = 0; ic < m; ic += MC) {{
                    const long long mc = (m - ic < MC) ? (m - ic) : MC;
                    pack_a(A + ic * k + pc, k, Ap, mc, kc);
                    for (long long jr = 0; jr < nc; jr += NR) {{
                        const long long cols = (nc - jr < NR) ? (nc - jr) : NR;
                        for (long long ir = 0; ir < mc; ir += MR) {{
                            const long long rows = (mc - ir < MR) ? (mc - ir) : MR;
                            T *cptr = C + (ic + ir) * n + jc + jr;
                            if (rows == MR && cols == NR) {{
                                micro(kc, Ap + ir * kc, Bp + jr * kc, cptr, n, accumulate);
                            }} else {{
                                T tmp[MR * NR] __attribute__((aligned(64)));
                                micro(kc, Ap + ir * kc, Bp + jr * kc, tmp, NR, 0);
                                for (long long r = 0; r < rows; r++)
                                    for (long long c = 0; c < cols; c++)
                                        cptr[r * n + c] = accumulate ? cptr[r * n + c] + tmp[r * NR + c]
                                                                     : tmp[r * NR + c];
                            }}
                        }}
                    }}
                }}
                free(Ap);
                }}
            }}"""


def _kloop(v: Variant) -> str:
    """Deroulage explicite de la boucle k du micro-noyau."""
    if v.unroll <= 1:
        return ""
    lines = [f"    for (; q + {v.unroll - 1} < kc; q += {v.unroll}) {{"]
    lines.append(f"        V b[{v.unroll}][NV];")
    for u in range(v.unroll):
        lines.append(f"        for (int j = 0; j < NV; j++) b[{u}][j] = VLOAD(Bp + (q + {u}) * NR + j * {v.lanes});")
    for u in range(v.unroll):
        lines.append(f"        for (int i = 0; i < MR; i++) {{")
        lines.append(f"            V a = VBCAST(Ap + (q + {u}) * MR + i);")
        lines.append(f"            for (int j = 0; j < NV; j++) c[i][j] = VFMA(a, b[{u}][j], c[i][j]);")
        lines.append(f"        }}")
    lines.append("    }")
    return "\n".join(lines)


def render(v: Variant) -> str:
    f64 = v.dtype == "f64"
    prefetch = ""
    if v.prefetch:
        prefetch = (f"                        __builtin_prefetch(cptr + {v.prefetch} * n, 1, 1);")
    body = (BODY_MT if v.omp else BODY_ST).format(prefetch=prefetch)
    return TEMPLATE.format(
        name=v.name,
        omp_include="#include <omp.h>" if v.omp else "",
        mr=v.mr, nr=v.nr, kc=v.kc, mc=v.mc, nc=v.nc,
        lanes=v.lanes,
        ctype="double" if f64 else "float",
        vtype="__m256d" if f64 else "__m256",
        vload="_mm256_loadu_pd" if f64 else "_mm256_loadu_ps",
        vstore="_mm256_storeu_pd" if f64 else "_mm256_storeu_ps",
        vadd="_mm256_add_pd" if f64 else "_mm256_add_ps",
        vzero="_mm256_setzero_pd" if f64 else "_mm256_setzero_ps",
        vfma="_mm256_fmadd_pd" if f64 else "_mm256_fmadd_ps",
        vbcast="_mm256_broadcast_sd" if f64 else "_mm256_broadcast_ss",
        kloop=_kloop(v),
        body=body,
    )
