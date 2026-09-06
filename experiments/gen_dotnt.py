"""Famille 2 — micro-noyau "dot-block" NT, **sans packing**.

Intuition : en convention NT (`C = A . B^T`, A(m,k) et B(n,k) row-major), la
dimension k est contigue *des deux cotes*. Un GEMM classique doit transposer et
packer B pour obtenir un micro-noyau broadcast ; ici on peut au contraire
garder l'acces naturel et bloquer **MR lignes de A x NR lignes de B** :

    pour q par pas de VLEN :
        av[i] = load(A[i] + q)        i < MR      (contigu)
        pour j < NR :
            bv    = load(B[j] + q)                (contigu)
            acc[i][j] = fma(av[i], bv, acc[i][j])
    C[i][j] = hsum(acc[i][j])                     (une seule fois, hors boucle k)

Cout des reductions horizontales : MR*NR hsum pour MR*NR*k/VLEN FMA — negligeable
des que k >= 64. Gains attendus vs le noyau actuel de SpearVM (1 x 4 sans reuse
de B) : la ligne de B chargee sert MR fois, la ligne de A sert NR fois, donc
MR*NR FMA pour MR+NR chargements (contre 4 FMA pour 5 chargements aujourd'hui).
Et zero octet copie, contre O(mk + nk) pour le packing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DotVariant:
    dtype: str = "f64"
    mr: int = 3          # lignes de A par tuile
    nr: int = 3          # lignes de B par tuile
    kc: int = 0          # 0 = pas de blocage en k
    nc: int = 64         # lignes de B gardees chaudes
    mc: int = 0          # 0 = pas de blocage en m
    hoist_b: bool = False  # garder les NR vecteurs de B vivants (plus de registres)
    prefetch: int = 0

    @property
    def lanes(self) -> int:
        return 4 if self.dtype == "f64" else 8

    @property
    def name(self) -> str:
        return (f"dotnt_{self.dtype}_mr{self.mr}_nr{self.nr}_kc{self.kc}"
                f"_nc{self.nc}_mc{self.mc}_h{int(self.hoist_b)}_pf{self.prefetch}")

    def valid(self) -> bool:
        regs = self.mr * self.nr + self.mr + (self.nr if self.hoist_b else 1)
        return 1 <= self.mr <= 8 and 1 <= self.nr <= 8 and regs <= 16


TEMPLATE = r"""
/* genere par experiments/gen_dotnt.py — variante {name} */
#include <immintrin.h>
#include <string.h>

#define MR {mr}
#define NR {nr}
#define VL {lanes}

typedef {ctype} T;
typedef {vtype} V;
#define VLOAD  {vload}
#define VZERO  {vzero}
#define VFMA   {vfma}

static inline T hsum(V v) {{
{hsum_body}
}}

/* C[MR x NR] (+)= A[MR x kc] . B[NR x kc]^T, lignes contigues des deux cotes */
static inline void micro(const T *restrict A, long long lda,
                         const T *restrict B, long long ldb,
                         T *restrict C, long long ldc,
                         long long kc, int accumulate) {{
    V acc[MR][NR];
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++) acc[i][j] = VZERO();

    long long q = 0;
    for (; q + VL - 1 < kc; q += VL) {{
        V av[MR];
        for (int i = 0; i < MR; i++) av[i] = VLOAD(A + i * lda + q);
{inner}
    }}

    T tail[MR][NR];
    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++) tail[i][j] = (T)0;
    for (long long t = q; t < kc; t++)
        for (int i = 0; i < MR; i++) {{
            const T a = A[i * lda + t];
            for (int j = 0; j < NR; j++) tail[i][j] += a * B[j * ldb + t];
        }}

    for (int i = 0; i < MR; i++)
        for (int j = 0; j < NR; j++) {{
            const T s = hsum(acc[i][j]) + tail[i][j];
            C[i * ldc + j] = accumulate ? C[i * ldc + j] + s : s;
        }}
}}

void {name}(const T *A, const T *B, T *C, long long m, long long k, long long n) {{
    const long long KC = {kc} ? {kc} : k;
    const long long NC = {nc} ? {nc} : n;
    const long long MC = {mc} ? {mc} : m;

    for (long long pc = 0; pc < k; pc += KC) {{
        const long long kc = (k - pc < KC) ? (k - pc) : KC;
        const int accumulate = (pc != 0);
        for (long long ic = 0; ic < m; ic += MC) {{
            const long long mcz = (m - ic < MC) ? (m - ic) : MC;
            for (long long jc = 0; jc < n; jc += NC) {{
                const long long ncz = (n - jc < NC) ? (n - jc) : NC;
                for (long long i = 0; i < mcz; i += MR) {{
                    const long long rows = (mcz - i < MR) ? (mcz - i) : MR;
                    for (long long j = 0; j < ncz; j += NR) {{
                        const long long cols = (ncz - j < NR) ? (ncz - j) : NR;
                        const T *ap = A + (ic + i) * k + pc;
                        const T *bp = B + (jc + j) * k + pc;
                        T *cp = C + (ic + i) * n + jc + j;
{prefetch}
                        if (rows == MR && cols == NR) {{
                            micro(ap, k, bp, k, cp, n, kc, accumulate);
                        }} else {{
                            /* bord : buffers pleins zero-remplis, recopie partielle */
                            T ab[MR * {maxk}] __attribute__((aligned(64)));
                            T bb[NR * {maxk}] __attribute__((aligned(64)));
                            T cb[MR * NR] __attribute__((aligned(64)));
                            if (kc <= {maxk}) {{
                                memset(ab, 0, sizeof(T) * MR * kc);
                                memset(bb, 0, sizeof(T) * NR * kc);
                                for (long long r = 0; r < rows; r++)
                                    memcpy(ab + r * kc, ap + r * k, sizeof(T) * kc);
                                for (long long c = 0; c < cols; c++)
                                    memcpy(bb + c * kc, bp + c * k, sizeof(T) * kc);
                                micro(ab, kc, bb, kc, cb, NR, kc, 0);
                                for (long long r = 0; r < rows; r++)
                                    for (long long c = 0; c < cols; c++)
                                        cp[r * n + c] = accumulate ? cp[r * n + c] + cb[r * NR + c]
                                                                   : cb[r * NR + c];
                            }} else {{
                                for (long long r = 0; r < rows; r++)
                                    for (long long c = 0; c < cols; c++) {{
                                        T s = 0;
                                        for (long long t = 0; t < kc; t++)
                                            s += ap[r * k + t] * bp[c * k + t];
                                        cp[r * n + c] = accumulate ? cp[r * n + c] + s : s;
                                    }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
}}
"""

HSUM_F64 = """    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));"""

HSUM_F32 = """    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x55));
    return _mm_cvtss_f32(lo);"""


def _inner(v: DotVariant) -> str:
    if v.hoist_b:
        lines = ["        V bv[NR];",
                 "        for (int j = 0; j < NR; j++) bv[j] = VLOAD(B + j * ldb + q);",
                 "        for (int i = 0; i < MR; i++)",
                 "            for (int j = 0; j < NR; j++) acc[i][j] = VFMA(av[i], bv[j], acc[i][j]);"]
    else:
        lines = ["        for (int j = 0; j < NR; j++) {",
                 "            const V bv = VLOAD(B + j * ldb + q);",
                 "            for (int i = 0; i < MR; i++) acc[i][j] = VFMA(av[i], bv, acc[i][j]);",
                 "        }"]
    return "\n".join(lines)


def render(v: DotVariant) -> str:
    f64 = v.dtype == "f64"
    prefetch = ""
    if v.prefetch:
        prefetch = ("                        __builtin_prefetch(ap + " + str(v.prefetch) + " * k, 0, 3);\n"
                    "                        __builtin_prefetch(bp + " + str(v.prefetch) + " * k, 0, 3);")
    return TEMPLATE.format(
        name=v.name, mr=v.mr, nr=v.nr, lanes=v.lanes,
        kc=v.kc, nc=v.nc, mc=v.mc,
        maxk=1024,
        ctype="double" if f64 else "float",
        vtype="__m256d" if f64 else "__m256",
        vload="_mm256_loadu_pd" if f64 else "_mm256_loadu_ps",
        vzero="_mm256_setzero_pd" if f64 else "_mm256_setzero_ps",
        vfma="_mm256_fmadd_pd" if f64 else "_mm256_fmadd_ps",
        hsum_body=HSUM_F64 if f64 else HSUM_F32,
        inner=_inner(v),
        prefetch=prefetch,
    )
