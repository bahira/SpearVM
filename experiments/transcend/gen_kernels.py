"""Generation du C : exp AVX2 minimax + trois strategies de softmax.

Les coefficients viennent de `fit_exp.py` (Remez en erreur relative) ; le degre
n'est pas cable, on peut en emettre plusieurs et les mesurer.

Trois strategies de softmax par ligne sont emises pour etre comparees :
  * `3pass`   : max, puis exp+somme, puis mise a l'echelle (le schema naif) ;
  * `online`  : max et somme calcules en une seule lecture avec rescale
                incremental (schema flash-attention), puis une passe d'ecriture ;
  * `masked`  : variante causale ou chaque ligne i n'a que `len[i]` entrees
                valides — c'est la forme dont l'attention a reellement besoin.
"""

from __future__ import annotations

HEADER = r"""
/* genere par experiments/transcend/gen_kernels.py */
#include <immintrin.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

/* ---- exp(x) AVX2, reduction x = k*ln2 + r puis minimax sur |r| <= ln2/2 ---- */
static inline __m256 spur_exp8_ps(__m256 x) {
    const __m256 LOG2E   = _mm256_set1_ps(1.4426950408889634f);
    const __m256 LN2_HI  = _mm256_set1_ps(0.693359375f);
    const __m256 LN2_LO  = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 LOWER   = _mm256_set1_ps(-87.33654f);   /* exp -> plus petit normal */
    x = _mm256_max_ps(x, LOWER);
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, LOG2E),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(k, LN2_HI, x);
    r = _mm256_fnmadd_ps(k, LN2_LO, r);
%(POLY_PS)s
    __m256i ki = _mm256_cvtps_epi32(k);
    __m256i pw = _mm256_slli_epi32(_mm256_add_epi32(ki, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pw));
}

static inline __m256d spur_exp4_pd(__m256d x) {
    const __m256d LOG2E  = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2_HI = _mm256_set1_pd(0.693145751953125);
    const __m256d LN2_LO = _mm256_set1_pd(1.42860682030941723212e-6);
    const __m256d LOWER  = _mm256_set1_pd(-708.396418);
    x = _mm256_max_pd(x, LOWER);
    __m256d k = _mm256_round_pd(_mm256_mul_pd(x, LOG2E),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, LN2_HI, x);
    r = _mm256_fnmadd_pd(k, LN2_LO, r);
%(POLY_PD)s
    /* 2^k : (k + 1023) << 52 en entiers 64 bits */
    __m128i ki32 = _mm256_cvtpd_epi32(k);
    __m256i ki64 = _mm256_cvtepi32_epi64(ki32);
    __m256i pw = _mm256_slli_epi64(_mm256_add_epi64(ki64, _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(pw));
}

void spur_exp_f32(const float* x, float* y, long long n) {
    long long i = 0;
    for (; i + 7 < n; i += 8) _mm256_storeu_ps(y + i, spur_exp8_ps(_mm256_loadu_ps(x + i)));
    for (; i < n; i++) y[i] = expf(x[i]);
}

void spur_exp_f64(const double* x, double* y, long long n) {
    long long i = 0;
    for (; i + 3 < n; i += 4) _mm256_storeu_pd(y + i, spur_exp4_pd(_mm256_loadu_pd(x + i)));
    for (; i < n; i++) y[i] = exp(x[i]);
}

/* ---- reductions horizontales ---- */
static inline float hmax8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
static inline float hsum8f(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
"""

SOFTMAX_3PASS = r"""
/* ---- softmax par ligne, schema naif en trois passes ---- */
void spur_softmax_3pass_f32(const float* X, float* Y, long long rows, long long cols) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < rows; i++) {
        const float* xr = X + i * cols;
        float* yr = Y + i * cols;
        __m256 vmax = _mm256_set1_ps(-INFINITY);
        long long j = 0;
        for (; j + 7 < cols; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(xr + j));
        float m = (cols >= 8) ? hmax8(vmax) : -INFINITY;
        for (; j < cols; j++) m = xr[j] > m ? xr[j] : m;

        __m256 vm = _mm256_set1_ps(m), vs = _mm256_setzero_ps();
        for (j = 0; j + 7 < cols; j += 8) {
            __m256 e = spur_exp8_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + j), vm));
            _mm256_storeu_ps(yr + j, e);
            vs = _mm256_add_ps(vs, e);
        }
        float s = (cols >= 8) ? hsum8f(vs) : 0.0f;
        for (; j < cols; j++) { yr[j] = expf(xr[j] - m); s += yr[j]; }

        __m256 vinv = _mm256_set1_ps(1.0f / s);
        for (j = 0; j + 7 < cols; j += 8)
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_loadu_ps(yr + j), vinv));
        for (; j < cols; j++) yr[j] *= 1.0f / s;
    }
}
"""

SOFTMAX_ONLINE = r"""
/* ---- softmax "online" : max et somme en une seule lecture de X ----
   A chaque bloc on met a jour (m, s) : si le nouveau maximum depasse l'ancien,
   la somme accumulee est remise a l'echelle par exp(m_ancien - m_nouveau).
   Une seule relecture de X ensuite pour ecrire le resultat normalise.      */
void spur_softmax_online_f32(const float* X, float* Y, long long rows, long long cols) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < rows; i++) {
        const float* xr = X + i * cols;
        float* yr = Y + i * cols;
        __m256 vm = _mm256_set1_ps(-INFINITY), vs = _mm256_setzero_ps();
        long long j = 0;
        for (; j + 7 < cols; j += 8) {
            __m256 v = _mm256_loadu_ps(xr + j);
            __m256 nm = _mm256_max_ps(vm, v);
            vs = _mm256_add_ps(_mm256_mul_ps(vs, spur_exp8_ps(_mm256_sub_ps(vm, nm))),
                               spur_exp8_ps(_mm256_sub_ps(v, nm)));
            vm = nm;
        }
        float m = (cols >= 8) ? hmax8(vm) : -INFINITY;
        for (long long t = (cols / 8) * 8; t < cols; t++) m = xr[t] > m ? xr[t] : m;
        /* fusion des 8 accumulateurs : chacun avait son propre maximum */
        float mv[8], sv[8];
        _mm256_storeu_ps(mv, vm); _mm256_storeu_ps(sv, vs);
        float s = 0.0f;
        if (cols >= 8) for (int t = 0; t < 8; t++) s += sv[t] * expf(mv[t] - m);
        for (long long t = (cols / 8) * 8; t < cols; t++) s += expf(xr[t] - m);

        __m256 vmf = _mm256_set1_ps(m), vinv = _mm256_set1_ps(1.0f / s);
        for (j = 0; j + 7 < cols; j += 8)
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(
                spur_exp8_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + j), vmf)), vinv));
        for (; j < cols; j++) yr[j] = expf(xr[j] - m) / s;
    }
}
"""

SOFTMAX_MASKED = r"""
/* ---- softmax causal : la ligne i n'a que len[i] entrees valides ----
   Forme dont l'attention a besoin : pas de -inf a materialiser, pas de masque
   a lire, la longueur suffit.                                              */
void spur_softmax_masked_f32(const float* X, float* Y, long long rows,
                             long long cols, const int* len) {
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < rows; i++) {
        const long long n = len[i] < cols ? len[i] : cols;
        const float* xr = X + i * cols;
        float* yr = Y + i * cols;
        if (n <= 0) { memset(yr, 0, (size_t)cols * sizeof(float)); continue; }

        __m256 vmax = _mm256_set1_ps(-INFINITY);
        long long j = 0;
        for (; j + 7 < n; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(xr + j));
        float m = (n >= 8) ? hmax8(vmax) : -INFINITY;
        for (; j < n; j++) m = xr[j] > m ? xr[j] : m;

        __m256 vm = _mm256_set1_ps(m), vs = _mm256_setzero_ps();
        for (j = 0; j + 7 < n; j += 8) {
            __m256 e = spur_exp8_ps(_mm256_sub_ps(_mm256_loadu_ps(xr + j), vm));
            _mm256_storeu_ps(yr + j, e);
            vs = _mm256_add_ps(vs, e);
        }
        float s = (n >= 8) ? hsum8f(vs) : 0.0f;
        for (; j < n; j++) { yr[j] = expf(xr[j] - m); s += yr[j]; }

        __m256 vinv = _mm256_set1_ps(1.0f / s);
        for (j = 0; j + 7 < n; j += 8)
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_loadu_ps(yr + j), vinv));
        for (; j < n; j++) yr[j] *= 1.0f / s;
        if (n < cols) memset(yr + n, 0, (size_t)(cols - n) * sizeof(float));
    }
}
"""


def horner(coeffs, var="r", vec="__m256", suffix="ps", indent=4) -> str:
    """Emet un schema de Horner en FMA a partir des coefficients c0..cd."""
    pad = " " * indent
    lines = [f"{pad}{vec} p = _mm256_set1_{suffix}({coeffs[-1]!r}{'f' if suffix == 'ps' else ''});"]
    for c in reversed(coeffs[:-1]):
        lit = f"{c!r}{'f' if suffix == 'ps' else ''}"
        lines.append(f"{pad}p = _mm256_fmadd_{suffix}(p, {var}, _mm256_set1_{suffix}({lit}));")
    return "\n".join(lines)


def render(coeffs_f32, coeffs_f64) -> str:
    src = HEADER % {"POLY_PS": horner(coeffs_f32, vec="__m256", suffix="ps"),
                    "POLY_PD": horner(coeffs_f64, vec="__m256d", suffix="pd")}
    return src + SOFTMAX_3PASS + SOFTMAX_ONLINE + SOFTMAX_MASKED
