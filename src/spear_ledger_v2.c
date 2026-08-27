/* spear_ledger_v2.c — implémentation batch AVX2 des noyaux SPEAR v2.
   Formes rationnelles impaires (fit Lawson quasi-minimax), queues saturées.
   API : spv2_batch_{tanh,erf,gelu,sigmoid}(const double* x, double* out,
                                            long long n);
   Thread-safe, sans allocation, sans appel libm dans la boucle chaude.   */
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include "../include/spear_ledger_v2.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- noyau scalaire commun : rationnel impair saturé -------------------
   R(u) = u * P(y)/D(y), y=u^2 ; saturation +-1 au-dela de |cut|.
   Horner complet sur TOUS les indices (den[0]=1 inclus en fin de chaîne). */
static inline double spv2_rat_odd_v(double u, const double* P, int np,
                                    const double* D, int nd, double cut){
    if(u > cut)  return 1.0;
    if(u < -cut) return -1.0;
    double y = u * u, pn = 0.0, dn = 0.0;
    for(int i = np - 1; i >= 0; i--) pn = fma(pn, y, P[i]);
    for(int i = nd - 1; i >= 0; i--) dn = fma(dn, y, D[i]);
    return u * (pn / dn);
}

double spv2_tanh(double x){
    extern const double SPV2_TANH_N[SPV2_TANH_NNUM];
    extern const double SPV2_TANH_D[SPV2_TANH_NDEN];
    return spv2_rat_odd_v(x, SPV2_TANH_N, SPV2_TANH_NNUM,
                             SPV2_TANH_D, SPV2_TANH_NDEN, 4.0);
}

const double SPV2_TANH_N[SPV2_TANH_NNUM] = SPV2_TANH_NUM;
const double SPV2_TANH_D[SPV2_TANH_NDEN] = SPV2_TANH_DEN;
const double SPV2_ERF_N [SPV2_ERF_NNUM ] = SPV2_ERF_NUM;
const double SPV2_ERF_D [SPV2_ERF_NDEN ] = SPV2_ERF_DEN;

double spv2_erf(double x){
    return spv2_rat_odd_v(x, SPV2_ERF_N, SPV2_ERF_NNUM,
                             SPV2_ERF_D, SPV2_ERF_NDEN, 3.5);
}
double spv2_gelu(double x){
    double t = x * x;
    double u = fma(SPV2_GELU_C3 * t, x, SPV2_GELU_C1 * x);
    double h = spv2_rat_odd_v(u, SPV2_TANH_N, SPV2_TANH_NNUM,
                                 SPV2_TANH_D, SPV2_TANH_NDEN, 4.0);
    return 0.5 * x * (1.0 + h);
}
double spv2_sigmoid(double x){
    double h = spv2_rat_odd_v(0.5 * x, SPV2_TANH_N, SPV2_TANH_NNUM,
                                        SPV2_TANH_D, SPV2_TANH_NDEN, 4.0);
    return 0.5 * (1.0 + h);
}

/* ===================== AVX2 batch ======================================== */
#define SPV2_LANE_FMA(P, Y, K) _mm256_fmadd_pd((P), (Y), (K))

typedef __m256d v4;
static inline v4 rat_odd_vec(v4 u, const double* P, int np,
                             const double* D, int nd, v4 cutmask_hi, v4 cutmask_lo,
                             v4 one, v4 zero){
    /* saturation vectorielle par blends */
    v4 gt = _mm256_cmp_pd(u, cutmask_hi, _CMP_GT_OQ);
    v4 lt = _mm256_cmp_pd(u, cutmask_lo, _CMP_LT_OQ);
    v4 y = _mm256_mul_pd(u, u);
    v4 pn = _mm256_setzero_pd();
    for(int i = np - 1; i >= 0; i--)
        pn = _mm256_fmadd_pd(pn, y, _mm256_set1_pd(P[i]));
    v4 dn = _mm256_setzero_pd();
    for(int i = nd - 1; i >= 0; i--)
        dn = _mm256_fmadd_pd(dn, y, _mm256_set1_pd(D[i]));
    v4 r = _mm256_mul_pd(u, _mm256_div_pd(pn, dn));
    r = _mm256_blendv_pd(r, one, gt);
    r = _mm256_blendv_pd(r, _mm256_sub_pd(zero, one), lt);
    return r;
}
static const double SPV2_CUT[2] = {4.0, -4.0};
static const double SPV2_CUT_E[2] = {3.5, -3.5};

void spv2_batch_tanh(const double* x, double* out, long long n){
    long long vec = n & ~3LL;
    v4 one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    v4 chi = _mm256_set1_pd(4.0), clo = _mm256_set1_pd(-4.0);
    #pragma omp parallel for schedule(static)
    for(long long i = 0; i < vec; i += 4){
        v4 vx = _mm256_loadu_pd(x + i);
        _mm256_storeu_pd(out + i,
            rat_odd_vec(vx, SPV2_TANH_N, SPV2_TANH_NNUM, SPV2_TANH_D,
                        SPV2_TANH_NDEN, chi, clo, one, zero));
    }
    for(long long i = vec; i < n; i++) out[i] = spv2_tanh(x[i]);
}

void spv2_batch_erf(const double* x, double* out, long long n){
    long long vec = n & ~3LL;
    v4 one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    v4 chi = _mm256_set1_pd(3.5), clo = _mm256_set1_pd(-3.5);
    #pragma omp parallel for schedule(static)
    for(long long i = 0; i < vec; i += 4){
        v4 vx = _mm256_loadu_pd(x + i);
        _mm256_storeu_pd(out + i,
            rat_odd_vec(vx, SPV2_ERF_N, SPV2_ERF_NNUM, SPV2_ERF_D,
                        SPV2_ERF_NDEN, chi, clo, one, zero));
    }
    for(long long i = vec; i < n; i++) out[i] = spv2_erf(x[i]);
}

void spv2_batch_gelu(const double* x, double* out, long long n){
    long long vec = n & ~3LL;
    v4 one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    v4 chi = _mm256_set1_pd(4.0), clo = _mm256_set1_pd(-4.0);
    v4 c1 = _mm256_set1_pd(SPV2_GELU_C1), c3 = _mm256_set1_pd(SPV2_GELU_C3);
    v4 half = _mm256_set1_pd(0.5);
    #pragma omp parallel for schedule(static)
    for(long long i = 0; i < vec; i += 4){
        v4 vx = _mm256_loadu_pd(x + i);
        v4 t  = _mm256_mul_pd(vx, vx);
        v4 u  = _mm256_fmadd_pd(_mm256_mul_pd(c3, t), vx, _mm256_mul_pd(c1, vx));
        v4 h  = rat_odd_vec(u, SPV2_TANH_N, SPV2_TANH_NNUM, SPV2_TANH_D,
                            SPV2_TANH_NDEN, chi, clo, one, zero);
        v4 g  = _mm256_mul_pd(_mm256_mul_pd(half, vx),
                              _mm256_add_pd(one, h));
        _mm256_storeu_pd(out + i, g);
    }
    for(long long i = vec; i < n; i++) out[i] = spv2_gelu(x[i]);
}

void spv2_batch_sigmoid(const double* x, double* out, long long n){
    long long vec = n & ~3LL;
    v4 one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    v4 chi = _mm256_set1_pd(4.0), clo = _mm256_set1_pd(-4.0);
    v4 half = _mm256_set1_pd(0.5);
    #pragma omp parallel for schedule(static)
    for(long long i = 0; i < vec; i += 4){
        v4 vx = _mm256_loadu_pd(x + i);
        v4 u  = _mm256_mul_pd(half, vx);
        v4 h  = rat_odd_vec(u, SPV2_TANH_N, SPV2_TANH_NNUM, SPV2_TANH_D,
                            SPV2_TANH_NDEN, chi, clo, one, zero);
        v4 s  = _mm256_mul_pd(_mm256_add_pd(one, h), half);
        _mm256_storeu_pd(out + i, s);
    }
    for(long long i = vec; i < n; i++) out[i] = spv2_sigmoid(x[i]);
}
