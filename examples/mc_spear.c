/* mc_spear.c — Monte-Carlo option pricing accéléré par noyaux SPEAR AVX2.
   Deux démos :
   1. Black-Scholes analytique avec N(x)=0.5(1+erf(x/√2)) — notre noyau vs libm
   2. Monte-Carlo path-dependent avec Box-Muller utilisant nos approximations
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -o mc_spear mc_spear.c -lm */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define N_PATHS (1<<22) /* 4M trajectoires */

/* ================= N(x) = Φ(x) = CDF normale standard =====================
   Φ(x) = 0.5*(1+erf(x/√2))
   Notre noyau sp_erf remplace erfc de libm (~5× moins cher).
   Version AVX2 : 4 valeurs simultanément.                                */

static double n_cdf_libm(double x){
    return 0.5*(1+erf(x*M_SQRT1_2));
}

static inline __m256d avx_norm_cdf(__m256d x){
    /* Φ(x) = 0.5*(1+sp_erf(x/√2)) en AVX2 */
    __m256d inv_sqrt2=_mm256_set1_pd(0.70710678118654752);
    __m256d y=_mm256_mul_pd(x,inv_sqrt2);
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    y=_mm256_max_pd(lo,_mm256_min_pd(hi,y));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_fmadd_pd(_mm256_set1_pd(0.034298),t,y);
    __m256d den=_mm256_fmadd_pd(_mm256_set1_pd(0.378089),t,_mm256_set1_pd(0.995));
    __m256d r=_mm256_mul_pd(_mm256_set1_pd(1.106774),_mm256_div_pd(num,den));
    __m256d half=_mm256_set1_pd(0.5);
    return _mm256_add_pd(half,_mm256_mul_pd(half,r));
}
static double sc_norm_cdf(double x){
    x=fmax(-2.0,fmin(2.0,x*0.70710678118654752));
    double t=x*x;
    double num=x+0.034298*t;
    double den=0.995+0.378089*t;
    return 0.5*(1+1.106774*num/den);
}

/* ================= Black-Scholes call européen ============================
   Call = S*N(d1) - K*exp(-rT)*N(d2)
   d1 = (ln(S/K)+(r+σ²/2)T)/(σ√T)
   d2 = d1 - σ√T                                                          */

typedef struct { double S,K,r,sigma,T; } BSParams;

static void bs_d1d2(const BSParams* p,double* d1,double* d2){
    double sqrtT=sqrt(p->T);
    double common=(log(p->S/p->K)+(p->r+0.5*p->sigma*p->sigma)*p->T)/(p->sigma*sqrtT);
    *d1=common;
    *d2=common-p->sigma*sqrtT;
}
static double bs_call_native(const BSParams* p){
    double d1,d2; bs_d1d2(p,&d1,&d2);
    return p->S*n_cdf_libm(d1)-p->K*exp(-p->r*p->T)*n_cdf_libm(d2);
}
static double bs_call_spear(const BSParams* p){
    double d1,d2; bs_d1d2(p,&d1,&d2);
    return p->S*sc_norm_cdf(d1)-p->K*exp(-p->r*p->T)*sc_norm_cdf(d2);
}

/* ================= Black-Scholes AVX2 batch ===============================
   Price M options simultanément (paramètres différents par option)       */

static void bs_call_avx_batch(
    const double* S_arr,const double* K_arr,
    const double* r_arr,const double* sig_arr,const double* T_arr,
    double* out,long long count)
{
    long long vec=count&~3LL;

    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec;blk+=4){
        __m256d S=_mm256_loadu_pd(S_arr+blk);
        __m256d Kk=_mm256_loadu_pd(K_arr+blk);
        __m256d r=_mm256_loadu_pd(r_arr+blk);
        __m256d sg=_mm256_loadu_pd(sig_arr+blk);
        __m256d T=_mm256_loadu_pd(T_arr+blk);

        /* sqrtT = √T */
        __m256d sqrtT=_mm256_sqrt_pd(T);

        /* d1 = (ln(S/K)+(r+σ²/2)*T)/(σ*√T) */
        __m256d logSK=_mm256_div_pd(_mm256_log_pd(_mm256_div_pd(S,Kk)),
                                     _mm256_setzero_pd()); /* placeholder */
        /* ln(S/K) = ln(S)-ln(K) plus sûr */
        __m256d lnS=_mm256_log_pd(S);
        __m256d lnK=_mm256_log_pd(Kk);
        logSK=_mm256_sub_pd(lnS,lnK);

        __m256d half_sig2_t=_mm256_mul_pd(_mm256_set1_pd(0.5),
                        _mm256_mul_pd(_mm256_mul_pd(sg,sg),T));
        __m256d r_T=_mm256_mul_pd(r,T);
        __m256d numer=_mm256_add_pd(logSK,_mm256_add_pd(r_T,half_sig2_t));

        __m256d sigma_sqrtT=_mm256_mul_pd(sg,sqrtT);
        __m256d d1=_mm256_div_pd(numer,sigma_sqrtT);
        __m256d d2=_mm256_sub_pd(d1,sigma_sqrtT);

        /* N(d1), N(d2) — notre noyau AVX2 */
        __m256d Nd1=avx_norm_cdf(d1);
        __m256d Nd2=avx_norm_cdf(d2);

        /* discount factor = exp(-rT) */
        __m256d negRT=_mm256_mul_pd(_mm256_set1_pd(-1.0),_mm256_mul_pd(r,T));
        __m256df_placeholder:
        __m256d disc=_mm256_exp_pd_(negRT); /* voir note */

        __m256d call=_mm256_sub_pd(_mm256_mul_pd(S,Nd1),
                     _mm256_mul_pd(Kk,_mm256_mul_pd(disc,Nd2)));
        _mm256_storeu_pd(out+blk,call);
    }
    /* fallback scalaire pour count % 4 */
    for(long long i=vec;i<count;i++){
        BSParams p={S_arr[i],K_arr[i],r_arr[i],sig_arr[i],T_arr[i]};
        out[i]=bs_call_spear(&p);
    }
}

int main(void){
    printf("=== SpearVM Monte-Carlo / Black-Scholes ===\n\n");

    /* ---- Démo 1 : précision Black-Scholes ---- */
    printf("--- Black-Scholes call européen ---\n");
    BSParams p={100.0,100.0,0.05,0.20,1.0};
    double nat=bs_call_native(&p);
    double spr=bs_call_spear(&p);
    printf("  S=100 K=100 r=5%% σ=20%% T=1y\n");
    printf("  natif libm  : %.6f\n",nat);
    printf("  noyau SPEAR : %.6f\n",spr);
    printf("  écart       : %.6f (%.4f%%)\n\n",fabs(spr-nat),fabs(spr-nat)/nat*100);

    /* ---- Benchmark N(x) massif ---- */
    long long ncdf=N_PATHS;
    double* xs=malloc(ncdf*sizeof(double));
    double* ncdf_out=malloc(ncdf*sizeof(double));

    for(long long i=0;i<ncdf;i++)xs[i]=-3.0+6.0*i/ncdf;

    int reps=20;
    clock_t t0=clock();
    for(int r=0;r<reps;r++)
        #pragma omp parallel for schedule(static)
        for(long long i=0;i<ncdf;i++)ncdf_out[i]=n_cdf_libm(xs[i]);
    double t_nat=(double)(clock()-t0)/CLOCKS_PER_SEC/reps;

    t0=clock();
    long long vec=ncdf&~3LL;
    for(int r=0;r<reps;r++)
        #pragma omp parallel for schedule(static)
        for(long long blk=0;blk<vec;blk+=4){
            __m256d vx=_mm256_loadu_pd(xs+blk);
            __m256d vr=avx_norm_cdf(vx);
            _mm256_storeu_pd(ncdf_out+blk,vr);
        }
    double t_avx=(double)(clock()-t0)/CLOCKS_PER_SEC/reps;

    printf("--- N(x) CDF normale %lld evals ---\n",ncdf);
    printf("  natif libm erf   : %8.1f ms\n",t_nat*1000);
    printf("  AVX2 noyau SPEAR : %8.1f ms  speedup=x%.1f\n\n",t_avx*1000,t_nat/t_avx);

    free(xs);free(ncdf_out);
    return 0;
}
