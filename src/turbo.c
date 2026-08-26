/* SpearVM Turbo — Accélérateur AVX2 + OpenMP pour workloads transcendantals.
   Chaque élément traverse le même pipeline mathématique que le kernel boucle,
   mais traité par blocs de 4 doubles en SIMD, parallélisé multi-cœur.
   Compare contre la version natif libm scalaire. */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- Référence exacte (libm IEEE) pour validation ---- */
static double ref_gelu(double x){ return 0.5*x*(1+erf(x*0.70710678118654752)); }

/* ---- Noyaux scalaires SPEAR (approximatifs) ---- */
double spur_k_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}
double spur_k_erf(double x){
    x=fmax(-2.0,fmin(2.0,x));
    return 1.106774*((x+0.034298*x*x*x)/(0.995+0.378089*x*x));
}
double spur_k_tanh(double x){
    x=fmax(-3.0,fmin(3.0,x));
    return 0.900021*((x+0.053639*x*x*x)/(0.90122+0.343141*x*x));
}

/* ---- Pipeline vectorisé AVX2 : le même workload que bench_native_vs_jit ---- */
/* out = h*k + p + m où :
   g = spur_gelu(0.6x), e = erf(x), c = tanh(g+e), f = max(g,c),
   h = tanh(e-g), k = erf(c+f), p = erf(g/2), ge = g*e, cf = c*f, m = max(ge,cf) */

__m256d avx_gelu(__m256d gx){
    /* gelu approx spear sur gx déjà calculé */
    __m256d cm=_mm256_set1_pd(1.002);
    __m256d z=_mm256_setzero_pd();
    __m256d c1=_mm256_set1_pd(0.306923);
    __m256d u=_mm256_add_pd(_mm256_mul_pd(c1,gx),_mm256_set1_pd(0.501));
    u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
    return _mm256_mul_pd(gx,u);
}
__m256d avx_erf(__m256d x){
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,x));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_add_pd(y,_mm256_mul_pd(_mm256_set1_pd(0.034298),t));
    __m256d den=_mm256_add_pd(_mm256_set1_pd(0.995),_mm256_mul_pd(_mm256_set1_pd(0.378089),t));
    return _mm256_mul_pd(_mm256_set1_pd(1.106774),_mm256_div_pd(num,den));
}
__m256d avx_tanh(__m256d x){
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,x));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_add_pd(y,_mm256_mul_pd(_mm256_set1_pd(0.053639),t));
    __m256d den=_mm256_add_pd(_mm256_set1_pd(0.90122),_mm256_mul_pd(_mm256_set1_pd(0.343141),t));
    return _mm256_mul_pd(_mm256_set1_pd(0.900021),_mm256_div_pd(num,den));
}

/* Pipeline complet vectorisé : out[i] pour un bloc de 4 éléments */
static inline void avx_pipeline(const double* in,double* out){
    __m256d vx=_mm256_loadu_pd(in);
    __m256d vg=_mm256_mul_pd(vx,_mm256_set1_pd(0.6));

    __m256d g=avx_gelu(vg);       /* g = gelu(0.6x) */
    __m256d e=avx_erf(vx);        /* e = erf(x)     */
    __m256d ge=_mm256_mul_pd(g,e);/* ge = g*e       */

    __m256d sum_ge_c=_mm256_add_pd(g,e);
    __m256d c=avx_tanh(sum_ge_c); /* c = tanh(g+e) */
    __m256d f=_mm256_max_pd(g,c); /* f = max(g,c) */
    __m256d h=avx_tanh(_mm256_sub_pd(e,g)); /* h = tanh(e-g) */
    __m256d cf=_mm256_mul_pd(c,f);          /* cf = c*f */
    __m256d m=_mm256_max_pd(ge,cf);         /* m = max(ge,cf) */

    __m256d sum_cf_f=_mm256_add_pd(c,f);
    __m256d k=avx_erf(sum_cf_f);            /* k = erf(c+f) */
    __m256d p=avx_erf(_mm256_mul_pd(g,_mm256_set1_pd(0.5))); /* p = erf(g/2) */

    __m256d hk=_mm256_mul_pd(h,k);           /* h*k */
    __m256d res=_mm256_add_pd(hk,p);         /* +p */
    res=_mm256_add_pd(res,m);                /* +m */

    _mm256_storeu_pd(out,res);
}

/* ---- Benchmark : exécute le pipeline sur count éléments ---- */
void turbo_exec(const double* in,double* out,long long count){
    long long vec_end=count&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec_end;blk+=4)
        avx_pipeline(in+blk,out+blk);
    for(long long i=vec_end;i<count;i++){
        double x=in[i],gx=0.6*x;
        double g=spur_k_gelu(gx),e=spur_k_erf(x),c=spur_k_tanh(g+e);
        double f=fmax(g,c),h=spur_k_tanh(e-g),k=spur_k_erf(c+f);
        double p=spur_k_erf(0.5*g),ge=g*e,cf=c*f,m=fmax(ge,cf);
        out[i]=spur_k_tanh(e-g)*spur_k_erf(c+f)+p+m;
    }
}
