/* bench_512.c — EXPERIENCE AVX-512 vs AVX2 sur le même rationnel tanh/gelu.
   SpearVM avait différé le dispatch AVX-512 "aucun CPU de test dispo" ;
   cette machine expose avx512f -> on mesure enfin l'effet réel.
   Deux builds :
     gcc -O3 -mavx2 -mfma -fopenmp bench_512.c -o b_avx2   && ./b_avx2
     gcc -O3 -mavx2 -mfma -mavx512f -fopenmp bench_512.c -o b_avx512
   Le chemin __m512d n'est compilé que si __AVX512F__ est défini.         */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N (4 * 1024 * 1024)

/* coefficients tanh_v2 (copie locale du header généré) */
static const double TN[3] = {9.99671553747818797e-01, 9.69818098308345700e-02, 5.34898030509189373e-04};
static const double TD[3] = {1.00000000000000000e+00, 4.28929626818397136e-01, 1.13140132579288184e-02};

static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e9+ts.tv_nsec;
}
static volatile double sink;

/* ---------- AVX2 ---------- */
static void tanh_256(const double* x, double* out, long long n){
    long long vec = n & ~3LL;
    const __m256d one = _mm256_set1_pd(1.0), zero = _mm256_setzero_pd();
    const __m256d chi = _mm256_set1_pd(4.0), clo = _mm256_set1_pd(-4.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d u = _mm256_loadu_pd(x+i);
        __m256d gt = _mm256_cmp_pd(u, chi, _CMP_GT_OQ);
        __m256d lt = _mm256_cmp_pd(u, clo, _CMP_LT_OQ);
        __m256d y = _mm256_mul_pd(u,u), pn=_mm256_setzero_pd(), dn=_mm256_setzero_pd();
        for(int k=2;k>=0;k--) pn = _mm256_fmadd_pd(pn,y,_mm256_set1_pd(TN[k]));
        for(int k=2;k>=0;k--) dn = _mm256_fmadd_pd(dn,y,_mm256_set1_pd(TD[k]));
        __m256d r = _mm256_mul_pd(u,_mm256_div_pd(pn,dn));
        r = _mm256_blendv_pd(r,one,gt);
        r = _mm256_blendv_pd(r,_mm256_sub_pd(zero,one),lt);
        _mm256_storeu_pd(out+i,r);
    }
    for(long long i=vec;i<n;i++){
        double u=x[i]; out[i]= fabs(u)>4.0?(u>0?1.0:-1.0):u*( (TN[2]*(u*u)+TN[1])*(u*u)+TN[0] ) / ( ((TD[2]*(u*u))+TD[1])*(u*u)+TD[0] );
    }
}

#ifdef __AVX512F__
/* ---------- AVX-512 : mêmes formules, largeur double ---------- */
static void tanh_512(const double* x, double* out, long long n){
    long long vec = n & ~7LL;
    const __m512d one = _mm512_set1_pd(1.0), zero = _mm512_setzero_pd();
    const __m512d chi = _mm512_set1_pd(4.0), clo = _mm512_set1_pd(-4.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m512d u = _mm512_loadu_pd(x+i);
        __mmask8 gt = _mm512_cmp_pd_mask(u, chi, _CMP_GT_OQ);
        __mmask8 lt = _mm512_cmp_pd_mask(u, clo, _CMP_LT_OQ);
        __m512d y = _mm512_mul_pd(u,u), pn=_mm512_setzero_pd(), dn=_mm512_setzero_pd();
        for(int k=2;k>=0;k--) pn = _mm512_fmadd_pd(pn,y,_mm512_set1_pd(TN[k]));
        for(int k=2;k>=0;k--) dn = _mm512_fmadd_pd(dn,y,_mm512_set1_pd(TD[k]));
        __m512d r = _mm512_mul_pd(u,_mm512_div_pd(pn,dn));
        r = _mm512_mask_blend_pd(gt, r, one);
        r = _mm512_mask_blend_pd(lt, r, _mm512_sub_pd(zero,one));
        _mm512_storeu_pd(out+i,r);
    }
    for(long long i=vec;i<n;i++) out[i]=tanh(x[i]);
}
#endif

static void fill(double* x){
    for(long long i=0;i<N;i++) x[i]=-6.0+12.0*(double)i/(N-1);
}
static double med(void (*fn)(const double*,double*,long long),
                  const double* x,double* out,int reps){
    double t[15];
    for(int r=0;r<reps;r++){
        double t0=now_ns(); fn(x,out,N); t[r]=now_ns()-t0;
        sink+=out[r];
    }
    for(int a=0;a<reps;a++)for(int b=a+1;b<reps;b++)
        if(t[b]<t[a]){double w=t[a];t[a]=t[b];t[b]=w;}
    return t[reps/2]/N;
}

int main(void){
    printf("build: %s\n",
#ifdef __AVX512F__
        "AVX-512 (__m512d)");
#else
        "AVX2 (__m256d)");
#endif
    double* x = aligned_alloc(64,N*sizeof(double));
    double* o = aligned_alloc(64,N*sizeof(double));
    fill(x);
    /* vérification */
    tanh_256(x,o,N>>0); 
    double e2=0;
    for(long long i=0;i<N;i++){double d=fabs(o[i]-tanh(x[i]));if(d>e2)e2=d;}
    double ns2 = med(tanh_256,x,o,9);
    printf("AVX2  : err=%.3e  ns/elem=%.3f\n", e2, ns2);
#ifdef __AVX512F__
    tanh_512(x,o,N); double e5=0;
    for(long long i=0;i<N;i++){double d=fabs(o[i]-tanh(x[i]));if(d>e5)e5=d;}
    double ns5 = med(tanh_512,x,o,9);
    printf("AVX512: err=%.3e  ns/elem=%.3f\n", e5, ns5);
    printf("speedup AVX512 vs AVX2 : x%.3f\n", ns2/ns5);
#else
    printf("(pas de chemin 512 dans ce build)\n");
#endif
    FILE* f=fopen(
#ifdef __AVX512F__
        "/home/z/my-project/spear-unified/results/bench_512.json"
#else
        "/home/z/my-project/spear-unified/results/bench_avx2_baseline.json"
#endif
        ,"w");
    if(f){
#ifdef __AVX512F__
        fprintf(f,"{\"variant\":\"avx512\",\"N\":%d,\"ns_per_elem\":%.4f,\"max_err\":%.3e}\n",N,
                med(tanh_512,x,o,9),e5);
#else
        fprintf(f,"{\"variant\":\"avx2\",\"N\":%d,\"ns_per_elem\":%.4f,\"max_err\":%.3e}\n",N,ns2,e2);
#endif
        fclose(f);
    }
    return 0;
}
