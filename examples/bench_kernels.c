/* bench_kernels.c — Benchmark par noyau : scalaire vs AVX2.
   Chaque noyau SPEAR appliqué à 4M éléments, comparé au natif libm.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -o bench_kernels bench_kernels.c -lm */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#ifdef __AVX2__
#include <immintrin.h>
#define HAS_AVX2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define N (1<<22)

/* ================= Références exactes ================= */
static double exact_gelu(double x){ return 0.5*x*(1+erf(x*0.70710678118654752)); }

/* ================= GELU ================= */
static void scalar_gelu(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++)
        out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}
#ifdef __AVX2__
static void avx_gelu(const double* x,double* out,long long n){
    __m256d c306=_mm256_set1_pd(0.306923),c501=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002),z=_mm256_setzero_pd();
    __m256d ck=_mm256_set1_pd(0.997729),cb=_mm256_set1_pd(-0.004004);
    long long vec=n&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec;blk+=4){
        __m256d vx=_mm256_loadu_pd(x+blk);
        __m256d u=_mm256_fmadd_pd(c306,vx,c501);
        u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
        _mm256_storeu_pd(out+blk,_mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(vx,u),ck),cb));
    }
    for(long long i=vec;i<n;i++) out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}
#endif

/* ================= ERF ================= */
static void scalar_erf(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++){
        double xx=fmax(-2.0,fmin(2.0,x[i]));
        out[i]=1.106774*((xx+0.034298*xx*xx*xx)/(0.995+0.378089*xx*xx));
    }
}
#ifdef __AVX2__
static void avx_erf(const double* x,double* out,long long n){
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d c3=_mm256_set1_pd(0.034298),cn=_mm256_set1_pd(1.106774);
    __m256d b2=_mm256_set1_pd(0.378089),b0=_mm256_set1_pd(0.995);
    long long vec=n&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec;blk+=4){
        __m256d vx=_mm256_loadu_pd(x+blk);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        __m256d num=_mm256_fmadd_pd(c3,t,y);
        __m256d den=_mm256_fmadd_pd(b2,t,b0);
        _mm256_storeu_pd(out+blk,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-2.0,fmin(2.0,x[i]));
        out[i]=1.106774*((xx+0.034298*xx*xx*xx)/(0.995+0.378089*xx*xx));
    }
}
#endif

/* ================= TANH ================= */
static void scalar_tanh(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++){
        double xx=fmax(-3.0,fmin(3.0,x[i]));
        out[i]=0.900021*((xx+0.053639*xx*xx*xx)/(0.90122+0.343141*xx*xx));
    }
}
#ifdef __AVX2__
static void avx_tanh(const double* x,double* out,long long n){
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d c3=_mm256_set1_pd(0.053639),cn=_mm256_set1_pd(0.900021);
    __m256d b2=_mm256_set1_pd(0.343141),b0=_mm256_set1_pd(0.90122);
    long long vec=n&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec;blk+=4){
        __m256d vx=_mm256_loadu_pd(x+blk);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        __m256d num=_mm256_fmadd_pd(c3,t,y);
        __m256d den=_mm256_fmadd_pd(b2,t,b0);
        _mm256_storeu_pd(out+blk,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-3.0,fmin(3.0,x[i]));
        out[i]=0.900021*((xx+0.053639*xx*xx*xx)/(0.90122+0.343141*xx*xx));
    }
}
#endif

/* ================= Benchmark générique ================= */
typedef void (*kernel_fn)(const double*,double*,long long);

static double bench(kernel_fn fn,const double* x,double* out,long long n,int reps){
    clock_t best=clock()+CLOCKS_PER_SEC;
    for(int r=0;r<reps;r++){
        clock_t t0=clock();
        fn(x,out,n);
        clock_t t1=clock();
        if(t1-t0<best)best=t1-t0;
    }
    return (double)best/CLOCKS_PER_SEC;
}

int main(void){
    printf("=== SpearVM : Benchmark par noyau ===\n");
    printf("N=%d M éléments, 10 reps, meilleur temps\n\n",N/1000000);

    double* x=malloc(N*sizeof(double));
    double* out_a=malloc(N*sizeof(double));
    double* out_b=malloc(N*sizeof(double));

    /* trajectoire [-1,3] */
    for(long long i=0;i<N;i++)x[i]=-1.0+(double)i/N*4.0;

    int reps=10;

    /* warmup */
    scalar_gelu(x,out_a,100);scalar_erf(x,out_a,100);scalar_tanh(x,out_a,100);

    struct { const char*name; kernel_fn scalar; kernel_fn avx; double ref_scale; } kernels[] = {
        {"GELU",scalar_gelu,NULL,1.0},
        {"ERF", scalar_erf, NULL,1.0},
        {"TANH",scalar_tanh,NULL,1.0},
    };
#ifdef __AVX2__
    kernels[0].avx=avx_gelu;
    kernels[1].avx=avx_erf;
    kernels[2].avx=avx_tanh;
#endif

    printf("%-6s %12s %12s %10s\n","kernel","scalaire","AVX2","speedup");
    printf("%-6s %12s %12s %10s\n","------","-----------","-----------","---------");

    for(int k=0;k<3;k++){
        /* warmup */
        kernels[k].scalar(x,out_a,100);
#ifdef __AVX2__
        if(kernels[k].avx)kernels[k].avx(x,out_b,100);
#endif
        /* timing */
        double ts=bench(kernels[k].scalar,x,out_a,N,reps);
#ifdef __AVX2__
        double ta=kernels[k].avx?bench(kernels[k].avx,x,out_b,N,reps):0;
#else
        double ta=0;
#endif
        printf("%-6s %10.1fms",(const char*)kernels[k].name,ts*1000);
#ifdef __AVX2__
        if(ta>0)printf(" %10.1fms %8.1fx",ta*1000,ts/ta);
        else printf(" %10s","n/a");
#endif
        printf("\n");
    }

    printf("\n");
    free(x);free(out_a);free(out_b);
    return 0;
}
