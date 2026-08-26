/* bench_turbo.c — Benchmark simple : chaque noyau SPEAR vs libm natif.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -o bt bench_turbo.c -lm */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef __AVX2__
#include <immintrin.h>
#define HAS_AVX 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define N 4000000
#define REPS 20

static double sp_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}

/* ---- Référence exacte ---- */
static double exact_gelu(double x){
    return 0.5*x*(1+erf(x*0.70710678118654752));
}

/* ================= GELU scalaire ================= */
static void scalar_gelu(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++)
        out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}

/* ================= GELU AVX2 ================= */
static void avx_gelu(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d c1=_mm256_set1_pd(0.306923),c2=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002),z=_mm256_setzero_pd();
    __m256d ck=_mm256_set1_pd(0.997729),cb=_mm256_set1_pd(-0.004004);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d u=_mm256_fmadd_pd(c1,vx,c2);
        u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
        __m256d r=_mm256_mul_pd(vx,u);
        _mm256_storeu_pd(out+i,_mm256_add_pd(_mm256_mul_pd(r,ck),cb));
    }
    for(long long i=vec;i<n;i++)
        out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}

/* ================= ERF scalaire ================= */
static void scalar_erf(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++)out[i]=erf(x[i]);
}

/* ================= ERF AVX2 ================= */
static void avx_erf(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d cn=_mm256_set1_pd(1.106774),c3=_mm256_set1_pd(0.034298);
    __m256d b2=_mm256_set1_pd(0.378089),b0=_mm256_set1_pd(0.995);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        __m256d num=_mm256_add_pd(y,_mm256_mul_pd(c3,t));
        __m256d den=_mm256_add_pd(b0,_mm256_mul_pd(b2,t));
        _mm256_storeu_pd(out+i,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-2.0,fmin(2.0,x[i]));
        out[i]=1.106774*((xx+0.034298*xx*xx*xx)/(0.995+0.378089*xx*xx));
    }
}

/* ================= TANH scalaire ================= */
static void scalar_tanh(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++)out[i]=tanh(x[i]);
}

/* ================= TANH AVX2 ================= */
static void avx_tanh(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d cn=_mm256_set1_pd(0.900021),c3=_mm256_set1_pd(0.053639);
    __m256d b2=_mm256_set1_pd(0.343141),b0=_mm256_set1_pd(0.90122);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        __m256d num=_mm256_add_pd(y,_mm256_mul_pd(c3,t));
        __m256d den=_mm256_add_pd(b0,_mm256_mul_pd(b2,t));
        _mm256_storeu_pd(out+i,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-3.0,fmin(3.0,x[i]));
        out[i]=0.900021*((xx+0.053639*xx*xx*xx)/(0.90122+0.343141*xx*xx));
    }
}

int main(void){
    printf("=== SpearVM Turbo : Benchmark noyaux ===\n\n");
    double* x=malloc(N*sizeof(double));
    double* out=malloc(N*sizeof(double));

    /* trajectoire : [-3,3] */
    for(long long i=0;i<N;i++)x[i]=-3.0+6.0*i/N;

    /* warmup */
    scalar_gelu(x,out,100);

    int reps=20;

    /* ---- benchmark chaque noyau : scalaire vs AVX2 ---- */
    struct { const char*name;
             void(*sc)(const double*,double*,long long);
             void(*ax)(const double*,double*,long long);
           } kernels[] = {
        {"GELU", scalar_gelu, avx_gelu},
        {"ERF",  scalar_erf,  avx_erf},
        {"TANH", scalar_tanh, avx_tanh},
    };

    printf("%-6s %10s %10s %8s\n","noyau","natif(ms)","avx2(ms)","speedup");
    printf("%-6s %10s %10s %8s\n","------","---------","--------","-------");

    double total_nat=0,total_avx=0;
    for(int k=0;k<3;k++){
        double bn=1e18,ba=1e18;
        for(int r=0;r<reps;r++){
            clock_t t=clock();kernels[k].sc(x,out,N);double dt=(double)(clock()-t)/CLOCKS_PER_SEC;
            if(dt<bn)bn=dt;
            t=clock();kernels[k].ax(x,out,N);dt=(double)(clock()-t)/CLOCKS_PER_SEC;
            if(dt<ba)ba=dt;
        }
        total_nat+=bn;total_avx+=ba;
        printf("%-6s %10.1fms %10.1fms %7s\n",
               kernels[k].name,bn*1000,ba*1000,"");
    }

    /* validation gelu approx vs exact */
    double max_err=0;
    for(long long i=0;i<N;i+=1000){
        double gx=0.6*x[i];
        double ref=0.5*gx*(1+erf(gx*0.70710678118654752));
        double got=sp_gelu(gx);
        double err=fabs(got-ref);
        if(err>max_err)max_err=err;
    }

    printf("\ntotal natif=%6.1fms  total avx2=%6.1fms\n",total_nat*1000,total_avx*1000);
    printf("speedup global = x%.1f\n",total_nat/total_avx);
    printf("validation err_max=%.6f\n",max_err);

    free(x);free(out);
    return 0;
}
