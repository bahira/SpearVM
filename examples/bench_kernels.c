/* bench_kernels.c — Benchmark : chaque noyau SPEAR, scalaire vs AVX2.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp bench_kernels.c -o bench_kernels -lm */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N (1<<22)

/* ---- déclarations des kernels AVX2 (dans spur_kernels.c) ---- */
void spur_batch_gelu(const double*,double*,long long);
void spur_batch_erf(const double*,double*,long long);
void spur_batch_tanh(const double*,double*,long long);

/* ---- implémentations scalaires (libm exact) ---- */
static void scalar_gelu(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++){
        double gx=0.6*x[i];
        out[i]=0.5*gx*(1+erf(gx*0.70710678118654752));
    }
}
static void scalar_erf(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++) out[i]=erf(x[i]);
}
static void scalar_tanh(const double* x,double* out,long long n){
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;i++) out[i]=tanh(x[i]);
}

static double bench(void(*fn)(const double*,double*,long long),
                    const double* x,double* out,long long n,int reps){
    clock_t best=clock()+CLOCKS_PER_SEC;
    fn(x,out,n); /* warmup */
    for(int r=0;r<reps;r++){
        clock_t t0=clock();
        fn(x,out,n);
        clock_t t1=clock();
        if(t1-t0<best)best=t1-t0;
    }
    return (double)best/CLOCKS_PER_SEC;
}

int main(void){
    printf("=== SpearVM Kernels AVX2 ===\n\n");

    double* x=malloc(N*sizeof(double));
    double* a=malloc(N*sizeof(double));
    double* b=malloc(N*sizeof(double));

    for(long long i=0;i<N;i++)x[i]=-3.0+6.0*i/N;

    int reps=10;
    struct {
        const char*name;
        void(*nat)(const double*,double*,long long);
        void(*avx)(const double*,double*,long long);
    } ks[]={
        {"GELU",scalar_gelu,spur_batch_gelu},
        {"ERF", scalar_erf, spur_batch_erf},
        {"TANH",scalar_tanh,spur_batch_tanh},
    };

    printf("%-6s %12s %12s %8s\n","kernel","native(ms)","AVX2(ms)","speedup");
    printf("%-6s %12s %12s %8s\n","------","----------","--------","-------");

    for(int k=0;k<3;k++){
        double tn=bench(ks[k].nat,x,a,N,reps);
        double ta=bench(ks[k].avx,x,b,N,reps);
        printf("%-6s %10.1fms %10.1fms %7.1fx\n",ks[k].name,tn*1000,ta*1000,tn/ta);
    }

    free(x);free(a);free(b);
    return 0;
}
