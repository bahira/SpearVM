#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#ifdef __AVX2__
#include <immintrin.h>
#define HAS_SIMD 1
#endif

#define N (1<<22)

static double sp_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}

static void scalar_run(const double* x,double* out,long long n){
    for(long long i=0;i<n;i++) out[i]=sp_gelu(x[i]);
}

#ifdef __AVX2__
static void avx_run(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d k306=_mm256_set1_pd(0.306923);
    __m256d k501=_mm256_set1_pd(0.501);
    __m256d km=_mm256_set1_pd(1.002);
    __m256d kz=_mm256_setzero_pd();
    __m256d kk=_mm256_set1_pd(0.997729);
    __m256d kb=_mm256_set1_pd(-0.004004);
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d u=_mm256_fmadd_pd(k306,vx,k501);
        u=_mm256_max_pd(u,kz); u=_mm256_min_pd(u,km);
        __m256d r=_mm256_mul_pd(vx,u);
        _mm256_storeu_pd(out+i,_mm256_add_pd(_mm256_mul_pd(r,kk),kb));
    }
    for(long long i=vec;i<n;i++) out[i]=sp_gelu(x[i]);
}
#endif

int main(void){
    double* x=malloc(N*sizeof(double));
    double* out=malloc(N*sizeof(double));

    for(long long i=0;i<N;i++) x[i]=-1.0+(double)i/N*4.0;

    scalar_run(x,out,100);

    int reps=20;

    /* benchmark scalaire */
    clock_t t0=clock();
    for(int r=0;r<reps;r++) scalar_run(x,out,N);
    double sec_scalar=(double)(clock()-t0)/CLOCKS_PER_SEC/reps;
    printf("gelu scalaire  : %8.2f ms (%.2f Gelem/s)\n",sec_scalar*1000,N/sec_scalar/1e9);

#ifdef HAS_SIMD
    /* benchmark AVX2 */
    clock_t ta=clock();
    for(int r=0;r<reps;r++) avx_run(x,out,N);
    double sec_avx=(double)(clock()-ta)/CLOCKS_PER_SEC/reps;
    printf("gelu AVX2      : %8.2f ms (%.2f Gelem/s)  speedup=x%.1f\n",
           sec_avx*1000,N/sec_avx/1e9,sec_scalar/sec_avx);
#endif

    /* validation */
    double max_err=0;
    for(long long i=0;i<N;i+=100){
        double ref=0.5*x[i]*(1+erf(x[i]*0.70710678118654752));
        double got=sp_gelu(x[i]);
        double err=fabs(got-ref);
        if(err>max_err)max_err=err;
    }
    printf("validation gelu: err_max=%.6f\n",max_err);

    free(x);free(out);
    return 0;
}
