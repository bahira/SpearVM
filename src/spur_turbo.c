/* spur_turbo.c — Bibliothèque de kernels AVX2 pour inférence accélérée.
   Chaque fonction traite un tableau élément-par-élément en SIMD.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -o spur_turbo.dll spur_turbo.c -lm */
#include <math.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================= GELU ================= */
void spur_gelu(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d c306=_mm256_set1_pd(0.306923),c501=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002),z=_mm256_setzero_pd();
    __m256d ck=_mm256_set1_pd(0.997729),cb=_mm256_set1_pd(-0.004004);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d u=_mm256_fmadd_pd(c306,vx,c501);
        u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
        __m256d r=_mm256_mul_pd(vx,u);
        _mm256_storeu_pd(out+i,_mm256_add_pd(_mm256_mul_pd(r,ck),cb));
    }
    for(long long i=vec;i<n;i++)
        out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}

/* ================= ERF ================= */
void spur_erf(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d cn=_mm256_set1_pd(1.106774);
    __m256d c3=_mm256_set1_pd(0.034298),b2=_mm256_set1_pd(0.378089),b0=_mm256_set1_pd(0.995);
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

/* ================= TANH ================= */
void spur_tanh(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d cn=_mm256_set1_pd(0.900021);
    __m256d c3=_mm256_set1_pd(0.053639),b2=_mm256_set1_pd(0.343141),b0=_mm256_set1_pd(0.90122);
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

/* ================= LSE2 ================= */
void spur_lse2(const double* a,const double* b,double* out,long long n){
    long long vec=n&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d va=_mm256_loadu_pd(a+i);
        __m256d vb=_mm256_loadu_pd(b+i);
        _mm256_storeu_pd(out+i,_mm256_max_pd(va,vb));
    }
    for(long long i=vec;i<n;i++)out[i]=fmax(a[i],b[i]);
}

/* ================= SIGMOID ================= */
void spur_sigmoid(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d half=_mm256_set1_pd(0.5),one=_mm256_set1_pd(1.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        /* sigmoid(x) = 0.5*(1+tanh(x/2)) */
        __m256d half_x=_mm256_mul_pd(vx,half);
        __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,half_x));
        __m256d t=_mm256_mul_pd(y,y);
        __m256d num=_mm256_add_pd(y,_mm256_mul_pd(_mm256_set1_pd(0.053639),t));
        __m256d den=_mm256_add_pd(_mm256_set1_pd(0.90122),_mm256_mul_pd(_mm256_set1_pd(0.343141),t));
        __m256d r=_mm256_mul_pd(_mm256_set1_pd(0.900021),_mm256_div_pd(num,den));
        _mm256_storeu_pd(out+i,_mm256_add_pd(half,_mm256_mul_pd(half,r)));
    }
    for(long long i=vec;i<n;i++)out[i]=0.5*(1+tanh(0.5*x[i]));
}
