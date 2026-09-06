/* SpearVM Kernels — 4 noyaux transcendantals vectorisés AVX2 + OpenMP.
   Chaque kernel traite un tableau élément-par-élément en SIMD.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -o spur_kernels.dll spur_kernels.c -lm */
#include <math.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <stdint.h>

/* ================= AVX-512 runtime dispatch ================================
   Compile avec -mavx2 (binaire portable), active AVX-512 a l'execution
   si le CPU le supporte. Env vars :
     SPUR_FORCE_AVX2=1   -> tout en AVX2
     SPUR_FORCE_AVX512=1 -> tout en AVX-512
   Policy mesuree : erf/tanh (division-bound) gagnent ~20% en AVX-512,
   gelu (FMA-chain) reste en AVX2 (downclock). */
#if defined(__x86_64__) || defined(_M_X64)
#define SPIR_HAS_TARGET_ATTR 1
#endif

static int spur_cpu_avx512(void){
    static int cached = -1;
    if (cached < 0)
        cached = (__builtin_cpu_supports("avx2")
               && __builtin_cpu_supports("fma")
               && __builtin_cpu_supports("avx512f")) ? 1 : 0;
    return cached;
}

#define SPUR_VEX_AUTO 0
#define SPUR_VEX_AVX2 1
#define SPUR_VEX_512  2

static int spur_vex_override(void){
    static int cached = -1;
    if (cached < 0){
        const char* f2=getenv("SPUR_FORCE_AVX2");
        const char* f512=getenv("SPUR_FORCE_AVX512");
        cached = (f2 && f2[0]=='1')   ? SPUR_VEX_AVX2
               : (f512 && f512[0]=='1') ? SPUR_VEX_512
               : SPUR_VEX_AUTO;
    }
    return cached;
}

static int spur_use_avx512(int prefers_512){
    switch (spur_vex_override()){
        case SPUR_VEX_AVX2: return 0;
        case SPUR_VEX_512:  return 1;
        default:            return prefers_512 && spur_cpu_avx512();
    }
}

#ifdef SPIR_HAS_TARGET_ATTR
__attribute__((target("avx512f,avx512vl")))
#endif
static void spur_batch_gelu_avx512(const double* x,double* out,long long n){
    long long vec=n&~7LL;
    __m512d c306=_mm512_set1_pd(0.306923);
    __m512d c501=_mm512_set1_pd(0.501);
    __m512d cm=_mm512_set1_pd(1.002);
    __m512d z=_mm512_setzero_pd();
    __m512d ck=_mm512_set1_pd(0.997729);
    __m512d cb=_mm512_set1_pd(-0.004004);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);
        __m512d u=_mm512_fmadd_pd(c306,vx,c501);
        u=_mm512_max_pd(u,z); u=_mm512_min_pd(u,cm);
        __m512d r=_mm512_mul_pd(vx,u);
        _mm512_storeu_pd(out+i,_mm512_add_pd(_mm512_mul_pd(r,ck),cb));
    }
    for(long long i=vec;i<n;i++)
        out[i]=0.997729*(x[i]*fmin(1.002,fmax(0.0,0.306923*x[i]+0.501)))-0.004004;
}

#ifdef SPIR_HAS_TARGET_ATTR
__attribute__((target("avx512f,avx512vl")))
#endif
static void spur_rat_avx512(const double* x,double* out,long long n,
                            double lo,double hi,double cn,
                            double c3,double b0,double b2){
    long long vec=n&~7LL;
    __m512d vlo=_mm512_set1_pd(lo),vhi=_mm512_set1_pd(hi);
    __m512d vcn=_mm512_set1_pd(cn),vc3=_mm512_set1_pd(c3);
    __m512d vb0=_mm512_set1_pd(b0),vb2=_mm512_set1_pd(b2);
    __m512d one=_mm512_set1_pd(1.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);
        __m512d y=_mm512_max_pd(vlo,_mm512_min_pd(vhi,vx));
        __m512d t=_mm512_mul_pd(y,y);
        __m512d num=_mm512_mul_pd(y,_mm512_add_pd(one,_mm512_mul_pd(vc3,t)));
        __m512d den=_mm512_add_pd(vb0,_mm512_mul_pd(vb2,t));
        _mm512_storeu_pd(out+i,_mm512_mul_pd(vcn,_mm512_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double y=fmax(lo,fmin(hi,x[i]));
        out[i]=cn*((y+c3*y*y*y)/(b0+b2*y*y));
    }
}

#ifdef SPIR_HAS_TARGET_ATTR
__attribute__((target("avx512f,avx512vl")))
#endif
static void spur_batch_gelu_f32_avx512(const float* x,float* out,long long n){
    long long vec=n&~15LL;
    __m512 c306=_mm512_set1_ps(0.306923f),c501=_mm512_set1_ps(0.501f);
    __m512 cm=_mm512_set1_ps(1.002f),z=_mm512_setzero_ps();
    __m512 ck=_mm512_set1_ps(0.997729f),cb=_mm512_set1_ps(-0.004004f);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=16){
        __m512 vx=_mm512_loadu_ps(x+i);
        __m512 u=_mm512_fmadd_ps(c306,vx,c501);
        u=_mm512_max_ps(u,z); u=_mm512_min_ps(u,cm);
        __m512 r=_mm512_mul_ps(vx,u);
        _mm512_storeu_ps(out+i,_mm512_add_ps(_mm512_mul_ps(r,ck),cb));
    }
    for(long long i=vec;i<n;i++){
        float u=0.306923f*x[i]+0.501f;
        u=fminf(fmaxf(u,0.0f),1.002f);
        out[i]=0.997729f*(x[i]*u)-0.004004f;
    }
}

/* ================= GELU ================= */
void spur_batch_gelu(const double* x,double* out,long long n){
    if(__builtin_expect(spur_use_avx512(0),0)){
        spur_batch_gelu_avx512(x,out,n); return;
    }
    long long vec=n&~3LL;
    __m256d c306=_mm256_set1_pd(0.306923);
    __m256d c501=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002);
    __m256d z=_mm256_setzero_pd();
    __m256d ck=_mm256_set1_pd(0.997729);
    __m256d cb=_mm256_set1_pd(-0.004004);
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

/* ================= GELU v2 quintique (smoothstep certifie) ===============
   t = clip(0.200055340257*x + 0.5, 0, 1)
   GELU(x) = x*t^3*(6t^2 - 15t + 10) - 0.01104961
   Linf 0.0174 sur [-3.5,3.5] ; MSE 1.35e-4 sur [-4,4] ; queue bornee sur R. */
#define SPV2_GELUQ_A    0.200055340257
#define SPV2_GELUQ_OFF  0.01104961

double spur_k_gelu_quintic(double x){
    double t = SPV2_GELUQ_A*x + 0.5;
    if(t < 0.0) return -SPV2_GELUQ_OFF;
    if(t > 1.0) return x - SPV2_GELUQ_OFF;
    double t2 = t*t;
    return fma(x, t2*t*(6.0*t2 - 15.0*t + 10.0), -SPV2_GELUQ_OFF);
}

#ifdef SPIR_HAS_TARGET_ATTR
__attribute__((target("avx512f,avx512vl")))
#endif
static void spur_batch_gelu_quintic_avx512(const double* x,double* out,long long n){
    long long vec=n&~7LL;
    __m512d a=_mm512_set1_pd(SPV2_GELUQ_A), half=_mm512_set1_pd(0.5);
    __m512d off=_mm512_set1_pd(SPV2_GELUQ_OFF), one=_mm512_set1_pd(1.0);
    __m512d c6=_mm512_set1_pd(6.0),c15=_mm512_set1_pd(15.0),c10=_mm512_set1_pd(10.0);
    __m512d z=_mm512_setzero_pd();
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m512d vx=_mm512_loadu_pd(x+i);
        __m512d t=_mm512_add_pd(_mm512_mul_pd(a,vx),half);
        __mmask8 lo=_mm512_cmp_pd_mask(t,z,_CMP_LT_OQ);
        __mmask8 hi=_mm512_cmp_pd_mask(t,one,_CMP_GT_OQ);
        t=_mm512_min_pd(_mm512_max_pd(t,z),one);
        __m512d t2=_mm512_mul_pd(t,t);
        __m512d s=_mm512_mul_pd(_mm512_mul_pd(t2,t),
            _mm512_fmadd_pd(c6,t2,_mm512_sub_pd(c10,_mm512_mul_pd(c15,t))));
        __m512d r=_mm512_sub_pd(_mm512_mul_pd(vx,s),off);
        __m512d tail=_mm512_sub_pd(vx,off), sato=_mm512_sub_pd(z,off);
        r=_mm512_mask_blend_pd(hi,r,tail);
        r=_mm512_mask_blend_pd(lo,r,sato);
        _mm512_storeu_pd(out+i,r);
    }
    for(long long i=vec;i<n;i++) out[i]=spur_k_gelu_quintic(x[i]);
}

void spur_batch_gelu_quintic(const double* x,double* out,long long n){
    if(__builtin_expect(spur_use_avx512(0),0)){
        spur_batch_gelu_quintic_avx512(x,out,n); return;
    }
    long long vec=n&~3LL;
    __m256d a=_mm256_set1_pd(SPV2_GELUQ_A), half=_mm256_set1_pd(0.5);
    __m256d off=_mm256_set1_pd(SPV2_GELUQ_OFF), one=_mm256_set1_pd(1.0);
    __m256d c6=_mm256_set1_pd(6.0),c15=_mm256_set1_pd(15.0),c10=_mm256_set1_pd(10.0);
    __m256d z=_mm256_setzero_pd();
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d t=_mm256_add_pd(_mm256_mul_pd(a,vx),half);
        __m256d lo=_mm256_cmp_pd(t,z,_CMP_LT_OQ);
        __m256d hi=_mm256_cmp_pd(t,one,_CMP_GT_OQ);
        t=_mm256_min_pd(_mm256_max_pd(t,z),one);
        __m256d t2=_mm256_mul_pd(t,t);
        __m256d s=_mm256_mul_pd(_mm256_mul_pd(t2,t),
            _mm256_fmadd_pd(c6,t2,_mm256_sub_pd(c10,_mm256_mul_pd(c15,t))));
        __m256d r=_mm256_sub_pd(_mm256_mul_pd(vx,s),off);
        __m256d tail=_mm256_sub_pd(vx,off), sato=_mm256_sub_pd(z,off);
        r=_mm256_blendv_pd(r,tail,hi);
        r=_mm256_blendv_pd(r,sato,lo);
        _mm256_storeu_pd(out+i,r);
    }
    for(long long i=vec;i<n;i++) out[i]=spur_k_gelu_quintic(x[i]);
}

/* ================= GELU haute precision via erf_v2 =======================
   GELU = 0.5*x*(1+erf(x/sqrt2)). Reutilise le rationnel erf_v2 certifie
   (max_err 2.3e-5 sur [-6,6]) : precision mesuree linff 2.05e-5, MSE 8.3e-11
   sur grille 400k pts — ~850x plus precis que le smoothstep v2. 100% ALU.
   Clamp erf a 3.5 -> sature (GELU~x) pour x>~4.95, correct asymptotiquement. */
static const double SPV2_ERF_N[5]={1.12841751266903279e+00,1.83482771948230095e-01,
    5.73373674730976793e-02,2.48430060206610405e-03,3.72785350475749968e-06};
static const double SPV2_ERF_D[6]={1.0,4.96471589671860558e-01,1.14910282096263028e-01,
    1.61717422205343367e-02,1.86656477609649336e-04,-1.74401807407079551e-07};

double spur_k_gelu_erf(double x){
    double u=x*0.7071067811865476, cut=3.5;
    if(u>cut) return x;
    if(u<-cut) return 0.0;
    double y=u*u, pn=0.0, dn=0.0;
    for(int i=4;i>=0;i--) pn=fma(pn,y,SPV2_ERF_N[i]);
    for(int i=5;i>=0;i--) dn=fma(dn,y,SPV2_ERF_D[i]);
    double h=u*(pn/dn);
    return 0.5*x*(1.0+h);
}

void spur_batch_gelu_erf(const double* x,double* out,long long n){
    long long vec=n&~3LL;
    __m256d inv=_mm256_set1_pd(0.7071067811865476);
    __m256d chi=_mm256_set1_pd(3.5),clo=_mm256_set1_pd(-3.5);
    __m256d one=_mm256_set1_pd(1.0),zero=_mm256_setzero_pd(),half=_mm256_set1_pd(0.5);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d u=_mm256_mul_pd(vx,inv);
        __m256d gt=_mm256_cmp_pd(u,chi,_CMP_GT_OQ);
        __m256d lt=_mm256_cmp_pd(u,clo,_CMP_LT_OQ);
        __m256d y=_mm256_mul_pd(u,u);
        __m256d pn=_mm256_setzero_pd(),dn=_mm256_setzero_pd();
        for(int k=4;k>=0;k--) pn=_mm256_fmadd_pd(pn,y,_mm256_set1_pd(SPV2_ERF_N[k]));
        for(int k=5;k>=0;k--) dn=_mm256_fmadd_pd(dn,y,_mm256_set1_pd(SPV2_ERF_D[k]));
        __m256d h=_mm256_mul_pd(u,_mm256_div_pd(pn,dn));
        h=_mm256_blendv_pd(h,one,gt); h=_mm256_blendv_pd(h,_mm256_sub_pd(zero,one),lt);
        __m256d g=_mm256_mul_pd(_mm256_mul_pd(half,vx),_mm256_add_pd(one,h));
        _mm256_storeu_pd(out+i,g);
    }
    for(long long i=vec;i<n;i++) out[i]=spur_k_gelu_erf(x[i]);
}

/* ================= ERF ================= */
void spur_batch_erf(const double* x,double* out,long long n){
    if(__builtin_expect(spur_use_avx512(1),0)){
        spur_rat_avx512(x,out,n,-2.0,2.0,1.106774,0.034298,0.995,0.378089);
        return;
    }
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d cn=_mm256_set1_pd(1.106774),b0=_mm256_set1_pd(0.995);
    __m256d c3=_mm256_set1_pd(0.034298),b2=_mm256_set1_pd(0.378089);
    __m256d one=_mm256_set1_pd(1.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        /* num = y + c3*y^3 = y*(1 + c3*y^2) — PAS y + c3*y^2 ! */
        __m256d num=_mm256_mul_pd(y,_mm256_add_pd(one,_mm256_mul_pd(c3,t)));
        __m256d den=_mm256_add_pd(b0,_mm256_mul_pd(b2,t));
        _mm256_storeu_pd(out+i,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-2.0,fmin(2.0,x[i]));
        out[i]=1.106774*((xx+0.034298*xx*xx*xx)/(0.995+0.378089*xx*xx));
    }
}

/* ================= TANH ================= */
void spur_batch_tanh(const double* x,double* out,long long n){
    if(__builtin_expect(spur_use_avx512(1),0)){
        spur_rat_avx512(x,out,n,-3.0,3.0,0.900021,0.053639,0.90122,0.343141);
        return;
    }
    long long vec=n&~3LL;
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d cn=_mm256_set1_pd(0.900021),b0=_mm256_set1_pd(0.90122);
    __m256d c3=_mm256_set1_pd(0.053639),b2=_mm256_set1_pd(0.343141);
    __m256d one=_mm256_set1_pd(1.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,vx));
        __m256d t=_mm256_mul_pd(y,y);
        /* num = y + c3*y^3 = y*(1 + c3*y^2) — PAS y + c3*y^2 ! */
        __m256d num=_mm256_mul_pd(y,_mm256_add_pd(one,_mm256_mul_pd(c3,t)));
        __m256d den=_mm256_add_pd(b0,_mm256_mul_pd(b2,t));
        _mm256_storeu_pd(out+i,_mm256_mul_pd(cn,_mm256_div_pd(num,den)));
    }
    for(long long i=vec;i<n;i++){
        double xx=fmax(-3.0,fmin(3.0,x[i]));
        out[i]=0.900021*((xx+0.053639*xx*xx*xx)/(0.90122+0.343141*xx*xx));
    }
}

/* ================= LSE2 (hard-max) ================= */
void spur_batch_lse2(const double* a,const double* b,double* out,long long n){
    long long vec=n&~3LL;
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d va=_mm256_loadu_pd(a+i);
        __m256d vb=_mm256_loadu_pd(b+i);
        _mm256_storeu_pd(out+i,_mm256_max_pd(va,vb));
    }
    for(long long i=vec;i<n;i++)out[i]=fmax(a[i],b[i]);
}

/* ================= MATMUL (C = A . B^T, convention BLAS NT) ================
   C[i,j] = sum_k A[i,k]*B[j,k]  — C m x n, A m x k, B n x k (row-major).
   - blocage registres 4 lignes : chaque ligne de B sert 4 rangs de A,
     4 chaines FMA independantes ;
   - TUILLAGE CACHE (KC x NC) : les tuiles de B restent en L2 au lieu de
     re-streamer tout B depuis la RAM a chaque bloc de lignes de A.
   Valide err<=1e-15 vs numpy sur toutes tailles (tests/test_train.py).     */
static inline double hs256(__m256d v){
    __m128d lo=_mm256_castpd256_pd128(v),hi=_mm256_extractf128_pd(v,1);
    lo=_mm_add_pd(lo,hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo,_mm_unpackhi_pd(lo,lo)));
}
/* gelu_s : formule evoluee SPEAR (superspear ledger "gelu", MSE 5.3e-4,
   x6.57 vs GELU-tanh) -- https://github.com/bahira/superspear
   Contrat datasheet : err <= 0.079 sur [-2,2], sature proprement au-dela
   (err bornee ~0.002*|x|, verifie globalement jusqu'a +/-100). */
static inline double gelu_s(double x){
    double u=0.306923*x+0.501;
    if(u<0.0)u=0.0; if(u>1.002)u=1.002;
    return 0.997729*(x*u)-0.004004;
}

/* Check CPU : evite SIGILL si le CPU ne supporte pas AVX2+FMA.
   Les bindings Python appellent spur_cpu_ok() avant tout binding.        */
int spur_cpu_ok(void){
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}


static inline float hsum8(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v),hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi);
    __m128 s=_mm_movehdup_ps(lo);
    __m128 s2=_mm_add_ps(lo,s);
    __m128 shuf=_mm_add_ps(s2,_mm_movehl_ps(s2,s2));
    return _mm_cvtss_f32(shuf);
}
static inline float gelu_f32_scalar(float x){
    float u=0.306923f*x+0.501f;
    if(u<0.0f)u=0.0f; if(u>1.002f)u=1.002f;
    return 0.997729f*(x*u)-0.004004f;
}

#define MM_KC 256   /* profondeur tuile : bande A de 4*KC*8 = 8 Ko          */
#define MM_NC 256   /* largeur tuile  : tuile B de NC*KC*8  = 512 Ko (L2)   */

void spur_matmul_nt_legacy(const double* A,const double* B,double* C,
                           long long m,long long k,long long n){
    memset(C,0,(size_t)m*n*sizeof(double));
    for(long long kb=0;kb<k;kb+=MM_KC){
        const long long ke=(kb+MM_KC<k)?kb+MM_KC:k;
        for(long long jb=0;jb<n;jb+=MM_NC){
            const long long je=(jb+MM_NC<n)?jb+MM_NC:n;
            /* sweep complet des lignes de A sur la tuile B courante */
            #pragma omp parallel for schedule(static)
            for(long long i0=0;i0<m/4;i0++){
                const double* ar=A+(size_t)i0*4*k;
                double* cr=C+(size_t)i0*4*n;
                for(long long j=jb;j<je;j++){
                    const double* br=B+(size_t)j*k;
                    __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
                    __m256d v2=_mm256_setzero_pd(),v3=_mm256_setzero_pd();
                    long long q=kb;
                    for(;q+3<ke;q+=4){
                        __m256d bv=_mm256_loadu_pd(br+q);
                        v0=_mm256_fmadd_pd(_mm256_loadu_pd(ar+q),bv,v0);
                        v1=_mm256_fmadd_pd(_mm256_loadu_pd(ar+k+q),bv,v1);
                        v2=_mm256_fmadd_pd(_mm256_loadu_pd(ar+2*k+q),bv,v2);
                        v3=_mm256_fmadd_pd(_mm256_loadu_pd(ar+3*k+q),bv,v3);
                    }
                    double d0=0,d1=0,d2=0,d3=0;
                    for(;q<ke;q++){
                        double b=br[q];
                        d0+=ar[q]*b; d1+=ar[k+q]*b;
                        d2+=ar[2*k+q]*b; d3+=ar[3*k+q]*b;
                    }
                    cr[j]      +=hs256(v0)+d0;
                    cr[n+j]    +=hs256(v1)+d1;
                    cr[2*n+j]  +=hs256(v2)+d2;
                    cr[3*n+j]  +=hs256(v3)+d3;
                }
            }
            /* queue : lignes restantes m%4 (accumulation identique) */
            #pragma omp parallel for schedule(static)
            for(long long i=(m/4)*4;i<m;i++){
                const double* xr=A+(size_t)i*k;
                double* tr=C+(size_t)i*n;
                for(long long j=jb;j<je;j++){
                    const double* wr=B+(size_t)j*k;
                    __m256d acc=_mm256_setzero_pd();
                    long long q=kb;
                    for(;q+3<ke;q+=4)
                        acc=_mm256_fmadd_pd(_mm256_loadu_pd(xr+q),
                                            _mm256_loadu_pd(wr+q),acc);
                    double s=hs256(acc);
                    for(;q<ke;q++) s+=xr[q]*wr[q];
                    tr[j]+=s;
                }
            }
        }
    }
}

/* Variante fusionnee : C = gelu(A . B^T). La gelu est non-lineaire :
   k ne peut PAS etre coupe -> blocage colonnes seul (tuile B de NC x k
   reste chaude pendant le sweep des lignes de A).                          */
void spur_matmul_nt_gelu_legacy(const double* A,const double* B,
                                const double* bias,double* C,
                                long long m,long long k,long long n){
    /* bias : NULL = sans biais, sinon tableau de n (ajoute AVANT gelu)   */
    for(long long jb=0;jb<n;jb+=MM_NC){
        const long long je=(jb+MM_NC<n)?jb+MM_NC:n;
        #pragma omp parallel for schedule(static)
        for(long long i0=0;i0<m/4;i0++){
            const double* ar=A+(size_t)i0*4*k;
            double* cr=C+(size_t)i0*4*n;
            for(long long j=jb;j<je;j++){
                const double* br=B+(size_t)j*k;
                __m256d v0=_mm256_setzero_pd(),v1=_mm256_setzero_pd();
                __m256d v2=_mm256_setzero_pd(),v3=_mm256_setzero_pd();
                long long q=0;
                for(;q+3<k;q+=4){
                    __m256d bv=_mm256_loadu_pd(br+q);
                    v0=_mm256_fmadd_pd(_mm256_loadu_pd(ar+q),bv,v0);
                    v1=_mm256_fmadd_pd(_mm256_loadu_pd(ar+k+q),bv,v1);
                    v2=_mm256_fmadd_pd(_mm256_loadu_pd(ar+2*k+q),bv,v2);
                    v3=_mm256_fmadd_pd(_mm256_loadu_pd(ar+3*k+q),bv,v3);
                }
                double d0=0,d1=0,d2=0,d3=0;
                for(;q<k;q++){
                    double b=br[q];
                    d0+=ar[q]*b; d1+=ar[k+q]*b;
                    d2+=ar[2*k+q]*b; d3+=ar[3*k+q]*b;
                }
                cr[j]=gelu_s(hs256(v0)+d0+(bias?bias[j]:0.0));
                cr[n+j]=gelu_s(hs256(v1)+d1+(bias?bias[j]:0.0));
                cr[2*n+j]=gelu_s(hs256(v2)+d2+(bias?bias[j]:0.0));
                cr[3*n+j]=gelu_s(hs256(v3)+d3+(bias?bias[j]:0.0));
            }
        }
        #pragma omp parallel for schedule(static)
        for(long long i=(m/4)*4;i<m;i++){
            const double* xr=A+(size_t)i*k;
            double* tr=C+(size_t)i*n;
            for(long long j=jb;j<je;j++){
                const double* wr=B+(size_t)j*k;
                __m256d acc=_mm256_setzero_pd();
                long long q=0;
                for(;q+3<k;q+=4)
                    acc=_mm256_fmadd_pd(_mm256_loadu_pd(xr+q),
                                        _mm256_loadu_pd(wr+q),acc);
                double s=hs256(acc);
                for(;q<k;q++) s+=xr[q]*wr[q];
                tr[j]=gelu_s(s+(bias?bias[j]:0.0));
            }
        }
    }
}
/* ================= GELU BACKWARD (training) ================================
   dX = dY * gelu'(x) pour r(x)=ck*(x*clamp(c306*x+c5,0,cm))+cb :
   interieur : ck*(u + c306*x) ; u<=0 : ~0 ; u>=cm : ck*cm.               */
void spur_batch_gelu_backward(const double* dY,const double* x,double* dX,
                              long long n){
    long long vec=n&~3LL;
    const double c306=0.306923,c5=0.501,cm=1.002,ck=0.997729;
    __m256d vc=_mm256_set1_pd(c306),vb=_mm256_set1_pd(c5);
    __m256d vm=_mm256_set1_pd(cm),vk=_mm256_set1_pd(ck),vz=_mm256_setzero_pd();
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d vd=_mm256_loadu_pd(dY+i);
        __m256d u=_mm256_fmadd_pd(vc,vx,vb);
        __m256d mi=_mm256_and_pd(
            _mm256_cmp_pd(u,vz,_CMP_GT_OQ),
            _mm256_cmp_pd(u,vm,_CMP_LT_OQ));
        __m256d uc=_mm256_max_pd(vz,_mm256_min_pd(vm,u));
        /* derivee = ck*(uc + x*c306*mask_interieur) */
        __m256d g=_mm256_mul_pd(vk,
            _mm256_add_pd(uc,_mm256_and_pd(_mm256_mul_pd(vx,vc),mi)));
        _mm256_storeu_pd(dX+i,_mm256_mul_pd(vd,g));
    }
    for(long long i=vec;i<n;i++){
        double u=c306*x[i]+c5;
        double g=(u<=0.0)?0.0:((u>=cm)?ck*cm:ck*(u+c306*x[i]));
        dX[i]=dY[i]*g;
    }
}

/* ================= BACKWARD erf / tanh / sigmoid ==========================
   Derivees exactes des approximations rationnelles certifiees :
   r(x) = cn*(x+c3*x^3)/(b0+b2*x^2)  =>
   r'(x) = cn*[(1+3c3x^2)(b0+b2x^2) - (x+c3x^3)*2b2x] / (b0+b2x^2)^2
   clamp [lo,hi] : derivee nulle hors bornes.                              */
static void rat_backward(const double* dY,const double* x,double* out,
                         long long n,
                         double cn,double c3,double b0,double b2,
                         double lo,double hi){
    long long vec=n&~3LL;
    __m256d vcn=_mm256_set1_pd(cn),vc3=_mm256_set1_pd(c3);
    __m256d vb0=_mm256_set1_pd(b0),vb2=_mm256_set1_pd(b2);
    __m256d vlo=_mm256_set1_pd(lo),vhi=_mm256_set1_pd(hi);
    __m256d two=_mm256_set1_pd(2.0),three=_mm256_set1_pd(3.0);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=4){
        __m256d vx=_mm256_loadu_pd(x+i);
        __m256d vd=_mm256_loadu_pd(dY+i);
        __m256d y=_mm256_max_pd(vlo,_mm256_min_pd(vhi,vx));
        /* dans la zone clampee : derivee nulle */
        __m256d inside=_mm256_and_pd(
            _mm256_cmp_pd(vx,vlo,_CMP_GT_OQ),
            _mm256_cmp_pd(vx,vhi,_CMP_LT_OQ));
        __m256d x2=_mm256_mul_pd(y,y);
        __m256d den=_mm256_add_pd(vb0,_mm256_mul_pd(vb2,x2));
        __m256d num=_mm256_add_pd(y,_mm256_mul_pd(vc3,
                    _mm256_mul_pd(y,x2)));
        __m256d np_=_mm256_add_pd(_mm256_set1_pd(1.0),
                          _mm256_mul_pd(three,_mm256_mul_pd(vc3,x2)));
        __m256d dp=_mm256_mul_pd(two,_mm256_mul_pd(vb2,y));
        __m256d g=_mm256_div_pd(
            _mm256_sub_pd(_mm256_mul_pd(np_,den),_mm256_mul_pd(num,dp)),
            _mm256_mul_pd(den,den));
        /* masque : AND (pas MUL ! les bits du masque sont 0/all-ones) */
        g=_mm256_and_pd(g,inside);
        _mm256_storeu_pd(out+i,_mm256_mul_pd(vcn,
            _mm256_mul_pd(vd,g)));
    }
    for(long long i=vec;i<n;i++){
        double xi=x[i];
        if(xi<=lo||xi>=hi){ out[i]=0.0; continue; }
        double xc=xi<lo?lo:(xi>hi?hi:xi);
        double x2=xc*xc;
        double num=xc+c3*xc*x2;
        double den=b0+b2*x2;
        double np_=1.0+3.0*c3*x2;
        double dp=2.0*b2*xc;
        double g=(np_*den-num*dp)/(den*den);
        out[i]=dY[i]*cn*g;
    }
}

void spur_batch_erf_backward(const double* dY,const double* x,double* out,
                             long long n){
    rat_backward(dY,x,out,n,1.106774,0.034298,0.995,0.378089,-2.0,2.0);
}
void spur_batch_tanh_backward(const double* dY,const double* x,double* out,
                              long long n){
    rat_backward(dY,x,out,n,0.900021,0.053639,0.90122,0.343141,-3.0,3.0);
}
/* sigmoid = 0.5 + 0.5*tanh_a(x/2) => s' = 0.25*tanh_a'(x/2) */
void spur_batch_sigmoid_backward(const double* dY,const double* x,
                                 double* out,long long n){
    /* implementation directe via tanh_backward sur x/2 */
    {
        double* xs=(double*)malloc((size_t)n*sizeof(double));
        double* g =(double*)malloc((size_t)n*sizeof(double));
        if(!xs||!g){ free(xs); free(g);
            for(long long i=0;i<n;i++) out[i]=0.0; return; }
        for(long long i=0;i<n;i++) xs[i]=0.5*x[i];
        spur_batch_tanh_backward(dY,xs,g,n);
        for(long long i=0;i<n;i++) out[i]=0.25*g[i];
        free(xs); free(g);
    }
}

/* ================= FLOAT32 — 8 lanes/vector, ~x2 debit ==================== */
void spur_batch_gelu_f32(const float* x,float* out,long long n){
    if(__builtin_expect(spur_use_avx512(0),0)){
        spur_batch_gelu_f32_avx512(x,out,n); return;
    }
    long long vec=n&~7LL;
    __m256 c306=_mm256_set1_ps(0.306923f),c501=_mm256_set1_ps(0.501f);
    __m256 cm=_mm256_set1_ps(1.002f),z=_mm256_setzero_ps();
    __m256 ck=_mm256_set1_ps(0.997729f),cb=_mm256_set1_ps(-0.004004f);
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m256 vx=_mm256_loadu_ps(x+i);
        __m256 u=_mm256_fmadd_ps(c306,vx,c501);
        u=_mm256_max_ps(u,z); u=_mm256_min_ps(u,cm);
        __m256 r=_mm256_mul_ps(vx,u);
        _mm256_storeu_ps(out+i,_mm256_add_ps(_mm256_mul_ps(r,ck),cb));
    }
    for(long long i=vec;i<n;i++){
        float u=0.306923f*x[i]+0.501f;
        u=fminf(fmaxf(u,0.0f),1.002f);
        out[i]=0.997729f*(x[i]*u)-0.004004f;
    }
}

void spur_matmul_nt_f32_legacy(const float* A,const float* B,float* C,
                        long long m,long long k,long long n){
    memset(C,0,(size_t)m*n*sizeof(float));
    for(long long kb=0;kb<k;kb+=MM_KC){
        const long long ke=(kb+MM_KC<k)?kb+MM_KC:k;
        for(long long jb=0;jb<n;jb+=MM_NC){
            const long long je=(jb+MM_NC<n)?jb+MM_NC:n;
            #pragma omp parallel for schedule(static)
            for(long long i0=0;i0<m/8;i0++){   /* 8 lignes f32 par bloc */
                const float* ar=A+(size_t)i0*8*k;
                float* cr=C+(size_t)i0*8*n;
                for(long long j=jb;j<je;j++){
                    const float* br=B+(size_t)j*k;
                    __m256 v0=_mm256_setzero_ps(),v1=_mm256_setzero_ps();
                    __m256 v2=_mm256_setzero_ps(),v3=_mm256_setzero_ps();
                    __m256 v4=_mm256_setzero_ps(),v5=_mm256_setzero_ps();
                    __m256 v6=_mm256_setzero_ps(),v7=_mm256_setzero_ps();
                    long long q=kb;
                    for(;q+7<ke;q+=8){
                        __m256 av0=_mm256_loadu_ps(ar+q);
                        __m256 av1=_mm256_loadu_ps(ar+k+q);
                        __m256 av2=_mm256_loadu_ps(ar+2*k+q);
                        __m256 av3=_mm256_loadu_ps(ar+3*k+q);
                        __m256 av4=_mm256_loadu_ps(ar+4*k+q);
                        __m256 av5=_mm256_loadu_ps(ar+5*k+q);
                        __m256 av6=_mm256_loadu_ps(ar+6*k+q);
                        __m256 av7=_mm256_loadu_ps(ar+7*k+q);
                        __m256 bv=_mm256_loadu_ps(br+q);
                        v0=_mm256_fmadd_ps(av0,bv,v0);
                        v1=_mm256_fmadd_ps(av1,bv,v1);
                        v2=_mm256_fmadd_ps(av2,bv,v2);
                        v3=_mm256_fmadd_ps(av3,bv,v3);
                        v4=_mm256_fmadd_ps(av4,bv,v4);
                        v5=_mm256_fmadd_ps(av5,bv,v5);
                        v6=_mm256_fmadd_ps(av6,bv,v6);
                        v7=_mm256_fmadd_ps(av7,bv,v7);
                    }
                    /* queue k%8 : sans elle les k non multiples de 8
                       perdaient jusqu'a 7 produits par tuile (resultat faux) */
                    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0;
                    for(;q<ke;q++){
                        const float b=br[q];
                        d0+=ar[q]*b;       d1+=ar[k+q]*b;
                        d2+=ar[2*k+q]*b;   d3+=ar[3*k+q]*b;
                        d4+=ar[4*k+q]*b;   d5+=ar[5*k+q]*b;
                        d6+=ar[6*k+q]*b;   d7+=ar[7*k+q]*b;
                    }
                    /* reduction horizontale f32 */
                    cr[j]     +=hsum8(v0)+d0; cr[n+j]    +=hsum8(v1)+d1;
                    cr[2*n+j] +=hsum8(v2)+d2; cr[3*n+j]  +=hsum8(v3)+d3;
                    cr[4*n+j] +=hsum8(v4)+d4; cr[5*n+j]  +=hsum8(v5)+d5;
                    cr[6*n+j] +=hsum8(v6)+d6; cr[7*n+j]  +=hsum8(v7)+d7;
                }
            }
            #pragma omp parallel for schedule(static)
            for(long long i=(m/8)*8;i<m;i++){
                const float* xr=A+(size_t)i*k;
                float* tr=C+(size_t)i*n;
                for(long long j=jb;j<je;j++){
                    const float* wr=B+(size_t)j*k;
                    __m256 acc=_mm256_setzero_ps();
                    long long q=kb;
                    for(;q+7<ke;q+=8)
                        acc=_mm256_fmadd_ps(_mm256_loadu_ps(xr+q),
                                            _mm256_loadu_ps(wr+q),acc);
                    float s=hsum8(acc);
                    for(;q<ke;q++) s+=xr[q]*wr[q];
                    tr[j]+=s;
                }
            }
        }
    }
}

void spur_matmul_nt_gelu_f32_legacy(const float* A,const float* B,
                                    const float* bias,float* C,
                                    long long m,long long k,long long n){
    for(long long jb=0;jb<n;jb+=MM_NC){
        const long long je=(jb+MM_NC<n)?jb+MM_NC:n;
        #pragma omp parallel for schedule(static)
        for(long long i0=0;i0<m/8;i0++){
            const float* ar=A+(size_t)i0*8*k;
            float* cr=C+(size_t)i0*8*n;
            for(long long j=jb;j<je;j++){
                const float* br=B+(size_t)j*k;
                /* blocage registres 8 lignes : la ligne de B est chargee UNE
                   fois et alimente 8 chaines FMA independantes (avant : 8
                   relectures completes de B -> ~x8 de trafic memoire). */
                __m256 v0=_mm256_setzero_ps(),v1=_mm256_setzero_ps();
                __m256 v2=_mm256_setzero_ps(),v3=_mm256_setzero_ps();
                __m256 v4=_mm256_setzero_ps(),v5=_mm256_setzero_ps();
                __m256 v6=_mm256_setzero_ps(),v7=_mm256_setzero_ps();
                long long q=0;
                for(;q+7<k;q+=8){
                    __m256 bv=_mm256_loadu_ps(br+q);
                    v0=_mm256_fmadd_ps(_mm256_loadu_ps(ar+q),bv,v0);
                    v1=_mm256_fmadd_ps(_mm256_loadu_ps(ar+k+q),bv,v1);
                    v2=_mm256_fmadd_ps(_mm256_loadu_ps(ar+2*k+q),bv,v2);
                    v3=_mm256_fmadd_ps(_mm256_loadu_ps(ar+3*k+q),bv,v3);
                    v4=_mm256_fmadd_ps(_mm256_loadu_ps(ar+4*k+q),bv,v4);
                    v5=_mm256_fmadd_ps(_mm256_loadu_ps(ar+5*k+q),bv,v5);
                    v6=_mm256_fmadd_ps(_mm256_loadu_ps(ar+6*k+q),bv,v6);
                    v7=_mm256_fmadd_ps(_mm256_loadu_ps(ar+7*k+q),bv,v7);
                }
                float s[8];
                s[0]=hsum8(v0); s[1]=hsum8(v1); s[2]=hsum8(v2); s[3]=hsum8(v3);
                s[4]=hsum8(v4); s[5]=hsum8(v5); s[6]=hsum8(v6); s[7]=hsum8(v7);
                const float bj=bias?bias[j]:0.0f;
                for(long long r2=0;r2<8;r2++){
                    const float* arr=ar+(size_t)r2*k;
                    float sr=s[r2];
                    for(long long t=q;t<k;t++) sr+=arr[t]*br[t];  /* queue k%8 */
                    cr[(size_t)r2*n+j]=gelu_f32_scalar(sr+bj);
                }
            }
        }
        #pragma omp parallel for schedule(static)
        for(long long i=(m/8)*8;i<m;i++){
            const float* xr=A+(size_t)i*k;
            float* tr=C+(size_t)i*n;
            for(long long j=jb;j<je;j++){
                const float* wr=B+(size_t)j*k;
                __m256 acc=_mm256_setzero_ps();
                long long q=0;
                for(;q+7<k;q+=8)
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(xr+q),
                                        _mm256_loadu_ps(wr+q),acc);
                float s=hsum8(acc);
                for(;q<k;q++) s+=xr[q]*wr[q];
                tr[j]=gelu_f32_scalar(s+(bias?bias[j]:0.0f));
            }
        }
    }
}

void spur_batch_gelu_backward_f32(const float* dY,const float* x,float* dX,
                                  long long n){
    long long vec=n&~7LL;
    const float c306=0.306923f,c5=0.501f,cm=1.002f,ck=0.997729f;
    __m256 vc=_mm256_set1_ps(c306),vb=_mm256_set1_ps(c5);
    __m256 vm=_mm256_set1_ps(cm),vk=_mm256_set1_ps(ck);
    __m256 vz=_mm256_setzero_ps();
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<vec;i+=8){
        __m256 vx=_mm256_loadu_ps(x+i);
        __m256 vd=_mm256_loadu_ps(dY+i);
        __m256 u=_mm256_fmadd_ps(vc,vx,vb);
        __m256 mi=_mm256_and_ps(
            _mm256_cmp_ps(u,vz,_CMP_GT_OQ),
            _mm256_cmp_ps(u,vm,_CMP_LT_OQ));
        __m256 uc=_mm256_max_ps(vz,_mm256_min_ps(vm,u));
        __m256 g=_mm256_mul_ps(vk,
            _mm256_add_ps(uc,_mm256_and_ps(_mm256_mul_ps(vx,vc),mi)));
        _mm256_storeu_ps(dX+i,_mm256_mul_ps(vd,g));
    }
    for(long long i=vec;i<n;i++){
        float u=c306*x[i]+c5;
        float g=(u<=0.0f)?0.0f:((u>=cm)?ck*cm:ck*(u+c306*x[i]));
        dX[i]=dY[i]*g;
    }
}

/* ============================================================================
   GEMM NT v2 — noyaux issus de l'autotuning (docs/EXPERIMENTS.md)
   ----------------------------------------------------------------------------
   902 variantes compilees, verifiees et chronometrees sur cette machine
   (2 vCPU AVX2/FMA) ont degage deux familles gagnantes, complementaires :

   [P] "pack"  micro-noyau broadcast facon BLIS : A et B sont packes en
       panneaux contigus (Bp transpose la convention NT), la maille interne
       accumule MR x NR/VL registres -> zero reduction horizontale.
       Champions mesures : f64 MR=4 NR=12 KC=576 MC=64 NC=2048
                           f32 MR=4 NR=24 KC=384 MC=128 NC=2048
       -> f64 41.8 GF (x2.02 vs legacy), f32 83.8 GF (x1.43) en mono-thread.

   [D] "dot"   bloc de produits scalaires, **zero packing** : en NT, k est
       contigu des deux cotes, donc MR lignes de A x NR lignes de B tiennent
       dans 12 accumulateurs (MR=3, NR=4) et une seule hsum finale par case.
       -> f64 38.7 GF / f32 75.9 GF, mais surtout imbattable quand une des
          dimensions est petite (le packing y coute plus qu'il ne rapporte).

   Aiguillage (mesure, cf. results/shapes.csv) : min(m,n) <= 32, ou tuile
   64x64 et moins -> [D] ; sinon -> [P].
   SPUR_MM_LEGACY=1 force l'ancien noyau (garde-fou A/B en production).
   ========================================================================== */

#if defined(_WIN32)
#include <malloc.h>
#define SPUR_AALLOC(bytes) _aligned_malloc((size_t)(bytes), 64)
#define SPUR_AFREE(p)      _aligned_free(p)
#else
static void* spur_aalloc(size_t bytes){
    void* p = NULL;
    return posix_memalign(&p, 64, bytes) ? NULL : p;
}
#define SPUR_AALLOC(bytes) spur_aalloc((size_t)(bytes))
#define SPUR_AFREE(p)      free(p)
#endif

static int spur_mm_legacy(void){
    static int cached = -1;
    if (cached < 0){
        const char* e = getenv("SPUR_MM_LEGACY");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* MC effectif : le parallelisme du noyau [P] porte sur les blocs de MC lignes.
   Si m <= MC il n'y a qu'un bloc, donc un seul thread travaille : on retrecit
   MC pour garantir au moins un bloc par thread (mesure : f32 128x128x128 a
   2 threads passe de x0.78 a x1.2+ du legacy).                              */
static inline long long spur_mm_mc(long long m, int nth, long long mc_max,
                                   long long mr){
    if (nth <= 1) return mc_max;
    long long want = (m + nth - 1) / nth;
    want = ((want + mr - 1) / mr) * mr;
    if (want < mr) want = mr;
    return (want < mc_max) ? want : mc_max;
}

static int spur_mm_threads(void){
#ifdef _OPENMP
    /* Appele depuis une region parallele (ex. attention multi-tetes) : le GEMM
       s'execute alors sur un seul thread, il ne faut ni retrecir MC ni
       reserver des tampons pour des threads qui n'existent pas ici. */
    if (omp_in_parallel()) return 1;
    int t = omp_get_max_threads();
    return t < 1 ? 1 : t;
#else
    return 1;
#endif
}

/* Aiguillage [L]egacy / [P]ack / [D]ot.
   Calibre par recherche exhaustive sur 149 formes x 2 precisions mesurees
   dans la .so livree (experiments/sweep_routing.py -> results/routing.csv).
   Contrainte imposee a la recherche : **aucune regression** (pire cas >= 1.00
   du noyau legacy) ; sous cette contrainte on maximise le gain moyen sur les
   formes de vraie taille (m*k*n >= 2^23) -> f64 x1.65, f32 x1.40.
   Les seuils different par precision parce que le micro-noyau f32 est deux
   fois plus large (NR=24 contre 12) : il lui faut plus de colonnes pour
   amortir le packing.                                                       */
#define SPUR_MM_ROUTE_LEGACY 0
#define SPUR_MM_ROUTE_DOT    1
#define SPUR_MM_ROUTE_PACK   2

static inline int spur_mm_route(long long m, long long k, long long n,
                                long long min_m, long long min_n,
                                long long dot_lo, long long min_k_tile){
    if (m < min_m || n < min_n) return SPUR_MM_ROUTE_LEGACY;  /* trop etroit */
    if (m <= 64 && n <= 64)     /* petite tuile : les MR*NR hsum finales ne sont
                                   amorties que si k est assez profond
                                   (results/smalltile.csv) */
        return (k >= min_k_tile) ? SPUR_MM_ROUTE_DOT : SPUR_MM_ROUTE_LEGACY;
    const long long lo = (m < n) ? m : n;
    if (lo <= dot_lo) return SPUR_MM_ROUTE_DOT;  /* dimension etroite : ne pas packer */
    return SPUR_MM_ROUTE_PACK;
}

/* seuils (min_m, min_n, dot_lo, min_k_tile) retenus par la calibration */
#define SPUR_MM_F64_MIN_M 24
#define SPUR_MM_F64_MIN_N 32
#define SPUR_MM_F64_DOTLO 32
#define SPUR_MM_F64_MINKT 96
#define SPUR_MM_F32_MIN_M 64
#define SPUR_MM_F32_MIN_N 48
#define SPUR_MM_F32_DOTLO 32
#define SPUR_MM_F32_MINKT 512

/* ------------------------------ f64 [P] ---------------------------------- */
#define D_MR  4
#define D_NR  12
#define D_NV  3                       /* D_NR / 4 lanes */
#define D_KC  576
#define D_MC  64
#define D_NC  2048
#define D_MCP (((D_MC)+D_MR-1)/D_MR*D_MR)
#define D_NCP (((D_NC)+D_NR-1)/D_NR*D_NR)

static void spur_pack_a_d(const double* restrict A, long long lda,
                          double* restrict Ap, long long mc, long long kc){
    for (long long i = 0; i < mc; i += D_MR){
        const long long rows = (mc - i < D_MR) ? (mc - i) : D_MR;
        double* dst = Ap + i * kc;
        for (long long q = 0; q < kc; q++){
            for (long long r = 0; r < rows; r++)  dst[q*D_MR + r] = A[(i+r)*lda + q];
            for (long long r = rows; r < D_MR; r++) dst[q*D_MR + r] = 0.0;
        }
    }
}

static void spur_pack_b_d(const double* restrict B, long long ldb,
                          double* restrict Bp, long long nc, long long kc){
    for (long long j = 0; j < nc; j += D_NR){
        const long long cols = (nc - j < D_NR) ? (nc - j) : D_NR;
        double* dst = Bp + j * kc;
        for (long long q = 0; q < kc; q++){
            for (long long c = 0; c < cols; c++)  dst[q*D_NR + c] = B[(j+c)*ldb + q];
            for (long long c = cols; c < D_NR; c++) dst[q*D_NR + c] = 0.0;
        }
    }
}

static inline void spur_micro_d(long long kc, const double* restrict Ap,
                                const double* restrict Bp, double* restrict C,
                                long long ldc, int accumulate){
    __m256d c[D_MR][D_NV];
    for (int i = 0; i < D_MR; i++)
        for (int j = 0; j < D_NV; j++) c[i][j] = _mm256_setzero_pd();

    for (long long q = 0; q < kc; q++){
        __m256d b[D_NV];
        for (int j = 0; j < D_NV; j++) b[j] = _mm256_loadu_pd(Bp + q*D_NR + j*4);
        for (int i = 0; i < D_MR; i++){
            __m256d a = _mm256_broadcast_sd(Ap + q*D_MR + i);
            for (int j = 0; j < D_NV; j++) c[i][j] = _mm256_fmadd_pd(a, b[j], c[i][j]);
        }
    }
    if (accumulate){
        for (int i = 0; i < D_MR; i++)
            for (int j = 0; j < D_NV; j++)
                _mm256_storeu_pd(C + i*ldc + j*4,
                                 _mm256_add_pd(_mm256_loadu_pd(C + i*ldc + j*4), c[i][j]));
    } else {
        for (int i = 0; i < D_MR; i++)
            for (int j = 0; j < D_NV; j++) _mm256_storeu_pd(C + i*ldc + j*4, c[i][j]);
    }
}

/* exporte : permet l'A/B des trois noyaux depuis les bancs (experiments/) */
void spur_gemm_pack_f64(const double* A, const double* B, double* C,
                               long long m, long long k, long long n){
    const int nth = spur_mm_threads();
    double* Bp = (double*)SPUR_AALLOC(sizeof(double) * (size_t)D_KC * D_NCP);
    double* Ap = (double*)SPUR_AALLOC(sizeof(double) * (size_t)nth * D_KC * D_MCP);
    if (!Bp || !Ap){                       /* OOM : repli sur le noyau legacy */
        if (Bp) SPUR_AFREE(Bp);
        if (Ap) SPUR_AFREE(Ap);
        spur_matmul_nt_legacy(A, B, C, m, k, n);
        return;
    }
    const long long MCE = spur_mm_mc(m, nth, D_MC, D_MR);
    for (long long jc = 0; jc < n; jc += D_NC){
        const long long nc = (n - jc < D_NC) ? (n - jc) : D_NC;
        for (long long pc = 0; pc < k; pc += D_KC){
            const long long kc = (k - pc < D_KC) ? (k - pc) : D_KC;
            const int accumulate = (pc != 0);
            spur_pack_b_d(B + jc*k + pc, k, Bp, nc, kc);
            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                double* Apt = Ap + (size_t)tid * D_KC * D_MCP;
                #pragma omp for schedule(static)
                for (long long ic = 0; ic < m; ic += MCE){
                    const long long mc = (m - ic < MCE) ? (m - ic) : MCE;
                    spur_pack_a_d(A + ic*k + pc, k, Apt, mc, kc);
                    for (long long jr = 0; jr < nc; jr += D_NR){
                        const long long cols = (nc - jr < D_NR) ? (nc - jr) : D_NR;
                        for (long long ir = 0; ir < mc; ir += D_MR){
                            const long long rows = (mc - ir < D_MR) ? (mc - ir) : D_MR;
                            double* cptr = C + (ic + ir)*n + jc + jr;
                            __builtin_prefetch(cptr + 4*n, 1, 1);
                            if (rows == D_MR && cols == D_NR){
                                spur_micro_d(kc, Apt + ir*kc, Bp + jr*kc, cptr, n, accumulate);
                            } else {
                                double tmp[D_MR * D_NR] __attribute__((aligned(64)));
                                spur_micro_d(kc, Apt + ir*kc, Bp + jr*kc, tmp, D_NR, 0);
                                for (long long r = 0; r < rows; r++)
                                    for (long long c = 0; c < cols; c++)
                                        cptr[r*n + c] = accumulate ? cptr[r*n + c] + tmp[r*D_NR + c]
                                                                   : tmp[r*D_NR + c];
                            }
                        }
                    }
                }
            }
        }
    }
    SPUR_AFREE(Ap);
    SPUR_AFREE(Bp);
}

/* ------------------------------ f64 [D] ---------------------------------- */
#define DD_MR 3
#define DD_NR 4
#define DD_KC 1024
#define DD_NC 96

static inline void spur_dot_tile_d(const double* restrict ap, const double* restrict bp,
                                   double* restrict cp, long long lda, long long ldb,
                                   long long ldc, long long rows, long long cols,
                                   long long kc, int accumulate){
    if (rows == DD_MR && cols == DD_NR){
        __m256d acc[DD_MR][DD_NR];
        for (int i = 0; i < DD_MR; i++)
            for (int j = 0; j < DD_NR; j++) acc[i][j] = _mm256_setzero_pd();
        long long q = 0;
        for (; q + 3 < kc; q += 4){
            __m256d av[DD_MR];
            for (int i = 0; i < DD_MR; i++) av[i] = _mm256_loadu_pd(ap + i*lda + q);
            for (int j = 0; j < DD_NR; j++){
                const __m256d bv = _mm256_loadu_pd(bp + j*ldb + q);
                for (int i = 0; i < DD_MR; i++) acc[i][j] = _mm256_fmadd_pd(av[i], bv, acc[i][j]);
            }
        }
        double tail[DD_MR][DD_NR];
        for (int i = 0; i < DD_MR; i++)
            for (int j = 0; j < DD_NR; j++) tail[i][j] = 0.0;
        for (long long t = q; t < kc; t++)
            for (int i = 0; i < DD_MR; i++){
                const double a = ap[i*lda + t];
                for (int j = 0; j < DD_NR; j++) tail[i][j] += a * bp[j*ldb + t];
            }
        for (int i = 0; i < DD_MR; i++)
            for (int j = 0; j < DD_NR; j++){
                const double s = hs256(acc[i][j]) + tail[i][j];
                cp[i*ldc + j] = accumulate ? cp[i*ldc + j] + s : s;
            }
        return;
    }
    /* bord : produit scalaire vectorise case par case (pas de padding) */
    for (long long i = 0; i < rows; i++)
        for (long long j = 0; j < cols; j++){
            __m256d v = _mm256_setzero_pd();
            long long q = 0;
            for (; q + 3 < kc; q += 4)
                v = _mm256_fmadd_pd(_mm256_loadu_pd(ap + i*lda + q),
                                    _mm256_loadu_pd(bp + j*ldb + q), v);
            double s = hs256(v);
            for (; q < kc; q++) s += ap[i*lda + q] * bp[j*ldb + q];
            cp[i*ldc + j] = accumulate ? cp[i*ldc + j] + s : s;
        }
}

/* exporte : permet l'A/B des trois noyaux depuis les bancs (experiments/) */
void spur_gemm_dot_f64(const double* A, const double* B, double* C,
                              long long m, long long k, long long n){
    for (long long pc = 0; pc < k; pc += DD_KC){
        const long long kc = (k - pc < DD_KC) ? (k - pc) : DD_KC;
        const int accumulate = (pc != 0);
        for (long long jc = 0; jc < n; jc += DD_NC){
            const long long ncz = (n - jc < DD_NC) ? (n - jc) : DD_NC;
            /* on parallelise la dimension qui offre le plus de tuiles */
            if (m / DD_MR >= ncz / DD_NR){
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < m; i += DD_MR){
                    const long long rows = (m - i < DD_MR) ? (m - i) : DD_MR;
                    for (long long j = 0; j < ncz; j += DD_NR){
                        const long long cols = (ncz - j < DD_NR) ? (ncz - j) : DD_NR;
                        __builtin_prefetch(A + (i + DD_MR)*k + pc, 0, 3);
                        spur_dot_tile_d(A + i*k + pc, B + (jc + j)*k + pc,
                                        C + i*n + jc + j, k, k, n, rows, cols, kc, accumulate);
                    }
                }
            } else {
                for (long long i = 0; i < m; i += DD_MR){
                    const long long rows = (m - i < DD_MR) ? (m - i) : DD_MR;
                    #pragma omp parallel for schedule(static)
                    for (long long j = 0; j < ncz; j += DD_NR){
                        const long long cols = (ncz - j < DD_NR) ? (ncz - j) : DD_NR;
                        spur_dot_tile_d(A + i*k + pc, B + (jc + j)*k + pc,
                                        C + i*n + jc + j, k, k, n, rows, cols, kc, accumulate);
                    }
                }
            }
        }
    }
}

void spur_matmul_nt(const double* A, const double* B, double* C,
                    long long m, long long k, long long n){
    if (m <= 0 || n <= 0) return;
    if (k <= 0){ memset(C, 0, (size_t)m*n*sizeof(double)); return; }
    if (spur_mm_legacy()){ spur_matmul_nt_legacy(A, B, C, m, k, n); return; }
    switch (spur_mm_route(m, k, n, SPUR_MM_F64_MIN_M, SPUR_MM_F64_MIN_N,
                          SPUR_MM_F64_DOTLO, SPUR_MM_F64_MINKT)){
        case SPUR_MM_ROUTE_DOT:  spur_gemm_dot_f64(A, B, C, m, k, n);  break;
        case SPUR_MM_ROUTE_PACK: spur_gemm_pack_f64(A, B, C, m, k, n); break;
        default:                 spur_matmul_nt_legacy(A, B, C, m, k, n); break;
    }
}

/* ------------------------------ f32 [P] ---------------------------------- */
#define S_MR  4
#define S_NR  24
#define S_NV  3                       /* S_NR / 8 lanes */
#define S_KC  384
#define S_MC  128
#define S_NC  2048
#define S_MCP (((S_MC)+S_MR-1)/S_MR*S_MR)
#define S_NCP (((S_NC)+S_NR-1)/S_NR*S_NR)

static void spur_pack_a_s(const float* restrict A, long long lda,
                          float* restrict Ap, long long mc, long long kc){
    for (long long i = 0; i < mc; i += S_MR){
        const long long rows = (mc - i < S_MR) ? (mc - i) : S_MR;
        float* dst = Ap + i * kc;
        for (long long q = 0; q < kc; q++){
            for (long long r = 0; r < rows; r++)  dst[q*S_MR + r] = A[(i+r)*lda + q];
            for (long long r = rows; r < S_MR; r++) dst[q*S_MR + r] = 0.0f;
        }
    }
}

static void spur_pack_b_s(const float* restrict B, long long ldb,
                          float* restrict Bp, long long nc, long long kc){
    for (long long j = 0; j < nc; j += S_NR){
        const long long cols = (nc - j < S_NR) ? (nc - j) : S_NR;
        float* dst = Bp + j * kc;
        for (long long q = 0; q < kc; q++){
            for (long long c = 0; c < cols; c++)  dst[q*S_NR + c] = B[(j+c)*ldb + q];
            for (long long c = cols; c < S_NR; c++) dst[q*S_NR + c] = 0.0f;
        }
    }
}

static inline void spur_micro_s(long long kc, const float* restrict Ap,
                                const float* restrict Bp, float* restrict C,
                                long long ldc, int accumulate){
    __m256 c[S_MR][S_NV];
    for (int i = 0; i < S_MR; i++)
        for (int j = 0; j < S_NV; j++) c[i][j] = _mm256_setzero_ps();

    for (long long q = 0; q < kc; q++){
        __m256 b[S_NV];
        for (int j = 0; j < S_NV; j++) b[j] = _mm256_loadu_ps(Bp + q*S_NR + j*8);
        for (int i = 0; i < S_MR; i++){
            __m256 a = _mm256_broadcast_ss(Ap + q*S_MR + i);
            for (int j = 0; j < S_NV; j++) c[i][j] = _mm256_fmadd_ps(a, b[j], c[i][j]);
        }
    }
    if (accumulate){
        for (int i = 0; i < S_MR; i++)
            for (int j = 0; j < S_NV; j++)
                _mm256_storeu_ps(C + i*ldc + j*8,
                                 _mm256_add_ps(_mm256_loadu_ps(C + i*ldc + j*8), c[i][j]));
    } else {
        for (int i = 0; i < S_MR; i++)
            for (int j = 0; j < S_NV; j++) _mm256_storeu_ps(C + i*ldc + j*8, c[i][j]);
    }
}

/* exporte : permet l'A/B des trois noyaux depuis les bancs (experiments/) */
void spur_gemm_pack_f32(const float* A, const float* B, float* C,
                               long long m, long long k, long long n){
    const int nth = spur_mm_threads();
    float* Bp = (float*)SPUR_AALLOC(sizeof(float) * (size_t)S_KC * S_NCP);
    float* Ap = (float*)SPUR_AALLOC(sizeof(float) * (size_t)nth * S_KC * S_MCP);
    if (!Bp || !Ap){
        if (Bp) SPUR_AFREE(Bp);
        if (Ap) SPUR_AFREE(Ap);
        spur_matmul_nt_f32_legacy(A, B, C, m, k, n);
        return;
    }
    const long long MCE = spur_mm_mc(m, nth, S_MC, S_MR);
    for (long long jc = 0; jc < n; jc += S_NC){
        const long long nc = (n - jc < S_NC) ? (n - jc) : S_NC;
        for (long long pc = 0; pc < k; pc += S_KC){
            const long long kc = (k - pc < S_KC) ? (k - pc) : S_KC;
            const int accumulate = (pc != 0);
            spur_pack_b_s(B + jc*k + pc, k, Bp, nc, kc);
            #pragma omp parallel
            {
                int tid = 0;
#ifdef _OPENMP
                tid = omp_get_thread_num();
#endif
                float* Apt = Ap + (size_t)tid * S_KC * S_MCP;
                #pragma omp for schedule(static)
                for (long long ic = 0; ic < m; ic += MCE){
                    const long long mc = (m - ic < MCE) ? (m - ic) : MCE;
                    spur_pack_a_s(A + ic*k + pc, k, Apt, mc, kc);
                    for (long long jr = 0; jr < nc; jr += S_NR){
                        const long long cols = (nc - jr < S_NR) ? (nc - jr) : S_NR;
                        for (long long ir = 0; ir < mc; ir += S_MR){
                            const long long rows = (mc - ir < S_MR) ? (mc - ir) : S_MR;
                            float* cptr = C + (ic + ir)*n + jc + jr;
                            __builtin_prefetch(cptr + 4*n, 1, 1);
                            if (rows == S_MR && cols == S_NR){
                                spur_micro_s(kc, Apt + ir*kc, Bp + jr*kc, cptr, n, accumulate);
                            } else {
                                float tmp[S_MR * S_NR] __attribute__((aligned(64)));
                                spur_micro_s(kc, Apt + ir*kc, Bp + jr*kc, tmp, S_NR, 0);
                                for (long long r = 0; r < rows; r++)
                                    for (long long c = 0; c < cols; c++)
                                        cptr[r*n + c] = accumulate ? cptr[r*n + c] + tmp[r*S_NR + c]
                                                                   : tmp[r*S_NR + c];
                            }
                        }
                    }
                }
            }
        }
    }
    SPUR_AFREE(Ap);
    SPUR_AFREE(Bp);
}

/* ------------------------------ f32 [D] ---------------------------------- */
#define SD_MR 3
#define SD_NR 4
#define SD_KC 1024
#define SD_NC 256

static inline void spur_dot_tile_s(const float* restrict ap, const float* restrict bp,
                                   float* restrict cp, long long lda, long long ldb,
                                   long long ldc, long long rows, long long cols,
                                   long long kc, int accumulate){
    if (rows == SD_MR && cols == SD_NR){
        __m256 acc[SD_MR][SD_NR];
        for (int i = 0; i < SD_MR; i++)
            for (int j = 0; j < SD_NR; j++) acc[i][j] = _mm256_setzero_ps();
        long long q = 0;
        for (; q + 7 < kc; q += 8){
            __m256 av[SD_MR];
            for (int i = 0; i < SD_MR; i++) av[i] = _mm256_loadu_ps(ap + i*lda + q);
            for (int j = 0; j < SD_NR; j++){
                const __m256 bv = _mm256_loadu_ps(bp + j*ldb + q);
                for (int i = 0; i < SD_MR; i++) acc[i][j] = _mm256_fmadd_ps(av[i], bv, acc[i][j]);
            }
        }
        float tail[SD_MR][SD_NR];
        for (int i = 0; i < SD_MR; i++)
            for (int j = 0; j < SD_NR; j++) tail[i][j] = 0.0f;
        for (long long t = q; t < kc; t++)
            for (int i = 0; i < SD_MR; i++){
                const float a = ap[i*lda + t];
                for (int j = 0; j < SD_NR; j++) tail[i][j] += a * bp[j*ldb + t];
            }
        for (int i = 0; i < SD_MR; i++)
            for (int j = 0; j < SD_NR; j++){
                const float s = hsum8(acc[i][j]) + tail[i][j];
                cp[i*ldc + j] = accumulate ? cp[i*ldc + j] + s : s;
            }
        return;
    }
    for (long long i = 0; i < rows; i++)
        for (long long j = 0; j < cols; j++){
            __m256 v = _mm256_setzero_ps();
            long long q = 0;
            for (; q + 7 < kc; q += 8)
                v = _mm256_fmadd_ps(_mm256_loadu_ps(ap + i*lda + q),
                                    _mm256_loadu_ps(bp + j*ldb + q), v);
            float s = hsum8(v);
            for (; q < kc; q++) s += ap[i*lda + q] * bp[j*ldb + q];
            cp[i*ldc + j] = accumulate ? cp[i*ldc + j] + s : s;
        }
}

/* exporte : permet l'A/B des trois noyaux depuis les bancs (experiments/) */
void spur_gemm_dot_f32(const float* A, const float* B, float* C,
                              long long m, long long k, long long n){
    for (long long pc = 0; pc < k; pc += SD_KC){
        const long long kc = (k - pc < SD_KC) ? (k - pc) : SD_KC;
        const int accumulate = (pc != 0);
        for (long long jc = 0; jc < n; jc += SD_NC){
            const long long ncz = (n - jc < SD_NC) ? (n - jc) : SD_NC;
            if (m / SD_MR >= ncz / SD_NR){
                #pragma omp parallel for schedule(static)
                for (long long i = 0; i < m; i += SD_MR){
                    const long long rows = (m - i < SD_MR) ? (m - i) : SD_MR;
                    for (long long j = 0; j < ncz; j += SD_NR){
                        const long long cols = (ncz - j < SD_NR) ? (ncz - j) : SD_NR;
                        __builtin_prefetch(A + (i + 2*SD_MR)*k + pc, 0, 3);
                        spur_dot_tile_s(A + i*k + pc, B + (jc + j)*k + pc,
                                        C + i*n + jc + j, k, k, n, rows, cols, kc, accumulate);
                    }
                }
            } else {
                for (long long i = 0; i < m; i += SD_MR){
                    const long long rows = (m - i < SD_MR) ? (m - i) : SD_MR;
                    #pragma omp parallel for schedule(static)
                    for (long long j = 0; j < ncz; j += SD_NR){
                        const long long cols = (ncz - j < SD_NR) ? (ncz - j) : SD_NR;
                        spur_dot_tile_s(A + i*k + pc, B + (jc + j)*k + pc,
                                        C + i*n + jc + j, k, k, n, rows, cols, kc, accumulate);
                    }
                }
            }
        }
    }
}

void spur_matmul_nt_f32(const float* A, const float* B, float* C,
                        long long m, long long k, long long n){
    if (m <= 0 || n <= 0) return;
    if (k <= 0){ memset(C, 0, (size_t)m*n*sizeof(float)); return; }
    if (spur_mm_legacy()){ spur_matmul_nt_f32_legacy(A, B, C, m, k, n); return; }
    switch (spur_mm_route(m, k, n, SPUR_MM_F32_MIN_M, SPUR_MM_F32_MIN_N,
                          SPUR_MM_F32_DOTLO, SPUR_MM_F32_MINKT)){
        case SPUR_MM_ROUTE_DOT:  spur_gemm_dot_f32(A, B, C, m, k, n);  break;
        case SPUR_MM_ROUTE_PACK: spur_gemm_pack_f32(A, B, C, m, k, n); break;
        default:                 spur_matmul_nt_f32_legacy(A, B, C, m, k, n); break;
    }
}

/* ============================================================================
   GEMM + GELU v2 : GEMM autotune suivi d'un epilogue biais+gelu vectorise.
   ----------------------------------------------------------------------------
   L'ancien noyau fusionnait la gelu dans la boucle j pour eviter une relecture
   de C. Mesure : cette fusion coute bien plus cher qu'elle ne rapporte, parce
   qu'elle interdit le blocage en k (donc le micro-noyau packe). L'epilogue
   relit m*n elements — memoire pure, quelques % — tandis que le GEMM gagne un
   facteur ~2. On garde l'ancien chemin sous *_legacy (et SPUR_MM_LEGACY=1).
   ========================================================================== */

static void spur_gelu_bias_rows_f64(double* C, long long m, long long n,
                                    const double* bias){
    const __m256d c306 = _mm256_set1_pd(0.306923), c501 = _mm256_set1_pd(0.501);
    const __m256d cmax = _mm256_set1_pd(1.002),   zero = _mm256_setzero_pd();
    const __m256d ck   = _mm256_set1_pd(0.997729), cb  = _mm256_set1_pd(-0.004004);
    const int hb = (bias != NULL);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < m; i++){
        double* row = C + i*n;
        long long j = 0;
        for (; j + 3 < n; j += 4){
            __m256d x = _mm256_loadu_pd(row + j);
            if (hb) x = _mm256_add_pd(x, _mm256_loadu_pd(bias + j));
            __m256d u = _mm256_fmadd_pd(c306, x, c501);
            u = _mm256_min_pd(_mm256_max_pd(u, zero), cmax);
            _mm256_storeu_pd(row + j,
                             _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(x, u), ck), cb));
        }
        for (; j < n; j++) row[j] = gelu_s(row[j] + (hb ? bias[j] : 0.0));
    }
}

static void spur_gelu_bias_rows_f32(float* C, long long m, long long n,
                                    const float* bias){
    const __m256 c306 = _mm256_set1_ps(0.306923f), c501 = _mm256_set1_ps(0.501f);
    const __m256 cmax = _mm256_set1_ps(1.002f),   zero = _mm256_setzero_ps();
    const __m256 ck   = _mm256_set1_ps(0.997729f), cb  = _mm256_set1_ps(-0.004004f);
    const int hb = (bias != NULL);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < m; i++){
        float* row = C + i*n;
        long long j = 0;
        for (; j + 7 < n; j += 8){
            __m256 x = _mm256_loadu_ps(row + j);
            if (hb) x = _mm256_add_ps(x, _mm256_loadu_ps(bias + j));
            __m256 u = _mm256_fmadd_ps(c306, x, c501);
            u = _mm256_min_ps(_mm256_max_ps(u, zero), cmax);
            _mm256_storeu_ps(row + j,
                             _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(x, u), ck), cb));
        }
        for (; j < n; j++) row[j] = gelu_f32_scalar(row[j] + (hb ? bias[j] : 0.0f));
    }
}

void spur_matmul_nt_gelu(const double* A, const double* B, const double* bias,
                         double* C, long long m, long long k, long long n){
    if (m <= 0 || n <= 0) return;
    if (spur_mm_legacy()){ spur_matmul_nt_gelu_legacy(A, B, bias, C, m, k, n); return; }
    spur_matmul_nt(A, B, C, m, k, n);
    spur_gelu_bias_rows_f64(C, m, n, bias);
}

void spur_matmul_nt_gelu_f32(const float* A, const float* B, const float* bias,
                             float* C, long long m, long long k, long long n){
    if (m <= 0 || n <= 0) return;
    if (spur_mm_legacy()){ spur_matmul_nt_gelu_f32_legacy(A, B, bias, C, m, k, n); return; }
    spur_matmul_nt_f32(A, B, C, m, k, n);
    spur_gelu_bias_rows_f32(C, m, n, bias);
}

/* ============================================================================
   EXP AVX2 + SOFTMAX — la brique qui manquait pour l'attention
   ----------------------------------------------------------------------------
   exp(x) = 2^k * exp(r), k = round(x/ln2), |r| <= ln2/2.
   ln2 est scinde en (hi, lo) : hi n'a que 11 bits de mantisse, donc k*hi est
   exact et la soustraction x - k*hi l'est aussi (Sterbenz). Le polynome sur r
   est un minimax en **erreur relative** obtenu par iterations de Remez
   (experiments/transcend/fit_exp.py).

   Precision mesuree contre la reference float64 (500 001 points) :
       f32, degre 5  : 2.02e-07 = 1.69 ulp   (numpy/libm : 1.66 ulp)
       f64, degre 10 : 4.72e-16 = 2.12 ulp
   Debit mesure : jusqu'a 3.0 G elements/s en f32 (x1.2 a x1.9 vs numpy).

   Le softmax par ligne en decoule. Trois passes battent le schema "online"
   facon flash-attention sur cette machine (mesure : x2 a x4 d'ecart) : le
   rescale incremental coute plus cher que la relecture d'une ligne deja en L1.
   ========================================================================== */

static inline __m256 spur_exp8_ps(__m256 x){
    const __m256 LOG2E  = _mm256_set1_ps(1.4426950408889634f);
    const __m256 LN2_HI = _mm256_set1_ps(0.693359375f);
    const __m256 LN2_LO = _mm256_set1_ps(-2.12194440e-4f);
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.33654f));
    __m256 k = _mm256_round_ps(_mm256_mul_ps(x, LOG2E),
                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(k, LN2_HI, x);
    r = _mm256_fnmadd_ps(k, LN2_LO, r);
    __m256 p = _mm256_set1_ps(0.008297655080344314f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.04191538199170809f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.16667574728755657f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.49998894851221815f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.9999996919915163f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0000000716546822f));
    __m256i pw = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(k), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(pw));
}

static inline __m256d spur_exp4_pd(__m256d x){
    const __m256d LOG2E  = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2_HI = _mm256_set1_pd(0.693145751953125);
    const __m256d LN2_LO = _mm256_set1_pd(1.42860682030941723212e-6);
    x = _mm256_max_pd(x, _mm256_set1_pd(-708.396418));
    __m256d k = _mm256_round_pd(_mm256_mul_pd(x, LOG2E),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256d r = _mm256_fnmadd_pd(k, LN2_HI, x);
    r = _mm256_fnmadd_pd(k, LN2_LO, r);
    __m256d p = _mm256_set1_pd(2.756128125488038e-07);
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.763753106961904e-06));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(2.4801689582726192e-05));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.00019841177154153087));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.0013888888750558034));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.00833333337961161));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.041666666667305056));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.16666666666574947));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(0.49999999999998845));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0000000000000047));
    p = _mm256_fmadd_pd(p, r, _mm256_set1_pd(1.0));
    __m256i ki = _mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(k));
    __m256i pw = _mm256_slli_epi64(_mm256_add_epi64(ki, _mm256_set1_epi64x(1023)), 52);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(pw));
}

void spur_batch_exp_f32(const float* x, float* y, long long n){
    long long i = 0;
    #pragma omp parallel for schedule(static)
    for (long long b = 0; b < n / 8; b++)
        _mm256_storeu_ps(y + b * 8, spur_exp8_ps(_mm256_loadu_ps(x + b * 8)));
    for (i = (n / 8) * 8; i < n; i++) y[i] = expf(x[i]);
}

void spur_batch_exp(const double* x, double* y, long long n){
    long long i = 0;
    #pragma omp parallel for schedule(static)
    for (long long b = 0; b < n / 4; b++)
        _mm256_storeu_pd(y + b * 4, spur_exp4_pd(_mm256_loadu_pd(x + b * 4)));
    for (i = (n / 4) * 4; i < n; i++) y[i] = exp(x[i]);
}

static inline float spur_hmax8(__m256 v){
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

/* softmax par ligne. `len` : nombre d'entrees valides par ligne (NULL = cols).
   Les entrees au-dela de len[i] sont mises a zero — c'est la forme causale
   dont l'attention a besoin, sans materialiser de -inf.                     */
static void spur_softmax_core_f32(const float* X, float* Y, long long rows,
                                  long long cols, const int* len, float scale){
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < rows; i++){
        const long long n = len ? (len[i] < cols ? len[i] : cols) : cols;
        const float* xr = X + i * cols;
        float* yr = Y + i * cols;
        if (n <= 0){ memset(yr, 0, (size_t)cols * sizeof(float)); continue; }

        __m256 vmax = _mm256_set1_ps(-INFINITY);
        long long j = 0;
        for (; j + 7 < n; j += 8) vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(xr + j));
        float m = (n >= 8) ? spur_hmax8(vmax) : -INFINITY;
        for (; j < n; j++) m = xr[j] > m ? xr[j] : m;

        /* exp(a(x-m)) : l'echelle 1/sqrt(d) de l'attention est absorbee ici,
           sans passe supplementaire sur la matrice de scores. */
        __m256 vm = _mm256_set1_ps(m), vs = _mm256_setzero_ps();
        __m256 va = _mm256_set1_ps(scale);
        for (j = 0; j + 7 < n; j += 8){
            __m256 e = spur_exp8_ps(_mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(xr + j), vm), va));
            _mm256_storeu_ps(yr + j, e);
            vs = _mm256_add_ps(vs, e);
        }
        float s = (n >= 8) ? hsum8(vs) : 0.0f;
        for (; j < n; j++){ yr[j] = expf((xr[j] - m) * scale); s += yr[j]; }

        const float inv = 1.0f / s;
        __m256 vinv = _mm256_set1_ps(inv);
        for (j = 0; j + 7 < n; j += 8)
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_loadu_ps(yr + j), vinv));
        for (; j < n; j++) yr[j] *= inv;
        if (n < cols) memset(yr + n, 0, (size_t)(cols - n) * sizeof(float));
    }
}

void spur_softmax_rows_f32(const float* X, float* Y, long long rows,
                           long long cols, const int* len){
    spur_softmax_core_f32(X, Y, rows, cols, len, 1.0f);
}

/* ---------------------------------------------------------------------------
   Tuile d'attention : O = softmax(scale * Q.K^T, masque causal) . V
   Les deux produits sont en convention NT, celle des noyaux GEMM v2 :
       scores = Q (tq,d) . K^T ou K est (tk,d)
       O      = scores (tq,tk) . V ou V est fourni **transpose** (d,tk)
   L'echelle 1/sqrt(d) est absorbee par le softmax (aucune passe de plus).
   `lens[i]` = nombre de cles visibles par la requete i (NULL = toutes).
   `scratch` : tampon (tq*tk) fourni par l'appelant, ou NULL pour allouer.   */
void spur_attention_tile_f32(const float* Q, const float* K, const float* Vt,
                             float* O, long long tq, long long tk, long long d,
                             float scale, const int* lens, float* scratch){
    if (tq <= 0 || tk <= 0 || d <= 0) return;
    float* S = scratch;
    float* owned = NULL;
    if (!S){
        owned = (float*)SPUR_AALLOC(sizeof(float) * (size_t)tq * tk);
        if (!owned) return;
        S = owned;
    }
    spur_matmul_nt_f32(Q, K, S, tq, d, tk);          /* scores (tq, tk)      */
    spur_softmax_core_f32(S, S, tq, tk, lens, scale); /* poids, en place     */
    spur_matmul_nt_f32(S, Vt, O, tq, tk, d);         /* O = poids . V (tq,d) */
    if (owned) SPUR_AFREE(owned);
}

void spur_softmax_rows(const double* X, double* Y, long long rows,
                       long long cols, const int* len){
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < rows; i++){
        const long long n = len ? (len[i] < cols ? len[i] : cols) : cols;
        const double* xr = X + i * cols;
        double* yr = Y + i * cols;
        if (n <= 0){ memset(yr, 0, (size_t)cols * sizeof(double)); continue; }

        double m = -INFINITY;
        for (long long j = 0; j < n; j++) if (xr[j] > m) m = xr[j];
        __m256d vm = _mm256_set1_pd(m), vs = _mm256_setzero_pd();
        long long j = 0;
        for (; j + 3 < n; j += 4){
            __m256d e = spur_exp4_pd(_mm256_sub_pd(_mm256_loadu_pd(xr + j), vm));
            _mm256_storeu_pd(yr + j, e);
            vs = _mm256_add_pd(vs, e);
        }
        double s = (n >= 4) ? hs256(vs) : 0.0;
        for (; j < n; j++){ yr[j] = exp(xr[j] - m); s += yr[j]; }

        const double inv = 1.0 / s;
        __m256d vinv = _mm256_set1_pd(inv);
        for (j = 0; j + 3 < n; j += 4)
            _mm256_storeu_pd(yr + j, _mm256_mul_pd(_mm256_loadu_pd(yr + j), vinv));
        for (; j < n; j++) yr[j] *= inv;
        if (n < cols) memset(yr + n, 0, (size_t)(cols - n) * sizeof(double));
    }
}

/* ============================================================================
   ATTENTION MULTI-TETES — une seule descente en C
   ----------------------------------------------------------------------------
   Motivation mesuree : appelee tete par tete depuis Python, la tuile
   d'attention plafonne a ~12 GFLOPS sur les petites formes (tq=4) alors
   qu'elle atteint 123 GFLOPS sur les grandes : c'est le cout fixe par appel
   (ctypes, allocation du scratch, packing) qui domine, pas le calcul.

   Ici tout se passe en C :
     * phase 1 : les tetes K/V sont packees UNE fois (K en (tk,d), V transpose
       en (d,tk)). En GQA (H_kv < H_q) plusieurs tetes de requetes partagent la
       meme tete de cles : on ne la packe donc qu'une fois pour tout le groupe ;
     * phase 2 : parallelisme OpenMP **sur les tetes de requetes**, chaque
       thread disposant de ses propres tampons (scores et sortie).

   Disposition memoire attendue (celle des simulations et de QSA) :
     Q : (tq, H_q,  d)   K, V : (tk, H_kv, d)   O : (tq, H_q, d)
   `lens[i]` = nombre de cles visibles par la requete i (NULL = toutes).
   ========================================================================== */
/* Choix de noyau interne a l'attention.
   Le dispatcher global ne peut pas savoir qu'il s'agit d'une tuile
   d'attention ; ici la forme est connue et les deux GEMM ont des regimes
   OPPOSES (mesure : results/att_m_grid.json, results/tile_gemm_shapes.json) :

     scores = Q.K^T  -> k = d (court), n = tk (long)
     PV     = W.V^T  -> k = tk (long), n = d (court)

   Le noyau dot accumule le long de k avec une seule reduction horizontale par
   case : il gagne quand k est profond (PV, jusqu'a x4.2) et perd quand k est
   court et n large, sauf a tres peu de lignes. Le noyau legacy, lui, bloque 8
   lignes : m multiple de 8 est sa zone de force.

   Regle retenue par recherche sur 62 formes de tuiles : moyenne x1.28,
   **pire cas x1.00** (aucune regression). La regle d'aiguillage GLOBALE n'est
   pas touchee : elle est deja Pareto-optimale sous la meme contrainte.       */
static inline void spur_att_gemm_f32(const float* A, const float* B, float* C,
                                     long long m, long long k, long long n){
    if (m < 64 && k >= 128){
        const int off8 = (m & 7) != 0;                  /* hors zone du legacy */
        const int pv_like = (k >= n) && (k >= 1024) && (off8 || k >= 4096);
        const int scores_like = off8 && (m < 16) && (n >= 256);
        if (pv_like || scores_like){
            spur_gemm_dot_f32(A, B, C, m, k, n);
            return;
        }
    }
    spur_matmul_nt_f32(A, B, C, m, k, n);
}

/* ---- cache KV packe, reutilisable ----------------------------------------
   Pour tq petit (decodage, ou micro-blocs QSA), le packing de K/V domine le
   cout de l'appel : il est en O(tk.d.h_kv) alors que le calcul n'est qu'en
   O(tq.tk.d.h_q). Or les memes cles servent a des dizaines de tuiles de
   requetes successives. On expose donc le packing pour l'amortir.
   Disposition du tampon : [K packe (h_kv x tk x d)][V^T packe (h_kv x d x tk)].
   ------------------------------------------------------------------------- */
long long spur_kv_pack_size_f32(long long tk, long long d, int h_kv){
    return (long long)2 * tk * d * h_kv;      /* en nombre de float */
}

void spur_kv_pack_f32(const float* K, const float* V, long long tk, long long d,
                      int h_kv, float* packed){
    const size_t kv_sz = (size_t)tk * d;
    float* Kp = packed;
    float* Vp = packed + kv_sz * (size_t)h_kv;
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < h_kv; g++){
        float* kd = Kp + kv_sz * (size_t)g;      /* (tk, d) */
        float* vd = Vp + kv_sz * (size_t)g;      /* (d, tk) */
        for (long long t = 0; t < tk; t++){
            const float* ks = K + (t * h_kv + g) * d;
            const float* vs = V + (t * h_kv + g) * d;
            memcpy(kd + t * d, ks, (size_t)d * sizeof(float));
            for (long long c = 0; c < d; c++) vd[c * tk + t] = vs[c];
        }
    }
}

void spur_attention_mha_packed_f32(const float* Q, const float* packed, float* O,
                                   long long tq, long long tk, long long d,
                                   int h_q, int h_kv, float scale, const int* lens);

void spur_attention_mha_f32(const float* Q, const float* K, const float* V,
                            float* O, long long tq, long long tk, long long d,
                            int h_q, int h_kv, float scale, const int* lens){
    if (tq <= 0 || tk <= 0 || d <= 0 || h_q <= 0 || h_kv <= 0) return;
    if (h_q % h_kv) return;                      /* groupes GQA mal formes    */
    const int rep = h_q / h_kv;

    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
    if (nth > h_kv) nth = h_kv;
    if (nth < 1) nth = 1;
#endif
    const int par_groups = (h_kv >= nth) && (nth > 1);
    const int slots = par_groups ? nth : 1;

    const size_t kv_sz = (size_t)tk * d;
    const long long mb = (long long)rep * tq;    /* lignes par GEMM groupe    */
    float* Kp = (float*)SPUR_AALLOC(sizeof(float) * kv_sz * (size_t)h_kv * 2);
    const size_t per = (size_t)mb * d + (size_t)mb * tk + (size_t)mb * d;
    float* work = (float*)SPUR_AALLOC(sizeof(float) * per * (size_t)slots);
    int* lbuf = (int*)SPUR_AALLOC(sizeof(int) * (size_t)mb * (size_t)slots);
    if (!Kp || !work || !lbuf){
        if (Kp) SPUR_AFREE(Kp);
        if (work) SPUR_AFREE(work);
        if (lbuf) SPUR_AFREE(lbuf);
        return;
    }
    float* Vp = Kp + kv_sz * (size_t)h_kv;
    spur_kv_pack_f32(K, V, tk, d, h_kv, Kp);

    spur_attention_mha_packed_f32(Q, Kp, O, tq, tk, d, h_q, h_kv, scale, lens);
    SPUR_AFREE(lbuf);
    SPUR_AFREE(work);
    SPUR_AFREE(Kp);
}

/* Meme calcul, mais sur un cache KV deja packe (cf. spur_kv_pack_f32). */
void spur_attention_mha_packed_f32(const float* Q, const float* packed, float* O,
                                   long long tq, long long tk, long long d,
                                   int h_q, int h_kv, float scale, const int* lens){
    if (tq <= 0 || tk <= 0 || d <= 0 || h_q <= 0 || h_kv <= 0) return;
    if (h_q % h_kv) return;
    const int rep = h_q / h_kv;
    const size_t kv_sz = (size_t)tk * d;
    const float* Kp = packed;
    const float* Vp = packed + kv_sz * (size_t)h_kv;

    int nth = 1;
#ifdef _OPENMP
    nth = omp_get_max_threads();
    if (nth > h_kv) nth = h_kv;
    if (nth < 1) nth = 1;
#endif
    const int par_groups = (h_kv >= nth) && (nth > 1);
    const int slots = par_groups ? nth : 1;
    const long long mb = (long long)rep * tq;
    const size_t per = (size_t)mb * d + (size_t)mb * tk + (size_t)mb * d;
    float* work = (float*)SPUR_AALLOC(sizeof(float) * per * (size_t)slots);
    int* lbuf = (int*)SPUR_AALLOC(sizeof(int) * (size_t)mb * (size_t)slots);
    if (!work || !lbuf){
        if (work) SPUR_AFREE(work);
        if (lbuf) SPUR_AFREE(lbuf);
        return;
    }

    /* Les `rep` tetes de requetes d'un groupe GQA partagent la meme tete de
       cles : on les empile en un seul GEMM de rep*tq lignes au lieu de rep
       GEMM de tq lignes. A tq=4 et rep=4, m passe de 4 a 16 — d'un regime
       domine par les couts fixes a un regime de calcul.                      */
    #pragma omp parallel for schedule(static) if(par_groups)
    for (int g = 0; g < h_kv; g++){
        int slot = 0;
#ifdef _OPENMP
        if (par_groups) slot = omp_get_thread_num();
#endif
        float* buf = work + per * (size_t)slot;
        float* Qb = buf;
        float* S  = Qb + (size_t)mb * d;
        float* Ob = S + (size_t)mb * tk;
        int* lb = lbuf + (size_t)mb * (size_t)slot;

        for (int r = 0; r < rep; r++){
            const int h = g * rep + r;
            for (long long t = 0; t < tq; t++){
                memcpy(Qb + ((long long)r * tq + t) * d, Q + (t * h_q + h) * d,
                       (size_t)d * sizeof(float));
                if (lens) lb[(long long)r * tq + t] = lens[t];
            }
        }

        spur_att_gemm_f32(Qb, Kp + kv_sz * (size_t)g, S, mb, d, tk);
        spur_softmax_core_f32(S, S, mb, tk, lens ? lb : NULL, scale);
        spur_att_gemm_f32(S, Vp + kv_sz * (size_t)g, Ob, mb, tk, d);

        for (int r = 0; r < rep; r++){
            const int h = g * rep + r;
            for (long long t = 0; t < tq; t++)
                memcpy(O + (t * h_q + h) * d, Ob + ((long long)r * tq + t) * d,
                       (size_t)d * sizeof(float));
        }
    }
    SPUR_AFREE(lbuf);
    SPUR_AFREE(work);
}

/* ============================================================================
   CACHE KV EN BF16 — moitie moins de trafic memoire
   ----------------------------------------------------------------------------
   A grand tk l'attention n'est plus limitee par le calcul mais par la lecture
   de K et V (8.4 Mo a tk=8192, d=64, 2 tetes : au-dela de tout cache). Le bf16
   (1 signe, 8 exposant, 7 mantisse) est la troncature naturelle du float32 :
   la conversion est un decalage de 16 bits, et la reconversion aussi.

   Le GEMM ci-dessous garde A en float32 (les requetes et les poids d'attention,
   peu volumineux) et lit B en bf16 : c'est la ou passe la bande passante.
   ========================================================================== */

static inline uint16_t spur_f32_to_bf16(float f){
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(x >> 16);   /* NaN / inf */
    const uint32_t r = ((x >> 16) & 1u) + 0x7FFFu;                /* arrondi au plus proche pair */
    return (uint16_t)((x + r) >> 16);
}

static inline __m256 spur_load8_bf16(const uint16_t* p){
    __m128i h = _mm_loadu_si128((const __m128i*)p);
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

/* C = A . B^T avec A float32 (m,k) et B bf16 (n,k). Bloc 3x4, une reduction
   horizontale par case, comme le noyau dot f32.                              */
static void spur_gemm_nt_bf16b_f32(const float* restrict A, const uint16_t* restrict B,
                                   float* restrict C, long long m, long long k,
                                   long long n){
    const long long MR = 3, NR = 4;
    #pragma omp parallel for schedule(static) if(m >= 3 * omp_get_max_threads())
    for (long long i0 = 0; i0 < m; i0 += MR){
        const long long rows = (m - i0 < MR) ? (m - i0) : MR;
        for (long long j0 = 0; j0 < n; j0 += NR){
            const long long cols = (n - j0 < NR) ? (n - j0) : NR;
            __m256 acc[3][4];
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 4; b++) acc[a][b] = _mm256_setzero_ps();
            long long q = 0;
            for (; q + 7 < k; q += 8){
                __m256 av[3];
                for (long long r = 0; r < rows; r++) av[r] = _mm256_loadu_ps(A + (i0 + r) * k + q);
                for (long long c = 0; c < cols; c++){
                    const __m256 bv = spur_load8_bf16(B + (j0 + c) * k + q);
                    for (long long r = 0; r < rows; r++)
                        acc[r][c] = _mm256_fmadd_ps(av[r], bv, acc[r][c]);
                }
            }
            for (long long r = 0; r < rows; r++)
                for (long long c = 0; c < cols; c++){
                    float s = hsum8(acc[r][c]);
                    for (long long t = q; t < k; t++){
                        uint32_t w = (uint32_t)B[(j0 + c) * k + t] << 16;
                        float bf;
                        memcpy(&bf, &w, sizeof(bf));
                        s += A[(i0 + r) * k + t] * bf;
                    }
                    C[(i0 + r) * n + j0 + c] = s;
                }
        }
    }
}

long long spur_kv_pack_size_bf16(long long tk, long long d, int h_kv){
    return (long long)2 * tk * d * h_kv;          /* en nombre de uint16 */
}

void spur_kv_pack_bf16(const float* K, const float* V, long long tk, long long d,
                       int h_kv, uint16_t* packed){
    const size_t kv_sz = (size_t)tk * d;
    uint16_t* Kp = packed;
    uint16_t* Vp = packed + kv_sz * (size_t)h_kv;
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < h_kv; g++){
        uint16_t* kd = Kp + kv_sz * (size_t)g;
        uint16_t* vd = Vp + kv_sz * (size_t)g;
        for (long long t = 0; t < tk; t++){
            const float* ks = K + (t * h_kv + g) * d;
            const float* vs = V + (t * h_kv + g) * d;
            for (long long c = 0; c < d; c++){
                kd[t * d + c] = spur_f32_to_bf16(ks[c]);
                vd[c * tk + t] = spur_f32_to_bf16(vs[c]);
            }
        }
    }
}

void spur_attention_mha_packed_bf16(const float* Q, const uint16_t* packed, float* O,
                                    long long tq, long long tk, long long d,
                                    int h_q, int h_kv, float scale, const int* lens){
    if (tq <= 0 || tk <= 0 || d <= 0 || h_q <= 0 || h_kv <= 0) return;
    if (h_q % h_kv) return;
    const int rep = h_q / h_kv;
    const size_t kv_sz = (size_t)tk * d;
    const uint16_t* Kp = packed;
    const uint16_t* Vp = packed + kv_sz * (size_t)h_kv;

    const long long mb = (long long)rep * tq;
    const size_t per = (size_t)mb * d + (size_t)mb * tk + (size_t)mb * d;
    float* work = (float*)SPUR_AALLOC(sizeof(float) * per);
    int* lb = (int*)SPUR_AALLOC(sizeof(int) * (size_t)mb);
    if (!work || !lb){
        if (work) SPUR_AFREE(work);
        if (lb) SPUR_AFREE(lb);
        return;
    }
    float* Qb = work;
    float* S  = Qb + (size_t)mb * d;
    float* Ob = S + (size_t)mb * tk;

    for (int g = 0; g < h_kv; g++){
        for (int r = 0; r < rep; r++){
            const int h = g * rep + r;
            for (long long t = 0; t < tq; t++){
                memcpy(Qb + ((long long)r * tq + t) * d, Q + (t * h_q + h) * d,
                       (size_t)d * sizeof(float));
                if (lens) lb[(long long)r * tq + t] = lens[t];
            }
        }
        spur_gemm_nt_bf16b_f32(Qb, Kp + kv_sz * (size_t)g, S, mb, d, tk);
        spur_softmax_core_f32(S, S, mb, tk, lens ? lb : NULL, scale);
        spur_gemm_nt_bf16b_f32(S, Vp + kv_sz * (size_t)g, Ob, mb, tk, d);
        for (int r = 0; r < rep; r++){
            const int h = g * rep + r;
            for (long long t = 0; t < tq; t++)
                memcpy(O + (t * h_q + h) * d, Ob + ((long long)r * tq + t) * d,
                       (size_t)d * sizeof(float));
        }
    }
    SPUR_AFREE(lb);
    SPUR_AFREE(work);
}
