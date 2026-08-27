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

void spur_matmul_nt(const double* A,const double* B,double* C,
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
void spur_matmul_nt_gelu(const double* A,const double* B,
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

void spur_matmul_nt_f32(const float* A,const float* B,float* C,
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
                    /* reduction horizontale f32 */
                    cr[j]     +=hsum8(v0); cr[n+j]    +=hsum8(v1);
                    cr[2*n+j] +=hsum8(v2); cr[3*n+j]  +=hsum8(v3);
                    cr[4*n+j] +=hsum8(v4); cr[5*n+j]  +=hsum8(v5);
                    cr[6*n+j] +=hsum8(v6); cr[7*n+j]  +=hsum8(v7);
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

void spur_matmul_nt_gelu_f32(const float* A,const float* B,
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
                __m256 acc=_mm256_setzero_ps();
                long long q=0;
                for(;q+7<k;q+=8)
                    acc=_mm256_fmadd_ps(_mm256_loadu_ps(ar+q),
                                        _mm256_loadu_ps(br+q),acc);
                float s=hsum8(acc);
                for(;q<k;q++) s+=ar[q]*br[q];
                cr[j]=gelu_f32_scalar(s+(bias?bias[j]:0.0f));
                /* lignes 1..7 du bloc : memes offsets que spur_matmul_nt_f32 */
                for(long long r2=1;r2<8;r2++){
                    __m256 vr=_mm256_setzero_ps();
                    const float* arr=ar+(size_t)r2*k;
                    for(q=0;q+7<k;q+=8)
                        vr=_mm256_fmadd_ps(_mm256_loadu_ps(arr+q),
                                           _mm256_loadu_ps(br+q),vr);
                    float sr=hsum8(vr);
                    for(;q<k;q++) sr+=arr[q]*br[q];
                    cr[(size_t)r2*n+j]=gelu_f32_scalar(sr+(bias?bias[j]:0.0f));
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
