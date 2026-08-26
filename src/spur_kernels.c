/* SpearVM Kernels — 4 noyaux transcendantals vectorisés AVX2 + OpenMP.
   Chaque kernel traite un tableau élément-par-élément en SIMD.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -o spur_kernels.dll spur_kernels.c -lm */
#include <math.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================= GELU ================= */
void spur_batch_gelu(const double* x,double* out,long long n){
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

/* ================= ERF ================= */
void spur_batch_erf(const double* x,double* out,long long n){
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
static inline double gelu_s(double x){
    double u=0.306923*x+0.501;
    if(u<0.0)u=0.0; if(u>1.002)u=1.002;
    return 0.997729*(x*u)-0.004004;
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
void spur_matmul_nt_gelu(const double* A,const double* B,double* C,
                         long long m,long long k,long long n){
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
                cr[j]=gelu_s(hs256(v0)+d0);
                cr[n+j]=gelu_s(hs256(v1)+d1);
                cr[2*n+j]=gelu_s(hs256(v2)+d2);
                cr[3*n+j]=gelu_s(hs256(v3)+d3);
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
                tr[j]=gelu_s(s);
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
