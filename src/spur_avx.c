/* SpearVM Turbo — VM coprocesseur SPEAR vectorisée AVX2 + OpenMP.
   Chaque instruction traite 4 doubles simultanément (registres __m256d).
   Parallélisé multi-cœur via OpenMP si disponible.
   Compile : gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared -o spur_turbo.dll spur_avx.c -lm */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- Noyaux scalaires ---- */
static inline double clampd(double x,double lo,double hi){ return x<lo?lo:(x>hi?hi:x); }
double spur_k_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}
double spur_k_erf(double x){
    x=clampd(x,-2,2);
    return 1.106774*((x+0.034298*x*x*x)/(0.995+0.378089*x*x));
}
double spur_k_tanh(double x){
    x=clampd(x,-3,3);
    return 0.900021*((x+0.053639*x*x*x)/(0.90122+0.343141*x*x));
}
double spur_k_lse2(double a,double b){ return fmax(a,b); }

/* ---- Noyaux vectorisés AVX2 ---- */
static inline __m256d v_gelu(__m256d x){
    /* gelu(x) : 0.997729*(x*min(1.002,max(0,0.306923x+0.501)))-0.004004 */
    __m256d c306=_mm256_set1_pd(0.306923),c501=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002),ck=_mm256_set1_pd(0.997729),cb=_mm256_set1_pd(-0.004004);
    __m256d z=_mm256_setzero_pd();
    __m256d u=_mm256_fmadd_pd(c306,x,c501);
    u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
    __m256d r=_mm256_mul_pd(x,u);
    return _mm256_add_pd(_mm256_mul_pd(r,ck),cb);
}
static inline __m256d v_erf(__m256d x){
    /* erf(x) rationnel sur [-2,2] */
    __m256d hi=_mm256_set1_pd(2.0),lo=_mm256_set1_pd(-2.0);
    __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,x));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_fmadd_pd(_mm256_set1_pd(0.034298),t,y);
    __m256d den=_mm256_fmadd_pd(_mm256_set1_pd(0.378089),t,_mm256_set1_pd(0.995));
    return _mm256_mul_pd(_mm256_set1_pd(1.106774),_mm256_div_pd(num,den));
}
static inline __m256d v_tanh(__m256d x){
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,x));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_fmadd_pd(_mm256_set1_pd(0.053639),t,y);
    __m256d den=_mm256_fmadd_pd(_mm256_set1_pd(0.343141),t,_mm256_set1_pd(0.90122));
    return _mm256_mul_pd(_mm256_set1_pd(0.900021),_mm256_div_pd(num,den));
}

/* ---- Gestion des kernels ---- */
#define MAXK 16
typedef struct { SpurIns* prog; int n; } SpurKernel;
static SpurKernel K[MAXK];
static int nk = 0;

int spur_map_build(const SpurIns* body,int n){
    if(n<1||n>256||nk>=MAXK) return -1;
    K[nk].prog=(SpurIns*)malloc(n*sizeof(SpurIns));
    if(!K[nk].prog) return -1;
    memcpy(K[nk].prog,body,n*sizeof(SpurIns));
    K[nk].n=n;
    return nk++;
}

/* ---- Éxecution AVX2 vectorisée + parallèle ---- */
void spur_map_exec(int handle,const double* restrict in,double* restrict out,long long count){
    if(handle<0||handle>=nk||count<1) return;
    const SpurIns* P=K[handle].prog;
    const int ni=K[handle].n;

    long long vec_end=count&~3LL;

    #pragma omp parallel for schedule(static)
    for(long long blk=0;blk<vec_end;blk+=4){
        /* registres virtuels AVX2 */
        __m256d v0,v1,v2,v3,v4,v5,v6,v7;
        v0=v1=v2=v3=v4=v5=v6=v7=_mm256_setzero_pd();

        const double* src=in+blk;

        /* interprétation du programme pour CE bloc de 4 éléments */
        for(int pc=0;pc<ni;pc++){
            const SpurIns* I=&P[pc];
            switch(I->op){
            case MOVI:  v[I->dst]=_mm256_set1_pd(I->imm); break;
            case ADDI:  v[I->dst]=_mm256_add_pd(v[I->a],_mm256_set1_pd(I->imm)); break;
            case MULI:  v[I->dst]=_mm256_mul_pd(v[I->a],_mm256_set1_pd(I->imm)); break;
            case SUBI:  v[I->dst]=_mm256_sub_pd(v[I->a],_mm256_set1_pd(I->imm)); break;
            case ADD:   v[I->dst]=_mm256_add_pd(v[I->a],v[I->b]); break;
            case SUB:   v[I->dst]=_mm256_sub_pd(v[I->a],v[I->b]); break;
            case MUL:   v[I->dst]=_mm256_mul_pd(v[I->a],v[I->b]); break;
            case GELU:  v[I->dst]=v_gelu(_mm256_mul_pd(v[I->a],_mm256_set1_pd(I->imm))); break;
            case ERF:   v[I->dst]=v_erf(_mm256_mul_pd(v[I->a],_mm256_set1_pd(I->imm))); break;
            case TANH:  v[I->dst]=v_tanh(_mm256_mul_pd(v[I->a],_mm256_set1_pd(I->imm))); break;
            case LSE2:  v[I->dst]=_mm256_max_pd(v[I->a],v[I->b]); break;
            case TANHA: { /* tanh(a+b) */
                __m256d sum=_mm256_add_pd(v[I->a],v[I->b]);
                v[I->dst]=v_tanh(sum);
            } break;
            case TANHS: { /* tanh(a-b) */
                __m256d dif=_mm256_sub_pd(v[I->a],v[I->b]);
                v[I->dst]=v_tanh(dif);
            } break;
            case ERFA: { /* erf(a+b) */
                __m256d sum=_mm256_add_pd(v[I->a],v[I->b]);
                v[I->dst]=v_erf(sum);
            } break;
            default: break;
            }
        }

        /* stocke le résultat de v5 (dernier calcul) dans out[blk..blk+3] */
        _mm256_storeu_pd(out+blk,r[5]);
    }

    /* fallback scalaire pour count % 4 éléments restants */
    for(long long i=vec_end;i<count;i++){
        double v[8]; memset(v,0,sizeof(v));
        double x=in[i];
        for(int pc=0;pc<ni;pc++){
            const SpurIns* I=&P[pc];
            switch(I->op){
            case MOVI: v[I->dst]=I->imm; break;
            case ADDI: v[I->dst]=v[I->a]+I->imm; break;
            case MULI: v[I->dst]=v[I->a]*I->imm; break;
            case SUBI: v[I->dst]=v[I->a]-I->imm; break;
            case ADD:  v[I->dst]=v[I->a]+v[I->b]; break;
            case SUB:  v[I->dst]=v[I->a]-v[I->b]; break;
            case MUL:  v[I->dst]=v[I->a]*v[I->b]; break;
            case GELU: v[I->dst]=spur_k_gelu(v[I->a]*I->imm); break;
            case ERF:  v[I->dst]=spur_k_erf(v[I->a]*I->imm); break;
            case TANH: v[I->dst]=spur_k_tanh(v[I->a]*I->imm); break;
            case LSE2: v[I->dst]=fmax(v[I->a],v[I->b]); break;
            default: break;
            }
        }
        out[i]=v[5];
    }
}
