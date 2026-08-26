/* spur_nn.c — Pipeline d'inférence NN vectorisé AVX2.
   Chaque couche : produit matrice-vecteur + activation SPEAR.
   Usage autonome : gcc -O3 -march=native -mavx2 -mfma -fopenmp -o spur_nn spur_nn.c -lm */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MAX_LAYERS 8

/* ================= Types d'activation ================= */
enum { ACT_GELU=0, ACT_SIGMOID=1, ACT_TANH=2 };

/* ================= Noyaux scalaires ================= */
static double sc_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}
static double sc_erf(double x){ return erf(x); }
static double sc_tanh(double x){ return tanh(x); }

/* ================= Noyaux vectorisés AVX2 ================= */
static inline __m256d v_gelu(__m256d x){
    __m256d c1=_mm256_set1_pd(0.306923),c2=_mm256_set1_pd(0.501);
    __m256d cm=_mm256_set1_pd(1.002),ck=_mm256_set1_pd(0.997729),cb=_mm256_set1_pd(-0.004004);
    __m256d z=_mm256_setzero_pd();
    __m256d u=_mm256_fmadd_pd(c1,x,c2);
    u=_mm256_max_pd(u,z); u=_mm256_min_pd(u,cm);
    return _mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(x,u),ck),cb);
}
static inline __m256d v_tanh(__m256d x){
    __m256d hi=_mm256_set1_pd(3.0),lo=_mm256_set1_pd(-3.0);
    __m256d y=_mm256_max_pd(lo,_mm256_min_pd(hi,x));
    __m256d t=_mm256_mul_pd(y,y);
    __m256d num=_mm256_add_pd(y,_mm256_mul_pd(_mm256_set1_pd(0.053639),t));
    __m256d den=_mm256_add_pd(_mm256_set1_pd(0.90122),_mm256_mul_pd(_mm256_set1_pd(0.343141),t));
    return _mm256_mul_pd(_mm256_set1_pd(0.900021),_mm256_div_pd(num,den));
}

/* ---- Activation par type sur un vecteur AVX2 ---- */
static inline __m256d v_activate(__m256d x,int act){
    switch(act){
    case ACT_GELU: return v_gelu(x);
    case ACT_TANH: return v_tanh(x);
    default: return x;
    }
}
static inline double sc_activate(double x,int act){
    switch(act){
    case ACT_GELU: return sc_gelu(x);
    case ACT_TANH: return tanh(x);
    default: return x;
    }
}

/* ================= Couche réseau ========================================== */
typedef struct {
    double* W;       /* [n_out × n_in] row-major */
    double* b;       /* [n_out]                  */
    int n_in,n_out;
    int act;
} Layer;

static Layer net[MAX_LAYERS];
static int nl = 0;

void nn_add_layer(const double* W,const double* bias,int n_out,int n_in,int act){
    if(nl>=MAX_LAYERS) return;
    net[nl].W=(double*)malloc(n_out*n_in*sizeof(double));
    net[nl].b=(double*)malloc(n_out*sizeof(double));
    memcpy(net[nl].W,W,n_out*n_in*sizeof(double));
    memcpy(net[nl].b,bias,n_out*sizeof(double));
    net[nl].n_in=n_in; net[nl].n_out=n_out; net[nl].act=act;
    nl++;
}

/* ---- Couche linéaire + activation : version native libm ---- */
static void layer_native(const Layer* l,const double* x,double* out){
    for(int j=0;j<l->n_out;j++){
        const double* w=l->W+j*l->n_in;
        double dot=0;
        for(int k=0;k<l->n_in;k++) dot+=w[k]*x[k];
        out[j]=tanh(dot+l->b[j]);
    }
}

/* ---- Couche linéaire + activation : version AVX2 ---- */
static void layer_avx(const Layer* l,const double* x,double* out){
    int ni=l->n_in,no=l->n_out;
    #pragma omp parallel for schedule(static)
    for(int j=0;j<no;j++){
        const double* w=l->W+j*ni;
        __m256d acc=_mm256_setzero_pd();
        int k=0;
        for(;k+3<ni;k+=4){
            __m256d wv=_mm256_loadu_pd(w+k);
            __m256d xv=_mm256_loadu_pd(x+k);
            acc=_mm256_fmadd_pd(wv,xv,acc);
        }
        double dot=0;
        /* réduction horizontale */
        double tmp[4]; _mm256_storeu_pd(tmp,acc);
        for(int q=0;q<4;q++)dot+=tmp[q];
        for(;k<ni;k++) dot+=w[k]*x[k];
        out[j]=sc_activate(dot+l->b[j],l->act);
    }
}

/* ================= Forward pass multi-couches ============================= */
static void nn_forward_native(const double* input,double* output,long long count){
    if(nl==0)return;
    double* buf_a=malloc(count*net[0].n_out*sizeof(double));
    double* buf_b=malloc(count*net[0].n_out*sizeof(double));
    double* cur=buf_a;
    /* première couche */
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<count;i++)
        layer_native(&net[0],input+i*net[0].n_in,cur+i*net[0].n_out);
    /* couches suivantes */
    for(int l=1;l<nl;l++){
        double* nxt=(l==nl-1)?output:(l%2==1?buf_b:buf_a);
        #pragma omp parallel for schedule(static)
        for(long long i=0;i<count;i++)
            layer_native(&net[l],cur+i*net[l].n_in,nxt+i*net[l].n_out);
        cur=nxt;
    }
    free(buf_a);free(buf_b);
}

static void nn_forward_avx(const double* input,double* output,long long count){
    if(nl==0)return;
    double* buf_a=malloc(count*net[0].n_out*sizeof(double));
    double* buf_b=malloc(count*net[0].n_out*sizeof(double));
    double* cur=buf_a;
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<count;i++)
        layer_avx(&net[0],input+i*net[0].n_in,cur+i*net[0].n_out);
    for(int l=1;l<nl;l++){
        double* nxt=(l==nl-1)?output:(l%2==1?buf_b:buf_a);
        #pragma omp parallel for schedule(static)
        for(long long i=0;i<count;i++)
            layer_avx(&net[l],cur+i*net[l].n_in,nxt+i*net[l].n_out);
        cur=nxt;
    }
    free(buf_a);free(buf_b);
}

/* ================= Démo / Benchmark ======================================= */
int main(void){
    printf("=== SpearVM NN Inference ===\n\n");

    /* Architecture : 64 -> 32 -> 16 -> 1 */
    int dims[]={64,32,16,1};
    int acts[]={ACT_GELU,ACT_TANH,ACT_SIGMOID};
    int nlayers=3;

    srand(42);

    /* construction des couches */
    for(int l=0;l<nlayers;l++){
        int ni=dims[l],no=dims[l+1];
        double* W=malloc(no*ni*sizeof(double));
        double* b=malloc(no*sizeof(double));
        for(int i=0;i<no*ni;i++) W[i]=((double)rand()/RAND_MAX-0.5)*2.0/sqrt(ni);
        for(int j=0;j<no;j++) b[j]=((double)rand()/RAND_MAX-0.5)*0.1;
        nn_add_layer(W,b,no,ni,acts[l]);
        free(W);free(b);
        printf("layer %d: %d -> %d (act=%d)\n",l,ni,no,acts[l]);
    }

    long long batch_size=100000; /* nombre d'échantillons à inférer */
    double* input=malloc(batch_size*dims[0]*sizeof(double));
    double* out_nat=malloc(batch_size*dims[nlayers]*sizeof(double));
    double* out_avx=malloc(batch_size*dims[nlayers]*sizeof(double));

    for(long long i=0;i<batch_size*dims[0];i++)
        input[i]=((double)rand()/RAND_MAX-0.5)*2.0;

    /* warmup */
    nn_forward_native(input,out_nat,10);
    nn_forward_avx(input,out_avx,10);

    /* benchmark native */
    clock_t t0=clock();
    for(int r=0;r<3;r++) nn_forward_native(input,out_nat,batch_size);
    double sec_nat=(double)(clock()-t0)/CLOCKS_PER_SEC/3;

    /* benchmark AVX2 */
    t0=clock();
    for(int r=0;r<3;r++) nn_forward_avx(input,out_avx,batch_size);
    double sec_avx=(double)(clock()-t0)/CLOCKS_PER_SEC/3;

    printf("\nnative libm : %8.1f ms (%lld samples)\n",sec_nat*1000,batch_size);
    printf("AVX2 turbo   : %8.1f ms (%lld samples)\n",sec_avx*1000,batch_size);
    printf("throughput   : %.0f samples/sec (AVX2)\n",batch_size/sec_avx);

    /* validation : compare les deux implémentations */
    double max_diff=0;
    for(long long i=0;i<batch_size*dims[nlayers];i++){
        double df=fabs(out_nat[i]-out_avx[i]);
        if(df>max_diff)max_diff=df;
    }
    printf("validation max_diff=%.9f %s\n",max_diff,max_diff<1e-12?"OK":"DIFF");

    free(input);free(out_nat);free(out_avx);
    for(int l=0;l<nl;l++){free(net[l].W);free(net[l].b);}
    return 0;
}
