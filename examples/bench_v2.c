/* bench_v2.c — précision + débit des noyaux SPEAR v2 vs v1 (SpearVM) vs libm.
   Écrit results/bench_v2.json
   Build :
     gcc -O3 -mavx2 -mfma -fopenmp -c spear_ledger_v2.c -o spear_ledger_v2.o
     gcc -O3 -mavx2 -mfma -fopenmp bench_v2.c spear_ledger_v2.o \
         ../repos/SpearVM/bin/spur_kernels.so -o bench_v2 -lm
   Exportés par spur_kernels.so : spur_batch_gelu/erf/tanh (v1 AVX2).      */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

void spur_batch_gelu(const double*, double*, long long);
void spur_batch_erf (const double*, double*, long long);
void spur_batch_tanh(const double*, double*, long long);

void spv2_batch_gelu(const double*, double*, long long);
void spv2_batch_erf (const double*, double*, long long);
void spv2_batch_tanh(const double*, double*, long long);
void spv2_batch_sigmoid(const double*, double*, long long);

#define N  (4 * 1024 * 1024)
#define GRID 2000001

static double *x, *out;

static double now_ns(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e9 + ts.tv_nsec;
}
static volatile double sink;
static void fill_domain(void){
    for(long long i=0;i<N;i++) x[i] = -6.0 + 12.0*(double)i/(N-1);
    /* plus un tirage quasi-uniforme pour couvrir le domaine entier */
}

static double med_of(void (*fn)(const double*, double*, long long), int reps){
    double t[15];
    for(int r=0;r<reps;r++){
        double t0 = now_ns();
        fn(x, out, N);
        double t1 = now_ns();
        t[r] = t1 - t0;
        sink += out[(r*7919) & (N-1)];
    }
    /* médiane */
    for(int a=0;a<reps;a++) for(int b=a+1;b<reps;b++)
        if(t[b]<t[a]){double tmp=t[a];t[a]=t[b];t[b]=tmp;}
    return t[reps/2] / N;   /* ns / élément */
}
static double med_scalar(double (*fn)(double), int reps){
    double t[15];
    for(int r=0;r<reps;r++){
        double acc=0; double t0 = now_ns();
        for(long long i=0;i<N;i++) acc += fn(x[i]);
        double t1 = now_ns();
        t[r] = t1-t0; sink += acc;
    }
    for(int a=0;a<reps;a++) for(int b=a+1;b<reps;b++)
        if(t[b]<t[a]){double tmp=t[a];t[a]=t[b];t[b]=tmp;}
    return t[reps/2] / N;
}

/* références IEEE strictes */
static double ref_gelu(double v){ return 0.5*v*(1.0+erf(v*0.70710678118654752)); }
static double ref_sig (double v){ return 1.0/(1.0+exp(-v)); }

typedef struct { char name[24]; double err; } ErrRow;

static double err_vs(void (*fn)(const double*, double*, long long),
                     double (*ref)(double)){
    fn(x, out, N);
    double worst = 0;
    for(long long i=0;i<N;i++){
        double e = fabs(out[i]-ref(x[i]));
        if(e>worst) worst=e;
    }
    return worst;
}
static double err_scalar(double (*fn)(double), double (*ref)(double)){
    double worst=0;
    for(long long i=0;i<N;i++){
        double e=fabs(fn(x[i])-ref(x[i]));
        if(e>worst)worst=e;
    }
    return worst;
}

int main(void){
    x = aligned_alloc(64, N*sizeof(double));
    out = aligned_alloc(64, N*sizeof(double));
    fill_domain();

    printf("=== SPEAR unified : precision + debit (N=%d, OMP=%s) ===\n", N,
           getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "auto");

    /* ---------- precision ---------- */
    ErrRow errs[] = {
        {"gelu_v1(SpearVM)", err_vs(spur_batch_gelu, ref_gelu)},
        {"gelu_v2(spv2)",    err_vs(spv2_batch_gelu, ref_gelu)},
        {"erf_v1(SpearVM)",  err_vs(spur_batch_erf, erf)},
        {"erf_v2(spv2)",     err_vs(spv2_batch_erf, erf)},
        {"tanh_v1(SpearVM)", err_vs(spur_batch_tanh, tanh)},
        {"tanh_v2(spv2)",    err_vs(spv2_batch_tanh, tanh)},
        {"sigmoid_v2(spv2)", err_vs(spv2_batch_sigmoid, ref_sig)},
    };
    double g_libm_err = err_scalar(ref_gelu, ref_gelu);
    (void)g_libm_err;

    /* ---------- debit ---------- */
    double t_gelu_v1   = med_of(spur_batch_gelu, 7);
    double t_gelu_v2   = med_of(spv2_batch_gelu, 7);
    double t_erf_v1    = med_of(spur_batch_erf, 7);
    double t_erf_v2    = med_of(spv2_batch_erf, 7);
    double t_tanh_v1   = med_of(spur_batch_tanh, 7);
    double t_tanh_v2   = med_of(spv2_batch_tanh, 7);
    double t_sig_v2    = med_of(spv2_batch_sigmoid, 7);
    double s_gelu_ref  = med_scalar(ref_gelu, 3);
    double s_erf       = med_scalar(erf, 3);
    double s_tanh      = med_scalar(tanh, 3);

    printf("\n%-22s %12s %14s\n","kernel","max_err","ns/elem");
    printf("----------------------------------------------\n");
    printf("%-22s %12.3e %14s\n","gelu v1 SpearVM", errs[0].err, "AVX2");
    printf("%-22s %12.3e %14.3f\n","gelu v2 spv2",      errs[1].err, t_gelu_v2);
    printf("%-22s %12.3e %14s\n","erf v1 SpearVM",  errs[2].err, "AVX2");
    printf("%-22s %12.3e %14.3f\n","erf v2 spv2",      errs[3].err, t_erf_v2);
    printf("%-22s %12.3e %14s\n","tanh v1 SpearVM", errs[4].err, "AVX2");
    printf("%-22s %12.3e %14.3f\n","tanh v2 spv2",      errs[5].err, t_tanh_v2);
    printf("%-22s %12.3e %14.3f\n","sigmoid v2 spv2",   errs[6].err, t_sig_v2);
    printf("\nlibm scalaire ns/elem : gelu=%.2f erf=%.2f tanh=%.2f\n",
           s_gelu_ref, s_erf, s_tanh);
    printf("v1 batch AVX2 ns/elem : gelu=%.3f erf=%.3f tanh=%.3f\n",
           t_gelu_v1, t_erf_v1, t_tanh_v1);
    printf("v2 batch AVX2 ns/elem : gelu=%.3f erf=%.3f tanh=%.3f sig=%.3f\n",
           t_gelu_v2, t_erf_v2, t_tanh_v2, t_sig_v2);
    printf("speedup v2 vs libm    : gelu=x%.2f erf=x%.2f tanh=x%.2f\n",
           s_gelu_ref/t_gelu_v2, s_erf/t_erf_v2, s_tanh/t_tanh_v2);
    printf("cout precision v1->v2 : gelu=x%.2f plus lent, erf=x%.2f, tanh=x%.2f\n",
           t_gelu_v2/t_gelu_v1, t_erf_v2/t_erf_v1, t_tanh_v2/t_tanh_v1);

    /* ---------- JSON ---------- */
    FILE* f = fopen("/home/z/my-project/spear-unified/results/bench_v2.json","w");
    if(!f){ perror("json"); return 1; }
    fprintf(f,
"{\n  \"N\": %d,\n  \"omp_threads\": \"%s\",\n  \"rows\": [\n", N,
getenv("OMP_NUM_THREADS")?getenv("OMP_NUM_THREADS"):"auto");
    const char* nms[]={"gelu_v1","gelu_v2","erf_v1","erf_v2","tanh_v1","tanh_v2","sigmoid_v2"};
    double er[] ={errs[0].err,errs[1].err,errs[2].err,errs[3].err,errs[4].err,errs[5].err,errs[6].err};
    double tt[] ={t_gelu_v1,t_gelu_v2,t_erf_v1,t_erf_v2,t_tanh_v1,t_tanh_v2,t_sig_v2};
    for(int i=0;i<7;i++){
        fprintf(f,"    {\"kernel\":\"%s\",\"max_err\":%.3e,\"ns_per_elem\":%.3f}%s\n",
                nms[i],er[i],tt[i],i==6?"":",");
    }
    fprintf(f,"  ],\n  \"libm_scalar_ns_per_elem\":{\"gelu\":%.3f,\"erf\":%.3f,\"tanh\":%.3f},\n"
              "  \"speedup_v2_vs_libm\":{\"gelu\":%.2f,\"erf\":%.2f,\"tanh\":%.2f}\n}\n",
        s_gelu_ref,s_erf,s_tanh,
        s_gelu_ref/t_gelu_v2, s_erf/t_erf_v2, s_tanh/t_tanh_v2);
    fclose(f);
    printf("\n[ok] results/bench_v2.json\n");
    return 0;
}
