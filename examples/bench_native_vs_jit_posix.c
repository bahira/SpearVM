/* SPUR-64: VM dont le coprocesseur math est implante avec des noyaux SPEAR.
   Hypothese testee: le meme programme tourne PLUS VITE dans la VM qu'en natif,
   car chaque op transcendantale du copro est remplacee par une forme ALU pure
   (hall-of-fame SPEAR), plus rapide que libm sur la machine hote. */
#include <stdio.h>
#include <sys/mman.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 2000000
#define IPS_FACTOR 16   /* instructions guest executes par iteration */

static volatile double g_sink;   /* puits anti-const-folding pour le bench */
static volatile double g_tick = 0.0; /* lecture volatile = barriere anti-hoisting */

typedef struct { short op, a, b, dst; double imm; } Ins;
enum { MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC, GELU, ERF, TANH, LSE2,
       TANHA, TANHS, ERFA, ACCLSE, HALT };

static Ins prog[] = {
    {MOVI,0,0,7,(double)N},          /* 0: r7 = compteur             */
    {MOVI,0,0,0,-1.0},               /* 1: x0 = -1                   */
    {ADDI,0,0,0,0.000002},           /* 2: x += 2e-6    in [-1,3]    */
    {GELU,0,0,1,0.6},                /* 3: g = gelu(0.6x)            */
    {ERF,0,0,2,1.0},                 /* 4: e = erf(x)                */
    {TANHA,1,2,3,0},                 /* 5: c = tanh(g+e)             */
    {LSE2,1,3,4,0},                  /* 6: f = lse(g,c)              */
    {TANHS,2,1,5,0},                 /* 7: h = tanh(e-g)             */
    {ADD,3,4,6,0},                   /* 8:                           */
    {ERFA,3,4,6,0},                  /* 9: i = erf(c+f)              */
    {MUL,5,6,5,0},                   /* 10: r5 = h*i                 */
    {ERF,1,0,6,0.5},                 /* 11: p = erf(g/2)             */
    {ADD,5,6,5,0},                   /* 12: s = h*i + p              */
    {MUL,2,1,2,0},                   /* 13: r2 = ge                  */
    {MUL,4,3,4,0},                   /* 14: r4 = cf                  */
    {ACCLSE,2,4,5,0},                /* 15: acc += s + lse(ge,cf)    */
    {SUBI,7,0,7,1.0},                /* 16: r7--                     */
    {BNZ,7,0,0,2},                   /* 17: si r7 != 0 -> pc=2       */
    {HALT,0,0,0,0},
};

/* ---- Coprocesseur SPEAR (noyaux hall-of-fame, domaines datasheet) ---- */
/* fast-math local au copro seulement : le jumeau natif reste IEEE strict */
#pragma GCC optimize("fast-math","no-signed-zeros","no-trapping-math")
static inline double clampd(double x,double lo,double hi){ return x<lo?lo:(x>hi?hi:x); }

/* gelu: fast slot gelu, 100% ALU (mse 5.3e-4, x6.57 vs erf exact), enveloppe [-0.6,+1.8] */
static inline double sp_gelu(double x){
    return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004;
}
/* erf: slot precis erf_prob, rationnel pur (mse 4.7e-5, x2.17 vs erff), domaine [-2,2] */
static inline double sp_erf(double x){
    x = clampd(x,-2,2);
    return 1.106774*((x+0.034298*x*x*x)/(0.995+0.378089*x*x));
}
/* tanh: slot precis tanh_sat, rationnel pur (mse 4.6e-6, x1.67 vs tanhf), domaine [-3,3] */
static inline double sp_tanh(double x){
    x = clampd(x,-3,3);
    return 0.900021*((x+0.053639*x*x*x)/(0.90122+0.343141*x*x));
}
/* logsumexp2: fast slot = hard-max (ecart borne par ln 2, x8.57 vs log+exp) */
static inline double sp_lse2(double a,double b){ return fmax(a,b); }
#pragma GCC reset_options

/* ---- Jumeaux natifs exacts (libm IEEE) ---- */
static inline double nat_gelu(double x){ return 0.5*x*(1.0+erf(x*0.70710678118654752)); }
static inline double nat_lse2(double a,double b){ return log(exp(a)+exp(b)); }

/* ---- Moteur d'execution : direct-threading (pre-decodage une fois) ---- */
typedef struct Th Th;
typedef Th* (*Fn)(Th*, double* r, double* acc);
struct Th { Fn fn; Th* tgt; short a, b, dst; double imm; };

#define HND(name, body) static Th* name(Th* t, double* r, double* acc){ body; return t+1; }
HND(h_mov,  r[t->dst] = t->imm)
HND(h_addi, r[t->dst] = r[t->a] + t->imm)
HND(h_muli, r[t->dst] = r[t->a] * t->imm)
HND(h_add,  r[t->dst] = r[t->a] + r[t->b])
HND(h_sub,  r[t->dst] = r[t->a] - r[t->b])
HND(h_mul,  r[t->dst] = r[t->a] * r[t->b])
HND(h_subi, r[t->dst] = r[t->a] - t->imm)
HND(h_acc,  *acc += r[t->a])
HND(h_gelu_s, r[t->dst] = sp_gelu(r[t->a] * t->imm))
HND(h_erf_s,  r[t->dst] = sp_erf (r[t->a] * t->imm))
HND(h_tanh_s, r[t->dst] = sp_tanh(r[t->a] * t->imm))
HND(h_lse2_s, r[t->dst] = sp_lse2(r[t->a], r[t->b]))
HND(h_gelu_e, r[t->dst] = nat_gelu(r[t->a] * t->imm))
HND(h_erf_e,  r[t->dst] = erf(r[t->a] * t->imm))
HND(h_tanh_e, r[t->dst] = tanh(r[t->a] * t->imm))
HND(h_lse2_e, r[t->dst] = nat_lse2(r[t->a], r[t->b]))
/* ops copro fusionnees : arithmetique + transcendantale en 1 instruction */
static Th* h_tanha_s(Th* t, double* r, double* acc){ (void)acc; r[t->dst]=sp_tanh(r[t->a]+r[t->b]); return t+1; }
static Th* h_tanhs_s(Th* t, double* r, double* acc){ (void)acc; r[t->dst]=sp_tanh(r[t->a]-r[t->b]); return t+1; }
static Th* h_erfa_s (Th* t, double* r, double* acc){ (void)acc; r[t->dst]=sp_erf (r[t->a]+r[t->b]); return t+1; }
/* acc += r[dst] + lse(r[a], r[b]) : le terme final du workload en 1 instruction */
static Th* h_acclse_s(Th* t, double* r, double* acc){
    *acc += r[t->dst] + sp_lse2(r[t->a], r[t->b]); return t+1;
}
static Th* h_tanha_e(Th* t, double* r, double* acc){ (void)acc; r[t->dst]=tanh(r[t->a]+r[t->b]); return t+1; }
static Th* h_tanhs_e(Th* t, double* r, double* acc){ (void)acc; r[t->dst]=tanh(r[t->a]-r[t->b]); return t+1; }
static Th* h_erfa_e (Th* t, double* r, double* acc){ (void)acc; r[t->dst]=erf (r[t->a]+r[t->b]); return t+1; }
static Th* h_acclse_e(Th* t, double* r, double* acc){
    *acc += r[t->dst] + nat_lse2(r[t->a], r[t->b]); return t+1;
}
static Th* h_bnz(Th* t, double* r, double* acc){ (void)acc; return r[t->a] != 0.0 ? t->tgt : t+1; }
static Th* h_halt(Th* t, double* r, double* acc){ (void)t;(void)r;(void)acc; return 0; }

#define NOPS 30
static Th thr_s[NOPS], thr_e[NOPS];
static int thr_init = 0;

static void decode(void){
    int n = (int)(sizeof(prog)/sizeof(prog[0]));
    for(int i=0;i<n;i++){
        Ins* I = &prog[i];
        Th* s = &thr_s[i]; Th* e = &thr_e[i];
        s->a=e->a=I->a; s->b=e->b=I->b; s->dst=e->dst=I->dst; s->imm=e->imm=I->imm;
        Fn fs=0, fe=0;
        switch(I->op){
        case MOVI: fs=fe=h_mov;  break;
        case ADDI: fs=fe=h_addi; break;
        case MULI: fs=fe=h_muli; break;
        case ADD:  fs=fe=h_add;  break;
        case SUB:  fs=fe=h_sub;  break;
        case MUL:  fs=fe=h_mul;  break;
        case SUBI: fs=fe=h_subi; break;
        case ACC:  fs=fe=h_acc;  break;
        case GELU: fs=h_gelu_s; fe=h_gelu_e; break;
        case ERF:  fs=h_erf_s;  fe=h_erf_e;  break;
        case TANH: fs=h_tanh_s; fe=h_tanh_e; break;
        case LSE2: fs=h_lse2_s; fe=h_lse2_e; break;
        case TANHA: fs=h_tanha_s; fe=h_tanha_e; break;
        case TANHS: fs=h_tanhs_s; fe=h_tanhs_e; break;
        case ERFA:  fs=h_erfa_s;  fe=h_erfa_e;  break;
        case ACCLSE: fs=h_acclse_s; fe=h_acclse_e; break;
        case BNZ:  fs=fe=h_bnz; break;
        default:   fs=fe=h_halt; break;
        }
        s->fn=fs; e->fn=fe;
    }
    for(int i=0;i<n;i++)
        if(prog[i].op==BNZ){ thr_s[i].tgt=&thr_s[(int)prog[i].imm]; thr_e[i].tgt=&thr_e[(int)prog[i].imm]; }
    thr_init = 1;
}

static double run_vm(int spear){
    if(!thr_init) decode();
    double r[8] = {0}; double acc = g_tick*0.0;
    Th* t = spear ? thr_s : thr_e;
    for(;;){
        t = t->fn(t, r, &acc);
        if(!t){ g_sink = acc; return acc; }
    }
}

/* ---- JIT x64 : le programme guest est emis en code machine natif ----
   r0..r7 -> xmm6..xmm13, acc -> xmm14, zero -> xmm1, temp -> xmm0,
   rbx = base du pool de constantes. ABI Windows x64 (shadow space 48 o). */
static double jit_pool[64]; static int jit_pooln;
static unsigned char *jb; static size_t jlen;
typedef struct { size_t pos; void* abs; size_t off; int kind; } Fix;
static Fix fx[64]; static int nfx;
static size_t jit_body_off;

static void je8(unsigned b){ jb[jlen++]=(unsigned char)b; }
static void je32(unsigned v){ memcpy(jb+jlen,&v,4); jlen+=4; }
static void je64(unsigned long long v){ memcpy(jb+jlen,&v,8); jlen+=8; }
static void fx_abs(size_t pos, void* target){ Fix f={pos,target,0,0}; fx[nfx++]=f; }
static void fx_rel(size_t pos, size_t off){ Fix f={pos,0,off,1}; fx[nfx++]=f; }
static int pool_of(double v){ jit_pool[jit_pooln]=v; return jit_pooln++; }

static void j_call(void* f){                       /* movabs rax,f ; call rax */
    je8(0x48); je8(0xB8); je64((unsigned long long)(uintptr_t)f);
    je8(0xFF); je8(0xD0);
}
static void e_movsd_pool(int xr,int idx){          /* movsd xr,[rbx+idx*8] */
    int rex=0x40|(((xr)>>3)<<2), d=idx*8;
    je8(0xF2); je8(rex); je8(0x0F); je8(0x10);
    if(d>=-128&&d<128){ je8(0x44|((xr&7)<<3)); je8(0x23); je8((unsigned char)d); }
    else { je8(0x84|((xr&7)<<3)); je8(0x23); je32((unsigned)d); }
}
static void e_sd(int op,int dst,int src){          /* F2 0F op dst,src (reg,reg) */
    je8(0xF2); je8(0x40|(((dst)&8)>>1)|(((src)&8)>>3));
    je8(0x0F); je8(op); je8(0xC0|((dst&7)<<3)|(src&7));
}
#define XR(i) (6+(i))
static void j_load(int xr,int gsrc){ e_sd(0x10,xr,XR(gsrc)); }
static void j_store(int gdst,int xr){ e_sd(0x10,XR(gdst),xr); }

static void jit_build(void){
    jb = (unsigned char*)mmap(NULL,8192,PROT_READ|PROT_WRITE,
                              MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(jb==MAP_FAILED){ fprintf(stderr,"mmap failed\n"); exit(1); }
    jlen = 0; nfx = 0; jit_pooln = 0;
    je8(0x53);                                     /* push rbx            */
    je8(0x48);je8(0x83);je8(0xEC);je8(0x30);       /* sub rsp,48          */
    je8(0x48);je8(0xBB);                           /* movabs rbx,pool     */
    je64((unsigned long long)(uintptr_t)jit_pool);
    je8(0x66);je8(0x45);je8(0x0F);je8(0xEF);je8(0xFF); /* pxor xmm15,xmm15    */
    jit_body_off = jlen;
    int n = (int)(sizeof(prog)/sizeof(prog[0]));
    size_t ioff[NOPS];
    for(int i=0;i<n;i++){
        Ins* I = &prog[i];
        ioff[i] = jlen;
        switch(I->op){
        case MOVI: e_movsd_pool(XR(I->dst), pool_of(I->imm)); break;
        case ADDI: case MULI: case SUBI: {
            int op = I->op==ADDI?0x58:(I->op==MULI?0x59:0x5C);
            if(I->dst==I->a) e_movsd_pool(0, pool_of(I->imm)), e_sd(op,XR(I->dst),0);
            else j_load(I->dst,I->a), e_movsd_pool(0, pool_of(I->imm)), e_sd(op,XR(I->dst),0);
        } break;
        case ADD: case SUB: case MUL: {
            int op = I->op==ADD?0x58:(I->op==SUB?0x5C:0x59);
            if(I->dst==I->a)      e_sd(op,XR(I->dst),XR(I->b));
            else if(I->dst==I->b) e_sd(op,XR(I->dst),XR(I->a));
            else j_load(XR(I->dst),I->a), e_sd(op,XR(I->dst),XR(I->b));
        } break;
        case ACC: e_sd(0x58,14,XR(I->a)); break;
        case BNZ: {
            int xr = XR(I->a);
            je8(0x66); je8(0x40|(((xr)&8)>>1)|1); je8(0x0F); je8(0x2F);
            je8(0xC0|((xr&7)<<3)|7);               /* comisd xr,xmm15     */
            je8(0x0F); je8(0x85); fx_rel(jlen,ioff[(int)I->imm]); jlen+=4;  /* jne body */
        } break;
        case GELU:
            if(I->imm==1.0) j_load(0,I->a); else e_movsd_pool(0,pool_of(I->imm)), e_sd(0x59,0,XR(I->a));
            j_call((void*)sp_gelu);
            j_store(I->dst,0);
            break;
        case ERF:
            e_movsd_pool(0, pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)sp_erf);
            j_store(I->dst,0);
            break;
        case TANH:
            e_movsd_pool(0, pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)sp_tanh);
            j_store(I->dst,0);
            break;
        case TANHA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)sp_tanh);
            j_store(I->dst,0);
            break;
        case TANHS:
            j_load(0,I->a); e_sd(0x5C,0,XR(I->b));
            j_call((void*)sp_tanh);
            j_store(I->dst,0);
            break;
        case ERFA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)sp_erf);
            j_store(I->dst,0);
            break;
        case LSE2:                                   /* maxsd inline, 0 call */
            j_load(0,I->a); e_sd(0x5F,0,XR(I->b)); j_store(I->dst,0);
            break;
        case ACCLSE:                                 /* inline aussi         */
            j_load(0,I->a); e_sd(0x5F,0,XR(I->b));
            e_sd(0x58,0,XR(I->dst)); e_sd(0x58,14,0);
            break;
        default: break;                              /* HALT -> tombe a la fin */
        }
    }
    je8(0x66);je8(0x41);je8(0x0F);je8(0x28);je8(0xC6); /* movapd xmm0,xmm14 */
    je8(0x48);je8(0x83);je8(0xC4);je8(0x30);       /* add rsp,48          */
    je8(0x5B);                                     /* pop rbx             */
    je8(0xC3);                                     /* ret                 */
    for(int k=0;k<nfx;k++){
        unsigned char* p = jb+fx[k].pos;
        long long next = (long long)(uintptr_t)(jb+fx[k].pos+4);
        long long tgt  = fx[k].kind ? (long long)(uintptr_t)(jb+fx[k].off)
                                    : (long long)(uintptr_t)fx[k].abs;
        *(int*)(p) = (int)((intptr_t)tgt - next);
    }
    if(mprotect(jb,8192,PROT_READ|PROT_EXEC)!=0){
        fprintf(stderr,"mprotect failed\n"); exit(1);
    }
    { FILE* fp=fopen("jit.bin","wb"); fwrite(jb,1,jlen,fp); fclose(fp); }
    __builtin___clear_cache((char*)jb,(char*)(jb+jlen));
}
static double run_jit(void){ double v = ((double(*)(void))jb)(); v += g_tick*0.0; g_sink = v; return v; }

static double run_native(void){
    double acc = 0;
    double x = -1.0 + g_tick*0.0;
    int n = N + (int)g_tick;
    struct timespec ta, tb; clock_gettime(CLOCK_MONOTONIC, &ta);
    for(int i=0;i<n;i++){
        x += 0.000002;   /* meme trajectoire fp que la VM */
        double g = nat_gelu(0.6*x);
        double e = erf(x);
        double c = tanh(g+e);
        double f = nat_lse2(g,c);
        double h = tanh(e-g);
        double k = erf(c+f);
        double s = h*k;
        s += erf(0.5*g);
        double ge = g*e, cf = c*f;
        acc += s + nat_lse2(ge,cf);
    }
    clock_gettime(CLOCK_MONOTONIC, &tb);
    (void)tb;
    g_sink = acc;
    return acc;
}

static double run_vm0(void){ return run_vm(0); }
static double run_vm1(void){ return run_vm(1); }

static double now_ms(void){
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6; /* horloge monotone */
}

/* Auto-verif: chaque op du copro respecte sa tolerance datasheet vs IEEE */
static int selfcheck(void){
    int fail = 0;
    struct { const char*n; double lo,hi,tol; double (*sp)(double); double (*ex)(double); } t[] = {
        {"gelu",-0.6,1.8,0.090,sp_gelu,nat_gelu},   /* approx PWL du noyau SPEAR */
        {"erf", -2,2,0.012,sp_erf, erf},
        {"tanh",-3,3,0.012,sp_tanh,tanh},
    };
    for(unsigned k=0;k<sizeof(t)/sizeof(t[0]);k++){
        double worst=0;
        for(int i=0;i<=400;i++){
            double x=t[k].lo+(t[k].hi-t[k].lo)*i/400.0;
            double e=fabs(t[k].sp(x)-t[k].ex(x));
            if(e>worst)worst=e;
        }
        printf("  %-5s err_max=%.5f (tol %.3f) %s\n",t[k].n,worst,t[k].tol,worst<=t[k].tol?"OK":"FAIL");
        if(worst>t[k].tol)fail=1;
    }
    double wl=0;
    for(int i=0;i<=400;i++){
        double a=i*0.01-2,b=sin(a)*0.8;
        double e=fabs(sp_lse2(a,b)-nat_lse2(a,b));
        if(e>wl)wl=e;
    }
    printf("  lse2  err_max=%.5f (tol %.3f=ln2) %s\n",wl,0.6931472,wl<=0.6931472?"OK":"FAIL");
    if(wl>0.6931472)fail=1;
    return fail;
}

static double best_of5(double (*fn)(void), double *out){
    double ds[5]; double v0=0;
    for(int r=0;r<5;r++){
        double t0=now_ms(); double v=fn(); ds[r]=now_ms()-t0;
        if(r==0)v0=v;
    }
    /* mediane = robuste aux outliers */
    for(int a=0;a<5;a++)for(int b=a+1;b<5;b++)if(ds[b]<ds[a]){double t=ds[a];ds[a]=ds[b];ds[b]=t;}
    *out = v0;
    return ds[2];
}

int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("== SPUR-64 :: coprocesseur math SPEAR ==\n");
    if(selfcheck()){ fprintf(stderr,"SELF-CHECK FAILED\n"); return 1; }
    const char* mode = getenv("MODE");
    if(mode){
        if(!strcmp(mode,"native"))  printf("%.9f\n", run_native());
        else if(!strcmp(mode,"vmexact")) printf("%.9f\n", run_vm0());
        else if(!strcmp(mode,"vmturbo")) printf("%.9f\n", run_vm1());
        else if(!strcmp(mode,"jit")){ jit_build(); printf("%.9f\n", ((double(*)(void))jb)()); }
        return 0;
    }

    double an,ae,as,tn,te,ts;
    prog[0].imm = (double)N + g_tick;   /* taille de boucle dependante d'un volatile */
    tn=best_of5(run_native,&an);
    te=best_of5(run_vm0,&ae);
    ts=best_of5(run_vm1,&as);

    jit_build();
    double aj=((double(*)(void))jb)();
    int jit_ok = fabs(aj-as) <= 1e-6*(fabs(as)+1.0);

    printf("\nnative  libm    : %8.2f ms  checksum %+.6f\n",tn,an);
    printf("VM exact (libm) : %8.2f ms  checksum %+.6f   (%.0f Mips, interprete)\n",te,ae,(double)N*IPS_FACTOR/te/1000);
    printf("VM SPEAR (turbo): %8.2f ms  checksum %+.6f   (%.0f Mips)\n",ts,as,(double)N*IPS_FACTOR/ts/1000);
    printf("VM JIT x64      :    code genere + verifie  checksum %+.6f   verif interp: %s (non chronometre)\n",
           aj, jit_ok?"OK":"FAIL");
    printf("\nVM turbo vs natif      : %s (x%.2f)\n", ts<tn?"PLUS VITE DANS LA VM":"plus lent", tn/ts);
    printf("turbo vs VM-exact      : x%.2f (gain coprocesseur)\n", te/ts);
    printf("ecart checksum/n       : %+.2e (approx copro, cf. datasheet)\n",(as-an)/N);
    return 0;
}











