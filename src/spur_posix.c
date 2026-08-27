/* SpearVM JIT — PORTAGE POSIX/Linux (System V AMD64 ABI)
   Portage de src/spur.c (Win64) vers Linux/macOS :
     - windows.h        -> pthread.h + sys/mman.h
     - SRWLOCK          -> pthread_mutex_t
     - VirtualAlloc     -> mmap(PROT_READ|PROT_WRITE, MAP_ANONYMOUS)
     - VirtualProtect   -> mprotect(PROT_READ|PROT_EXEC)  (W^X propre)
     - VirtualFree      -> munmap
     - Convention Win64 -> System V AMD64 :
         kernel boucle : aucun argument runtime (pool par adresse absolue)
         kernel map    : rdi=ins[], rsi=out, rdx=count
   API publique : include/spur.h (identique) */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/mman.h>
#include "spur.h"

/* ================= Noyaux SPEAR (reference scalaire, fast-math local) ===== */
#pragma GCC optimize("fast-math","no-signed-zeros","no-trapping-math")
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
double spur_k_sig(double x){ return 0.5+0.5*spur_k_tanh(0.5*x); }
double spur_k_lse2(double a,double b){ return fmax(a,b); }
#pragma GCC reset_options

/* ================= Etat =================================================== */
#define MAXK 32
#define JCODE_CAP 16384
static struct { unsigned char* code; size_t len; double* pool; int used; } K[MAXK];
static int nk = 0;
/* Thread-safety : exec est reentrant (code pur), seuls les BUILDS partagent
   jb/jlen/jpool/nfx -> serialization via pthread_mutex (equiv. SRWLOCK). */
static pthread_mutex_t g_jit_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_slot = -1;

static double *jpool; static int jpooln;
static unsigned char *jb; static size_t jlen;

typedef struct { size_t pos; size_t off; } Fix;
static Fix fxc[64]; static int nfx;

#define XR(i)     (6+(i))          /* vregs -> xmm6..13            */
#define ACC_XR    14               /* accumulateur (kernel boucle) */
#define ZERO_XR   15               /* +0.0                         */

#define POOL_CONST0 3              /* constantes slots 3..47       */
#define POOL_IN0    48             /* ptr tableau in0              */
#define POOL_IN1    49             /* ptr tableau in1              */
#define POOL_OUT    50             /* ptr out                      */
#define POOL_CNT    51             /* count int32 (bytes 408-411)  */

/* ================= Emetteur bas niveau ==================================== */
static void je8(unsigned b){ jb[jlen++]=(unsigned char)b; }
static void je32(unsigned v){ memcpy(jb+jlen,&v,4); jlen+=4; }
static void je64(unsigned long long v){ memcpy(jb+jlen,&v,8); jlen+=8; }

static void fx_rel(size_t pos, size_t off){ Fix f={pos,off}; fxc[nfx++]=f; }

static int const_slot(double v){
    uint64_t bits; memcpy(&bits,&v,8);
    uint64_t* q = (uint64_t*)jpool;
    for(int i=POOL_CONST0;i<jpooln;i++)
        if(q[i]==bits) return i;
    if(jpooln>=48) return -1;
    q[jpooln]=bits; return jpooln++;
}

static void e_f2mem(int op,int xr,int idx,int store){
    unsigned rex=(unsigned)(0x40|(((xr)&8)>>1));
    int d=idx*8;
    (void)store;
    je8(0x48); je8(0xB8); je64((unsigned long long)(uintptr_t)jpool);
    je8(0xF2); je8((unsigned char)rex); je8(0x0F); je8(op);
    if(d>=-128&&d<128){ je8((unsigned char)(0x40|(((xr)&7)<<3))); je8((unsigned char)d); }
    else { je8((unsigned char)(0x80|(((xr)&7)<<3))); je32((unsigned)d); }
}
static void e_movsd_pool(int xr,int idx){ e_f2mem(0x10,xr,idx,0); }

/* F2 op xr_dst,xr_src */
static void e_sd(int op,int dst,int src){
    je8(0xF2); je8((unsigned char)(0x40|(((dst)&8)>>1)|(((src)&8)>>3)));
    je8(0x0F); je8(op); je8((unsigned char)(0xC0|(((dst)&7)<<3)|((src)&7)));
}
static void j_call(void* f){
    je8(0x48); je8(0xB8); je64((unsigned long long)(uintptr_t)f);
    je8(0xFF); je8(0xD0);
}
static void j_load(int vr_dst,int vsrc){ e_sd(0x10, vr_dst, XR(vsrc)); }
static void j_store(int vdst,int vr_src){ e_sd(0x10, XR(vdst), vr_src); }

/* ================= Prologue / epilogue ==================================== */
/* System V : rsp aligne 16 avant chaque call. Entree rsp%16=8,
   5 pushes (40) -> rsp%16=0, sub 32 -> reste aligne. Pas de shadow space
   requis (les appels ne sont pas variadiques).                              */
static void emit_prologue(int with_arrays){
    je8(0x53);                                     /* push rbx                    */
    je8(0x41);je8(0x54);                           /* push r12                    */
    je8(0x41);je8(0x55);                           /* push r13                    */
    je8(0x41);je8(0x56);                           /* push r14                    */
    je8(0x41);je8(0x57);                           /* push r15                    */
    je8(0x48);je8(0x83);je8(0xEC);je8(0x20);       /* sub rsp,32                  */
    je8(0x48);je8(0xBB);                           /* movabs rbx,pool             */
    je64((unsigned long long)(uintptr_t)jpool);
        je8(0x66);je8(0x45);je8(0x0F);je8(0xEF);je8(0xFF); /* pxor xmm15,xmm15        */
    je8(0x66);je8(0x43);je8(0x0F);je8(0xEF);je8(0xF6); /* pxor xmm14,xmm14 (acc)  */
        je8(0x45);je8(0x31);je8(0xFF);                     /* xor r15d,r15d (index i) */
    (void)with_arrays; /* branche map gere ses propres pointeurs (SysV ci-bas) */
}

static void emit_epilogue_ret_acc(void){
    je8(0x66);je8(0x41);je8(0x0F);je8(0x28);je8(0xC6); /* movapd xmm0,xmm14        */
    je8(0x48);je8(0x83);je8(0xC4);je8(0x20);       /* add rsp,32                  */
    je8(0x41);je8(0x5F);                           /* pop r15                     */
    je8(0x41);je8(0x5E);                           /* pop r14                     */
    je8(0x41);je8(0x5D);                           /* pop r13                     */
    je8(0x41);je8(0x5C);                           /* pop r12                     */
    je8(0x5B);                                     /* pop rbx                     */
    je8(0xC3);                                     /* ret                         */
}

/* ================= Copro inline SSE ======================================= */
static void rat_on_xmm0(double c3,double cnum,double cb2,double cb0,
                        double lo,double hi){
    int hiK=const_slot(hi), loK=const_slot(lo);
    int c3K=const_slot(c3), cnK=const_slot(cnum), b2K=const_slot(cb2), b0K=const_slot(cb0);
    e_movsd_pool(1,hiK); e_sd(0x5D,0,1);
    e_movsd_pool(1,loK); e_sd(0x5F,0,1);
    e_sd(0x10,1,0);
    e_sd(0x59,1,1);
    e_movsd_pool(2,b2K); e_sd(0x59,2,1);
    e_movsd_pool(2,b0K); e_sd(0x58,2,1);
    e_movsd_pool(1,c3K); e_sd(0x59,1,1);
    e_sd(0x59,1,0);
    e_sd(0x58,1,0);
    e_sd(0x5E,1,1);
    e_movsd_pool(1,cnK); e_sd(0x59,1,1);
}
static void emit_gelu_inline(int dst,int a,double scale){
    int iK=const_slot(scale), c1=const_slot(0.306923), c2=const_slot(0.501),
        c0=const_slot(0.0), cm=const_slot(1.002),
        ck=const_slot(0.997729), cb=const_slot(-0.004004);
    (void)c0;
    e_movsd_pool(0,iK); e_sd(0x59,0,XR(a));
    e_movsd_pool(1,c1); e_sd(0x58,1,0);
    e_movsd_pool(1,c2); e_sd(0x58,1,1);
    je8(0x66);je8(0x0F);je8(0xEF);je8(0xD2);       /* pxor xmm2,xmm2   */
    e_sd(0x5F,1,2);
    e_movsd_pool(1,cm); e_sd(0x5D,1,1);
    e_sd(0x59,0,1);
    e_movsd_pool(1,ck); e_sd(0x59,0,1);
    e_movsd_pool(1,cb); e_sd(0x5A,0,1);
    j_store(dst,0);
}

/* ================= Dispatch d'operations ================================== */
static void emit_op(const SpurIns* I){
    switch(I->op){
    case MOVI:
        e_movsd_pool(XR(I->dst), const_slot(I->imm)); break;
    case ADDI:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x58,XR(I->dst),0); break;
    case MULI:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x59,XR(I->dst),0); break;
    case SUBI:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x5C,XR(I->dst),0); break;
    case ADD: case SUB: case MUL: {
        int op = I->op==ADD?0x58:(I->op==SUB?0x5C:0x59);
        if(I->dst==I->a)      e_sd(op,XR(I->dst),XR(I->b));
        else if(I->dst==I->b) e_sd(op,XR(I->dst),XR(I->a));
        else { j_load(XR(I->dst),I->a); e_sd(op,XR(I->dst),XR(I->b)); }
    } break;
    case ACC:
        e_sd(0x58,ACC_XR,XR(I->a)); break;
    case GELU:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x59,0,XR(I->a));
        j_call((void*)spur_k_gelu); j_store(I->dst,0); break;
    case ERF:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x59,0,XR(I->a));
        j_call((void*)spur_k_erf); j_store(I->dst,0); break;
    case TANH:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x59,0,XR(I->a));
        j_call((void*)spur_k_tanh); j_store(I->dst,0); break;
    case SIGMOID:
        e_movsd_pool(0, const_slot(I->imm)); e_sd(0x59,0,XR(I->a));
        j_call((void*)spur_k_sig);
        e_movsd_pool(1, const_slot(1.0)); e_sd(0x58,XR(I->dst),0);
        break;
    case LSE2:
        j_load(XR(I->dst),I->a); e_sd(0x5F,XR(I->dst),XR(I->b)); break;
    case ACCLSE:
        e_sd(0x10,0,XR(I->a)); e_sd(0x5F,0,XR(I->b));
        e_sd(0x58,0,XR(I->dst)); e_sd(0x58,ACC_XR,0); break;
    default: break; /* HALT */
    }
}

/* ================= Kernel boucle (compteur guest + BNZ) =================== */
int spur_jit_build(const SpurIns* prog,int n){
    pthread_mutex_lock(&g_jit_lock);
    if(n<1||n>256){ pthread_mutex_unlock(&g_jit_lock); return -1; }
    { int slot=-1;
      for(int s=0;s<nk;s++) if(!K[s].used){ slot=s; break; }
      if(slot<0 && nk<MAXK) slot=nk++;
      if(slot<0){ pthread_mutex_unlock(&g_jit_lock); return -1; }
      g_slot=slot; }
    /* W^X : ecriture puis bascule executable (comme VirtualAlloc+VirtualProtect) */
    jb=(unsigned char*)mmap(NULL,JCODE_CAP,PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(jb==MAP_FAILED){ pthread_mutex_unlock(&g_jit_lock); return -1; }
    double* pl=(double*)calloc(64,sizeof(double));
    if(!pl){ munmap(jb,JCODE_CAP); pthread_mutex_unlock(&g_jit_lock); return -1; }
    jpool=pl; jlen=0; nfx=0; jpooln=POOL_CONST0;
    emit_prologue(0);
    static SpurIns g_body[256];
    (void)g_body;
    size_t ioff[256];
    int halted=0;
    for(int i=0;i<n && !halted;i++){
        const SpurIns* I=&prog[i];
        ioff[i]=jlen;
        if(I->op==HALT){ halted=1; continue; }
        if(I->op==BNZ){
            int xr=XR(I->a);
            je8(0x66); je8((unsigned char)(0x40|(((xr)&8)>>1))); je8(0x0F); je8(0x2F);
            je8((unsigned char)(0xC0|(((xr)&7)<<3)|7));   /* comisd xr,xmm15      */
            je8(0x0F); je8(0x85); fx_rel(jlen,ioff[(int)I->imm]); jlen+=4;
            continue;
        }
        emit_op(I);
    }
    emit_epilogue_ret_acc();
    for(int k=0;k<nfx;k++){
        long long tgt=(long long)(uintptr_t)(jb+fxc[k].off);
        long long nxt=(long long)(uintptr_t)(jb+fxc[k].pos+4);
        *(int*)(jb+fxc[k].pos)=(int)(tgt-nxt);
    }
    if(mprotect(jb,JCODE_CAP,PROT_READ|PROT_EXEC)!=0){
        munmap(jb,JCODE_CAP); free(pl);
        pthread_mutex_unlock(&g_jit_lock); return -1;
    }
    __builtin___clear_cache((char*)jb,(char*)(jb+jlen));
    K[g_slot].code=jb; K[g_slot].len=jlen; K[g_slot].pool=pl;
    K[g_slot].used=1;
    int ret=g_slot;
    pthread_mutex_unlock(&g_jit_lock);
    return ret;
}
double spur_exec(int h, const double* regs8){
    (void)regs8;
    if(h<0||h>=nk||!K[h].used) return 0.0/0.0;
    return ((double(*)(void))K[h].code)();
}

/* ================= Kernel map element-wise (ABI System V) ================= */
/* Y = F(in0[i], in1[i], ...) — args directs SysV :
     rdi=ins[]  rsi=out  rdx=count
   Sequencede chargement (l'ordre importe : rdi est ecrase en dernier usage) :
     mov r12,[rdi]      4C 8B 27
     mov r13,[rdi+8]    4C 8B 6F 08
     mov r14,rsi        49 89 F6      (out)
     mov rsi,[rdi+16]   48 8B 77 10   (ins[2])
     mov rdi,[rdi+24]   48 8B 7F 18   (ins[3])
     mov r15,rdx        49 89 D7      (count)
   m_ld which==2 -> [rsi], which==3 -> [rdi] : encodages inchanges.          */
static void m_ld(int xr,int which){
    unsigned rex=(unsigned)(0x40|(((xr)&8)>>1));
    if(which==0||which==1) rex|=0x01;   /* B : bases r12/r13 >= 8          */
    je8(0xF2); je8((unsigned char)rex); je8(0x0F); je8(0x10);
    switch(which){
    case 0:  je8((unsigned char)(0x04|((xr&7)<<3))); je8(0x24); break;  /* [r12]     */
    case 1:  je8((unsigned char)(0x45|((xr&7)<<3))); je8(0x00); break;  /* disp8[r13]*/
    case 2:  je8((unsigned char)(0x06|((xr&7)<<3))); break;             /* [rsi]     */
    default: je8((unsigned char)(0x07|((xr&7)<<3))); break;             /* [rdi]     */
    }
}
static void m_st(int xr){                    /* [r14] <- xr                  */
    je8(0xF2); je8((unsigned char)(0x41|(((xr)&8)>>1))); je8(0x0F); je8(0x11);
    je8((unsigned char)(0x04|((xr&7)<<3))); je8(0x26);
}
int spur_map_build(const SpurIns* body,int n){
    pthread_mutex_lock(&g_jit_lock);
    if(n<1||n>256){ pthread_mutex_unlock(&g_jit_lock); return -1; }
    { int slot=-1;
      for(int s=0;s<nk;s++) if(!K[s].used){ slot=s; break; }
      if(slot<0 && nk<MAXK) slot=nk++;
      if(slot<0){ pthread_mutex_unlock(&g_jit_lock); return -1; }
      g_slot=slot; }
    jb=(unsigned char*)mmap(NULL,JCODE_CAP,PROT_READ|PROT_WRITE,
                            MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(jb==MAP_FAILED){ pthread_mutex_unlock(&g_jit_lock); return -1; }
    double* pl=(double*)calloc(64,sizeof(double));
    if(!pl){ munmap(jb,JCODE_CAP); pthread_mutex_unlock(&g_jit_lock); return -1; }
    jpool=pl; jlen=0; nfx=0; jpooln=POOL_CONST0;
    emit_prologue(0);
    /* Chargement des pointeurs SysV (cf. en-tete de section) */
    je8(0x4C);je8(0x8B);je8(0x27);               /* mov r12,[rdi]           */
    je8(0x4C);je8(0x8B);je8(0x6F);je8(0x08);     /* mov r13,[rdi+8]         */
    je8(0x49);je8(0x89);je8(0xF6);               /* mov r14,rsi    (out)    */
    je8(0x48);je8(0x8B);je8(0x77);je8(0x10);     /* mov rsi,[rdi+16](ins[2])*/
    je8(0x48);je8(0x8B);je8(0x7F);je8(0x18);     /* mov rdi,[rdi+24](ins[3])*/
    je8(0x49);je8(0x89);je8(0xD7);               /* mov r15,rdx    (count)  */
    size_t body_off=jlen;
    for(int i=0;i<n;i++){
        const SpurIns* I=&body[i];
        switch(I->op){
        case MP_LD: m_ld(XR(I->dst),I->a); break;
        case MP_ST: m_st(XR(I->a)); break;
        default: emit_op(I); break;
        }
    }
    /* queue : avance les 3 pointeurs, decremente, boucle */
    je8(0x49);je8(0x83);je8(0xC4);je8(0x08);     /* add r12,8                */
    je8(0x49);je8(0x83);je8(0xC5);je8(0x08);     /* add r13,8                */
    je8(0x49);je8(0x83);je8(0xC6);je8(0x08);     /* add r14,8                */
    je8(0x48);je8(0x83);je8(0xC6);je8(0x08);     /* add rsi,8                */
    je8(0x48);je8(0x83);je8(0xC7);je8(0x08);     /* add rdi,8                */
    je8(0x49);je8(0xFF);je8(0xCF);               /* dec r15                  */
    je8(0x0F);je8(0x85); fx_rel(jlen,body_off); jlen+=4;  /* jnz corps       */
    /* pas de pop rsi/rdi : jamais pushes en SysV (caller-saved) */
    emit_epilogue_ret_acc();
    for(int k=0;k<nfx;k++){
        long long tgt=(long long)(uintptr_t)(jb+fxc[k].off);
        long long nxt=(long long)(uintptr_t)(jb+fxc[k].pos+4);
        *(int*)(jb+fxc[k].pos)=(int)(tgt-nxt);
    }
    if(mprotect(jb,JCODE_CAP,PROT_READ|PROT_EXEC)!=0){
        munmap(jb,JCODE_CAP); free(pl);
        pthread_mutex_unlock(&g_jit_lock); return -1;
    }
    __builtin___clear_cache((char*)jb,(char*)(jb+jlen));
    K[g_slot].code=jb; K[g_slot].len=jlen; K[g_slot].pool=pl;
    K[g_slot].used=1;
    int ret=g_slot;
    pthread_mutex_unlock(&g_jit_lock);
    return ret;
}

void spur_map_exec(int handle,const double* const* ins,double* out,
                   long long count){
    if(handle<0||handle>=nk||!K[handle].used||count<1) return;
    ((void(*)(const double* const*,double*,long long))K[handle].code)
        (ins,out,count);
}

void spur_free(int handle){
    if(handle<0||handle>=nk||!K[handle].used) return;
    pthread_mutex_lock(&g_jit_lock);
    if(K[handle].used){
        munmap(K[handle].code, JCODE_CAP);   /* meme taille que l'alloc */
        free(K[handle].pool);
        K[handle].code=NULL; K[handle].pool=NULL;
        K[handle].used=0;
    }
    pthread_mutex_unlock(&g_jit_lock);
}

const char* SPUR_MARKER="SPURMARKER_V2_XYZ_POSIX";
