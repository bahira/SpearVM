/* SpearVM ??? VM registres + coprocesseur math approximatif (noyaux SPEAR certifi??s).
   ??metteur x64 SSE2 : les kernels sont compil??s en code machine natif.
   API publique : include/spur.h ??? d??tails : README.md */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>
#include "spur.h"

/* ================= Noyaux SPEAR (r??f??rence scalaire, fast-math local) ===== */
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

/* ================= ??tat =================================================== */
#define MAXK 8
static struct { unsigned char* code; size_t len; double* pool; } K[MAXK];
static int nk = 0;

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

/* ================= ??metteur bas niveau ==================================== */
static void je8(unsigned b){ jb[jlen++]=(unsigned char)b; }
static void je32(unsigned v){ memcpy(jb+jlen,&v,4); jlen+=4; }
static void je64(unsigned long long v){ memcpy(jb+jlen,&v,8); jlen+=8; }

static void fx_rel(size_t pos, size_t off){ Fix f={pos,off}; fxc[nfx++]=f; }

static int const_slot(double v){
    /* BIT PATTERN, pas la valeur : (uint64_t)v convertirait 42.0 en entier 42,
       relu comme double = denormal -> tous les consts fausses silencieusement */
    uint64_t bits; memcpy(&bits,&v,8);
    uint64_t* q = (uint64_t*)jpool;
    for(int i=POOL_CONST0;i<jpooln;i++)
        if(q[i]==bits) return i;
    if(jpooln>=48) return -1;
    q[jpooln]=bits; return jpooln++;
}

/* SSE load constante pool : version ABSOLUE (movabs rax,jpool ; movsd xr,[rax+d]).
   Independante de rbx et des registres d'arguments Win64 (rcx/rdx/r8/r9),
   donc identique dans les contextes boucle ET map.                          */
static void e_f2mem(int op,int xr,int idx,int store){
    unsigned rex=(unsigned)(0x40|(((xr)&8)>>1));
    int d=idx*8;
    (void)store;
    je8(0x48); je8(0xB8); je64((unsigned long long)(uintptr_t)jpool);
    je8(0xF2); je8((unsigned char)rex); je8(0x0F); je8(op);
    if(d>=-128&&d<128){ je8((unsigned char)(0x40|(((xr)&7)<<3))); je8((unsigned char)d); } /* [rax+d8]  */
    else { je8((unsigned char)(0x80|(((xr)&7)<<3))); je32((unsigned)d); }                  /* [rax+d32] */
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

/* ================= Prologue / ??pilogue ==================================== */
/* 5 pushes (40) + sub 32 : rsp align?? 16 avant chaque call ; shadow 32 dispo */
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
    if(with_arrays){
        je8(0x4C);je8(0x8B);je8(0xA3);je8(0x80);je8(0x01);je8(0x00);je8(0x00); /* mov r12,[rbx+384]   */
        je8(0x4C);je8(0x8B);je8(0xAB);je8(0x88);je8(0x01);je8(0x00);je8(0x00); /* mov r13,[rbx+392]   */
        je8(0x4C);je8(0x8B);je8(0xB3);je8(0x90);je8(0x01);je8(0x00);je8(0x00); /* mov r14,[rbx+400]   */
    }
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

/* ================= Copro inline SSE (z?ro appel libm) ===================== */
/* rationnel sur xmm0 : r = cnum*(y+c3*y^3)/(cb0+cb2*y^2), clamp [lo,hi].
   Registres : xmm0 = y/num/r, xmm1 = y?/den. Aucun appel libm.              */
static void rat_on_xmm0(double c3,double cnum,double cb2,double cb0,
                        double lo,double hi){
    int hiK=const_slot(hi), loK=const_slot(lo);
    int c3K=const_slot(c3), cnK=const_slot(cnum), b2K=const_slot(cb2), b0K=const_slot(cb0);
    e_movsd_pool(1,hiK); e_sd(0x5D,0,1);      /* min(y,hi)        */
    e_movsd_pool(1,loK); e_sd(0x5F,0,1);      /* max(y,lo)        */
    e_sd(0x10,1,0);                           /* t = y            */
    e_sd(0x59,1,1);                           /* t = y?           */
    e_movsd_pool(2,b2K); e_sd(0x59,2,1);      /* cb2*y?           */
    e_movsd_pool(2,b0K); e_sd(0x58,2,1);      /* den = cb0+cb2*y? */
    e_movsd_pool(1,c3K); e_sd(0x59,1,1);      /* c3*y?            */
    e_sd(0x59,1,0);                           /* c3*y?            */
    e_sd(0x58,1,0);                           /* num = y+c3*y?    */
    e_sd(0x5E,1,1);                           /* num/den          */
    e_movsd_pool(1,cnK); e_sd(0x59,1,1);      /* *= cnum          */
}
static void emit_gelu_inline(int dst,int a,double scale){
    int iK=const_slot(scale), c1=const_slot(0.306923), c2=const_slot(0.501),
        c0=const_slot(0.0), cm=const_slot(1.002),
        ck=const_slot(0.997729), cb=const_slot(-0.004004);
    e_movsd_pool(0,iK); e_sd(0x59,0,XR(a));   /* y = scale*a      */
    e_movsd_pool(1,c1); e_sd(0x58,1,0);       /* 0.306923*y       */
    e_movsd_pool(1,c2); e_sd(0x58,1,1);       /* +0.501           */
    je8(0x66);je8(0x0F);je8(0xEF);je8(0xD2);  /* pxor xmm2,xmm2   */
    e_sd(0x5F,1,2);                           /* max(u,0)         */
    e_movsd_pool(1,cm); e_sd(0x5D,1,1);       /* min(u,1.002)     */
    e_sd(0x59,0,1);                           /* y*u              */
    e_movsd_pool(1,ck); e_sd(0x59,0,1);       /* *0.997729        */
    e_movsd_pool(1,cb); e_sd(0x5A,0,1);       /* -0.004004        */
    j_store(dst,0);
}

/* ================= Dispatch d'op??rations ================================== */
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
static SpurIns g_body[256];

int spur_jit_build(const SpurIns* prog,int n){
    if(n<1||n>256||nk>=MAXK) return -1;
    jb=VirtualAlloc(NULL,16384,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    if(!jb) return -1;
    double* pl=(double*)calloc(64,sizeof(double));
    if(!pl) return -1;
    jpool=pl; jlen=0; nfx=0; jpooln=POOL_CONST0;
    emit_prologue(0);
    size_t ioff[256];
    size_t body_off=jlen;
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
        DWORD old; VirtualProtect(jb,jlen,PAGE_EXECUTE_READ,&old);
    FlushInstructionCache(GetCurrentProcess(),jb,(DWORD)jlen);
    K[nk].code=jb; K[nk].len=jlen; K[nk].pool=pl;
    return nk++;
}
double spur_exec(int h, const double* regs8){
    if(h<0||h>=nk) return 0.0/0.0;
    return ((double(*)(void))K[h].code)();
}

/* ================= Kernel map element-wise (tableaux) ===================== */
/* Y = F(in0[i], in1[i]) — pointeurs passes en ARGS DIRECTS Win64 :
   rcx=in0 rdx=in1 r8=out r9=count. Encodages valides octet par octet
   par tests/test_encoding.c (17 assertions + execution reelle).
   Pieges REX/SIB couverts : [r12] exige SIB(base=100+B), [r13] en mod00
   serait RIP-relatif -> forme disp8, store SIB base=110(r14) pas 100.      */
static void m_ld(int xr,int which){          /* xr <- [r12] ou [r13+0]       */
    unsigned rex=(unsigned)(0x41|(((xr)&8)>>1));
    je8(0xF2); je8((unsigned char)rex); je8(0x0F); je8(0x10);
    if(which==0) je8((unsigned char)(0x04|((xr&7)<<3))), je8(0x24);
    else         je8((unsigned char)(0x45|((xr&7)<<3))), je8(0x00);
}
static void m_st(int xr){                    /* [r14] <- xr                  */
    je8(0xF2); je8((unsigned char)(0x41|(((xr)&8)>>1))); je8(0x0F); je8(0x11);
    je8((unsigned char)(0x04|((xr&7)<<3))); je8(0x26);
}
int spur_map_build(const SpurIns* body,int n){
    if(n<1||n>256||nk>=MAXK) return -1;
    jb=VirtualAlloc(NULL,16384,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    if(!jb) return -1;
    double* pl=(double*)calloc(64,sizeof(double));
    if(!pl) return -1;
    jpool=pl; jlen=0; nfx=0; jpooln=POOL_CONST0;
    emit_prologue(0);
    /* prologue map : recopie des args vers les pointeurs dedies */
    je8(0x49);je8(0x89);je8(0xCC);               /* mov r12,rcx  (rex 49!)   */
    je8(0x49);je8(0x89);je8(0xD5);               /* mov r13,rdx              */
    je8(0x4D);je8(0x89);je8(0xC6);               /* mov r14,r8               */
    je8(0x4D);je8(0x89);je8(0xCF);               /* mov r15,r9               */
    size_t body_off=jlen;
    for(int i=0;i<n;i++){
        const SpurIns* I=&body[i];
        switch(I->op){
        case MP_LD: m_ld(XR(I->dst),I->a); break;    /* a=0|1 choix entree   */
        case MP_ST: m_st(XR(I->a)); break;
        default: emit_op(I); break;
        }
    }
    /* queue : avance les 3 pointeurs, decremente, boucle */
    je8(0x49);je8(0x83);je8(0xC4);je8(0x08);     /* add r12,8                */
    je8(0x49);je8(0x83);je8(0xC5);je8(0x08);     /* add r13,8                */
    je8(0x49);je8(0x83);je8(0xC6);je8(0x08);     /* add r14,8                */
    je8(0x49);je8(0xFF);je8(0xCF);               /* dec r15                  */
    je8(0x0F);je8(0x85); fx_rel(jlen,body_off); jlen+=4;  /* jnz corps       */
    emit_epilogue_ret_acc();
    for(int k=0;k<nfx;k++){
        long long tgt=(long long)(uintptr_t)(jb+fxc[k].off);
        long long nxt=(long long)(uintptr_t)(jb+fxc[k].pos+4);
        *(int*)(jb+fxc[k].pos)=(int)(tgt-nxt);
    }
    DWORD old; VirtualProtect(jb,jlen,PAGE_EXECUTE_READ,&old);
    FlushInstructionCache(GetCurrentProcess(),jb,(DWORD)jlen);
    K[nk].code=jb; K[nk].len=jlen; K[nk].pool=pl;
    return nk++;
}
void spur_map_exec(int handle,const double* a0,const double* a1,double* out,long long count){
    if(handle<0||handle>=nk||count<1) return;
    ((void(*)(const double*,const double*,double*,long long))K[handle].code)
        (a0,a1,out,count);
}


const char* SPUR_MARKER="SPURMARKER_V2_XYZ";

