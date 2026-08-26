/* libSPUR : VM coprocesseur SPEAR exposee en DLL (FFI C / Python ctypes).
   Chaque "kernel" est un programme SPUR compile une fois en code x64 natif.
   API :
     int    spur_jit_build(const SpurIns* prog, int n)   -> handle (-1 si erreur)
     double spur_exec(int handle, const double* regs8)   -> acc ; regs8 = r0..r7 init
     double spur_k_gelu/erf/tanh(double), spur_k_lse2(a,b) -> noyaux directs */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <windows.h>

/* ---- Noyaux SPEAR (fast-math local : reciproques etc.) ---- */
#pragma GCC optimize("fast-math","no-signed-zeros","no-trapping-math")
static inline double clampd(double x,double lo,double hi){ return x<lo?lo:(x>hi?hi:x); }
double spur_k_gelu(double x){ return 0.997729*(x*fmin(1.002,fmax(0.0,0.306923*x+0.501)))-0.004004; }
double spur_k_erf (double x){ x=clampd(x,-2,2); return 1.106774*((x+0.034298*x*x*x)/(0.995+0.378089*x*x)); }
double spur_k_tanh(double x){ x=clampd(x,-3,3); return 0.900021*((x+0.053639*x*x*x)/(0.90122+0.343141*x*x)); }
double spur_k_lse2(double a,double b){ return fmax(a,b); }
#pragma GCC reset_options

enum { MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC, GELU, ERF, TANH, LSE2,
       TANHA, TANHS, ERFA, ACCLSE, HALT };

#ifdef __cplusplus
extern "C" {
#endif
typedef struct { short op, a, b, dst; double imm; } SpurIns;

int    spur_jit_build(const SpurIns* prog, int n);
double spur_exec(int handle, const double* regs8);

/* ---- Batch API : map element-wise sur tableaux ----
   Le programme (corps seulement, sans boucle) calcule out[i] depuis in[i].
   Op specifiques : MP_LD (dst <- in[i]), MP_ST (out[i] <- src). */
int  spur_map_build(const SpurIns* body, int n);
void spur_map_exec(int handle, const double* in, double* out, long long count);
#ifdef __cplusplus
}
#endif

/* ---- Emetteur x64 ---- */
#define MAXK 8
static struct { unsigned char* code; size_t len; double* pool; } K[MAXK];
static int nk = 0;

static double *jpool; static int jpooln;      /* pool du kernel EN COURS de build */
static unsigned char *jb; static size_t jlen;
typedef struct { size_t pos; void* abs; size_t off; int kind; } Fix;
static Fix fxc[64]; static int nfx;
#define XR(i) (6+(i))

static void je8(unsigned b){ jb[jlen++]=(unsigned char)b; }
static void je32(unsigned v){ memcpy(jb+jlen,&v,4); jlen+=4; }
static void je64(unsigned long long v){ memcpy(jb+jlen,&v,8); jlen+=8; }
static void fx_rel(size_t pos, size_t off){ Fix f={pos,0,off,1}; fxc[nfx++]=f; }
static int pool_of(double v){ jpool[jpooln]=v; return jpooln++; }

static void e_movsd_pool(int xr,int idx){
    int rex=0x40|(((xr)>>3)<<2), d=idx*8;
    je8(0xF2); je8(rex); je8(0x0F); je8(0x10);
    if(d>=-128&&d<128){ je8(0x44|((xr&7)<<3)); je8(0x23); je8((unsigned char)d); }
    else { je8(0x84|((xr&7)<<3)); je8(0x23); je32((unsigned)d); }
}
static void e_sd(int op,int dst,int src){
    je8(0xF2); je8(0x40|(((dst)&8)>>1)|(((src)&8)>>3));
    je8(0x0F); je8(op); je8(0xC0|((dst&7)<<3)|(src&7));
}
static void j_call(void* f){
    je8(0x48); je8(0xB8); je64((unsigned long long)(uintptr_t)f);
    je8(0xFF); je8(0xD0);
}
static void j_load(int xr,int gsrc){ e_sd(0x10,xr,XR(gsrc)); }
static void j_store(int gdst,int xr){ e_sd(0x10,XR(gdst),xr); }

int spur_jit_build(const SpurIns* prog, int n){
    if(n<1 || n>256 || nk>=MAXK) return -1;
    jb = VirtualAlloc(NULL,16384,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    if(!jb) return -1;
    double* pl = (double*)calloc(64,sizeof(double));
    if(!pl) return -1;
    jpool = pl;
    jlen=0; nfx=0; jpooln=0;                      /* cstes 0..47, entrees 56..63      */
    je8(0x53);                                    /* push rbx                         */
    je8(0x48);je8(0x83);je8(0xEC);je8(0x30);      /* sub rsp,48                       */
    je8(0x48);je8(0xBB);                          /* movabs rbx,pool                  */
    je64((unsigned long long)(uintptr_t)jpool);
    je8(0x66);je8(0x45);je8(0x0F);je8(0xEF);je8(0xFF); /* pxor xmm15,xmm15            */
    je8(0x66);je8(0x45);je8(0x0F);je8(0xEF);je8(0xF6); /* pxor xmm14,xmm14 (=acc)     */
    for(int i=0;i<8;i++) e_movsd_pool(XR(i),56+i);/* r0..r7 <- entrees                */

    size_t ioff[256];
    for(int i=0;i<n;i++){
        const SpurIns* I=&prog[i];
        ioff[i]=jlen;
        switch(I->op){
        case MOVI:
            e_movsd_pool(XR(I->dst), pool_of(I->imm)); break;
        case ADDI: case MULI: case SUBI: {
            int op = I->op==ADDI?0x58:(I->op==MULI?0x59:0x5C);
            j_load(0,I->a);
            e_movsd_pool(1, pool_of(I->imm));
            e_sd(op,XR(I->dst),1);
        } break;
        case ADD: case SUB: case MUL: {
            int op = I->op==ADD?0x58:(I->op==SUB?0x5C:0x59);
            if(I->dst==I->a)      e_sd(op,XR(I->dst),XR(I->b));
            else if(I->dst==I->b) e_sd(op,XR(I->dst),XR(I->a));
            else { j_load(XR(I->dst),I->a); e_sd(op,XR(I->dst),XR(I->b)); }
        } break;
        case ACC:
            e_sd(0x58,14,XR(I->a)); break;
        case BNZ: {
            int xr = XR(I->a);
            je8(0x66); je8(0x40|(((xr)&8)>>1)|1); je8(0x0F); je8(0x2F);
            je8(0xC0|((xr&7)<<3)|7);              /* comisd xr,xmm15                  */
            je8(0x0F); je8(0x85);                 /* jne rel32                        */
            if((int)I->imm >= i) return -1;       /* boucles arriere uniquement (v1)  */
            fx_rel(jlen, ioff[(int)I->imm]);
            jlen+=4;
        } break;
        case GELU:
            e_movsd_pool(0, pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_gelu); j_store(I->dst,0);
            break;
        case ERF:
            e_movsd_pool(0, pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_erf); j_store(I->dst,0);
            break;
        case TANH:
            e_movsd_pool(0, pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case LSE2:
            j_load(0,I->a); e_sd(0x5F,0,XR(I->b)); j_store(I->dst,0);
            break;
        case TANHA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case TANHS:
            j_load(0,I->a); e_sd(0x5C,0,XR(I->b));
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case ERFA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)spur_k_erf); j_store(I->dst,0);
            break;
        case ACCLSE:
            j_load(0,I->a); e_sd(0x5F,0,XR(I->b));
            e_sd(0x58,0,XR(I->dst)); e_sd(0x58,14,0);
            break;
        default: break;                           /* HALT -> ret                      */
        }
    }
    je8(0x66);je8(0x41);je8(0x0F);je8(0x28);je8(0xC6); /* movapd xmm0,xmm14       */
    /* sauvegarde r0..r7 DESACTIVEE pour bissect */
    je8(0x48);je8(0x83);je8(0xC4);je8(0x30);      /* add rsp,48                      */
    je8(0x5B);                                    /* pop rbx                         */
    je8(0xC3);                                    /* ret                             */
    for(int k=0;k<nfx;k++){
        long long tgt = (long long)(uintptr_t)(jb+fxc[k].off);
        long long nxt = (long long)(uintptr_t)(jb+fxc[k].pos+4);
        *(int*)(jb+fxc[k].pos) = (int)(tgt-nxt);
    }
    DWORD old; VirtualProtect(jb,jlen,PAGE_EXECUTE_READ,&old);
    { FILE* fp=fopen("jit_dll.bin","wb"); fwrite(jb,1,jlen,fp); fclose(fp); }
    FlushInstructionCache(GetCurrentProcess(),jb,(DWORD)jlen);
    K[nk].code=jb; K[nk].len=jlen; K[nk].pool=pl;
    return nk++;
}

/* ---- Batch map : JIT d'une boucle element-wise sur tableaux ----
   r12=in ptr, r13=out ptr, r14=index. vregs = xmm6..13.
   pool: slot0=ptr in, slot1=ptr out, slot2(int32)=count, constantes depuis slot 3. */
enum { MP_LD=18, MP_ST };

static void e_load_vr(int dst){                    /* movsd xr,[r12]                  */
    int rex=0x40|(((dst)&8)>>1)|1;
    je8(0xF2); je8(rex); je8(0x0F); je8(0x10);
    je8((unsigned char)(0x04|((dst&7)<<3))); je8(0x24);
}
static void e_store_vr(int src){                   /* movsd [r13],xr  (disp8=0)       */
    int rex=0x41|(((src)&8)>>1)|(((13)&8)>>3);
    je8(0xF2); je8(rex); je8(0x0F); je8(0x11);
    je8((unsigned char)(0x44|((src&7)<<3))); je8(0x25); je8(0x00);
}

int spur_map_build(const SpurIns* body, int n){
    if(n<1 || n>256 || nk>=MAXK) return -1;
    jb = VirtualAlloc(NULL,16384,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    if(!jb) return -1;
    double* pl = (double*)calloc(64,sizeof(double));
    if(!pl) return -1;
    jpool = pl;
    jlen=0; nfx=0; jpooln=3;                      /* slots 0-2 reserves (ptrs+count) */
    je8(0x53);                                     /* push rbx                        */
    je8(0x41);je8(0x54);                           /* push r12                        */
    je8(0x41);je8(0x55);                           /* push r13                        */
    je8(0x41);je8(0x56);                           /* push r14                        */
    je8(0x48);je8(0x83);je8(0xEC);je8(0x28);       /* sub rsp,40 -> pile alignee 16   */
    je8(0x48);je8(0xBB);                           /* movabs rbx,pool                 */
    je64((unsigned long long)(uintptr_t)jpool);
    je8(0x66);je8(0x45);je8(0x0F);je8(0xEF);je8(0xFF); /* pxor xmm15,xmm15            */
    je8(0x4C);je8(0x8B);je8(0x24);je8(0x23);       /* mov r12,[rbx+0]   (in ptr)      */
    je8(0x4C);je8(0x8B);je8(0x6C);je8(0x23);je8(0x08); /* mov r13,[rbx+8]   (out ptr) */
    je8(0x45);je8(0x31);je8(0xF6);                 /* xor r14,r14                     */
    size_t body_off = jlen;
    for(int i=0;i<n;i++){
        const SpurIns* I=&body[i];
        switch(I->op){
        case MP_LD:  e_load_vr(XR(I->dst)); break;
        case MP_ST:  e_store_vr(XR(I->a)); break;
        case MOVI: e_movsd_pool(XR(I->dst), pool_of(I->imm)); break;
        case ADDI: case MULI:
            j_load(0,I->a);
            e_movsd_pool(1, pool_of(I->imm));
            e_sd(I->op==ADDI?0x58:0x59,XR(I->dst),1);
            break;
        case ADD: case SUB: case MUL: {
            int op = I->op==ADD?0x58:(I->op==SUB?0x5C:0x59);
            if(I->dst==I->a)      e_sd(op,XR(I->dst),XR(I->b));
            else if(I->dst==I->b) e_sd(op,XR(I->dst),XR(I->a));
            else { j_load(XR(I->dst),I->a); e_sd(op,XR(I->dst),XR(I->b)); }
        } break;
        case GELU:
            e_movsd_pool(0,pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_gelu); j_store(I->dst,0);
            break;
        case ERF:
            e_movsd_pool(0,pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_erf); j_store(I->dst,0);
            break;
        case TANH:
            e_movsd_pool(0,pool_of(I->imm)); j_load(1,I->a); e_sd(0x59,0,1);
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case LSE2:
            j_load(0,I->a); e_sd(0x5F,0,XR(I->b)); j_store(I->dst,0);
            break;
        case TANHA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case TANHS:
            j_load(0,I->a); e_sd(0x5C,0,XR(I->b));
            j_call((void*)spur_k_tanh); j_store(I->dst,0);
            break;
        case ERFA:
            j_load(0,I->a); e_sd(0x58,0,XR(I->b));
            j_call((void*)spur_k_erf); j_store(I->dst,0);
            break;
        default: break;
        }
    }
    /* NOTE: le corps doit contenir son propre MP_ST ; pas de store implicite */
    /* avancement des pointeurs in/out */
    je8(0x49);je8(0x83);je8(0xC4);je8(0x08);       /* add r12,8                       */
    je8(0x49);je8(0x83);je8(0xC5);je8(0x08);       /* add r13,8                       */
    je8(0x49);je8(0xFF);je8(0xC6);                 /* inc r14                         */
    je8(0x44);je8(0x3B);je8(0x74);je8(0x23);je8(16); /* cmp r14d,[rbx+16]             */
    je8(0x0F);je8(0x8C); fx_rel(jlen,body_off); jlen+=4; /* jl body                   */
    je8(0x48);je8(0x83);je8(0xC4);je8(0x28);       /* add rsp,40                      */
    je8(0x41);je8(0x5E);                           /* pop r14                         */
    je8(0x41);je8(0x5D);                           /* pop r13                         */
    je8(0x41);je8(0x5C);                           /* pop r12                         */
    je8(0x5B);                                     /* pop rbx                         */
    je8(0xC3);                                     /* ret                             */
    for(int k=0;k<nfx;k++){
        long long tgt=(long long)(uintptr_t)(jb+fxc[k].off);
        long long nxt=(long long)(uintptr_t)(jb+fxc[k].pos+4);
        *(int*)(jb+fxc[k].pos)=(int)(tgt-nxt);
    }
    DWORD old; VirtualProtect(jb,jlen,PAGE_EXECUTE_READ,&old);
    { FILE* fp=fopen("jit_dll.bin","wb"); fwrite(jb,1,jlen,fp); fclose(fp); }
    FlushInstructionCache(GetCurrentProcess(),jb,(DWORD)jlen);
    K[nk].code=jb; K[nk].len=jlen; K[nk].pool=pl;
    return nk++;
}

void spur_map_exec(int handle, const double* in, double* out, long long count){
    if(handle<0 || handle>=nk || count<1 || count>0x7FFFFFFF) return;
    double* pl=K[handle].pool;
    memcpy(pl+0, &in,  sizeof(in));
    memcpy(pl+1, &out, sizeof(out));
    { int c32=(int)count; memcpy((char*)(pl+2), &c32, 4); }
    ((double(*)(void))K[handle].code)();
}

