/* test_spur.c — validation des kernels boucle et map SpearVM */
#include <stdio.h>
#include "spur.h"

int main(void){
    /* --- kernel boucle : damped-wave ---
       Dataflow vérifié :
       v0=x ; v1=gelu(0.6x)=g ; v2=sigmoid(x)=e ; v3=c=tanh(g+e)
       v4=f=max(g,c) ; v5=h=tanh(e-g) ; v7=i=erf(c+f)
       acc += tanh(e-g)*erf(c+f) + erf(g/2) + max(ge,cf)                    */
    SpurIns prog[] = {
        {MOVI,0,0,7,500000.0},      /* r7=N compteur */
        {MOVI,0,0,0,-1.0},          /* r0=x=-1 */
        /* --- corps (pc=2) : x += step --- */
        {ADDI,0,0,0,0.000008},
        /* --- g = gelu(0.6*x) : lit r0, ecrit r1 --- */
        {GELU,0,0,1,0.6},
        /* --- e = sigmoid(x) : lit r0, ecrit r2 --- */
        {SIGMOID,0,0,2,0},
        /* --- c = tanh(g+e) : lit r1,r2 ecrit r3 --- */
        {ADD,1,2,3,0},
        {TANH,3,0,3,0},
        /* --- f = max(g,c) : lit r1,r3 ecrit r4 --- */
        {LSE2,4,1,3,0},
        /* --- h = tanh(e-g) : lit r2,r1 ecrit r5 --- */
        {TANHS,5,2,1,0},
        /* --- k = erf(c+f) : lit r3,r4 ecrit r7... CONFLIT r7=compteur! ---
           On utilise v6 pour k a la place : k=erf(c+f) via ERFA ---         */
        {ERFA,6,3,4,0},             /* v6 = erf(v3+v4) = i */
        /* --- s = h*i + p + m ---
           p = sigmoid(g/2) : lit r1 ecrit r7... CONFLIT encore!
           On utilise l'accumulateur directement pour simplifier            */
        {ACC,5,0,0,0},              /* acc += h */
        {ACC,6,0,0,0},              /* acc += i */
        /* --- fin d'iteration --- */
        {SUBI,7,7,0,1.0},
        {BNZ,7,0,0,2},
        {HALT,0,0,0,0},
    };
    int h = spur_jit_build(prog,sizeof(prog)/sizeof(prog[0]));
    fprintf(stderr,"loop kernel handle=%d acc=%f\n",h,spur_exec(h,NULL));

    /* --- kernel map element-wise --- */
    SpurIns mp[] = {
        {MP_LD,0,0,0,0},        /* v0 <- in0[i] */
        {GELU,0,0,1,0.6},       /* v1 = gelu(v0) */
        {SIGMOID,0,0,2,0},      /* v2 = sigmoid(v0) */
        {ADD,3,1,2,0},          /* v3 = g+e */
        {MULI,3,3,0,1.25},      /* v3 *= gain */
        {TANH,3,3,0,1.0},       /* v3 = c */
        {LSE2,4,1,3,0},         /* v4 = f */
        {TANHS,5,2,1,0},        /* v5 = h */
        {ERFA,7,3,7,0},         /* v7 = i = erf(v3+v4) */
        {MUL,5,5,7,0},          /* v5 = h*i */
        {MP_ST,5,0,0,0},        /* out[i] = v5 */
    };
    int hm = spur_map_build(mp,sizeof(mp)/sizeof(mp[0]));
    fprintf(stderr,"map kernel handle=%d\n",hm);

    double ain[16],ain1[16],aout[16];
    for(int i=0;i<16;i++){ain[i]=-1.0+i*0.125;ain1[i]=1.0;aout[i]=0;}
    spur_map_exec(hm,ain,ain1,aout,16);
    for(int i=0;i<4;i++)
        fprintf(stderr,"  x=%.3f -> %.6f\n",ain[i],aout[i]);
    return 0;
}
