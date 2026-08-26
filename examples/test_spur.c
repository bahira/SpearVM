/* test_spur.c â€” validation des kernels boucle et map SpearVM */
#include <stdio.h>
#include <math.h>
#include "spur.h"

int main(void){
    /* --- kernel boucle : damped-wave (comme bench C historique) --- */
    SpurIns prog[] = {
        {MOVI,0,0,7,500000.0},
        {MOVI,0,0,0,-1.0},
        {ADDI,0,0,0,0.000008},
        {GELU,0,0,1,0.6},
        {SIGMOID,1,2,2,0},
        {ADD,3,1,2,0},
        {MULI,3,3,0,1.25},
        {TANH,3,3,0,1.0},
        {LSE2,4,1,3,0},
        {TANHS,5,2,1,0},
        {ACC,3,0,0,0},
        {SUBI,7,7,7,1.0},
        {BNZ,7,0,0,2},
        {HALT,0,0,0,0},
    };
    int h = spur_jit_build(prog,sizeof(prog)/sizeof(prog[0]));
    fprintf(stderr,"loop kernel handle=%d acc=%.6f\n",h,spur_exec(h,NULL));

    /* --- kernel map element-wise multi-entrees ---
       out[i] = h*i + p + max(ge,cf)
       g=gelu(0.6*x), e=sigmoid(x), c=tanh(1.25*(g+e)), f=max(g,c),
       h=tanh(e-g), i=erf(c+f), p=erf(g/2), ge=g*e, cf=c*f                */
    SpurIns mp[] = {
        {MP_LD,0,0,0,0},        /* v0 <- x   (in0) */
        {GELU,0,0,1,0.6},       /* v1 = g */
        {SIGMOID,0,0,2,0},      /* v2 = e = sigmoid(x) */
        {ADD,3,1,2,0},
        {MULI,3,3,0,1.25},
        {TANH,3,3,0,1.0},       /* v3 = c */
        {LSE2,4,1,3,0},         /* v4 = f */
        {TANHS,5,2,1,0},        /* v5 = h */
        {ERFA,7,3,7,0},         /* v7 = erf(c+f) */
        {MUL,5,5,7,0},          /* v5 = h*i */
        {MP_LD,6,1,0,0},        /* v6 = gain (in1) */
        {MULI,6,6,6,0.01},      /* v6 *= 0.01 -> petit terme de couplage */
        {ADD,5,5,6,0},          /* v5 = h*i + 0.01*gain */
        {ERF,6,1,0,0.5},        /* v6 = p = erf(g/2) */
        {ADD,5,6,5,0},          /* s += p */
        {MUL,2,1,2,0},          /* ge */
        {MUL,4,3,4,0},          /* cf */
        {LSE2,6,2,4,0},         /* m */
        {ADD,5,5,6,0},          /* s += m */
        {MP_ST,5,0,0,0},        /* out[i] = s */
    };
    int hm = spur_map_build(mp,sizeof(mp)/sizeof(mp[0]));
    fprintf(stderr,"map kernel handle=%d\n",hm);

    double in0[16],in1[16],out[16];
    for(int i=0;i<16;i++){ in0[i]=-1.0+i*0.125; in1[i]=1.0; }
    spur_map_exec(hm,in0,in1,out,16);
    for(int i=0;i<4;i++) fprintf(stderr,"  x=%.3f gain=%.1f -> %.6f\n",in0[i],in1[i],out[i]);
    return 0;
}

