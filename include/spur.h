/* spur.h — API publique SpearVM
   VM registres + coprocesseur math approximatif (noyaux SPEAR certifiés).
   Deux modes d'exécution JIT x64 :
     - kernel boucle  : programme avec compteur/BNZ, accumule dans acc
     - kernel map     : out[i] = F(in[i]) element-wise sur tableaux
*/
#ifndef SPUR_H
#define SPUR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Opcodes ----
   Convention operandes (struct SpurIns): a, b = sources ; dst = destination ;
   imm = constante. Les ops copro multiplient leur source par imm avant calcul. */
enum {
    MOVI=0, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC,
    GELU, ERF, TANH, LSE2, TANHA, TANHS, ERFA, ACCLSE, HALT,
    /* batch map uniquement */
    MP_LD, MP_ST,
    /* derives certified */
    SIGMOID
};

/* Instruction : NE PAS changer l'ordre des champs (ABI ctypes/FFI) */
typedef struct { short op, a, b, dst; double imm; } SpurIns;

/* ---- Kernel boucle ---- */
int    spur_jit_build(const SpurIns* prog, int n);         /* -> handle */
double spur_exec(int handle, const double* regs8);          /* -> acc, regs in/out */

/* ---- Kernel map element-wise ---- */
/* Corps sans boucle : MP_LD dst=a (charge in[a][i], a=0|1) ; MP_ST a=v_src.
   Deux tableaux d'entree max (a=0 -> premier tableau, a=1 -> second).      */
int  spur_map_build(const SpurIns* body, int n);
void spur_map_exec(int handle, const double* a0, const double* a1,
                   double* out, long long count);   /* a1 peut etre NULL */

/* ---- Noyaux directs ---- */
double spur_k_gelu(double x);
double spur_k_erf (double x);
double spur_k_tanh(double x);
double spur_k_lse2(double a,double b);

#ifdef __cplusplus
}
#endif
#endif /* SPUR_H */
