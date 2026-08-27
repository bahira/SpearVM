/* spur.h ??? API publique SpearVM
   VM registres + coprocesseur math approximatif (noyaux SPEAR certifi??s).
   Deux modes d'ex??cution JIT x64 :
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
/* Corps sans boucle : MP_LD dst=a (charge ins[a][i], a=0..3) ; MP_ST a=v_src.
   Jusqu'a 4 tableaux d'entree (ins[] = tableau de pointeurs).              */
int  spur_map_build(const SpurIns* body, int n);
void spur_map_exec(int handle, const double* const* ins,
                   double* out, long long count);

/* Libere un handle JIT (boucle ou map). Slot reutilisable. */
void spur_free(int handle);

/* ---- Noyaux directs ---- */
double spur_k_gelu(double x);
double spur_k_erf (double x);
double spur_k_tanh(double x);
double spur_k_lse2(double a,double b);

/* ---- Noyaux directs v2 quintique ---- */
double spur_k_gelu_quintic(double x);
void   spur_batch_gelu_quintic(const double* x, double* out, long long n);
double spv2_gelu_erf(double x);
void   spv2_batch_gelu_erf(const double* x, double* out, long long n);

/* ---- Noyaux v2 (fit Lawson quasi-minimax, precision x12-x453 vs v1) ---- */
void spv2_batch_tanh   (const double* x, double* out, long long n);
void spv2_batch_erf    (const double* x, double* out, long long n);
void spv2_batch_gelu   (const double* x, double* out, long long n);
void spv2_batch_sigmoid(const double* x, double* out, long long n);
double spv2_gelu_quintic(double x);
void   spv2_batch_gelu_quintic(const double* x, double* out, long long n);

#ifdef __cplusplus
}
#endif
#endif /* SPUR_H */
