/* spur.h — API publique SpearVM Kernels AVX2 */
#ifndef SPUR_KERNELS_H
#define SPUR_KERNELS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Applique gelu(x) élément-par-élément (approximation SPEAR certifiée) */
void spur_batch_gelu(const double* x, double* out, long long n);

/* Applique erf(x) approximatif (rationnel certifié [-2,2]) */
void spur_batch_erf(const double* x, double* out, long long n);

/* Applique tanh(x) approximatif (rationnel certifié [-3,3]) */
void spur_batch_tanh(const double* x, double* out, long long n);

/* max(a,b) élément-par-élément (hard-max approximation de lse2) */
void spur_batch_lse2(const double* a, const double* b,
                     double* out, long long n);

#ifdef __cplusplus
}
#endif
#endif /* SPUR_KERNELS_H */
