/* bench_nn.c — pipeline transformer FFN : Y = gelu(X . W^T)
   Matmul AVX4 bloc-4l puis batch_gelu SPEAR. Compare contre numpy via dump. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

void spur_batch_gelu(const double* x, double* out, long long n);

#define M 1024 /* batch */
#define K 768  /* d_model */
#define N 3072 /* d_ff */

static void fill_rand(double* m, size_t cnt) {
    for (size_t i = 0; i < cnt; i++) m[i] = (double)rand() / RAND_MAX - 0.5;
}

static void transpose(const double* src, double* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            dst[(size_t)j * rows + i] = src[(size_t)i * cols + j];
}

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}

static inline void dot_row4(const double* arow, const double* brow,
                            double* out, int k) {
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
    int kk = 0;
    for (; kk + 3 < k; kk += 4) {
        __m256d bv = _mm256_loadu_pd(brow + kk);
        v0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + kk), bv, v0);
        v1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k + kk), bv, v1);
        v2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + 2 * (size_t)k + kk), bv, v2);
        v3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + 3 * (size_t)k + kk), bv, v3);
    }
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; kk < k; kk++) {
        double b = brow[kk];
        s0 += arow[kk] * b;
        s1 += arow[k + kk] * b;
        s2 += arow[2 * (size_t)k + kk] * b;
        s3 += arow[3 * (size_t)k + kk] * b;
    }
    out[0] = hsum256(v0) + s0;
    out[1] = hsum256(v1) + s1;
    out[2] = hsum256(v2) + s2;
    out[3] = hsum256(v3) + s3;
}

/* T = X . W^T : T[i][j] = dot(X_ligne_i, W_ligne_j) => W sert directement,
   sans transposition (chaque ligne de W est lue sequentiellement)          */
static void matmul_avx4(const double* X, const double* Wm, double* T, int m,
                        int k, int n) {
#pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < m / 4; i0++) {
        const double* xrow = X + (size_t)i0 * 4 * k;
        double* trow = T + (size_t)i0 * 4 * n;
        double out[4];
        for (int j = 0; j < n; j++) {
            dot_row4(xrow, Wm + (size_t)j * k, out, k);
            trow[j] = out[0];
            trow[(size_t)n + j] = out[1];
            trow[2 * (size_t)n + j] = out[2];
            trow[3 * (size_t)n + j] = out[3];
        }
    }
    for (int i = (m / 4) * 4; i < m; i++) { /* queue */
        const double* xr = X + (size_t)i * k;
        double* tr = T + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double* wr = Wm + (size_t)j * k;
            __m256d vacc = _mm256_setzero_pd();
            int kk = 0;
            for (; kk + 3 < k; kk += 4)
                vacc = _mm256_fmadd_pd(_mm256_loadu_pd(xr + kk),
                                       _mm256_loadu_pd(wr + kk), vacc);
            double acc = 0.0;
            for (; kk < k; kk++) acc += xr[kk] * wr[kk];
            tr[j] = hsum256(vacc) + acc;
        }
    }
}

static double now_ms(void) { return 1000.0 * clock() / CLOCKS_PER_SEC; }

int main(void) {
    printf("=== SpearVM pipeline FFN : Y = gelu(X.W^T) ===\n");
#ifdef _OPENMP
    printf("OpenMP %d threads | M=%d K=%d N=%d\n\n", omp_get_max_threads(), M,
           K, N);
#endif

    double* X = malloc((size_t)M * K * 8);
    double* W = malloc((size_t)N * K * 8);

    double* T = malloc((size_t)M * N * 8);
    double* Y = malloc((size_t)M * N * 8);
    if (!X || !W || !T || !Y) return fprintf(stderr, "OOM\n"), 1;

    srand(42);
    fill_rand(X, (size_t)M * K);
    fill_rand(W, (size_t)N * K);

    /* warmup */
    matmul_avx4(X, W, T, M, K, N);

    /* --- etape 1 : matmul (T = produit brut, garde intact pour le dump) --- */
    double t0 = now_ms();
    int reps_mm = 5;
    for (int r = 0; r < reps_mm; r++) matmul_avx4(X, W, T, M, K, N);
    double t_mm = (now_ms() - t0) / reps_mm;

    /* --- etape 2 : gelu batch, lit T ecrit Y (jamais en place ici) --- */
    spur_batch_gelu(T, Y, (long long)M * N); /* warmup */
    t0 = now_ms();
    int reps_ge = 20;
    for (int r = 0; r < reps_ge; r++) spur_batch_gelu(T, Y, (long long)M * N);
    double t_ge = (now_ms() - t0) / reps_ge;

    double flops = 2.0 * M * K * N;
    printf("matmul  X.W^T : %8.2f ms  %6.2f GFLOPS\n", t_mm,
           flops / t_mm / 1e6);
    printf("gelu    batch  : %8.2f ms  %6.2f Gelem/s\n", t_ge,
           M * N / t_ge / 1e6);
    printf("TOTAL          : %8.2f ms\n", t_mm + t_ge);

    /* dump pour validation numpy */
    FILE* f = fopen("nn_io.bin", "wb");
    if (f) {
        fwrite(X, 8, (size_t)M * K, f);
        fwrite(W, 8, (size_t)N * K, f);
        fwrite(Y, 8, (size_t)M * N, f);
        fclose(f);
        printf("dump nn_io.bin OK (%.0f Mo)\n",
               ((size_t)M * K + (size_t)N * K + (size_t)M * N) * 8 / 1048576.0);
    }

    free(X); free(W); free(T); free(Y);
    return 0;
}
