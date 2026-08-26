/* bench_mm.c — matmul C=A.B : naif -> dot -> AVX2 -> AVX2 bloc-4l -> +OMP
   Validation de chaque variante contre le naif ; timing adaptatif (VM clock). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void fill_rand(double* m, int n) {
    for (int i = 0; i < n * n; i++) m[i] = (double)rand() / RAND_MAX - 0.5;
}

static void transpose(const double* src, double* dst, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dst[(size_t)j * n + i] = src[(size_t)i * n + j];
}

/* v0 : naif ikj sur B brute — reference exacte et baseline */
static void mm_naive(const double* A, const double* B, double* C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        const double* arow = A + (size_t)i * n;
        double* crow = C + (size_t)i * n;
        for (int k = 0; k < n; k++) {
            double a = arow[k];
            const double* brow = B + (size_t)k * n;
            for (int j = 0; j < n; j++) crow[j] += a * brow[j];
        }
    }
}

static inline double hsum256(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi); /* [r0+r2, r1+r3] */
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}

/* v1 : produits scalaires scalaires ijk sur BT */
static void mm_dot(const double* A, const double* BT, double* C, int n) {
    for (int i = 0; i < n; i++) {
        const double* arow = A + (size_t)i * n;
        double* crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double* brow = BT + (size_t)j * n;
            double acc = 0.0;
            for (int k = 0; k < n; k++) acc += arow[k] * brow[k];
            crow[j] = acc;
        }
    }
}

/* v2 : AVX2 FMA sur le produit scalaire, queue scalaire */
static void mm_avx(const double* A, const double* BT, double* C, int n) {
    for (int i = 0; i < n; i++) {
        const double* arow = A + (size_t)i * n;
        double* crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double* brow = BT + (size_t)j * n;
            __m256d vacc = _mm256_setzero_pd();
            int k = 0;
            for (; k + 3 < n; k += 4)
                vacc = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k),
                                       _mm256_loadu_pd(brow + k), vacc);
            double acc = 0.0;
            for (; k < n; k++) acc += arow[k] * brow[k];
            crow[j] = hsum256(vacc) + acc;
        }
    }
}

static inline void dot_row(const double* arow, const double* brow,
                           double* out, int n) {
    /* out[0..3] = dot(arow0..3, brow) pour une ligne BT donnee */
    __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
    __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
    __m256d bv;
    int k = 0;
    for (; k + 3 < n; k += 4) {
        bv = _mm256_loadu_pd(brow + k);
        v0 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k), bv, v0);
        v1 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + n + k), bv, v1);
        v2 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + 2 * (size_t)n + k), bv, v2);
        v3 = _mm256_fmadd_pd(_mm256_loadu_pd(arow + 3 * (size_t)n + k), bv, v3);
    }
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (; k < n; k++) {
        double b = brow[k];
        s0 += arow[k] * b;
        s1 += arow[n + k] * b;
        s2 += arow[2 * (size_t)n + k] * b;
        s3 += arow[3 * (size_t)n + k] * b;
    }
    out[0] = hsum256(v0) + s0;
    out[1] = hsum256(v1) + s1;
    out[2] = hsum256(v2) + s2;
    out[3] = hsum256(v3) + s3;
}

static void avx4_body(const double* A, const double* BT, double* C, int n,
                      int i0lo, int i0hi) {
    for (int i0 = i0lo; i0 < i0hi; i0++) {
        const double* arow = A + (size_t)i0 * 4 * n;
        double* crow = C + (size_t)i0 * 4 * n;
        double out[4];
        for (int j = 0; j < n; j++) {
            dot_row(arow, BT + (size_t)j * n, out, n);
            crow[j] = out[0];
            crow[(size_t)n + j] = out[1];
            crow[2 * (size_t)n + j] = out[2];
            crow[3 * (size_t)n + j] = out[3];
        }
    }
}

/* v3 : blocage registres 4 lignes ; OMP seulement si n assez grand */
static void mm_avx4(const double* A, const double* BT, double* C, int n) {
    int nb = n / 4;
    if (nb > 0) {
        if (n >= 192) {
#pragma omp parallel for schedule(static)
            for (int i0 = 0; i0 < nb; i0++)
                avx4_body(A, BT, C, n, i0, i0 + 1);
        } else {
            avx4_body(A, BT, C, n, 0, nb);
        }
    }
    /* lignes restantes si n%4 != 0 */
    for (int i = (n / 4) * 4; i < n; i++) {
        const double* arow = A + (size_t)i * n;
        double* crow = C + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            const double* brow = BT + (size_t)j * n;
            __m256d vacc = _mm256_setzero_pd();
            int k = 0;
            for (; k + 3 < n; k += 4)
                vacc = _mm256_fmadd_pd(_mm256_loadu_pd(arow + k),
                                       _mm256_loadu_pd(brow + k), vacc);
            double acc = 0.0;
            for (; k < n; k++) acc += arow[k] * brow[k];
            crow[j] = hsum256(vacc) + acc;
        }
    }
}

typedef void (*mmfn)(const double*, const double*, double*, int);

static double time_fn(mmfn fn, const double* X, const double* Y, double* C,
                      int n) {
    fn(X, Y, C, n); /* warmup */
    long reps = 1;
    for (;;) {
        clock_t t0 = clock();
        for (long r = 0; r < reps; r++) fn(X, Y, C, n);
        clock_t dt = clock() - t0;
        if (dt >= CLOCKS_PER_SEC / 4 || reps >= (1L << 20))
            return 1000.0 * (double)dt / ((double)reps * CLOCKS_PER_SEC);
        reps *= 4;
    }
}

static double bench(mmfn fn, const char* name, const double* A,
                    const double* XY, double* C, const double* R, int n,
                    double tnaive) {
    double t = time_fn(fn, A, XY, C, n);
    double err = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) {
        double d = fabs(C[i] - R[i]);
        if (d > err) err = d;
    }
    printf("%-9s %9.2f ms  %7.2f GFLOPS", name, t,
           2.0 * n * n * n / t / 1e6);
    if (tnaive > 0)
        printf("  x%5.2f vs naif", tnaive / t);
    printf("  err=%.1e\n", err);
    return t;
}

int main(void) {
    /* threads persistants : evite le cout de reveil a chaque region */
    _putenv("OMP_WAIT_POLICY=ACTIVE");
    _putenv("GOMP_SPINCOUNT=infinite");
    int sizes[] = {128, 256, 512};
#ifdef _OPENMP
    printf("=== Matmul C=A.B === OpenMP : %d threads\n",
           omp_get_max_threads());
#else
    printf("=== Matmul C=A.B === OpenMP INACTIF\n");
#endif
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        size_t sz = (size_t)n * n * sizeof(double);
        double *A = malloc(sz), *B = malloc(sz), *BT = malloc(sz),
               *C = malloc(sz), *R = malloc(sz);
        if (!A || !B || !BT || !C || !R) { fprintf(stderr, "OOM\n"); return 1; }

        srand(42);
        fill_rand(A, n);
        fill_rand(B, n);
        transpose(B, BT, n);
        mm_naive(A, B, R, n);

        printf("--- n=%d (%d Mo/matrice) ---\n", n, (n * n * 8) >> 20);
        double t0 = bench(mm_naive, "naif", A, B, C, R, n, 0);
        double t1 = bench(mm_dot, "dot", A, BT, C, R, n, t0);
        double t2 = bench(mm_avx, "AVX2", A, BT, C, R, n, t0);
        double t3 = bench(mm_avx4, "AVX4+OMP", A, BT, C, R, n, t0);
        printf("gain global naif -> AVX4+OMP : x%.2f\n\n", t0 / t3);

        free(A); free(B); free(BT); free(C); free(R);
    }
    return 0;
}
