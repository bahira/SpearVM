import time
import numpy as np
import spur_math as sm

rng = np.random.default_rng(0)
print(f"{'size':>12} {'spur GFLOPS':>12} {'numpy GFLOPS':>13}  ratio")
for s in (512, 1024, 2048):
    A = rng.normal(0, 1, (s, s))
    B = rng.normal(0, 1, (s, s))
    sm.matmul_nt(A, B)  # warmup
    t = float("inf")
    for _ in range(3 if s <= 1024 else 2):
        t0 = time.perf_counter()
        C = sm.matmul_nt(A, B)
        t = min(t, time.perf_counter() - t0)
    tnp = float("inf")
    for _ in range(3 if s <= 1024 else 2):
        t0 = time.perf_counter()
        R = A @ B.T
        tnp = min(tnp, time.perf_counter() - t0)
    err = float(np.max(np.abs(C - R)))
    gf = 2 * s**3 / t / 1e9
    gfn = 2 * s**3 / tnp / 1e9
    print(f"{s:>12} {gf:>12.2f} {gfn:>13.2f}  x{gf/gfn:.2f}  err={err:.1e}")
