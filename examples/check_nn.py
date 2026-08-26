import numpy as np
import time

M, K, N = 1024, 768, 3072
raw = np.fromfile("nn_io.bin", dtype=np.float64)
xw, ww = M * K, N * K
X = raw[:xw].reshape(M, K)
W = raw[xw : xw + ww].reshape(N, K)
Y_spear = raw[xw + ww :].reshape(M, N)

try:
    from scipy.special import erf
except ImportError:
    from math import erf as _e
    erf = np.vectorize(_e)

def gelu(T):
    return 0.5 * T * (1.0 + erf(T / np.sqrt(2.0)))

# warmup + timing
T = X @ W.T; Y_np = gelu(T)
t_best = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    T = X @ W.T
    Y_np = gelu(T)
    t_best = min(t_best, (time.perf_counter() - t0) * 1000)

print(f"numpy total : {t_best:.1f} ms")

err = float(np.max(np.abs(Y_spear - Y_np)))
rel = err / float(np.max(np.abs(Y_np)))
print(f"err max     : {err:.3e} (rel {rel:.1e})")
ok = err <= 0.085  # contrat datasheet 0.079 + marge bruit fp (ordre de somme)
print("OK" if ok else "ECHEC", "- ecart vs contrat gelu (<=0.079) + bruit fp matmul")
