"""Quickstart spur-math : forward, backward, training d'une mini couche.

pip install spur-math   puis   python quickstart.py
"""
import numpy as np

import spur_math as sm

rng = np.random.default_rng(0)

# --- donnees synthetiques : y = gelu(x . w) + bruit --------------------------
X = rng.uniform(-1, 1, (256, 4))
w_true = rng.uniform(-1, 1, 4)
y = sm.gelu(X @ w_true) + 0.05 * rng.normal(0, 1, 256)

# --- modele : W (1,4), biais scalaire-like -----------------------------------
W = rng.normal(0, 0.3, (1, 4))
b = np.zeros(1)
lr = 0.05

print("step  loss")
for step in range(200):
    Y = sm.matmul_nt_gelu(X, W, b).ravel()          # forward fusionne
    resid = Y - y
    mse = float(np.mean(resid ** 2))
    if step % 50 == 0:
        print(f"{step:4d}  {mse:.5f}")

    Tpre = sm.matmul_nt(X, W).ravel() + b           # pre-activation
    dLdT = sm.gelu_backward(2.0 * resid / 256, Tpre)
    _, dW = sm.matmul_backward(dLdT.reshape(-1, 1), X, W)  # (M,N) oblige
    W -= lr * dW
    b -= lr * np.array([dLdT.sum()])

print(f"final {mse:.5f}")
print(f"W appris : {np.round(W.ravel(), 3)}  b={np.round(b, 3)}")
print("(gelu sature : plusieurs (W,b) equivalents, la perte est le seul juge)")
