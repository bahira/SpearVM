"""Training/backprop : gradcheck numerique + convergence SGD reelle.
Fonctions pytest (pas de SystemExit au niveau module !)."""
import numpy as np
import pytest

import spur_math as sm


def test_gradcheck_gelu_backward():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    w = rng.uniform(-2, 2, 200)

    def f(v):
        return float(np.sum(sm.gelu(v) * w))

    eps = 1e-6
    num = np.zeros(200)
    for i in range(200):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        num[i] = (f(xp) - f(xm)) / (2 * eps)
    ana = sm.gelu_backward(w, x)
    assert float(np.max(np.abs(num - ana))) < 1e-5


def _layer_and_grads(X, W, G):
    T = sm.matmul_nt(X, W)
    dY = G * sm.gelu_backward(np.ones_like(T), T)
    dW = sm.matmul_nt(dY.T, X.T)   # (N,M) NT (K,M) -> (N,K)
    dX = sm.matmul_nt(dY, W.T)     # (M,N) NT (K,N) -> (M,K)
    return dW, dX


def test_gradcheck_layer_dW_dX():
    rng = np.random.default_rng(1)
    M, K, N = 8, 5, 3
    X = rng.normal(0, 0.5, (M, K))
    W = rng.normal(0, 0.5, (N, K))
    G = rng.normal(0, 1, (M, N))
    eps = 1e-6

    def loss(Xv, Wv):
        return float(np.sum(sm.matmul_nt_gelu(Xv, Wv) * G))

    dW, dX = _layer_and_grads(X, W, G)

    num_dW = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            num_dW[i, j] = (loss(X, Wp) - loss(X, Wm)) / (2 * eps)
    assert float(np.max(np.abs(num_dW - dW))) < 1e-5

    num_dX = np.zeros((M, K))
    for i in range(M):
        for j in range(K):
            Xp = X.copy(); Xp[i, j] += eps
            Xm = X.copy(); Xm[i, j] -= eps
            num_dX[i, j] = (loss(Xp, W) - loss(Xm, W)) / (2 * eps)
    assert float(np.max(np.abs(num_dX - dX))) < 1e-5


def test_training_converges():
    rng = np.random.default_rng(2)
    M, K = 256, 4
    Xtr = rng.uniform(-1, 1, (M, K))
    Wtrue = rng.uniform(-1, 1, K)
    ytr = sm.gelu(Xtr @ Wtrue) + 0.05 * rng.normal(0, 1, M)

    What = rng.normal(0, 0.3, (1, K))
    lr = 0.05
    first = None
    for step in range(300):
        Y = sm.matmul_nt_gelu(Xtr, What).ravel()
        resid = Y - ytr
        mse = float(np.mean(resid ** 2))
        if first is None:
            first = mse
        Tpre = sm.matmul_nt(Xtr, What).ravel()
        dLdT = sm.gelu_backward(2.0 * resid / M, Tpre)
        What -= lr * sm.matmul_nt(dLdT.reshape(1, -1), Xtr.T)
    assert mse < first * 0.5, f"loss {first:.4f} -> {mse:.4f}"


def test_matmul_nt_exact_vs_numpy():
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1, (16, 8))
    B = rng.normal(0, 1, (7, 8))
    err = float(np.max(np.abs(sm.matmul_nt(A, B) - A @ B.T)))
    assert err < 1e-12
