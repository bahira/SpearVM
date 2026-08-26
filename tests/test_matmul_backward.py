"""Gradients du matmul NT : dA/dB vs differences finies."""
import numpy as np

from spur_math import matmul_backward, matmul_nt_gelu


def test_matmul_backward_gradcheck():
    rng = np.random.default_rng(5)
    M, K, N = 7, 5, 4
    A = rng.normal(0, 0.5, (M, K))
    B = rng.normal(0, 0.5, (N, K))
    G = rng.normal(0, 1, (M, N))
    eps = 1e-6

    dA, dB = matmul_backward(G, A, B)

    def loss(Av, Bv):
        return float(np.sum((Av @ Bv.T) * G))

    num_dA = np.zeros_like(A)
    for i in range(M):
        for j in range(K):
            Ap = A.copy(); Ap[i, j] += eps
            Am = A.copy(); Am[i, j] -= eps
            num_dA[i, j] = (loss(Ap, B) - loss(Am, B)) / (2 * eps)
    assert float(np.max(np.abs(num_dA - dA))) < 1e-8

    num_dB = np.zeros_like(B)
    for i in range(N):
        for j in range(K):
            Bp = B.copy(); Bp[i, j] += eps
            Bm = B.copy(); Bm[i, j] -= eps
            num_dB[i, j] = (loss(A, Bp) - loss(A, Bm)) / (2 * eps)
    assert float(np.max(np.abs(num_dB - dB))) < 1e-8


def test_backward_through_fused_layer():
    """dL/dW complet de la couche fusionnee via matmul_backward+gelu_backward."""
    rng = np.random.default_rng(6)
    X = rng.normal(0, 0.5, (9, 6))
    W = rng.normal(0, 0.5, (4, 6))
    bias = rng.normal(0, 0.2, 4)
    G = rng.normal(0, 1, (9, 4))

    Tpre = X @ W.T + bias
    u = np.clip(0.306923 * Tpre + 0.501, 0, 1.002)
    dY = G * 0.997729 * np.where(
        (u > 0) & (u < 1.002), u + 0.306923 * Tpre,
        np.where(u <= 0, 0.0, 1.002),
    )
    _, dW = matmul_backward(dY, X, W)
    dbias = dY.sum(axis=0)

    eps = 1e-6

    def loss(Wv, bv):
        return float(np.sum(matmul_nt_gelu(X, Wv, bv) * G))

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy(); Wp[i, j] += eps
            Wm = W.copy(); Wm[i, j] -= eps
            num = (loss(Wp, bias) - loss(Wm, bias)) / (2 * eps)
            assert abs(num - dW[i, j]) < 1e-8, (i, j, num, dW[i, j])

    for j in range(len(bias)):
        bp = bias.copy(); bp[j] += eps
        bm = bias.copy(); bm[j] -= eps
        num = (loss(W, bp) - loss(W, bm)) / (2 * eps)
        assert abs(num - dbias[j]) < 1e-8, (j, num, dbias[j])
