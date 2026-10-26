"""PackedNT : memes resultats que matmul_nt / matmul_nt_gelu, poids pre-packes."""
import numpy as np
import pytest
import spur_math as sm

SHAPES = [(128, 384, 1536), (1, 384, 1536), (9, 33, 17), (40, 600, 300), (8, 16, 32), (5, 7, 65)]


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_packed_matmul_eq_plain(m, k, n):
    rng = np.random.default_rng(m + k + n)
    A = rng.standard_normal((m, k)).astype(np.float32)
    W = rng.standard_normal((n, k)).astype(np.float32)
    P = sm.PackedNT(W)
    C = P.matmul(A)
    R = A.astype(np.float64) @ W.astype(np.float64).T
    assert C.dtype == np.float32 and C.shape == (m, n)
    assert np.abs(C - R).max() <= 1e-5 * np.sqrt(k) * max(1.0, np.abs(R).max())


@pytest.mark.parametrize("m,k,n,bias", [(128, 384, 1536, 1), (9, 33, 17, 0), (1, 16, 40, 1),
                                         (40, 600, 300, 1)])
def test_packed_gelu_eq_plain(m, k, n, bias):
    rng = np.random.default_rng(3 * m + k)
    A = rng.standard_normal((m, k)).astype(np.float32)
    W = rng.standard_normal((n, k)).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32) if bias else None
    P = sm.PackedNT(W)
    C = P.matmul_gelu(A, b)
    ref = sm.matmul_nt_gelu(A, W, b)   # chemin non pre-packe
    assert np.abs(C - ref).max() < 1e-4


def test_packed_reuse_many_calls():
    rng = np.random.default_rng(9)
    W = rng.standard_normal((64, 48)).astype(np.float32)
    P = sm.PackedNT(W)
    for _ in range(20):
        A = rng.standard_normal((rng.integers(1, 50), 48)).astype(np.float32)
        assert np.abs(P.matmul(A) - A @ W.T).max() < 1e-4
