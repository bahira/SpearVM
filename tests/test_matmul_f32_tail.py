"""Regression : matmul_nt float32 doit etre exact quand k n'est pas multiple
de la largeur SIMD (bug : le reste k%8 etait ignore dans le bloc 8 lignes)."""
import numpy as np
import pytest
import spur_math as sm

SHAPES = [(8, 10, 8), (8, 7, 9), (16, 33, 17), (9, 300, 11),
          (64, 513, 40), (8, 257, 8), (8, 256, 8), (5, 1, 3), (17, 15, 23)]


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_matmul_nt_f32_tail(m, k, n):
    rng = np.random.default_rng(m * 1000 + k * 10 + n)
    A = rng.standard_normal((m, k)).astype(np.float32)
    B = rng.standard_normal((n, k)).astype(np.float32)
    C = sm.matmul_nt(A, B)
    R = A.astype(np.float64) @ B.astype(np.float64).T
    assert C.dtype == np.float32
    # tolerance fp32 relative a la norme (k termes)
    tol = 1e-5 * np.sqrt(k) * max(1.0, np.abs(R).max())
    assert np.abs(C - R).max() <= tol


@pytest.mark.parametrize("m,k,n", SHAPES)
def test_matmul_nt_gelu_f32_tail(m, k, n):
    rng = np.random.default_rng(7 + m + k + n)
    A = rng.standard_normal((m, k)).astype(np.float32)
    B = rng.standard_normal((n, k)).astype(np.float32)
    bias = rng.standard_normal(n).astype(np.float32)
    Y = sm.matmul_nt_gelu(A, B, bias)
    T = A.astype(np.float64) @ B.astype(np.float64).T + bias
    G = sm.gelu_erf(T)                       # forme exacte utilisee par le fusionne
    assert np.abs(Y - G).max() < 1e-4 * max(1.0, np.sqrt(k))


# ---- chemin AVX-512 vs AVX2 : memes resultats (dispatch a l'execution) ----
import os
import subprocess
import sys
import tempfile

_CHILD = r"""
import sys, numpy as np, spur_math as sm
m,k,n = map(int, sys.argv[1:4])
rng = np.random.default_rng(42)
A = rng.standard_normal((m,k)).astype(np.float32)
B = rng.standard_normal((n,k)).astype(np.float32)
np.save(sys.argv[4], sm.matmul_nt(A,B))
"""


def _run_mode(env_key, m, k, n, out):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]))
    env.pop("SPUR_FORCE_AVX2", None)
    env.pop("SPUR_FORCE_AVX512", None)
    if env_key:
        env[env_key] = "1"
    subprocess.check_call([sys.executable, "-c", _CHILD, str(m), str(k),
                           str(n), out], env=env)
    return np.load(out)


@pytest.mark.parametrize("m,k,n", [(40, 600, 300), (33, 257, 17), (8, 16, 16)])
def test_matmul_nt_f32_avx512_eq_avx2(m, k, n):
    with tempfile.TemporaryDirectory() as d:
        c512 = _run_mode("SPUR_FORCE_AVX512", m, k, n, os.path.join(d, "a.npy"))
        c2 = _run_mode("SPUR_FORCE_AVX2", m, k, n, os.path.join(d, "b.npy"))
    assert np.abs(c512 - c2).max() <= 1e-4 * np.sqrt(k)


_CHILD_GELU = r"""
import sys, numpy as np, spur_math as sm
m,k,n,usebias = map(int, sys.argv[1:5])
rng = np.random.default_rng(7)
A = rng.standard_normal((m,k)).astype(np.float32)
B = rng.standard_normal((n,k)).astype(np.float32)
bias = rng.standard_normal(n).astype(np.float32) if usebias else None
np.save(sys.argv[5], sm.matmul_nt_gelu(A,B,bias))
"""


@pytest.mark.parametrize("m,k,n,bias", [(40, 600, 300, 1), (33, 257, 17, 0),
                                         (8, 16, 16, 1), (9, 768, 37, 1)])
def test_matmul_nt_gelu_f32_avx512_eq_avx2(m, k, n, bias):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {}
    with tempfile.TemporaryDirectory() as d:
        for mode in ("SPUR_FORCE_AVX512", "SPUR_FORCE_AVX2"):
            env = dict(os.environ, PYTHONPATH=repo)
            env.pop("SPUR_FORCE_AVX2", None)
            env.pop("SPUR_FORCE_AVX512", None)
            env[mode] = "1"
            out = os.path.join(d, mode + ".npy")
            subprocess.check_call([sys.executable, "-c", _CHILD_GELU, str(m),
                                   str(k), str(n), str(bias), out], env=env)
            outs[mode] = np.load(out)
    assert np.abs(outs["SPUR_FORCE_AVX512"] - outs["SPUR_FORCE_AVX2"]).max() <= 1e-4 * np.sqrt(k)


@pytest.mark.parametrize("m,k,n", [(9, 5, 17), (8, 1, 16), (17, 300, 33),
                                    (40, 600, 300), (3, 7, 2)])
def test_matmul_nt_f64_tail(m, k, n):
    rng = np.random.default_rng(m + 31 * k + 97 * n)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((n, k))
    C = sm.matmul_nt(A, B)
    assert np.abs(C - A @ B.T).max() < 1e-11 * max(1, np.sqrt(k))


@pytest.mark.parametrize("m,k,n,bias", [(40, 600, 300, 1), (9, 33, 17, 0),
                                         (8, 16, 16, 1), (3, 7, 2, 1)])
def test_matmul_nt_gelu_f64_fused(m, k, n, bias):
    rng = np.random.default_rng(m * k + n)
    A = rng.standard_normal((m, k))
    B = rng.standard_normal((n, k))
    b = rng.standard_normal(n) if bias else None
    C = sm.matmul_nt_gelu(A, B, b)
    T = A @ B.T + (b if bias else 0.0)
    assert np.abs(C - sm.gelu_erf(T)).max() < 1e-11
