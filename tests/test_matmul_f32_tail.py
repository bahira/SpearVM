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
    G = sm.gelu(T) if hasattr(sm, "gelu") else None
    if G is None:
        pytest.skip("gelu indisponible")
    assert np.abs(Y - G).max() < 0.1 + 0.02 * np.sqrt(k)


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
