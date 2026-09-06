"""Correctness du GEMM NT v2 (noyaux autotunes + aiguillage pack/dot).

Points sensibles couverts :
  * les deux branches d'aiguillage (min(m,n) <= 32 -> dot, sinon pack) ;
  * les bords : m ou n non multiples de MR/NR (4/12 en f64, 4/24 en f32) ;
  * le blocage en k : k > KC force le chemin "accumulate" (576 f64, 384 f32,
    1024 pour le noyau dot) ;
  * C doit etre **entierement ecrit** : on le pre-remplit de NaN, aucun NaN ne
    doit survivre (un tile oublie ressortirait immediatement) ;
  * equivalence avec l'ancien noyau, toujours exporte sous *_legacy.
"""
import ctypes
import itertools

import numpy as np
import pytest

import spur_math as sm

_DLL = ctypes.CDLL(sm._dll_path)
_PD = ctypes.POINTER(ctypes.c_double)
_PF = ctypes.POINTER(ctypes.c_float)
for _name, _p in (("spur_matmul_nt_legacy", _PD), ("spur_matmul_nt_f32_legacy", _PF)):
    _fn = getattr(_DLL, _name)
    _fn.argtypes = [_p, _p, _p] + [ctypes.c_longlong] * 3
    _fn.restype = None


def _legacy(a, b):
    dt = a.dtype
    m, k = a.shape
    n = b.shape[0]
    c = np.zeros((m, n), dtype=dt)
    p = _PF if dt == np.float32 else _PD
    fn = _DLL.spur_matmul_nt_f32_legacy if dt == np.float32 else _DLL.spur_matmul_nt_legacy
    fn(a.ctypes.data_as(p), b.ctypes.data_as(p), c.ctypes.data_as(p), m, k, n)
    return c


def _rel_err(got, ref):
    scale = max(float(np.abs(ref).max()), 1e-30)
    return float(np.abs(got.astype(np.float64) - ref).max()) / scale


# tailles choisies autour des seuils d'aiguillage et des tailles de tuile
DIMS = [1, 2, 3, 4, 5, 7, 8, 13, 16, 24, 31, 32, 33, 47, 64, 65, 96, 129]
KS = [1, 2, 3, 7, 8, 15, 16, 64, 127, 385, 577, 1025, 1153]


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 2e-6)])
def test_matmul_nt_shapes(dtype, tol):
    rng = np.random.default_rng(20260906)
    worst = 0.0
    for i, (m, n) in enumerate(itertools.product(DIMS, DIMS)):
        k = KS[i % len(KS)]
        a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=dtype)
        b = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=dtype)
        ref = a.astype(np.float64) @ b.astype(np.float64).T
        got = sm.matmul_nt(a, b)
        assert np.isfinite(got).all(), f"NaN/inf residuel en {m}x{k}x{n}"
        worst = max(worst, _rel_err(got, ref))
    assert worst <= tol, f"erreur relative max {worst:.2e} > {tol:.0e}"


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 2e-6)])
def test_matmul_nt_c_fully_written(dtype, tol):
    """Le noyau ecrit C sans pre-mise a zero : NaN en entree -> aucun NaN en sortie."""
    rng = np.random.default_rng(7)
    for (m, k, n) in [(37, 577, 41), (64, 1153, 64), (5, 1025, 300), (256, 96, 10)]:
        a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=dtype)
        b = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=dtype)
        c = np.full((m, n), np.nan, dtype=dtype)
        p = _PF if dtype == np.float32 else _PD
        fn = sm._matmul_nt_f32 if dtype == np.float32 else sm._matmul_nt
        fn(a.ctypes.data_as(p), b.ctypes.data_as(p), c.ctypes.data_as(p), m, k, n)
        assert np.isfinite(c).all(), f"cases non ecrites en {m}x{k}x{n}"
        assert _rel_err(c, a.astype(np.float64) @ b.astype(np.float64).T) <= tol


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-13), (np.float32, 2e-6)])
def test_matches_legacy_kernel(dtype, tol):
    rng = np.random.default_rng(11)
    for (m, k, n) in [(1, 512, 512), (16, 512, 512), (100, 100, 100),
                      (129, 257, 333), (512, 16, 512)]:
        a = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=dtype)
        b = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=dtype)
        ref = a.astype(np.float64) @ b.astype(np.float64).T
        new, old = sm.matmul_nt(a, b), _legacy(a, b)
        assert _rel_err(new, ref) <= tol
        assert _rel_err(old, ref) <= tol
        assert _rel_err(new, old.astype(np.float64)) <= tol


def test_backward_still_consistent():
    """matmul_backward passe par matmul_nt : la chaine complete doit tenir."""
    rng = np.random.default_rng(3)
    dY = np.ascontiguousarray(rng.standard_normal((33, 17)))
    A = np.ascontiguousarray(rng.standard_normal((33, 9)))
    B = np.ascontiguousarray(rng.standard_normal((17, 9)))
    dA, dB = sm.matmul_backward(dY, A, B)
    assert _rel_err(dA, dY @ B) <= 1e-13
    assert _rel_err(dB, dY.T @ A) <= 1e-13
