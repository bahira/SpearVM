"""Le backend de noyaux doit respecter le datasheet, natif comme en repli."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spearvm_sim import kernels as K

_erf = np.vectorize(math.erf, otypes=[np.float64])


def exact_gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def test_backend_reporte_ses_capacites(kernels):
    caps = kernels.capabilities()
    assert caps["backend"] in {"spearvm-avx2", "numpy-reference"}
    assert isinstance(caps["cpu_flags"], list)
    assert caps["numpy"] == np.__version__
    if caps["backend"] == "spearvm-avx2":
        assert caps["native"] is True
        assert caps["library"]


@pytest.mark.parametrize(
    "name,lo,hi,tol",
    [
        ("gelu", -2.0, 2.0, 0.0798),         # datasheet v1 (0.079 annonce, 0.0797 mesure)
        ("gelu_quintic", -3.5, 3.5, 0.0175),  # datasheet v2 quintique
        ("gelu_erf", -3.5, 3.5, 2.1e-5),      # datasheet haute precision
    ],
)
def test_variantes_gelu_dans_le_contrat(kernels, name, lo, hi, tol):
    x = np.linspace(lo, hi, 4096)
    got = np.asarray(getattr(kernels, name)(x), dtype=np.float64)
    assert np.max(np.abs(got - exact_gelu(x))) <= tol


def test_erf_et_tanh_dans_le_contrat(kernels):
    x = np.linspace(-2.0, 2.0, 4096)
    assert np.max(np.abs(np.asarray(kernels.erf(x)) - _erf(x))) <= 0.011
    y = np.linspace(-3.0, 3.0, 4096)
    assert np.max(np.abs(np.asarray(kernels.tanh(y)) - np.tanh(y))) <= 0.009


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-10), (np.float32, 2e-3)])
def test_matmul_nt_vs_numpy(kernels, dtype, tol):
    rng = np.random.default_rng(0)
    a = np.ascontiguousarray(rng.standard_normal((67, 33)), dtype=dtype)
    b = np.ascontiguousarray(rng.standard_normal((41, 33)), dtype=dtype)
    got = kernels.matmul_nt(a, b)
    assert got.shape == (67, 41)
    assert np.max(np.abs(got - a @ b.T)) <= tol


def test_matmul_nt_gelu_est_le_pipeline_fusionne(kernels):
    """Fusion == matmul puis gelu, y compris avec biais et queues non alignees."""
    rng = np.random.default_rng(3)
    a = np.ascontiguousarray(rng.standard_normal((37, 24)), dtype=np.float32)
    b = np.ascontiguousarray(rng.standard_normal((29, 24)) * 0.2, dtype=np.float32)
    bias = np.ascontiguousarray(rng.standard_normal(29), dtype=np.float32)

    fused = kernels.matmul_nt_gelu(a, b, bias)
    split = kernels.gelu(np.ascontiguousarray(kernels.matmul_nt(a, b) + bias, dtype=np.float32))
    assert np.max(np.abs(fused - split)) <= 1e-5


def test_gelu_backward_gradcheck(kernels):
    rng = np.random.default_rng(5)
    x = np.ascontiguousarray(rng.standard_normal(2048) * 2.0)
    eps = 1e-4
    analytic = np.asarray(kernels.gelu_backward(np.ones_like(x), x), dtype=np.float64)
    numeric = (np.asarray(kernels.gelu(x + eps)) - np.asarray(kernels.gelu(x - eps))) / (2 * eps)
    assert np.max(np.abs(analytic - numeric)) < 5e-4


def test_matmul_backward_gradients(kernels):
    rng = np.random.default_rng(7)
    A = np.ascontiguousarray(rng.standard_normal((12, 9)))
    B = np.ascontiguousarray(rng.standard_normal((7, 9)))
    dY = np.ascontiguousarray(rng.standard_normal((12, 7)))
    dA, dB = kernels.matmul_backward(dY, A, B)
    assert np.allclose(dA, dY @ B, atol=1e-10)
    assert np.allclose(dB, dY.T @ A, atol=1e-10)


def test_backend_de_repli_reste_utilisable(monkeypatch):
    """Sans noyaux natifs le serveur doit continuer a tourner (mode degrade)."""
    monkeypatch.setenv("SPEARVM_FORCE_FALLBACK", "1")
    K.reset_kernels()
    try:
        fallback = K.get_kernels()
        assert fallback.native is False
        assert fallback.name == "numpy-reference"
        x = np.linspace(-2, 2, 512)
        assert np.max(np.abs(np.asarray(fallback.gelu(x)) - exact_gelu(x))) < 1e-9
    finally:
        monkeypatch.delenv("SPEARVM_FORCE_FALLBACK", raising=False)
        K.reset_kernels()
