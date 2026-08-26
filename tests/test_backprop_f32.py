"""Backward erf/tanh/sigmoid + fp32 : gradchecks et exactitude."""
import numpy as np
import pytest

import spur_math as sm


@pytest.mark.parametrize("name", ["erf", "tanh", "sigmoid"])
def test_backward_gradcheck(name):
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.5, 2.5, 300)

    if name == "sigmoid":
        fwd = lambda v: 0.5 + 0.5 * sm.tanh(0.5 * v)
    else:
        fwd = getattr(sm, name)
    bwd = getattr(sm, f"{name}_backward")

    w = rng.normal(0, 1, 300)
    eps = 1e-6
    num = np.zeros(300)
    for i in range(300):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        num[i] = (float(np.sum(fwd(xp) * w)) -
                  float(np.sum(fwd(xm) * w))) / (2 * eps)
    ana = bwd(w, x)
    assert float(np.nanmax(np.abs(num - ana))) < 1e-5


def test_matmul_nt_f32_exact():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (256, 128)).astype(np.float32)
    B = rng.normal(0, 1, (64, 128)).astype(np.float32)
    err = float(np.max(np.abs(sm.matmul_nt(A, B) - A @ B.T)))
    assert err < 2e-4


def test_f32_faster_than_f64():
    import time
    rng = np.random.default_rng(1)
    A64 = rng.normal(0, 1, (1024, 1024))
    A32 = A64.astype(np.float32)
    sm.matmul_nt(A64, A64); sm.matmul_nt(A32, A32)  # warmup
    t0 = time.perf_counter(); sm.matmul_nt(A64, A64); t64 = time.perf_counter() - t0
    t0 = time.perf_counter(); sm.matmul_nt(A32, A32); t32 = time.perf_counter() - t0
    assert t32 < t64  # fp32 doit gagner (x~2)
