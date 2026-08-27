"""Le module packagé spur_math._jit : compile_map/run_map/free (nouvelle ABI)."""
import os
import platform
import sys

import numpy as np
import pytest

if sys.platform != "win32" and not (
    sys.platform.startswith("linux")
    and platform.machine() in ("x86_64", "amd64")):
    pytest.skip("JIT : Windows x64 ou Linux x86-64", allow_module_level=True)

try:
    import spur_math._jit as jit
except ImportError as e:
    pytest.skip(f"spur_jit indisponible : {e}", allow_module_level=True)


def gelu_np(v):
    return 0.997729 * (v * np.clip(0.306923 * v + 0.501, 0, 1.002)) - 0.004004


def test_pkg_jit_gelu_mul():
    rng = np.random.default_rng(11)
    prog = [(jit.MP_LD, 0, 0, 0, 0.0),
            (jit.GELU, 0, 0, 0, 1.0),
            (jit.MP_LD, 1, 0, 1, 0.0),
            (jit.MUL, 0, 1, 0, 0.0),
            (jit.MP_ST, 0, 0, 0, 0.0)]
    h = jit.compile_map(prog)
    x0 = rng.normal(0, 0.4, 300)
    x1 = rng.normal(0, 0.4, 300)
    out = np.zeros(300)
    jit.run_map(h, [x0, x1], out)
    assert float(np.max(np.abs(out - gelu_np(x0) * x1))) < 1e-12
    jit.free(h)


def test_pkg_jit_four_inputs():
    rng = np.random.default_rng(12)
    ins = [rng.uniform(-2, 2, 100) for _ in range(4)]
    p4 = [(jit.MP_LD, k, 0, k, 0.0) for k in range(4)]
    p4 += [(jit.ADD, 0, 1, 0, 0.0), (jit.ADD, 2, 3, 2, 0.0),
           (jit.ADD, 0, 2, 0, 0.0), (jit.MP_ST, 0, 0, 0, 0.0)]
    h = jit.compile_map(p4)
    out = np.zeros(100)
    jit.run_map(h, ins, out)
    assert float(np.max(np.abs(out - sum(ins)))) < 1e-13
