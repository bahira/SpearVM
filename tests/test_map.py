"""Map kernel multi-entrees : add 2-in bit-exact, gelu*in1, 4-in somme, free."""
import os
import platform
import sys

import numpy as np
import pytest

_JIT_OK = os.path.exists("bin/spur.dll") or os.path.exists("bin/spur.so")
_JIT_OK = _JIT_OK and (
    sys.platform == "win32"
    or (sys.platform.startswith("linux") and platform.machine() in ("x86_64", "amd64")))
if not _JIT_OK:
    pytest.skip("JIT map kernel : x64 (Windows, ou Linux avec bin/spur.so build via make jit)",
                allow_module_level=True)

import ctypes
if sys.platform == "win32":
    lib = ctypes.CDLL("bin/spur.dll")
else:
    lib = ctypes.CDLL("bin/spur.so")


class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]


MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC = range(9)
GELU, ERF, TANH, LSE2 = 9, 10, 11, 12
MP_LD, MP_ST = 18, 19

lib.spur_map_build.restype = ctypes.c_int
lib.spur_map_build.argtypes = [ctypes.POINTER(SpurIns), ctypes.c_int]
lib.spur_map_exec.restype = None
lib.spur_map_exec.argtypes = [ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_longlong]
lib.spur_free.restype = None
lib.spur_free.argtypes = [ctypes.c_int]

n = 500
rng = np.random.default_rng(42)
out = np.zeros(n)


def run(prog, ins):
    arr = (SpurIns * len(prog))()
    for i, t in enumerate(prog):
        arr[i] = SpurIns(*t)
    h = lib.spur_map_build(arr, len(prog))
    assert h >= 0, f"build failed ({h})"
    arrays = [np.ascontiguousarray(a) for a in ins]
    ptrs = (ctypes.c_void_p * 4)()
    for i, a in enumerate(arrays[:4]):
        ptrs[i] = a.ctypes.data
    lib.spur_map_exec(h, ctypes.cast(ptrs, ctypes.c_void_p),
                      out.ctypes.data, n)
    return out.copy()


def gelu_np(v):
    return 0.997729 * (v * np.clip(0.306923 * v + 0.501, 0, 1.002)) - 0.004004


def test_add_two_inputs_bitexact():
    a = rng.normal(0, .3, n)
    b = rng.normal(0, .3, n)
    got = run([(MP_LD, 0, 0, 0, 0.0), (MP_LD, 1, 0, 1, 0.0),
               (ADD, 0, 1, 0, 0.0), (MP_ST, 0, 0, 0, 0.0)], [a, b])
    assert np.array_equal(got, a + b)


def test_gelu_mul():
    a = rng.normal(0, .4, n)
    b = rng.normal(0, .4, n)
    got = run([(MP_LD, 0, 0, 0, 0.0), (GELU, 0, 0, 0, 1.0),
               (MP_LD, 1, 0, 1, 0.0), (MUL, 0, 1, 0, 0.0),
               (MP_ST, 0, 0, 0, 0.0)], [a, b])
    assert float(np.max(np.abs(got - gelu_np(a) * b))) < 1e-12


def test_four_inputs_sum():
    ins = [rng.uniform(-2, 2, n) for _ in range(4)]
    prog = []
    for k in range(4):
        prog.append((MP_LD, k, 0, k, 0.0))       # vk <- ins[k][i]
    prog += [(ADD, 0, 1, 0, 0.0), (ADD, 2, 3, 2, 0.0),
             (ADD, 0, 2, 0, 0.0), (MP_ST, 0, 0, 0, 0.0)]
    got = run(prog, ins)
    ref = sum(ins)
    assert float(np.max(np.abs(got - ref))) < 1e-13


def test_free_and_reuse():
    a = np.ones(n)
    h1 = None
    prog = [(MP_LD, 0, 0, 0, 0.0), (MULI, 0, 0, 0, 2.0),
            (MP_ST, 0, 0, 0, 0.0)]
    arr = (SpurIns * len(prog))()
    for i, t in enumerate(prog):
        arr[i] = SpurIns(*t)
    h1 = lib.spur_map_build(arr, len(prog))
    lib.spur_free(h1)
    h2 = lib.spur_map_build(arr, len(prog))   # slot reutilise
    assert h2 >= 0
    run_ok = np.zeros(n)
    ptrs = (ctypes.c_void_p * 4)(a.ctypes.data, 0, 0, 0)
    lib.spur_map_exec(h2, ctypes.cast(ptrs, ctypes.c_void_p),
                      run_ok.ctypes.data, n)
    assert np.array_equal(run_ok, 2 * a)


def tanh_np(v):
    v = np.clip(v, -3, 3)
    return 0.900021 * ((v + 0.053639 * v**3) / (0.90122 + 0.343141 * v**2))


def test_tanh_plus_in1():
    a = rng.uniform(-3, 3, n)
    b = rng.normal(0, .3, n)
    got = run([(MP_LD, 0, 0, 0, 0.0), (TANH, 0, 0, 0, 1.0),
               (MP_LD, 1, 0, 1, 0.0), (ADD, 0, 1, 0, 0.0),
               (MP_ST, 0, 0, 0, 0.0)], [a, b])
    assert float(np.max(np.abs(got - (tanh_np(a) + b)))) < 1e-12
