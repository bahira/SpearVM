import ctypes
import os
import sys

import numpy as np
import pytest

if sys.platform != "win32" or not os.path.exists("bin/spur.dll"):
    pytest.skip("JIT map kernel : Windows x64 avec bin/spur.dll requis",
                allow_module_level=True)

lib = ctypes.CDLL("bin/spur.dll")

class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]

# opcodes (include/spur.h)
MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC = 0, 1, 2, 3, 4, 5, 6, 7, 8
GELU, ERF, TANH, LSE2, TANHA, TANHS, ERFA, ACCLSE, HALT = 9, 10, 11, 12, 13, 14, 15, 16, 17
MP_LD, MP_ST, SIGMOID = 18, 19, 20

lib.spur_map_build.restype = ctypes.c_int
lib.spur_map_build.argtypes = [ctypes.POINTER(SpurIns), ctypes.c_int]
lib.spur_map_exec.restype = None
lib.spur_map_exec.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_longlong]

n = 1000
rng = np.random.default_rng(42)
in0 = rng.standard_normal(n) * 0.3
in1 = rng.standard_normal(n) * 0.3
out = np.zeros(n)

def run(prog, a0, a1):
    arr = (SpurIns * len(prog))()
    for i, ins in enumerate(prog):
        arr[i] = SpurIns(*ins)
    h = lib.spur_map_build(arr, len(prog))
    assert h >= 0, f"build failed ({h})"
    lib.spur_map_exec(h, a0.ctypes.data, a1.ctypes.data,
                      out.ctypes.data, n)
    return out.copy()

ok = True

# P1 : out = in0 + in1
prog1 = [(MP_LD, 0, 0, 0, 0.0),      # v0 <- in0[i]
         (MP_LD, 1, 0, 1, 0.0),      # v1 <- in1[i]
         (ADD, 0, 1, 0, 0.0),        # v0 += v1
         (MP_ST, 0, 0, 0, 0.0)]      # out[i] <- v0
got = run(prog1, in0, in1)
err = float(np.max(np.abs(got - (in0 + in1))))
print(f"P1 add     err={err:.2e}", "OK" if err == 0 else "FAIL")
ok &= err == 0

# P2 : out = gelu(in0) * in1
prog2 = [(MP_LD, 0, 0, 0, 0.0),
         (GELU, 0, 0, 0, 1.0),       # v0 = gelu(v0)
         (MP_LD, 1, 0, 1, 0.0),
         (MUL, 0, 1, 0, 0.0),
         (MP_ST, 0, 0, 0, 0.0)]
got = run(prog2, in0, in1)
ref = 0.997729 * (in0 * np.clip(0.306923 * in0 + 0.501, 0, 1.002)) - 0.004004
ref *= in1
err = float(np.max(np.abs(got - ref)))
print(f"P2 gelu*in1 err={err:.2e}", "OK" if err < 1e-12 else "FAIL")
ok &= err < 1e-12

# P3 : out = tanh(in0) + in1  (op via appel C certifie)
prog3 = [(MP_LD, 0, 0, 0, 0.0),
         (TANH, 0, 0, 0, 1.0),
         (MP_LD, 1, 0, 1, 0, 0.0) if False else (MP_LD, 1, 0, 1, 0.0),
         (ADD, 0, 1, 0, 0.0),
         (MP_ST, 0, 0, 0, 0.0)]
got = run(prog3, in0, in1)


def k_tanh(x):
    x = np.clip(x, -3, 3)
    return 0.900021 * ((x + 0.053639 * x**3) / (0.90122 + 0.343141 * x**2))

ref = k_tanh(in0) + in1
err = float(np.max(np.abs(got - ref)))
print(f"P3 tanh+in1 err={err:.2e}", "OK" if err < 1e-9 else "FAIL")
ok &= err < 1e-9

print("=== TOUS PASS ===" if ok else "=== ECHEC ===")
raise SystemExit(0 if ok else 1)
