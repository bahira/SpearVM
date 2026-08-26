"""Bindings JIT (map kernel multi-entrees) — Windows x64 uniquement.

Compile le DLL si absent :
    gcc -O3 -mavx2 -mfma -shared -o spur_math/spur_jit.dll src/spur.c -lm
"""
import ctypes
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_dll_path = os.path.join(_HERE, "spur_jit.dll")
if not os.path.exists(_dll_path):
    raise ImportError(
        "spur_jit.dll introuvable (JIT Windows x64). "
        "gcc -O2 -mavx2 -mfma -shared -o spur_math/spur_jit.dll src/spur.c -lm"
    )

_dll = ctypes.CDLL(_dll_path)


class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]


# opcodes (include/spur.h)
MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC = range(9)
GELU, ERF, TANH, LSE2, TANHA, TANHS, ERFA, ACCLSE, HALT = range(9, 18)
MP_LD, MP_ST, SIGMOID = 18, 19, 20

_dll.spur_map_build.restype = ctypes.c_int
_dll.spur_map_build.argtypes = [ctypes.POINTER(SpurIns), ctypes.c_int]
_dll.spur_map_exec.restype = None
_dll.spur_map_exec.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 3 + [
    ctypes.c_longlong]


def compile_map(prog):
    """prog : liste de tuples (op,a,b,dst,imm) -> handle JIT."""
    arr = (SpurIns * len(prog))()
    for i, ins in enumerate(prog):
        arr[i] = SpurIns(*ins)
    h = _dll.spur_map_build(arr, len(prog))
    if h < 0:
        raise RuntimeError(f"spur_map_build a echoue ({h})")
    return h


def run_map(handle, in0, in1, out):
    """out[i] = F(in0[i], in1[i]) element-wise. in1 peut etre in0."""
    n = out.size
    _dll.spur_map_exec(handle,
                       in0.ctypes.data, in1.ctypes.data,
                       out.ctypes.data, n)
