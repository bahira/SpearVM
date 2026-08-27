"""Bindings JIT (map kernel multi-entrees) — Linux x86-64 & Windows x64.

Compile automatiquement la lib si absente (spur_posix.c sous Linux/SysV,
spur.c sous Windows/Win64) :
    gcc -O2 -mavx2 -mfma -fPIC -shared -Iinclude -o spur_math/spur_jit.{dll,so} src/spur_posix.c -lm
"""
import ctypes
import os
import platform
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    _lib_name = "spur_jit.dll"
else:
    _lib_name = "spur_jit.so"

_dll_path = os.path.join(_HERE, _lib_name)

if not os.path.exists(_dll_path):
    # portage POSIX (SysV ABI) pour Linux ; spur.c (Win64 ABI) sur Windows
    _src_name = "spur.c" if sys.platform == "win32" else "spur_posix.c"
    src = os.path.join(os.path.dirname(_HERE), "src", _src_name)
    inc = os.path.join(os.path.dirname(_HERE), "include")
    cmd = (f"gcc -O2 -mavx2 -mfma -fPIC -shared -I{inc} "
           f"-o {_dll_path} {src} -lm")
    print(f"spur_jit absent, compilation automatique...\n  {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        raise ImportError(
            f"Echec compilation JIT ({e}). Installez gcc ou compilez manuellement :\n  {cmd}"
        ) from e

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
_dll.spur_map_exec.argtypes = [ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_longlong]
_dll.spur_free.restype = None
_dll.spur_free.argtypes = [ctypes.c_int]


def compile_map(prog):
    """prog : liste de tuples (op,a,b,dst,imm) -> handle JIT."""
    arr = (SpurIns * len(prog))()
    for i, ins in enumerate(prog):
        arr[i] = SpurIns(*ins)
    h = _dll.spur_map_build(arr, len(prog))
    if h < 0:
        raise RuntimeError(f"spur_map_build a echoue ({h})")
    return h


def run_map(handle, ins, out):
    """out[i] = F(ins[0][i], ..., ins[3][i]). ins : sequence de 1 a 4 arrays."""
    arrays = [np.ascontiguousarray(a, dtype=np.float64) for a in ins]
    ptrs = (ctypes.c_void_p * 4)()
    for i, a in enumerate(arrays[:4]):
        ptrs[i] = a.ctypes.data
    _dll.spur_map_exec(handle, ctypes.cast(ptrs, ctypes.c_void_p),
                       out.ctypes.data, out.size)


def free(handle):
    _dll.spur_free(handle)
