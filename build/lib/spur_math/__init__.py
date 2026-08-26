import ctypes
import os

import numpy as np

_dll_path = os.path.join(os.path.dirname(__file__), "spur_kernels.dll")
if not os.path.exists(_dll_path):
    _dll_path = os.path.join(os.path.dirname(__file__), "libspur_kernels.so")
if not os.path.exists(_dll_path):
    raise ImportError(
        "spur_kernels DLL/SO introuvable a cote du package. "
        "Compilez-le: gcc -O3 -march=native -mavx2 -mfma -fopenmp -shared "
        "-o spur_math/spur_kernels.dll src/spur_kernels.c"
    )

_dll = ctypes.CDLL(os.path.abspath(_dll_path))


def _bind(name):
    f = getattr(_dll, name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.c_longlong]
    f.restype = None
    return f


_batch_gelu = _bind("spur_batch_gelu")
_batch_erf = _bind("spur_batch_erf")
_batch_tanh = _bind("spur_batch_tanh")


def gelu(x):
    """GELU approximatif AVX2. Erreur max 0.079 sur [-2, 2], sature proprement au-dela."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_gelu
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(xc))
    return out


def erf(x):
    """erf approximatif AVX2. Erreur max 0.011 sur [-2, 2]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_erf
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(xc))
    return out


def tanh(x):
    """tanh approximatif AVX2. Erreur max 0.008 sur [-3, 3]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_tanh
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(xc))
    return out
