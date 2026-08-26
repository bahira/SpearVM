"""spur_math — Noyaux mathématiques AVX2 accélérés (SPEAR certifiés)."""
import ctypes, os
import numpy as np

_dll_path = os.path.join(os.path.dirname(__file__), "spur_kernels.dll")
if not os.path.exists(_dll_path):
    _dll_path = os.path.join(os.path.dirname(__file__), "..", "bin", "spur_kernels.dll")
_dll = ctypes.CDLL(os.path.abspath(_dll_path))

def _bind_batch(fn_name):
    f = getattr(_dll, fn_name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double), ctypes.c_longlong]
    f.restype = None
    return f

_batch_gelu = _bind_batch("spur_batch_gelu")
_batch_erf = _bind_batch("spur_batch_erf")
_batch_tanh = _bind_batch("spur_batch_tanh")
_batch_lse2 = _bind_batch("spur_batch_lse2")

def _run(fn, x):
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(xc))
    return out

def gelu(x):
    """GELU approximatif AVX2. err ≤ 0.079 sur [-2,2]."""
    return _run(_batch_gelu, x)

def erf(x):
    """erf approximatif AVX2. err ≤ 0.011 sur [-2,2]."""
    return _run(_batch_erf, x)

def tanh(x):
    """tanh approximatif AVX2. err ≤ 0.008 sur [-3,3]."""
    return _run(_batch_tanh, x)

def lse2(a, b):
    """LSE2 hard-max. err ≤ ln2."""
    return np.maximum(a, b)
