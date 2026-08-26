"""spur_math — Noyaux mathématiques accélérés AVX2 (polynômes certifiés SPEAR).
×4-47 plus rapide que libm scalaire pour les opérations transcendantales."""
import ctypes, os, sys

_dll_path = os.path.join(os.path.dirname(__file__), "spur_kernels.dll")
if not os.path.exists(_dll_path):
    # essaie le répertoire parent (dev mode)
    _dll_path = os.path.join(os.path.dirname(__file__), "..", "spur_kernels.dll")

_dll = ctypes.CDLL(os.path.abspath(_dll_path))

# configure les signatures
for fn_name in ("spur_batch_gelu","spur_batch_erf","spur_batch_tanh"):
    f = getattr(_dll, fn_name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double), ctypes.c_longlong]
    f.restype = None

def spur_batch_lse2(a,b,out,n):
    f = _dll.spur_batch_lse2 if hasattr(_dll,'spur_batch_lse2') else None
    if f is None:
        f = ctypes.CDLL(_dll_path).spur_batch_lse2
    f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),ctypes.c_longlong]
    f.restype=None
    f(a,b,out,n)

def _run(fn, x):
    """Convertit numpy array → pointeurs C, appelle le kernel, retourne out."""
    import numpy as np
    n = len(x)
    x_c = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    xp = x_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    op = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fn(xp, op, n)
    return out

def gelu(x):
    """GELU approximatif AVX2. err ≤ 0.079 sur [-2,2]."""
    return _run(_dll.spur_batch_gelu, x)

def erf(x):
    """erf approximatif AVX2. err ≤ 0.011 sur [-2,2]."""
    return _run(_dll.spur_batch_erf, x)

def tanh(x):
    """tanh approximatif AVX2. err ≤ 0.008 sur [-3,3]."""
    return _run(_dll.spur_batch_tanh, x)
