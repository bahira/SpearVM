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

_pkg_dir = os.path.dirname(os.path.abspath(_dll_path))
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_pkg_dir)  # runtimes MinGW a cote du .dll

_dll = ctypes.CDLL(_dll_path)

# refus propre (au lieu de SIGILL) sur CPU sans AVX2+FMA
_dll.spur_cpu_ok.restype = ctypes.c_int
if not _dll.spur_cpu_ok():
    raise ImportError(
        "spur_math : ce paquet requiert un CPU avec AVX2 et FMA "
        "(Intel Haswell 2013+, AMD Zen 2017+)."
    )


def _bind(name):
    f = getattr(_dll, name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.c_longlong]
    f.restype = None
    return f


_batch_gelu = _bind("spur_batch_gelu")
_batch_erf = _bind("spur_batch_erf")
_batch_tanh = _bind("spur_batch_tanh")


def _bind_mm(name, with_bias=False):
    f = getattr(_dll, name)
    if with_bias:
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                      ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
    else:
        f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                      ctypes.POINTER(ctypes.c_double),
                      ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
    f.restype = None
    return f


def _bind_bw(name):
    f = getattr(_dll, name)
    f.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                  ctypes.POINTER(ctypes.c_double), ctypes.c_longlong]
    f.restype = None
    return f


_matmul_nt = _bind_mm("spur_matmul_nt")
_matmul_nt_gelu = _bind_mm("spur_matmul_nt_gelu", with_bias=True)
_gelu_backward = _bind_bw("spur_batch_gelu_backward")

_PF = ctypes.POINTER(ctypes.c_float)
_batch_gelu_f32 = _bind("spur_batch_gelu_f32")

_matmul_nt_f32 = getattr(_dll, "spur_matmul_nt_f32")
_matmul_nt_f32.argtypes = [_PF, _PF, _PF] + [ctypes.c_longlong] * 3
_matmul_nt_f32.restype = None

_matmul_nt_gelu_f32 = getattr(_dll, "spur_matmul_nt_gelu_f32")
_matmul_nt_gelu_f32.argtypes = [_PF, _PF, ctypes.c_void_p, _PF] + [
    ctypes.c_longlong] * 3
_matmul_nt_gelu_f32.restype = None

_gelu_backward_f32 = getattr(_dll, "spur_batch_gelu_backward_f32")
_gelu_backward_f32.argtypes = [_PF, _PF, _PF, ctypes.c_longlong]
_gelu_backward_f32.restype = None

for _n in ("spur_batch_erf_backward", "spur_batch_tanh_backward",
           "spur_batch_sigmoid_backward"):
    _f = getattr(_dll, _n)
    _f.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [ctypes.c_longlong]
    _f.restype = None
_erf_backward = _dll.spur_batch_erf_backward
_tanh_backward = _dll.spur_batch_tanh_backward
_sigmoid_backward = _dll.spur_batch_sigmoid_backward


def matmul_nt(a, b):
    """C = A . B^T. a:(m,k), b:(n,k) -> (m,n). float32 ou float64."""
    dt = np.float32 if np.asarray(a).dtype == np.float32 else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    b = np.ascontiguousarray(b, dtype=dt)
    m, k = a.shape
    n = b.shape[0]
    c = np.zeros((m, n), dtype=dt)
    if dt == np.float32:
        _matmul_nt_f32(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       m, k, n)
    else:
        _matmul_nt(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   m, k, n)
    return c


def matmul_nt_gelu(a, b, bias=None):
    """C = gelu(A . B^T + bias) fusionne. float32 ou float64. bias:(n,) ou None."""
    dt = np.float32 if np.asarray(a).dtype == np.float32 else np.float64
    a = np.ascontiguousarray(a, dtype=dt)
    b = np.ascontiguousarray(b, dtype=dt)
    m, k = a.shape
    n = b.shape[0]
    c = np.zeros((m, n), dtype=dt)
    bp = None
    if bias is not None:
        bias = np.ascontiguousarray(bias, dtype=dt)
        assert bias.shape == (n,), f"bias attendu ({n},), recu {bias.shape}"
        bp = bias.ctypes.data_as(ctypes.c_void_p)
    if dt == np.float32:
        _matmul_nt_gelu_f32(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            bp,
                            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            m, k, n)
    else:
        _matmul_nt_gelu(a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        bp,
                        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        m, k, n)
    return c


def gelu_backward(dY, x):
    """dX = dY * gelu'(x). float32 ou float64."""
    dt = np.float32 if np.asarray(x).dtype == np.float32 else np.float64
    dYc = np.ascontiguousarray(dY, dtype=dt)
    xc = np.ascontiguousarray(x, dtype=dt)
    out = np.zeros_like(xc)
    if dt == np.float32:
        _gelu_backward_f32(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                           xc.size)
    else:
        _gelu_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       xc.size)
    return out


def erf_backward(dY, x):
    """dX = dY * erf_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _erf_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                  xc.size)
    return out


def tanh_backward(dY, x):
    """dX = dY * tanh_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _tanh_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   xc.size)
    return out


def sigmoid_backward(dY, x):
    """dX = dY * sigmoid_approx'(x)."""
    dYc = np.ascontiguousarray(dY, dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _sigmoid_backward(dYc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                      xc.size)
    return out


def gelu(x):
    """GELU approximatif AVX2. Erreur max 0.079 sur [-2,2] (f32: +1e-6 bruit)."""
    if np.asarray(x).dtype == np.float32:
        xc = np.ascontiguousarray(x, dtype=np.float32)
        out = np.zeros_like(xc)
        _batch_gelu_f32(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        xc.size)
        return out
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    _batch_gelu(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out


def erf(x):
    """erf approximatif AVX2. Erreur max 0.011 sur [-2, 2]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_erf
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out


def tanh(x):
    """tanh approximatif AVX2. Erreur max 0.008 sur [-3, 3]."""
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.zeros_like(xc)
    fn = _batch_tanh
    fn(xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), xc.size)
    return out
