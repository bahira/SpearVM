import ctypes, os, math
import numpy as np
import pytest

# localise la DLL
dll_dir = os.path.join(os.path.dirname(__file__),"..","bin")
dll_path = os.path.join(dll_dir,"spur_kernels.dll")
if not os.path.exists(dll_path):
    dll_path = os.path.join(os.path.dirname(__file__),"..","spur_math","spur_kernels.dll")

lib = ctypes.CDLL(os.path.abspath(dll_path))
for fn_name in ("spur_batch_gelu","spur_batch_erf","spur_batch_tanh"):
    f = getattr(lib,fn_name)
    f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_longlong]
    f.restype=None

class TestGelu:
    def test_positive_range(self):
        x = np.linspace(0.1,2.0,1000)
        out = np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_gelu(xp,op,len(x))
        ref=0.5*x*(1+np.vectorize(math.erf)(x/np.sqrt(2)))
        assert np.max(np.abs(out-ref)) < 0.10

    def test_negative_range(self):
        x = np.linspace(-2.0,-0.1,1000)
        out = np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_gelu(xp,op,len(x))
        ref=0.5*x*(1+np.vectorize(math.erf)(x/np.sqrt(2)))
        assert np.max(np.abs(out-ref)) < 0.15

    def test_zero_input(self):
        x=np.array([0.0])
        out=np.zeros(1)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_gelu(xp,op,1)
        assert abs(out[0]-(-0.004004)) < 0.01  # gelu(0)≈-0.004

class TestErf:
    def test_accuracy(self):
        x=np.linspace(-2,2,10000)
        out=np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_erf(xp,op,len(x))
        ref=np.vectorize(math.erf)(x)
        assert np.max(np.abs(out-ref)) < 0.02

    def test_bounds(self):
        """erf est borné [-1,1]"""
        x=np.linspace(-2,2,10000)
        out=np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_erf(xp,op,len(x))
        assert np.all(np.abs(out)<1.01)

class TestTanh:
    def test_accuracy(self):
        x=np.linspace(-3,3,10000)
        out=np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_tanh(xp,op,len(x))
        ref=np.tanh(x)
        assert np.max(np.abs(out-ref)) < 0.01

    def test_monotonic(self):
        """tanh est croissante"""
        x=np.linspace(-3,3,1000)
        out=np.zeros_like(x)
        xp=x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        op=out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lib.spur_batch_tanh(xp,op,len(x))
        assert np.all(np.diff(out)>=-1e-10) or np.all(np.diff(out)<=1e-10)
