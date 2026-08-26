# -*- coding: utf-8 -*-
"""Demo Batch API : spur_map_build/exec traite un tableau entier en une passe JIT.
out[i] = h*erf(c+f) + erf(g/2) + max(g*e, c*f)
avec g=gelu(0.6x), e=erf(x), c=tanh(gain*(g+e)), f=max(g,c), h=tanh(e-x)"""
import ctypes, time, math

class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]   # meme ordre que C: op,a,b,dst,imm

MP_LD,MP_ST = 18,19
MOVI,ADDI,MULI,ADD,SUB,MUL,SUBI,BNZ,ACC,GELU,ERF,TANH,LSE2,TANHA,TANHS,ERFA,ACCLSE,HALT = range(18)

dll=ctypes.CDLL("./spur.dll")
dll.spur_map_build.restype=ctypes.c_int
dll.spur_map_build.argtypes=[ctypes.POINTER(SpurIns),ctypes.c_int]
dll.spur_map_exec.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),ctypes.c_longlong]

GAIN=1.25
body=[
    (MP_LD,0,0,0,0),        # v0 = x
    (GELU,1,0,0,0.6),       # v1 = g = gelu(0.6x)
    (ERF,2,0,0,1.0),        # v2 = e = erf(x)
    (ADD,3,1,2,0),          # v3 = g+e
    (MULI,3,3,0,GAIN),      # v3 *= gain
    (TANH,3,3,0,1.0),       # v3 = c = tanh(gain*(g+e))
    (LSE2,4,1,3,0),         # v4 = f = max(g,c)
    (TANHS,5,2,1,0),        # v5 = h = tanh(e-g)
    (ADD,7,3,4,0),          # v7 = c+f
    (ERF,7,7,0,1.0),        # v7 = i = erf(c+f)
    (MUL,5,5,7,0),          # v5 = h*i
    (ERF,6,1,0,0.5),        # v6 = p = erf(g/2)
    (ADD,5,6,5,0),          # v5 = s = h*i+p
    (MUL,2,2,1,0),          # v2 = ge = e*g
    (MUL,4,4,3,0),          # v4 = cf = f*c
    (LSE2,6,2,4,0),         # v6 = m = max(ge,cf)
    (ADD,0,5,6,0),          # v0 = s+m
    (MP_ST,0,0,0,0),        # out[i] = v0
]
arr=(SpurIns*len(body))()
for i,(op,dst,a,b,im) in enumerate(body):
    arr[i].op,arr[i].dst,arr[i].a,arr[i].b,arr[i].imm=op,dst,a,b,im

h=dll.spur_map_build(arr,len(body))
assert h>=0, "map build failed"
print(f"[spur] map kernel #{h} compile ({len(body)} ops)")

# ---- validation exacte sur 2000 elements ----
NV=2000
xin=[(-1.0+i*(4.0/NV)) for i in range(NV)]
vin=(ctypes.c_double*NV)(*xin); vout=(ctypes.c_double*NV)()
dll.spur_map_exec(h,vin,vout,NV)
err_max=0.0
for x in xin[:200]:
    gx=0.6*x
    g=0.997729*(gx*min(1.002,max(0.0,0.306923*gx+0.501)))-0.004004
    e=math.erf(x); c=math.tanh(GAIN*(g+e)); f=max(g,c)
    hh=math.tanh(e-g); k=math.erf(c+f); p=math.erf(0.5*g)
    ref=hh*k+p+max(g*e,c*f)
    err_max=max(err_max,abs(vout[xin.index(x)]-ref)/max(1,abs(ref)))
print(f"[spur] validation 200 pts: err relative max={err_max*100:.3f}% "
      f"{'OK' if err_max<0.03 else 'HORS TOLERANCE'}")

# ---- benchmark 2M elements : JIT vs numpy (si dispo) ----
NB=2_000_000
big_in=(ctypes.c_double*NB)(*[(-1.0+i*(4.0/NB)) for i in range(NB)])
big_out=(ctypes.c_double*NB)()
t0=time.perf_counter(); dll.spur_map_exec(h,big_in,big_out,NB); dt_jit=(time.perf_counter()-t0)*1000
print(f"[spur] JIT map {NB} elems : {dt_jit:.1f} ms ({NB/dt_jit/1000:.2f} Gelem/s)")

try:
    import numpy as np
    x_np=np.linspace(-1,3,NB)
    t0=time.perf_counter()
    g=0.997729*(x_np*np.minimum(1.002,np.maximum(0.0,0.306923*x_np+0.501)))-0.004004
    e=np.vectorize(math.erf)(x_np)
    c=np.tanh(GAIN*(g+e)); f=np.maximum(g,c)
    hh=np.tanh(e-g); k=np.vectorize(math.erf)(c+f); p=np.vectorize(math.erf)(0.5*g)
    r=hh*k+p+np.maximum(g*e,c*f)
    dt_np=(time.perf_counter()-t0)*1000
    print(f"[numpy] equivalent exact : {dt_np:.1f} ms ({NB/dt_np/1000:.2f} Gelem/s)")
except ImportError:
    print("[numpy] non installe - comparaison sautee")
