# -*- coding: utf-8 -*-
"""Benchmark SpearVM Turbo : noyaux AVX2 vs numpy exact.
Compile d'abord la DLL puis benchmark chaque noyau."""
import ctypes, time, math, os, subprocess
import numpy as np

N = 1_000_000

# ---- compile la DLL si nécessaire ----
src = os.path.join(os.path.dirname(__file__),"..","src","spur_turbo.c")
dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","bin","spur_turbo.dll"))

if not os.path.exists(dll_path) or os.path.getmtime(src) > os.path.getmtime(dll_path):
    cmd = ["gcc","-O3","-march=native","-mavx2","-mfma","-fopenmp",
           "-shared","-o",dll_path,src]
    r = subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(1)

print("[spur] DLL chargée")

lib = ctypes.CDLL(dll_path)
for fn in ("spur_gelu","spur_erf","spur_tanh"):
    f = getattr(lib,fn)
    f.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.c_longlong]
    f.restype = None

def spur_call(fn,x_arr,out_arr,n):
    xp = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    op = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    fn(xp,op,n)

# ---- données de test : trajectoire [-3,3] ----
x_np = np.linspace(-3,3,N).astype(np.float64)
out_spur = np.zeros(N,dtype=np.float64)

xp = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
op = out_spur.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

# ---- validation : compare vs numpy exact ----
print("\n=== Validation ===\n")
reps_val = 3

# gelu
spur_call(lib.spur_gelu,x_np,op,N)
g_np = 0.5*x_np*(1+np.vectorize(math.erf)(x_np/math.sqrt(2)))
err_gelu = np.max(np.abs(out_spur-g_np))
print(f"  gelu err_max = {err_gelu:.6f}")

# erf
spur_call(lib.spur_erf,x_np,op,N)
e_np = np.vectorize(math.erf)(x_np)
err_erf = np.max(np.abs(out_spur-e_np))
print(f"  erf  err_max = {err_erf:.6f}")

# tanh
spur_call(lib.spur_tanh,x_np,op,N)
t_np = np.tanh(x_np)
err_tanh = np.max(np.abs(out_spur-t_np))
print(f"  tanh err_max = {err_tanh:.6f}")

# ---- benchmark ----
print(f"\n=== Benchmark N={N:,} éléments ===\n")
results = {}

kernels = [
    ("gelu", lib.spur_gelu),
    ("erf",  lib.spur_erf),
    ("tanh", lib.spur_tanh),
]

np_funcs = {
    "gelu": lambda x: 0.5*x*(1+np.vectorize(math.erf)(x/math.sqrt(2))),
    "erf":  lambda x: np.vectorize(math.erf)(x),
    "tanh": np.tanh,
}

for name,fn in kernels:
    # warmup
    spur_call(fn,x_np,op,N)

    best = float('inf')
    for _ in range(10):
        t0 = time.perf_counter()
        spur_call(fn,x_np,op,N)
        dt = time.perf_counter()-t0
        if dt < best: best = dt
    results[name] = best*1000
    print(f"  SPUR {name:>4} : {best*1000:8.2f} ms ({N/best/1e9:.2f} Gelem/s)")

# numpy comparison (si scipy pour erf, sinon juste tanh)
try:
    from scipy.special import erf as np_erf
    has_scipy = True
except ImportError:
    has_scipy = False

print(f"\n{'kernel':>6} {'SPUR(ms)':>10} {'numpy(ms)':>10} {'speedup':>8}")
print("-"*40)

# tanh : comparable directement
best_tanh_spur = results["tanh"]
t0 = time.perf_counter()
for _ in range(10): y_np = np.tanh(x_np)
best_tanh_np = min((time.perf_counter()-t0) for _ in range(1))*1000
print(f"{'TANH':>6} {best_tanh_spur:>10.2f} {best_tanh_np:>10.2f} {best_tanh_np/best_tanh_spur:>7.1f}x")

print(f"\n[spur] Tous les benchmarks terminés.")
