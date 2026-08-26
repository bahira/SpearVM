# -*- coding: utf-8 -*-
"""Bindings ctypes SpearVM — kernel boucle JIT x64.

Le kernel s'écrit comme une liste d'instructions (op,a,b,dst,imm) :
  MOVI dst,imm      MOVI 7,0,0,7,N        ; compteur
  ADDI  a,imm       ADDI 0,0,0,0.000004   ; x += pas
  GELU  dst,a,imm   GELU 1,0,0,0.6        ; v1 = gelu(0.6*v0)
  ERF/TANH/LSE2     idem
  ACC   a           ACC 3                 ; acc += v3
  SUBI  a,dst,imm   SUBI 7,7,7,1.0
  BNZ   a,target    BNZ 7,0,0,body_off
Voir examples/bench_native_vs_jit.c pour la version C."""
import ctypes

class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]

(MOVI,ADDI,MULI,ADD,SUB,MUL,SUBI,BNZ,ACC,GELU,ERF,TANH,LSE2,
 TANHA,TANHS,ERFA,ACCLSE,SIGMOID,HALT) = range(19)

dll = ctypes.CDLL("./spur.dll")
dll.spur_jit_build.restype = ctypes.c_int
dll.spur_jit_build.argtypes = [ctypes.POINTER(SpurIns), ctypes.c_int]
dll.spur_exec.restype = ctypes.c_double
dll.spur_exec.argtypes = [ctypes.c_int]

def build(prog):
    arr = (SpurIns * len(prog))()
    for i,(op,a,b,dst,imm) in enumerate(prog):
        arr[i].op,arr[i].a,arr[i].b,arr[i].dst,arr[i].imm = op,a,b,dst,imm
    h = dll.spur_jit_build(arr,len(prog))
    assert h >= 0, "jit build failed"
    return h

def run(h):
    return dll.spur_exec(h)

# ---- demo : onde amortie, 500k iterations ----
N = 500_000
prog = [
    (MOVI,7,0,0,float(N)),
    (MOVI,0,0,0,-1.0),
    (ADDI,0,0,0,0.000008),
    (GELU,0,0,1,0.6),
    (SIGMOID,1,2,2,0),
    (ADD,  3,1,2,0),
    (MULI, 3,3,0,1.25),
    (TANH, 3,3,0,1.0),
    (LSE2, 4,1,3,0),
    (ACC,  3,0,0,0),
    (SUBI, 7,7,7,1.0),
    (BNZ,  7,0,0,2),
]
h = build(prog)
import time
t0=time.perf_counter(); acc=run(h); dt=(time.perf_counter()-t0)*1000
print(f"kernel boucle : acc={acc:,.3f}  [{dt:.0f} ms pour {N} iters]")
