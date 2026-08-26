import ctypes
import numpy as np

lib = ctypes.CDLL("bin/spur.dll")

class SpurIns(ctypes.Structure):
    _fields_ = [("op", ctypes.c_short), ("a", ctypes.c_short),
                ("b", ctypes.c_short), ("dst", ctypes.c_short),
                ("imm", ctypes.c_double)]

MOVI, ADDI, MULI, ADD, SUB, MUL, SUBI, BNZ, ACC = range(9)
GELU, ERF, TANH = 9, 10, 11
MP_LD, MP_ST = 18, 19

lib.spur_map_build.restype = ctypes.c_int
lib.spur_map_build.argtypes = [ctypes.POINTER(SpurIns), ctypes.c_int]
lib.spur_map_exec.restype = None
lib.spur_map_exec.argtypes = [ctypes.c_int] + [ctypes.c_void_p]*3 + [ctypes.c_longlong]

n = 4
x = np.array([0.5, -0.5, 2.0, -2.0])
out = np.zeros(n)

def run(prog, inp):
    arr = (SpurIns * len(prog))()
    for i, ins in enumerate(prog):
        arr[i] = SpurIns(*ins)
    h = lib.spur_map_build(arr, len(prog))
    lib.spur_map_exec(h, inp.ctypes.data, inp.ctypes.data,
                      out.ctypes.data, n)
    return out.copy()

def k_gelu(v):
    return 0.997729*(v*np.clip(0.306923*v+0.501, 0, 1.002))-0.004004

# sonde A : identite  ld -> st
print("A id   :", run([(MP_LD,0,0,0,0.0),(MP_ST,0,0,0,0.0)], x), "attendu", x)

# sonde B : ld -> muli 3 -> st
r = run([(MP_LD,0,0,0,0.0),(MULI,0,0,0,3.0),(MP_ST,0,0,0,0.0)], x)
print("B *3   :", r, "attendu", x*3)

# sonde C : ld -> gelu -> st
r = run([(MP_LD,0,0,0,0.0),(GELU,0,0,0,1.0),(MP_ST,0,0,0,0.0)], x)
print("C gelu :", r, "attendu", k_gelu(x))

# sonde D : MOVI seul -> st (charge pool sans rien d'autre)
r = run([(MOVI,0,0,1,42.0),(MP_ST,1,0,0,0.0)], x)
print("D movi :", r, "attendu", np.full(n, 42.0))

# sonde E : ld -> addi 1 -> st
r = run([(MP_LD,0,0,0,0.0),(ADDI,0,0,0,1.0),(MP_ST,0,0,0,0.0)], x)
print("E +1   :", r, "attendu", x+1)

# sonde F : ld v0 ; movi v1=3 ; mul dst=v1 -> st  (dst != a)
r = run([(MP_LD,0,0,0,0.0),(MOVI,0,0,1,3.0),(MUL,0,1,1,0.0),(MP_ST,1,0,0,0.0)], x)
print("F ld*3 :", r, "attendu", x*3)

