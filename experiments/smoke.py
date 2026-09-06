import sys, numpy as np
sys.path.insert(0, '.')
from gen_gemm import Variant, render
from harness import Case, bind_gemm, compile_source, make_operands, race, gflops, verify

cases = [Case(256,256,256), Case(512,512,512), Case(1024,768,1024)]
for v in (Variant(dtype="f64", mr=4, nr=12), Variant(dtype="f32", mr=4, nr=24)):
    lib = compile_source(render(v), v.name)
    call = bind_gemm(lib, v.name, v.dtype)
    err = verify(call, v.dtype)
    print(f"{v.name}: err_rel={err:.2e}")
    for case in cases:
        A,B,C = make_operands(case, v.dtype)
        t = race({"v": lambda: call(A,B,C), "np": lambda: np.dot(A,B.T)}, repeats=2)
        print(f"   {case.label():>16} : variante {gflops(case,t['v']):7.1f} GF | numpy {gflops(case,t['np']):7.1f} GF | ratio {t['np']/t['v']:.2f}")
