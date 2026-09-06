"""Etage 8 : GEMM+GELU fusionne — l'ancienne fusion valait-elle son prix ?

`spur_matmul_nt_gelu` fusionnait la non-linearite dans la boucle j, ce qui
interdit de couper k et donc d'utiliser un micro-noyau packe. La v2 fait
GEMM autotune puis un epilogue biais+gelu vectorise (une relecture de C).

On mesure les deux dans la .so livree, avec le meme protocole alterne, et on
verifie que les deux chemins donnent le meme resultat a la tolerance flottante.

Sortie : results/gelu_fused.csv
"""

from __future__ import annotations

import csv
import ctypes
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import Case, gflops, race  # noqa: E402

import spur_math as sm  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

_DLL = ctypes.CDLL(sm._dll_path)
_PD = ctypes.POINTER(ctypes.c_double)
_PF = ctypes.POINTER(ctypes.c_float)
for _nm, _p in (("spur_matmul_nt_gelu", _PD), ("spur_matmul_nt_gelu_legacy", _PD),
                ("spur_matmul_nt_gelu_f32", _PF), ("spur_matmul_nt_gelu_f32_legacy", _PF)):
    _fn = getattr(_DLL, _nm)
    _fn.argtypes = [_p, _p, ctypes.c_void_p, _p] + [ctypes.c_longlong] * 3
    _fn.restype = None

SHAPES = [Case(128, 784, 256), Case(256, 256, 256), Case(384, 384, 384),
          Case(512, 512, 512), Case(1024, 768, 1024), Case(64, 3072, 768),
          Case(32, 768, 3072), Case(2048, 128, 128), Case(512, 64, 512)]


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "st"
    print(f"[{tag}] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}\n")
    rows: list[dict] = []
    for dtype, np_t, ptr, suf in (("f64", np.float64, _PD, ""),
                                  ("f32", np.float32, _PF, "_f32")):
        v2 = getattr(_DLL, f"spur_matmul_nt_gelu{suf}")
        old = getattr(_DLL, f"spur_matmul_nt_gelu{suf}_legacy")
        print(f"{'forme':>16} {'legacy':>8} {'v2':>8} {'gain':>7}   {dtype}")
        for case in SHAPES:
            rng = np.random.default_rng(0)
            A = np.ascontiguousarray(rng.standard_normal((case.m, case.k)), dtype=np_t)
            B = np.ascontiguousarray(rng.standard_normal((case.n, case.k)), dtype=np_t)
            bias = np.ascontiguousarray(rng.standard_normal(case.n), dtype=np_t)
            C1 = np.zeros((case.m, case.n), dtype=np_t)
            C2 = np.zeros((case.m, case.n), dtype=np_t)
            pa, pb = A.ctypes.data_as(ptr), B.ctypes.data_as(ptr)
            pbias = bias.ctypes.data_as(ctypes.c_void_p)
            p1, p2 = C1.ctypes.data_as(ptr), C2.ctypes.data_as(ptr)
            args = (case.m, case.k, case.n)

            old(pa, pb, pbias, p1, *args)
            v2(pa, pb, pbias, p2, *args)
            gap = float(np.abs(C1.astype(np.float64) - C2.astype(np.float64)).max())
            scale = max(float(np.abs(C1).max()), 1e-30)

            t = race({"legacy": lambda: old(pa, pb, pbias, p1, *args),
                      "v2": lambda: v2(pa, pb, pbias, p2, *args)},
                     repeats=4, block=3)
            gf = {k: gflops(case, v) for k, v in t.items()}
            rows.append({"tag": tag, "dtype": dtype, "shape": case.label(),
                         "gf_legacy": round(gf["legacy"], 2), "gf_v2": round(gf["v2"], 2),
                         "speedup": round(gf["v2"] / gf["legacy"], 3),
                         "ecart_rel": gap / scale})
            print(f"{case.label():>16} {gf['legacy']:8.1f} {gf['v2']:8.1f} "
                  f"{gf['v2']/gf['legacy']:7.2f}   ecart {gap/scale:.1e}")
        print()

    out = RESULTS / f"gelu_fused_{tag}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for dtype in ("f64", "f32"):
        sub = [r["speedup"] for r in rows if r["dtype"] == dtype]
        print(f"  {dtype}: gain median x{sorted(sub)[len(sub)//2]:.2f} "
              f"(min x{min(sub):.2f}, max x{max(sub):.2f})")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
