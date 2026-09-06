"""Verification finale : le noyau **porte dans src/spur_kernels.c** tient-il les
promesses mesurees sur les prototypes ?

On compare, dans un meme processus et avec le meme harnais alterne :
  legacy  = spur_matmul_nt_legacy      (le noyau livre jusqu'ici)
  v2      = spur_matmul_nt             (aiguillage pack/dot autotune)
  numpy   = OpenBLAS

Le nombre de threads est celui de l'environnement (OMP_NUM_THREADS et
OPENBLAS_NUM_THREADS doivent etre regles a la meme valeur pour etre honnete).

Usage :
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python verify_port.py st
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python verify_port.py mt
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
for _name, _p in (("spur_matmul_nt_legacy", _PD), ("spur_matmul_nt_f32_legacy", _PF),
                  ("spur_matmul_nt", _PD), ("spur_matmul_nt_f32", _PF)):
    _fn = getattr(_DLL, _name)
    _fn.argtypes = [_p, _p, _p] + [ctypes.c_longlong] * 3
    _fn.restype = None

SHAPES = [
    Case(64, 64, 64), Case(128, 128, 128), Case(256, 256, 256),
    Case(384, 384, 384), Case(512, 512, 512), Case(768, 768, 768),
    Case(1024, 1024, 1024), Case(1024, 768, 1024),
    Case(1, 512, 512), Case(4, 512, 512), Case(16, 512, 512), Case(32, 768, 3072),
    Case(64, 3072, 768), Case(128, 784, 256), Case(256, 256, 10),
    Case(512, 64, 512), Case(512, 16, 512), Case(2048, 128, 128),
    Case(100, 100, 100), Case(333, 257, 129),
]


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "st"
    threads = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"[{tag}] OMP_NUM_THREADS={threads} "
          f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', '?')}\n")

    rows: list[dict] = []
    for dtype, np_t, ptr in (("f64", np.float64, _PD), ("f32", np.float32, _PF)):
        legacy = getattr(_DLL, "spur_matmul_nt_legacy" if dtype == "f64"
                         else "spur_matmul_nt_f32_legacy")
        v2 = getattr(_DLL, "spur_matmul_nt" if dtype == "f64" else "spur_matmul_nt_f32")
        print(f"{'forme':>16} {'legacy':>8} {'v2':>8} {'numpy':>8} {'x_v2/leg':>9} "
              f"{'v2/numpy':>9}   {dtype}", flush=True)
        for case in SHAPES:
            rng = np.random.default_rng(0)
            A = np.ascontiguousarray(rng.standard_normal((case.m, case.k)), dtype=np_t)
            B = np.ascontiguousarray(rng.standard_normal((case.n, case.k)), dtype=np_t)
            C = np.zeros((case.m, case.n), dtype=np_t)
            pa, pb, pc = (A.ctypes.data_as(ptr), B.ctypes.data_as(ptr),
                          C.ctypes.data_as(ptr))
            ref = A.astype(np.float64) @ B.astype(np.float64).T

            # A/B des deux noyaux C entre eux (meme allocation, meme C) ;
            # numpy est chronometre a part : melange dans la meme course, ses
            # threads/allocations polluent les formes de quelques dizaines de us.
            reps = 200 if case.flops < 5e6 else (40 if case.flops < 5e7 else 4)
            timing = race({
                "legacy": lambda: legacy(pa, pb, pc, case.m, case.k, case.n),
                "v2": lambda: v2(pa, pb, pc, case.m, case.k, case.n),
            }, repeats=reps, block=3)
            timing.update(race({"numpy": lambda: np.dot(A, B.T)},
                               repeats=min(reps, 12), block=3))
            err = float(np.abs(C.astype(np.float64) - ref).max()) / max(
                float(np.abs(ref).max()), 1e-30)
            gf = {k: gflops(case, v) for k, v in timing.items()}
            row = {"tag": tag, "dtype": dtype, "shape": case.label(),
                   **{f"gf_{k}": round(v, 2) for k, v in gf.items()},
                   "speedup": round(gf["v2"] / gf["legacy"], 3),
                   "frac_numpy": round(gf["v2"] / gf["numpy"], 3),
                   "err": err}
            rows.append(row)
            print(f"{case.label():>16} {gf['legacy']:8.1f} {gf['v2']:8.1f} "
                  f"{gf['numpy']:8.1f} {row['speedup']:9.2f} {row['frac_numpy']:9.2f}",
                  flush=True)
        print(flush=True)

    out = RESULTS / f"port_{tag}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for dtype in ("f64", "f32"):
        sub = [r for r in rows if r["dtype"] == dtype]
        sp = sorted(r["speedup"] for r in sub)
        med = sp[len(sp) // 2]
        print(f"  {dtype}: acceleration mediane x{med:.2f}, "
              f"min x{min(sp):.2f}, max x{max(sp):.2f}, "
              f"err max {max(r['err'] for r in sub):.2e}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
