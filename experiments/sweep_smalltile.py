"""Etage 7b : ou bascule la branche "petite tuile" en fonction de k ?

La calibration d'ensemble (sweep_routing.py) a ete faite a k fixe ; or pour une
tuile 64x64 le noyau dot ne gagne que si k est assez grand pour amortir les
MR*NR reductions horizontales finales. On mesure donc explicitement le plan
(m,n) petit x k variable, et on en tire le seuil min_k par precision.

Sortie : results/smalltile.csv
"""

from __future__ import annotations

import csv
import ctypes
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
KERNELS = {
    "f64": {"legacy": "spur_matmul_nt_legacy", "pack": "spur_gemm_pack_f64",
            "dot": "spur_gemm_dot_f64"},
    "f32": {"legacy": "spur_matmul_nt_f32_legacy", "pack": "spur_gemm_pack_f32",
            "dot": "spur_gemm_dot_f32"},
}
for _dt, _ks in KERNELS.items():
    _p = _PD if _dt == "f64" else _PF
    for _sym in _ks.values():
        _fn = getattr(_DLL, _sym)
        _fn.argtypes = [_p, _p, _p] + [ctypes.c_longlong] * 3
        _fn.restype = None

MN = [(32, 64), (48, 48), (48, 64), (64, 32), (64, 48), (64, 64)]
KS = [16, 32, 48, 64, 96, 128, 192, 256, 512, 1024]


def main() -> None:
    rows: list[dict] = []
    for dtype, np_t, ptr in (("f64", np.float64, _PD), ("f32", np.float32, _PF)):
        fns = {k: getattr(_DLL, v) for k, v in KERNELS[dtype].items()}
        print(f"\n### {dtype} — ratio noyau/legacy (>1 = gain)")
        print(f"{'k':>6} " + " ".join(f"{m}x{n:<7}" for m, n in MN))
        per_k: dict[int, list[float]] = {}
        for k in KS:
            cells = []
            for (m, n) in MN:
                case = Case(m, k, n)
                rng = np.random.default_rng(0)
                A = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np_t)
                B = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=np_t)
                C = np.zeros((m, n), dtype=np_t)
                pa, pb, pc = (A.ctypes.data_as(ptr), B.ctypes.data_as(ptr),
                              C.ctypes.data_as(ptr))
                t = race({name: (lambda f=fn: f(pa, pb, pc, m, k, n))
                          for name, fn in fns.items()}, repeats=8, block=4)
                gf = {name: gflops(case, v) for name, v in t.items()}
                rows.append({"dtype": dtype, "m": m, "k": k, "n": n,
                             **{f"gf_{a}": round(b, 3) for a, b in gf.items()}})
                cells.append(gf["dot"] / gf["legacy"])
            per_k[k] = cells
            print(f"{k:>6} " + " ".join(f"{c:8.2f}" for c in cells))
        print("  (colonne = forme, valeur = dot / legacy)")
        for k in KS:
            worst = min(per_k[k])
            print(f"   k={k:<5} pire dot/legacy = {worst:.2f}"
                  + ("   <- dot sur" if worst >= 1.0 else ""))

    with (RESULTS / "smalltile.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n-> results/smalltile.csv")


if __name__ == "__main__":
    main()
