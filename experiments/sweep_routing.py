"""Etage 7 : calibration de la regle d'aiguillage, mesuree dans le binaire livre.

Les trois noyaux (`legacy`, `pack`, `dot`) sont exportes par la .so : on peut
donc les mettre en concurrence *dans les conditions reelles de production*
(memes allocations, meme compilateur, meme .so) sur une grille de formes, et
lire directement quelle regle minimise le regret.

On evalue ensuite plusieurs regles candidates sur cette grille et on garde
celle qui maximise le pire cas (min du rapport gagnant/legacy) : l'objectif
n'est pas seulement d'aller vite en moyenne, c'est de **ne jamais regresser**.

Sortie : results/routing.csv + results/routing.json
"""

from __future__ import annotations

import csv
import ctypes
import json
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

# grille centree sur la zone de bascule (petites dimensions) + quelques grandes
GRID: list[Case] = []
for m in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512):
    for n in (1, 4, 10, 16, 32, 48, 64, 96, 128, 256, 512):
        k = 512 if (m * n) <= 4096 else 256
        GRID.append(Case(m, k, n))
GRID += [Case(768, 768, 768), Case(1024, 768, 1024), Case(64, 3072, 768),
         Case(512, 16, 512), Case(2048, 128, 128), Case(32, 768, 3072)]

RULES = {
    "toujours_pack": lambda m, k, n: "pack",
    "toujours_dot": lambda m, k, n: "dot",
    "min<=32": lambda m, k, n: "dot" if min(m, n) <= 32 else "pack",
    "min<=32_ou_64x64": lambda m, k, n: (
        "dot" if (min(m, n) <= 32 or (m <= 64 and n <= 64)) else "pack"),
    "v3": lambda m, k, n: (
        "legacy" if (min(m, n) <= 2 or (m <= 8 and n * k <= 1 << 20) or n <= 12)
        else "dot" if min(m, n) <= 32 or (m <= 64 and n <= 64)
        else "pack"),
    "v4": lambda m, k, n: (
        "legacy" if (n <= 12 or (m <= 8 and n <= 64))
        else "dot" if (min(m, n) <= 32 or (m <= 64 and n <= 64))
        else "pack"),
}


def main() -> None:
    rows: list[dict] = []
    for dtype, np_t, ptr in (("f64", np.float64, _PD), ("f32", np.float32, _PF)):
        fns = {k: getattr(_DLL, v) for k, v in KERNELS[dtype].items()}
        print(f"\n### {dtype} — {len(GRID)} formes", flush=True)
        for case in GRID:
            rng = np.random.default_rng(0)
            A = np.ascontiguousarray(rng.standard_normal((case.m, case.k)), dtype=np_t)
            B = np.ascontiguousarray(rng.standard_normal((case.n, case.k)), dtype=np_t)
            C = np.zeros((case.m, case.n), dtype=np_t)
            pa, pb, pc = (A.ctypes.data_as(ptr), B.ctypes.data_as(ptr),
                          C.ctypes.data_as(ptr))
            args = (case.m, case.k, case.n)
            timing = race({name: (lambda f=fn: f(pa, pb, pc, *args))
                           for name, fn in fns.items()}, repeats=3, block=2)
            gf = {k: gflops(case, v) for k, v in timing.items()}
            rows.append({"dtype": dtype, "m": case.m, "k": case.k, "n": case.n,
                         **{f"gf_{k}": round(v, 3) for k, v in gf.items()},
                         "best": max(gf, key=gf.get)})
        print(f"  mesure ok ({len([r for r in rows if r['dtype'] == dtype])} formes)",
              flush=True)

    report: dict = {}
    for dtype in ("f64", "f32"):
        sub = [r for r in rows if r["dtype"] == dtype]
        report[dtype] = {}
        for label, rule in RULES.items():
            ratios = []
            for r in sub:
                choice = rule(r["m"], r["k"], r["n"])
                ratios.append(r[f"gf_{choice}"] / r["gf_legacy"])
            ratios.sort()
            report[dtype][label] = {
                "pire": round(ratios[0], 3),
                "p10": round(ratios[len(ratios) // 10], 3),
                "mediane": round(ratios[len(ratios) // 2], 3),
                "moyenne": round(sum(ratios) / len(ratios), 3),
                "meilleur": round(ratios[-1], 3),
                "regressions": sum(1 for x in ratios if x < 0.98),
            }
        print(f"\n=== {dtype} : regles candidates (rapport vs legacy) ===")
        print(f"{'regle':>18} {'pire':>7} {'p10':>7} {'mediane':>8} {'moyenne':>8} "
              f"{'regr<0.98':>10}")
        for label, st in report[dtype].items():
            print(f"{label:>18} {st['pire']:7.2f} {st['p10']:7.2f} {st['mediane']:8.2f} "
                  f"{st['moyenne']:8.2f} {st['regressions']:10d}")

    with (RESULTS / "routing.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "routing.json").write_text(json.dumps(report, indent=2))
    print("\n-> results/routing.csv")


if __name__ == "__main__":
    main()
