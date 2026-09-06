"""Etage 6 : matrice de formes — quel noyau gagne, et ou ?

On compare sur une vingtaine de formes (carrees, minces, plates, k court,
tailles typiques d'un MLP) :
  * `spear`   : le noyau actuellement livre (spur_math.matmul_nt)
  * `pack`    : champion famille 1 (micro-noyau broadcast + packing)
  * `dot`     : champion famille 2 (dot-block, zero packing)
  * `numpy`   : OpenBLAS, la reference haute

Objectif : etablir la **regle d'aiguillage** (dispatch) a porter dans le C, au
lieu de choisir un noyau unique a l'aveugle.

Sortie : results/shapes.csv + results/shapes.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gen_dotnt import DotVariant  # noqa: E402
from gen_dotnt import render as render_dot  # noqa: E402
from gen_gemm import Variant  # noqa: E402
from gen_gemm import render as render_pack  # noqa: E402
from harness import (Case, bind_gemm, compile_source, gflops, make_operands,  # noqa: E402
                     race, verify)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

TOL = {"f64": 1e-13, "f32": 1e-5}

PACK = {
    "f64": Variant(dtype="f64", mr=4, nr=12, kc=576, mc=64, nc=2048, prefetch=4),
    "f32": Variant(dtype="f32", mr=4, nr=24, kc=384, mc=192, nc=2048, prefetch=4),
}
DOT = {
    "f64": DotVariant(dtype="f64", mr=3, nr=4, kc=1024, nc=96, mc=0, prefetch=1),
    "f32": DotVariant(dtype="f32", mr=3, nr=4, kc=1024, nc=256, mc=0, prefetch=2),
}

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
    import spur_math as sm  # noqa: PLC0415

    rows: list[dict] = []
    for dtype in ("f64", "f32"):
        dv = DOT[dtype]
        pv = PACK[dtype]
        pack_lib = compile_source(render_pack(pv), pv.name)
        dot_lib = compile_source(render_dot(dv), dv.name)
        pack = bind_gemm(pack_lib, pv.name, dtype)
        dot = bind_gemm(dot_lib, dv.name, dtype)
        e_pack, e_dot = verify(pack, dtype), verify(dot, dtype)
        print(f"[{dtype}] verif  pack err={e_pack:.2e}  dot err={e_dot:.2e}", flush=True)
        assert e_pack <= TOL[dtype] and e_dot <= TOL[dtype]

        print(f"\n{'forme':>18} {'spear':>8} {'pack':>8} {'dot':>8} {'numpy':>8}   "
              f"{'best':>5}  x_vs_spear", flush=True)
        for case in SHAPES:
            A, B, C = make_operands(case, dtype)
            timing = race({
                "spear": lambda: sm.matmul_nt(A, B),
                "pack": lambda: pack(A, B, C),
                "dot": lambda: dot(A, B, C),
                "numpy": lambda: np.dot(A, B.T),
            }, repeats=3, block=2)
            gf = {k: gflops(case, v) for k, v in timing.items()}
            best = max(("pack", "dot"), key=lambda k: gf[k])
            row = {"dtype": dtype, "shape": case.label(), "m": case.m, "k": case.k,
                   "n": case.n, **{f"gf_{k}": round(v, 2) for k, v in gf.items()},
                   "best_new": best,
                   "speedup_vs_spear": round(gf[best] / gf["spear"], 3),
                   "frac_of_numpy": round(gf[best] / gf["numpy"], 3)}
            rows.append(row)
            print(f"{case.label():>18} {gf['spear']:8.1f} {gf['pack']:8.1f} "
                  f"{gf['dot']:8.1f} {gf['numpy']:8.1f}   {best:>5}  "
                  f"x{row['speedup_vs_spear']:.2f}", flush=True)

    with (RESULTS / "shapes.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "shapes.json").write_text(json.dumps(rows, indent=2))
    print("\n-> results/shapes.csv", flush=True)


if __name__ == "__main__":
    main()
