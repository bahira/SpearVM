"""Autotuner GEMM NT : recherche etagee sur des centaines de variantes.

Strategie (coordinate descent, chaque etage garde les meilleurs) :
  1. forme du micro-noyau (MR x NR) sous contrainte de 16 registres YMM
  2. deroulage k et prefetch de C sur les meilleures formes
  3. grille des tuiles KC x MC x NC
  4. raffinement local autour du champion

Chaque variante est **verifiee** (erreur relative vs numpy sur des tailles non
alignees) avant d'etre chronometree ; une variante fausse est rejetee.
Le score est la moyenne geometrique des GFLOPS sur les cas de test, pour ne pas
sur-optimiser une seule taille.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_gemm import Variant, render  # noqa: E402
from harness import (Case, CompileError, bind_gemm, compile_source, gflops,  # noqa: E402
                     make_operands, race, verify)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

CASES = [Case(256, 256, 256), Case(512, 512, 512), Case(1024, 768, 1024)]
TOL = {"f64": 1e-13, "f32": 1e-5}

_seen: set[str] = set()
_rows: list[dict] = []
_start = time.time()


def score(measurements: dict[str, float]) -> float:
    """Moyenne geometrique des GFLOPS (robuste aux ecarts d'echelle)."""
    values = [v for v in measurements.values() if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def evaluate(v: Variant, repeats: int = 2) -> dict | None:
    if v.name in _seen or not v.valid():
        return None
    _seen.add(v.name)
    try:
        lib = compile_source(render(v), v.name, extra=(["-fopenmp"] if v.omp else []))
    except CompileError as exc:
        print(f"  [compile ko] {v.name}: {str(exc)[:120]}", flush=True)
        return None
    call = bind_gemm(lib, v.name, v.dtype)
    err = verify(call, v.dtype)
    if not (err <= TOL[v.dtype]):
        print(f"  [faux] {v.name}: err={err:.2e}", flush=True)
        return None

    per_case: dict[str, float] = {}
    for case in CASES:
        A, B, C = make_operands(case, v.dtype)
        timing = race({"v": lambda: call(A, B, C)}, repeats=repeats, block=2)
        per_case[case.label()] = gflops(case, timing["v"])

    row = {
        "name": v.name, "dtype": v.dtype, "mr": v.mr, "nr": v.nr, "kc": v.kc,
        "mc": v.mc, "nc": v.nc, "unroll": v.unroll, "prefetch": v.prefetch,
        "omp": int(v.omp), "err": err, "score": score(per_case),
        **{f"gf_{key}": round(val, 2) for key, val in per_case.items()},
        "t": round(time.time() - _start, 1),
    }
    _rows.append(row)
    print(f"  {row['score']:7.2f} GF  {v.name}  "
          + " ".join(f"{k.split('_')[1]}={val:.1f}" for k, val in row.items() if k.startswith("gf_")),
          flush=True)
    return row


def stage(title: str, variants: list[Variant], keep: int, repeats: int = 2) -> list[dict]:
    print(f"\n=== {title} ({len(variants)} variantes) ===", flush=True)
    rows = [r for r in (evaluate(v, repeats) for v in variants) if r]
    rows.sort(key=lambda r: -r["score"])
    for row in rows[:keep]:
        print(f"  -> retenu {row['score']:7.2f} GF  {row['name']}", flush=True)
    return rows[:keep]


def variant_from_row(row: dict, **overrides) -> Variant:
    base = dict(dtype=row["dtype"], mr=row["mr"], nr=row["nr"], kc=row["kc"],
                mc=row["mc"], nc=row["nc"], unroll=row["unroll"],
                prefetch=row["prefetch"], omp=bool(row["omp"]))
    base.update(overrides)
    return Variant(**base)


def baseline(dtype: str) -> dict:
    """Reference : noyau SpearVM actuel + numpy BLAS, memes cas, meme harnais."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import spur_math as sm  # noqa: PLC0415

    out = {}
    for case in CASES:
        A, B, C = make_operands(case, dtype)
        timing = race({
            "spear": lambda: sm.matmul_nt(A, B),
            "numpy": lambda: np.dot(A, B.T),
        }, repeats=3, block=3)
        out[case.label()] = {
            "spear_gf": round(gflops(case, timing["spear"]), 2),
            "numpy_gf": round(gflops(case, timing["numpy"]), 2),
        }
        print(f"  {case.label():>16} : spearvm {out[case.label()]['spear_gf']:7.2f} GF | "
              f"numpy {out[case.label()]['numpy_gf']:7.2f} GF", flush=True)
    return out


def search(dtype: str) -> dict:
    lanes = 4 if dtype == "f64" else 8
    print(f"\n########## AUTOTUNE {dtype} ##########", flush=True)
    print("--- baseline ---", flush=True)
    base = baseline(dtype)

    # etage 1 : forme du micro-noyau
    shapes: list[Variant] = []
    for nv in (1, 2, 3, 4):
        nr = nv * lanes
        for mr in (2, 3, 4, 6, 8, 10, 12):
            v = Variant(dtype=dtype, mr=mr, nr=nr)
            if v.valid():
                shapes.append(v)
    best_shapes = stage("etage 1 — forme MR x NR", shapes, keep=4)

    # etage 2 : deroulage + prefetch
    step2: list[Variant] = []
    for row in best_shapes:
        for unroll in (1, 2, 4):
            for prefetch in (0, 4):
                step2.append(variant_from_row(row, unroll=unroll, prefetch=prefetch))
    best_step2 = stage("etage 2 — deroulage k + prefetch C", step2, keep=3)

    # etage 3 : tuiles
    step3: list[Variant] = []
    for row in best_step2[:2]:
        for kc in (64, 128, 192, 256, 384, 512):
            for mc in (64, 128, 256, 512):
                for nc in (256, 512, 1024, 2048):
                    step3.append(variant_from_row(row, kc=kc, mc=mc, nc=nc))
    best_step3 = stage("etage 3 — tuiles KC x MC x NC", step3, keep=4)

    # etage 4 : raffinement local (voisinage du champion, mesures plus longues)
    step4: list[Variant] = []
    for row in best_step3[:2]:
        for dkc in (-64, 0, 64, 128):
            for dmc in (-64, 0, 64, 128):
                kc = row["kc"] + dkc
                mc = row["mc"] + dmc
                if kc < 32 or mc < 32:
                    continue
                step4.append(variant_from_row(row, kc=kc, mc=mc))
    best_step4 = stage("etage 4 — raffinement local", step4, keep=3, repeats=3)

    champions = sorted(best_step3 + best_step4, key=lambda r: -r["score"])
    champion = champions[0]
    print(f"\n>>> champion {dtype} : {champion['name']}  {champion['score']:.2f} GF", flush=True)
    return {"dtype": dtype, "baseline": base, "champion": champion}


def main() -> None:
    summary = {"cases": [c.label() for c in CASES], "runs": {}}
    for dtype in ("f64", "f32"):
        summary["runs"][dtype] = search(dtype)

    with (RESULTS / "gemm_sweep.csv").open("w", newline="") as fh:
        if _rows:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in _rows for k in r}))
            writer.writeheader()
            writer.writerows(_rows)
    (RESULTS / "gemm_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(_rows)} variantes mesurees en {time.time() - _start:.0f} s", flush=True)
    for dtype, run in summary["runs"].items():
        champ = run["champion"]
        print(f"  {dtype}: {champ['name']} -> {champ['score']:.1f} GF", flush=True)


if __name__ == "__main__":
    main()
