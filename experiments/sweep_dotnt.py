"""Autotuner famille 2 : noyau "dot-block" NT sans packing.

Meme protocole que sweep_gemm.py (verification avant chronometrage, moyenne
geometrique sur 3 cas), mais l'espace de recherche est different :

  1. forme MR x NR (contrainte 16 registres YMM) + hoist de B
  2. blocage NC (lignes de B gardees chaudes en L2) et MC
  3. blocage KC + prefetch
  4. raffinement local

Le resultat est ecrit dans results/dotnt_sweep.csv / dotnt_summary.json et se
compare directement au champion famille 1 (packing broadcast).
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_dotnt import DotVariant, render  # noqa: E402
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
    values = [v for v in measurements.values() if v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def evaluate(v: DotVariant, repeats: int = 2) -> dict | None:
    if v.name in _seen or not v.valid():
        return None
    _seen.add(v.name)
    try:
        lib = compile_source(render(v), v.name)
    except CompileError as exc:
        print(f"  [compile ko] {v.name}: {str(exc)[:160]}", flush=True)
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
        "mc": v.mc, "nc": v.nc, "hoist_b": int(v.hoist_b), "prefetch": v.prefetch,
        "err": err, "score": score(per_case),
        **{f"gf_{key}": round(val, 2) for key, val in per_case.items()},
        "t": round(time.time() - _start, 1),
    }
    _rows.append(row)
    print(f"  {row['score']:7.2f} GF  {v.name}  "
          + " ".join(f"{k[3:]}={val:.1f}" for k, val in row.items() if k.startswith("gf_")),
          flush=True)
    return row


def stage(title: str, variants: list[DotVariant], keep: int, repeats: int = 2) -> list[dict]:
    print(f"\n=== {title} ({len(variants)} variantes) ===", flush=True)
    rows = [r for r in (evaluate(v, repeats) for v in variants) if r]
    rows.sort(key=lambda r: -r["score"])
    for row in rows[:keep]:
        print(f"  -> retenu {row['score']:7.2f} GF  {row['name']}", flush=True)
    return rows[:keep]


def variant_from_row(row: dict, **overrides) -> DotVariant:
    base = dict(dtype=row["dtype"], mr=row["mr"], nr=row["nr"], kc=row["kc"],
                nc=row["nc"], mc=row["mc"], hoist_b=bool(row["hoist_b"]),
                prefetch=row["prefetch"])
    base.update(overrides)
    return DotVariant(**base)


def search(dtype: str) -> dict:
    print(f"\n########## DOT-BLOCK {dtype} ##########", flush=True)

    # etage 1 : forme MR x NR + hoist
    shapes: list[DotVariant] = []
    for mr in (1, 2, 3, 4, 5, 6):
        for nr in (1, 2, 3, 4, 5, 6):
            for hoist in (False, True):
                v = DotVariant(dtype=dtype, mr=mr, nr=nr, hoist_b=hoist,
                               kc=0, nc=64, mc=0)
                if v.valid():
                    shapes.append(v)
    best_shapes = stage("etage 1 — forme MR x NR (+hoist B)", shapes, keep=4)
    if not best_shapes:
        return {"dtype": dtype, "champion": None}

    # etage 2 : blocage NC / MC
    step2: list[DotVariant] = []
    for row in best_shapes:
        for nc in (0, 32, 64, 128, 256, 512):
            for mc in (0, 64, 128, 256):
                step2.append(variant_from_row(row, nc=nc, mc=mc))
    best_step2 = stage("etage 2 — blocage NC x MC", step2, keep=3)

    # etage 3 : blocage KC + prefetch
    step3: list[DotVariant] = []
    for row in best_step2:
        for kc in (0, 128, 256, 512, 1024):
            for prefetch in (0, 1, 2):
                step3.append(variant_from_row(row, kc=kc, prefetch=prefetch))
    best_step3 = stage("etage 3 — blocage KC + prefetch", step3, keep=4)

    # etage 4 : raffinement local
    step4: list[DotVariant] = []
    for row in best_step3[:2]:
        for dnc in (-32, -16, 0, 16, 32, 64):
            nc = row["nc"] + dnc
            if nc < 0:
                continue
            for dmc in (0, 32, 64):
                mc = row["mc"] + dmc
                step4.append(variant_from_row(row, nc=nc, mc=mc))
    best_step4 = stage("etage 4 — raffinement local", step4, keep=3, repeats=3)

    champions = sorted(best_step2 + best_step3 + best_step4, key=lambda r: -r["score"])
    champion = champions[0]
    print(f"\n>>> champion dot-block {dtype} : {champion['name']}  {champion['score']:.2f} GF",
          flush=True)
    return {"dtype": dtype, "champion": champion}


def main() -> None:
    summary = {"cases": [c.label() for c in CASES], "runs": {}}
    for dtype in ("f64", "f32"):
        summary["runs"][dtype] = search(dtype)

    with (RESULTS / "dotnt_sweep.csv").open("w", newline="") as fh:
        if _rows:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for r in _rows for k in r}))
            writer.writeheader()
            writer.writerows(_rows)
    (RESULTS / "dotnt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(_rows)} variantes mesurees en {time.time() - _start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
