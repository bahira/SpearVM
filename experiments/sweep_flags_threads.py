"""Etage 5 : options de compilation + montee en threads sur les champions.

Deux questions distinctes :
  1. le code genere gagne-t-il quelque chose a -march=native, -funroll-loops,
     -fno-signed-zeros... ? (mesure, pas croyance)
  2. le pilote OpenMP (parallelisation sur les blocs MC) tient-il l'echelle sur
     les 2 vCPU de la machine ?

Sortie : results/flags_threads.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_gemm import Variant, render  # noqa: E402
from harness import (Case, CompileError, bind_gemm, compile_source, gflops,  # noqa: E402
                     make_operands, race, verify)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

CASES = [Case(256, 256, 256), Case(512, 512, 512), Case(1024, 768, 1024)]
TOL = {"f64": 1e-13, "f32": 1e-5}

CHAMPS = {
    "f64": Variant(dtype="f64", mr=4, nr=12, kc=576, mc=64, nc=2048, prefetch=4),
    "f32": Variant(dtype="f32", mr=4, nr=24, kc=384, mc=192, nc=2048, prefetch=4),
}

FLAG_SETS = {
    "base": [],
    "native": ["-march=native"],
    "unroll": ["-funroll-loops"],
    "native+unroll": ["-march=native", "-funroll-loops"],
    "fast-nosigned": ["-fno-signed-zeros", "-fno-trapping-math", "-fassociative-math",
                      "-fno-math-errno", "-freciprocal-math"],
    "native+O3clone": ["-march=native", "-fipa-cp-clone"],
}


def measure(v: Variant, extra: list[str], repeats: int = 3, tag: str = "",
            threads: int | None = None) -> dict | None:
    name = v.name + ("_" + tag if tag else "")
    try:
        lib = compile_source(render(v), name,
                             extra=list(extra) + (["-fopenmp"] if v.omp else []))
    except CompileError as exc:
        print(f"  [compile ko] {name} {extra}: {str(exc)[:150]}", flush=True)
        return None
    if threads is not None:
        # OMP_NUM_THREADS n'est lu qu'a l'init de libgomp : on force par API.
        try:
            lib.omp_set_num_threads(int(threads))
        except AttributeError:
            print("  [omp] omp_set_num_threads introuvable", flush=True)
    call = bind_gemm(lib, v.name, v.dtype)
    err = verify(call, v.dtype)
    if not (err <= TOL[v.dtype]):
        print(f"  [faux] {name} {extra}: err={err:.2e}", flush=True)
        return None
    out = {}
    for case in CASES:
        A, B, C = make_operands(case, v.dtype)
        t = race({"v": lambda: call(A, B, C)}, repeats=repeats, block=2)
        out[case.label()] = round(gflops(case, t["v"]), 2)
    return out


def main() -> None:
    report: dict = {"cases": [c.label() for c in CASES], "flags": {}, "threads": {}}

    print("########## etage 5a — options de compilation ##########", flush=True)
    for dtype, champ in CHAMPS.items():
        report["flags"][dtype] = {}
        for label, extra in FLAG_SETS.items():
            got = measure(champ, extra, tag=label.replace("+", "_").replace("=", ""))
            if got:
                report["flags"][dtype][label] = got
                print(f"  {dtype} {label:16s} " +
                      " ".join(f"{k}={v:.1f}" for k, v in got.items()), flush=True)

    print("\n########## etage 5b — OpenMP (1 vs 2 threads) ##########", flush=True)
    for dtype, champ in CHAMPS.items():
        report["threads"][dtype] = {}
        mt = Variant(**{**champ.__dict__, "omp": True})
        for threads in (1, 2):
            got = measure(mt, [], tag=f"omp{threads}", threads=threads)
            if got:
                report["threads"][dtype][f"omp{threads}"] = got
                print(f"  {dtype} omp x{threads}   " +
                      " ".join(f"{k}={v:.1f}" for k, v in got.items()), flush=True)
        # variante MC plus petit : plus de blocs -> meilleur equilibrage a 2 threads
        for mc in (32, 64, 128):
            mt2 = Variant(**{**mt.__dict__, "mc": mc})
            got = measure(mt2, [], tag=f"omp2_mc{mc}", threads=2)
            if got:
                report["threads"][dtype][f"omp2_mc{mc}"] = got
                print(f"  {dtype} omp2 mc={mc:<4d}" +
                      " ".join(f"{k}={v:.1f}" for k, v in got.items()), flush=True)

    (RESULTS / "flags_threads.json").write_text(json.dumps(report, indent=2))
    print("\n-> results/flags_threads.json", flush=True)


if __name__ == "__main__":
    main()
