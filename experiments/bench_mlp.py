"""Etage 9 : impact bout-en-bout — une iteration d'entrainement MLP.

Les GFLOPS d'un noyau ne sont pas un livrable ; ce qui compte est le temps
d'une iteration reelle. On entraine un MLP 784-256-128-10 (batch 128) avec
l'API SpearVM (matmul_nt_gelu en avant, matmul_nt + gelu_backward en arriere)
et on chronometre l'iteration complete.

Le noyau est choisi par variable d'environnement (SPUR_MM_LEGACY=1 -> ancien),
lue une seule fois au premier appel : la comparaison se fait donc en lancant
deux processus (`python bench_mlp.py` orchestre les deux).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

LAYERS = [784, 256, 128, 10]
BATCH = 128
STEPS = 30


def run_once() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import spur_math as sm  # noqa: PLC0415

    dtype = np.float32 if os.environ.get("SPUR_BENCH_F32") == "1" else np.float64
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.standard_normal((BATCH, LAYERS[0])), dtype=dtype)
    Y = np.ascontiguousarray(rng.standard_normal((BATCH, LAYERS[-1])), dtype=dtype)
    W = [np.ascontiguousarray(rng.standard_normal((LAYERS[i + 1], LAYERS[i]))
                              / np.sqrt(LAYERS[i]), dtype=dtype)
         for i in range(len(LAYERS) - 1)]
    b = [np.zeros(LAYERS[i + 1], dtype=dtype) for i in range(len(LAYERS) - 1)]
    lr = dtype(0.01)

    def step():
        acts = [X]
        pre = []
        h = X
        for li, (w, bb) in enumerate(zip(W, b)):
            if li < len(W) - 1:
                pre.append(sm.matmul_nt(h, w) + bb)
                h = sm.matmul_nt_gelu(acts[-1], w, bb)
            else:
                pre.append(sm.matmul_nt(h, w) + bb)
                h = pre[-1]
            acts.append(h)
        d = (2.0 / BATCH) * (acts[-1] - Y)
        for li in range(len(W) - 1, -1, -1):
            if li < len(W) - 1:
                d = sm.gelu_backward(d, pre[li])
            gw = sm.matmul_nt(np.ascontiguousarray(d.T),
                              np.ascontiguousarray(acts[li].T))
            d = sm.matmul_nt(d, np.ascontiguousarray(W[li].T))
            W[li] -= lr * gw
        return float(np.abs(d).mean())

    step()
    best = float("inf")
    tot = 0.0
    for _ in range(STEPS):
        t0 = time.perf_counter()
        step()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        tot += dt
    return {"ms_min": best * 1e3, "ms_moy": tot / STEPS * 1e3,
            "iters_par_s": 1.0 / best}


def main() -> None:
    if os.environ.get("SPUR_BENCH_CHILD") == "1":
        print(json.dumps(run_once()))
        return

    report: dict = {}
    for dtype_tag, f32 in (("f64", "0"), ("f32", "1")):
        report[dtype_tag] = {}
        for label, legacy in (("legacy", "1"), ("v2", "0")):
            env = dict(os.environ, SPUR_BENCH_CHILD="1", SPUR_MM_LEGACY=legacy,
                       SPUR_BENCH_F32=f32)
            out = subprocess.run([sys.executable, __file__], env=env,
                                 capture_output=True, text=True, check=True)
            report[dtype_tag][label] = json.loads(out.stdout.strip().splitlines()[-1])
        old = report[dtype_tag]["legacy"]["ms_min"]
        new = report[dtype_tag]["v2"]["ms_min"]
        report[dtype_tag]["gain"] = round(old / new, 3)
        print(f"  {dtype_tag}: iteration MLP {LAYERS} batch {BATCH} — "
              f"legacy {old:.2f} ms | v2 {new:.2f} ms -> x{old / new:.2f}")

    report["config"] = {"layers": LAYERS, "batch": BATCH, "steps": STEPS,
                        "threads": os.environ.get("OMP_NUM_THREADS", "?")}
    tag = os.environ.get("OMP_NUM_THREADS", "x")
    out = RESULTS / f"mlp_e2e_t{tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"-> {out.name}")


if __name__ == "__main__":
    main()
