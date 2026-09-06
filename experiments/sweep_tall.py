"""Etage 11 : formes "hautes et minces" — le regime du champ implicite.

Un MLP evalue par point donne des GEMM tres inhabituels : m = grid^3 (dizaines
de milliers de lignes), k = 14 a 128, n = 1 a 128. Ce regime n'etait pas couvert
par la grille de calibration initiale (m <= 512) : on le mesure ici, noyau par
noyau, pour corriger l'aiguillage.

On mesure aussi l'effet du **decoupage en tuiles de lignes** (chunking) : au
lieu d'evaluer les 64 000 points d'un coup (activations intermediaires de
16 Mo qui font des aller-retours en RAM), on traite des paquets de lignes qui
tiennent en cache. C'est ce que fait tout moteur de rendu implicite serieux.

Sortie : results/tall.csv
"""

from __future__ import annotations

import csv
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import Case, gflops, race  # noqa: E402

import spur_math as sm  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

_DLL = ctypes.CDLL(sm._dll_path)
_PF = ctypes.POINTER(ctypes.c_float)
KERNELS = {"legacy": "spur_matmul_nt_f32_legacy", "pack": "spur_gemm_pack_f32",
           "dot": "spur_gemm_dot_f32"}
for _sym in KERNELS.values():
    _fn = getattr(_DLL, _sym)
    _fn.argtypes = [_PF, _PF, _PF] + [ctypes.c_longlong] * 3
    _fn.restype = None

MS = [8192, 32768, 64000]
KS = [8, 14, 32, 64, 128]
NS = [1, 16, 32, 64, 96, 128]


def bench_shapes() -> list[dict]:
    fns = {k: getattr(_DLL, v) for k, v in KERNELS.items()}
    rows: list[dict] = []
    print(f"{'m':>7}{'k':>5}{'n':>5} {'legacy':>8}{'pack':>8}{'dot':>8}  {'best':>7} "
          f"{'gain':>6}")
    for m in MS:
        for k in KS:
            for n in NS:
                case = Case(m, k, n)
                rng = np.random.default_rng(0)
                A = np.ascontiguousarray(rng.standard_normal((m, k)), dtype=np.float32)
                B = np.ascontiguousarray(rng.standard_normal((n, k)), dtype=np.float32)
                C = np.zeros((m, n), dtype=np.float32)
                pa, pb, pc = (A.ctypes.data_as(_PF), B.ctypes.data_as(_PF),
                              C.ctypes.data_as(_PF))
                t = race({name: (lambda f=fn: f(pa, pb, pc, m, k, n))
                          for name, fn in fns.items()}, repeats=5, block=3)
                gf = {name: gflops(case, v) for name, v in t.items()}
                best = max(gf, key=gf.get)
                rows.append({"m": m, "k": k, "n": n,
                             **{f"gf_{a}": round(b, 2) for a, b in gf.items()},
                             "best": best,
                             "gain": round(gf[best] / gf["legacy"], 3)})
                print(f"{m:>7}{k:>5}{n:>5} {gf['legacy']:8.1f}{gf['pack']:8.1f}"
                      f"{gf['dot']:8.1f}  {best:>7} {gf[best]/gf['legacy']:6.2f}")
    return rows


def bench_chunking() -> None:
    """MLP 14 -> 64 -> 64 -> 1 sur 64 000 points, entier vs par paquets."""
    n_pts, h, k_in = 64000, 64, 14
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.standard_normal((n_pts, k_in)), dtype=np.float32)
    W1 = np.ascontiguousarray(rng.standard_normal((h, k_in)), dtype=np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = np.ascontiguousarray(rng.standard_normal((h, h)), dtype=np.float32)
    b2 = np.zeros(h, dtype=np.float32)
    W3 = np.ascontiguousarray(rng.standard_normal((1, h)), dtype=np.float32)
    flops = 2.0 * n_pts * (k_in * h + h * h + h)

    def whole():
        h1 = sm.matmul_nt_gelu(X, W1, b1)
        h2 = sm.matmul_nt_gelu(h1, W2, b2)
        return sm.matmul_nt(h2, W3)

    def chunked(size: int):
        def run():
            out = np.empty((n_pts, 1), dtype=np.float32)
            for s in range(0, n_pts, size):
                blk = X[s:s + size]
                h1 = sm.matmul_nt_gelu(blk, W1, b1)
                h2 = sm.matmul_nt_gelu(h1, W2, b2)
                out[s:s + size] = sm.matmul_nt(h2, W3)
            return out
        return run

    ref = whole()
    print("\n--- MLP 14->64->64->1 sur 64 000 points ---")
    variants = {"entier": whole}
    for size in (1024, 2048, 4096, 8192, 16384):
        variants[f"paquets de {size}"] = chunked(size)
    for label, fn in variants.items():
        got = fn()
        err = float(np.abs(got - ref).max())
        fn(); best = float("inf")
        for _ in range(6):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        print(f"  {label:>18} : {best*1e3:7.2f} ms   {flops/1e9/best:6.1f} GF   "
              f"ecart {err:.1e}")


def main() -> None:
    rows = bench_shapes()
    with (RESULTS / "tall.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    bench_chunking()
    print("\n-> results/tall.csv")


if __name__ == "__main__":
    main()
