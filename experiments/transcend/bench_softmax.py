"""Banc : exp AVX2 et softmax par ligne, contre libm et contre numpy.

On verifie d'abord (erreur relative contre reference float64 sur les memes
valeurs arrondies), on chronometre ensuite. Les strategies de softmax sont
mises en concurrence par blocs alternes, comme les GEMM.

Deux precautions d'honnetete :
  * l'erreur du noyau est comparee a celle de **numpy en float32**, pas
    seulement a la reference float64 : la question utile est « fait-on pire que
    la bibliotheque ? », pas « perd-on des bits par rapport au reel » ;
  * la baseline du softmax causal est la version numpy **vectorisee** (matrice
    complete avec -inf), pas une boucle Python ligne a ligne.

`harness.race` renvoie des MILLISECONDES.

Sortie : results/softmax_bench.csv + results/softmax_bench.json
"""

from __future__ import annotations

import csv
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import gen_kernels  # noqa: E402
from fit_exp import LN2, remez_relative  # noqa: E402
from harness import compile_source, race  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

_F = ctypes.POINTER(ctypes.c_float)
_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)


def build(deg32: int = 5, deg64: int = 10):
    c32, _ = remez_relative(deg32, -LN2 / 2, LN2 / 2)
    c64, _ = remez_relative(deg64, -LN2 / 2, LN2 / 2)
    src = gen_kernels.render([float(v) for v in c32], [float(v) for v in c64])
    (RESULTS / "spur_softmax_generated.c").write_text(src)
    lib = compile_source(src, f"softmax_d{deg32}_{deg64}", extra=["-fopenmp"])
    ll = ctypes.c_longlong
    for nm, sig in [("spur_exp_f32", [_F, _F, ll]),
                    ("spur_exp_f64", [_D, _D, ll]),
                    ("spur_softmax_3pass_f32", [_F, _F, ll, ll]),
                    ("spur_softmax_online_f32", [_F, _F, ll, ll]),
                    ("spur_softmax_masked_f32", [_F, _F, ll, ll, _I])]:
        fn = getattr(lib, nm)
        fn.argtypes = sig
        fn.restype = None
    return lib, c32, c64


def np_softmax(x):
    m = x.max(axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=1, keepdims=True)


def verify_exp(lib):
    print("\n### 1. exp : precision contre la reference float64")
    out = {}
    for tag, dtype, ptr, fn in (("f32", np.float32, _F, lib.spur_exp_f32),
                                ("f64", np.float64, _D, lib.spur_exp_f64)):
        lo = -87.0 if tag == "f32" else -700.0
        x = np.linspace(lo, 0.0, 500001).astype(dtype)
        y = np.empty_like(x)
        fn(x.ctypes.data_as(ptr), y.ctypes.data_as(ptr), x.size)
        ref = np.exp(x.astype(np.float64))
        rel = np.abs(y.astype(np.float64) - ref) / ref
        rel_np = np.abs(np.exp(x).astype(np.float64) - ref) / ref
        eps = float(np.finfo(dtype).eps)
        out[tag] = {"linf_rel": float(rel.max()), "ulp": float(rel.max() / eps),
                    "linf_numpy": float(rel_np.max()),
                    "ulp_numpy": float(rel_np.max() / eps)}
        print(f"  {tag} : spur {rel.max():.3e} = {rel.max()/eps:5.2f} ulp   |   "
              f"numpy {rel_np.max():.3e} = {rel_np.max()/eps:.2f} ulp")
    return out


def bench_exp(lib):
    print("\n### 2. exp : debit contre numpy (libm vectorise)")
    rows = []
    for n in (1 << 14, 1 << 18, 1 << 22):
        for tag, dtype, ptr, fn in (("f32", np.float32, _F, lib.spur_exp_f32),
                                    ("f64", np.float64, _D, lib.spur_exp_f64)):
            x = np.random.default_rng(0).uniform(-20, 0, n).astype(dtype)
            y = np.empty_like(x)
            px, py = x.ctypes.data_as(ptr), y.ctypes.data_as(ptr)
            t = race({"spur": lambda: fn(px, py, n),
                      "numpy": lambda: np.exp(x)}, repeats=6, block=3)
            rows.append({"noyau": "exp", "dtype": tag, "n": n,
                         "spur_ms": round(t["spur"], 5),
                         "numpy_ms": round(t["numpy"], 5),
                         "gain": round(t["numpy"] / t["spur"], 3),
                         "M_elem_par_s": round(n / t["spur"] / 1e3, 1)})
            r = rows[-1]
            print(f"  {tag} n=2^{n.bit_length()-1:<2d} : spur {t['spur']:8.4f} ms  "
                  f"numpy {t['numpy']:8.4f} ms  -> x{r['gain']:.2f}  "
                  f"({r['M_elem_par_s']:.0f} M elem/s)")
    return rows


def bench_softmax(lib):
    print("\n### 3. softmax par ligne : deux strategies contre numpy")
    rows = []
    for rows_n, cols in [(64, 128), (256, 512), (1024, 1024), (4096, 512), (256, 8192)]:
        rng = np.random.default_rng(1)
        X = (rng.standard_normal((rows_n, cols)) * 4).astype(np.float32)
        Y1, Y2 = np.empty_like(X), np.empty_like(X)
        px = X.ctypes.data_as(_F)
        p1, p2 = Y1.ctypes.data_as(_F), Y2.ctypes.data_as(_F)
        ref = np_softmax(X.astype(np.float64))

        lib.spur_softmax_3pass_f32(px, p1, rows_n, cols)
        lib.spur_softmax_online_f32(px, p2, rows_n, cols)
        e1 = float(np.abs(Y1.astype(np.float64) - ref).max())
        e2 = float(np.abs(Y2.astype(np.float64) - ref).max())
        e_np = float(np.abs(np_softmax(X).astype(np.float64) - ref).max())

        t = race({"3pass": lambda: lib.spur_softmax_3pass_f32(px, p1, rows_n, cols),
                  "online": lambda: lib.spur_softmax_online_f32(px, p2, rows_n, cols),
                  "numpy": lambda: np_softmax(X)}, repeats=6, block=3)
        best = min(("3pass", "online"), key=lambda k: t[k])
        gb = rows_n * cols * 4 * 3 / 1e9
        rows.append({"noyau": "softmax", "rows": rows_n, "cols": cols,
                     "ms_3pass": round(t["3pass"], 5),
                     "ms_online": round(t["online"], 5),
                     "ms_numpy": round(t["numpy"], 5),
                     "gain_vs_numpy": round(t["numpy"] / t[best], 3),
                     "meilleur": best, "err_spur": e1, "err_numpy_f32": e_np,
                     "GB_par_s": round(gb / (t[best] / 1e3), 1)})
        r = rows[-1]
        print(f"  {rows_n:>5}x{cols:<5} 3pass {t['3pass']:8.4f}  online {t['online']:8.4f}"
              f"  numpy {t['numpy']:8.4f} ms  -> x{r['gain_vs_numpy']:.2f} ({best}, "
              f"{r['GB_par_s']:.0f} GB/s)  err {e1:.1e} vs numpy f32 {e_np:.1e}")
    return rows


def bench_masked(lib):
    print("\n### 4. softmax causal (longueurs variables) — la forme utile a l'attention")
    rows_n, cols = 2048, 1024
    rng = np.random.default_rng(2)
    X = (rng.standard_normal((rows_n, cols)) * 4).astype(np.float32)
    lens = np.minimum(np.arange(1, rows_n + 1), cols).astype(np.int32)
    Y = np.empty_like(X)
    px, py = X.ctypes.data_as(_F), Y.ctypes.data_as(_F)
    pl = lens.ctypes.data_as(_I)

    lib.spur_softmax_masked_f32(px, py, rows_n, cols, pl)
    err = 0.0
    for i in (0, 1, 7, 100, 1023, 2047):
        n = int(lens[i])
        ref = np_softmax(X[i:i + 1, :n].astype(np.float64))[0]
        err = max(err, float(np.abs(Y[i, :n] - ref).max()))
        assert np.all(Y[i, n:] == 0.0), "les entrees hors longueur doivent etre nulles"

    # baseline : la version numpy vectorisee (matrice complete masquee a -inf)
    mask = np.arange(cols)[None, :] < lens[:, None]
    Xm = np.where(mask, X, -np.inf).astype(np.float32)

    def numpy_masked():
        m = Xm.max(axis=1, keepdims=True)
        e = np.exp(Xm - m)
        return e / e.sum(axis=1, keepdims=True)

    t = race({"spur": lambda: lib.spur_softmax_masked_f32(px, py, rows_n, cols, pl),
              "numpy": numpy_masked}, repeats=5, block=3)
    print(f"  {rows_n}x{cols} triangulaire : spur {t['spur']:.4f} ms  "
          f"numpy vectorise {t['numpy']:.4f} ms  -> x{t['numpy']/t['spur']:.1f}"
          f"   err {err:.1e}")
    print("  (le noyau ne parcourt que les len[i] entrees valides ; numpy doit "
          "traiter la matrice entiere)")
    return {"ms_spur": t["spur"], "ms_numpy": t["numpy"],
            "gain": t["numpy"] / t["spur"], "err": err}


def main():
    print("=" * 78)
    print("exp AVX2 minimax + softmax : verification et debit")
    print("=" * 78)
    lib, c32, c64 = build()
    acc = verify_exp(lib)
    e = bench_exp(lib)
    s = bench_softmax(lib)
    m = bench_masked(lib)

    allrows = e + s
    with (RESULTS / "softmax_bench.csv").open("w", newline="") as fh:
        keys = sorted({k for r in allrows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)
    (RESULTS / "softmax_bench.json").write_text(json.dumps(
        {"precision_exp": acc, "exp": e, "softmax": s, "masked": m,
         "coeffs_f32": [float(v) for v in c32],
         "coeffs_f64": [float(v) for v in c64]}, indent=2))
    print("\n-> results/softmax_bench.csv / .json")


if __name__ == "__main__":
    main()
