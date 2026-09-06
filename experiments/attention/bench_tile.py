"""La tuile d'attention en C tient-elle contre numpy ?

`spur_attention_tile_f32` enchaine GEMM NT -> softmax masque -> GEMM NT sans
materialiser de masque ni repasser sur la matrice de scores (l'echelle
1/sqrt(d) est absorbee par le softmax). On verifie contre une reference numpy
float64, puis on chronometre sur les formes de tuile que produit le routage QSA.

Sortie : results/attention_tile.csv / .json
"""

from __future__ import annotations

import csv
import ctypes
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

from harness import race  # noqa: E402

import spur_math as sm  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)

_DLL = ctypes.CDLL(sm._dll_path)
_F = ctypes.POINTER(ctypes.c_float)
_I = ctypes.POINTER(ctypes.c_int)
_tile = _DLL.spur_attention_tile_f32
_tile.argtypes = [_F, _F, _F, _F, ctypes.c_longlong, ctypes.c_longlong,
                  ctypes.c_longlong, ctypes.c_float, _I, _F]
_tile.restype = None


def ref_attention(Q, K, V, scale, lens=None):
    Qd, Kd, Vd = Q.astype(np.float64), K.astype(np.float64), V.astype(np.float64)
    s = (Qd @ Kd.T) * scale
    if lens is not None:
        mask = np.arange(K.shape[0])[None, :] >= np.asarray(lens)[:, None]
        s = np.where(mask, -np.inf, s)
    s -= s.max(axis=1, keepdims=True)
    w = np.exp(s)
    w /= w.sum(axis=1, keepdims=True)
    return w @ Vd


def run_tile(Q, K, Vt, O, scratch, scale, lens_ptr):
    tq, d = Q.shape
    tk = K.shape[0]
    _tile(Q.ctypes.data_as(_F), K.ctypes.data_as(_F), Vt.ctypes.data_as(_F),
          O.ctypes.data_as(_F), tq, tk, d, scale, lens_ptr,
          scratch.ctypes.data_as(_F))


def main():
    print("=" * 78)
    print("Tuile d'attention en C : GEMM NT -> softmax masque -> GEMM NT")
    print("=" * 78)
    rows = []
    shapes = [(4, 512, 64), (16, 512, 64), (64, 1024, 64), (64, 2048, 128),
              (256, 2048, 128), (512, 512, 64)]
    print(f"\n{'tq':>5}{'tk':>6}{'d':>5} {'spur ms':>9} {'numpy ms':>9} {'gain':>6} "
          f"{'GF':>7}  {'err rel':>9}")
    for tq, tk, d in shapes:
        rng = np.random.default_rng(0)
        Q = (rng.standard_normal((tq, d)) / np.sqrt(d)).astype(np.float32)
        K = (rng.standard_normal((tk, d)) / np.sqrt(d)).astype(np.float32)
        V = rng.standard_normal((tk, d)).astype(np.float32)
        Vt = np.ascontiguousarray(V.T)
        O = np.zeros((tq, d), dtype=np.float32)
        scratch = np.zeros((tq, tk), dtype=np.float32)
        scale = float(1.0 / np.sqrt(d))
        lens = np.minimum(np.arange(tk - tq + 1, tk + 1), tk).astype(np.int32)
        lp = lens.ctypes.data_as(_I)

        run_tile(Q, K, Vt, O, scratch, scale, lp)
        ref = ref_attention(Q, K, V, scale, lens)
        err = float(np.abs(O.astype(np.float64) - ref).max() /
                    max(np.abs(ref).max(), 1e-30))

        def np_tile():
            s = (Q @ K.T) * np.float32(scale)
            m = np.arange(tk)[None, :] >= lens[:, None]
            s = np.where(m, np.float32(-np.inf), s)
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            return (e / e.sum(axis=1, keepdims=True)) @ V

        t = race({"spur": lambda: run_tile(Q, K, Vt, O, scratch, scale, lp),
                  "numpy": np_tile}, repeats=6, block=3)
        flops = 4.0 * tq * tk * d
        rows.append({"tq": tq, "tk": tk, "d": d,
                     "ms_spur": round(t["spur"], 5),
                     "ms_numpy": round(t["numpy"], 5),
                     "gain": round(t["numpy"] / t["spur"], 3),
                     "gflops": round(flops / (t["spur"] / 1e3) / 1e9, 1),
                     "err_rel": err})
        r = rows[-1]
        print(f"{tq:>5}{tk:>6}{d:>5} {t['spur']:9.4f} {t['numpy']:9.4f} "
              f"{r['gain']:6.2f} {r['gflops']:7.1f}  {err:9.1e}")

    gains = [r["gain"] for r in rows]
    print(f"\n  gain median x{sorted(gains)[len(gains)//2]:.2f} "
          f"(min x{min(gains):.2f}, max x{max(gains):.2f})")
    print(f"  erreur relative max : {max(r['err_rel'] for r in rows):.2e}")

    with (RESULTS / "attention_tile.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "attention_tile.json").write_text(json.dumps(rows, indent=2))
    print("-> results/attention_tile.csv")


if __name__ == "__main__":
    main()
