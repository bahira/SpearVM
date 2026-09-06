"""Deux questions restantes, mesurees a armes egales :

  1. la regle de noyau locale a l'attention vise les tuiles dont rep*tq n'est
     PAS multiple de 8 (hors zone de force du noyau legacy). Combien
     rapporte-t-elle vraiment sur ces configurations ?
  2. le cache KV en bf16 divise l'empreinte par deux — a partir de quelle
     taille de contexte devient-il aussi plus RAPIDE, et que coute-t-il en
     precision ?

Sortie : results/bf16_kv.csv / .json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import spur_math as sm  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


def best_ms(fn, reps=12):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def main():
    print("=" * 78)
    print("1. Effet de la regle de noyau selon rep*tq (multiple de 8 ou non)")
    print("=" * 78)
    print(f"\n{'tq':>4}{'Hq/Hkv':>8}{'rep*tq':>8}{'tk':>7}{'d':>5} {'ms':>8} {'GFLOPS':>8}  note")
    rowsA = []
    for tq, h_q, h_kv, tk, d in [(4, 8, 2, 4096, 64),    # rep*tq = 16 (multiple de 8)
                                 (4, 12, 4, 4096, 64),   # rep*tq = 12
                                 (4, 16, 8, 4096, 64),   # rep*tq =  8 (multiple de 8)
                                 (4, 24, 8, 4096, 64),   # rep*tq = 12
                                 (5, 12, 4, 4096, 64),   # rep*tq = 15
                                 (8, 8, 2, 4096, 64)]:   # rep*tq = 32 (multiple de 8)
        rng = np.random.default_rng(0)
        Q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
        K = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
        V = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
        cache = sm.KVCache(K, V)
        O = np.empty((tq, h_q, d), dtype=np.float32)
        ms = best_ms(lambda: cache.attend(Q, out=O))
        rep = h_q // h_kv
        mb = rep * tq
        gf = 4.0 * tq * tk * d * h_q / (ms / 1e3) / 1e9
        note = "multiple de 8 -> legacy" if mb % 8 == 0 else "-> noyau dot"
        rowsA.append({"tq": tq, "h_q": h_q, "h_kv": h_kv, "mb": mb, "tk": tk,
                      "d": d, "ms": round(ms, 4), "gflops": round(gf, 1),
                      "note": note})
        print(f"{tq:>4}{h_q:>5}/{h_kv:<2}{mb:>8}{tk:>7}{d:>5} {ms:8.4f} {gf:8.1f}  {note}")

    print("\n" + "=" * 78)
    print("2. Cache KV bf16 : empreinte, vitesse, precision")
    print("=" * 78)
    print(f"\n{'tk':>7}{'d':>5}{'Hq/Hkv':>8} {'Mo f32':>8}{'Mo bf16':>8} {'ms f32':>8}"
          f"{'ms bf16':>9} {'gain':>6} {'err rel':>9}")
    rowsB = []
    for tk, d, h_q, h_kv in [(1024, 64, 8, 2), (4096, 64, 8, 2), (8192, 64, 8, 2),
                             (16384, 64, 8, 2), (4096, 128, 12, 4),
                             (8192, 128, 12, 4), (16384, 128, 12, 4)]:
        rng = np.random.default_rng(1)
        K = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
        V = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
        Q = (rng.standard_normal((4, h_q, d)) / np.sqrt(d)).astype(np.float32)
        c32 = sm.KVCache(K, V)
        cbf = sm.KVCache(K, V, dtype="bf16")
        a, b = c32.attend(Q), cbf.attend(Q)
        err = float(np.abs(a - b).max() / np.abs(a).max())
        t32 = best_ms(lambda: c32.attend(Q))
        tbf = best_ms(lambda: cbf.attend(Q))
        rowsB.append({"tk": tk, "d": d, "h_q": h_q, "h_kv": h_kv,
                      "Mo_f32": round(c32.nbytes / 1e6, 2),
                      "Mo_bf16": round(cbf.nbytes / 1e6, 2),
                      "ms_f32": round(t32, 4), "ms_bf16": round(tbf, 4),
                      "gain": round(t32 / tbf, 3), "err_rel": err})
        r = rowsB[-1]
        print(f"{tk:>7}{d:>5}{h_q:>5}/{h_kv:<2} {r['Mo_f32']:8.2f}{r['Mo_bf16']:8.2f} "
              f"{t32:8.4f}{tbf:9.4f} {r['gain']:6.2f} {err:9.1e}")

    seuil = [r for r in rowsB if r["gain"] >= 1.0]
    print(f"\n  empreinte : divisee par 2 dans tous les cas")
    print(f"  vitesse   : bf16 devient gagnant a partir de "
          f"{min((r['Mo_f32'] for r in seuil), default=float('nan')):.1f} Mo de cache f32")
    print(f"  precision : erreur relative {max(r['err_rel'] for r in rowsB):.1e} "
          f"(bf16 = 8 bits de mantisse)")

    with (RESULTS / "bf16_kv.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rowsB[0]))
        w.writeheader()
        w.writerows(rowsB)
    (RESULTS / "bf16_kv.json").write_text(json.dumps(
        {"regle_noyau": rowsA, "bf16": rowsB}, indent=2))
    print("-> results/bf16_kv.csv / .json")


if __name__ == "__main__":
    main()
