"""Attention multi-tetes : le cout par appel est-il bien ce qui plafonnait ?

Trois implementations, memes entrees :
  * `boucle`  : `spur_math.attention_tile` appelee une fois par tete (l'etat
                precedent du depot) ;
  * `mha`     : `spur_math.attention_mha`, une seule descente en C, packing K/V
                partage par groupe GQA, parallelisme sur les tetes ;
  * `numpy`   : einsum groupe, la reference honnete.

On mesure le debit ET les GFLOPS effectifs, pour verifier que les petites
tuiles rejoignent le regime des grandes.

Sortie : results/mha_bench.csv / .json
"""

from __future__ import annotations

import csv
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

CASES = [
    # (tq, tk, d, H_q, H_kv) — tq=4 est la taille de micro-bloc de QSA
    (4, 512, 64, 8, 2),
    (4, 2048, 64, 8, 2),
    (16, 512, 64, 8, 2),
    (64, 1024, 64, 8, 2),
    (64, 2048, 128, 12, 4),
    (256, 2048, 128, 12, 4),
    (128, 512, 64, 16, 16),      # attention multi-tetes classique (pas de GQA)
]


def ref_mha(Q, K, V, lens):
    tq, h_q, d = Q.shape
    tk, h_kv, _ = K.shape
    rep = h_q // h_kv
    Kr = np.repeat(K, rep, axis=1).astype(np.float64)
    Vr = np.repeat(V, rep, axis=1).astype(np.float64)
    out = np.zeros((tq, h_q, d))
    for h in range(h_q):
        s = (Q[:, h].astype(np.float64) @ Kr[:, h].T) / np.sqrt(d)
        s = np.where(np.arange(tk)[None, :] >= lens[:, None], -np.inf, s)
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
        out[:, h] = w @ Vr[:, h]
    return out


def main():
    print("=" * 84)
    print("Attention multi-tetes : boucle par tete contre une seule descente C")
    print("=" * 84)
    print(f"\n{'tq':>4}{'tk':>6}{'d':>5}{'Hq/Hkv':>8} {'boucle':>9} {'mha':>9} "
          f"{'numpy':>9} {'x boucle':>9} {'x numpy':>8} {'GF boucle':>10} {'GF mha':>8}")
    rows = []
    for tq, tk, d, h_q, h_kv in CASES:
        rng = np.random.default_rng(0)
        Q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
        K = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
        V = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
        lens = np.minimum(np.arange(tk - tq + 1, tk + 1), tk).astype(np.int32)
        rep = h_q // h_kv
        O = np.empty((tq, h_q, d), dtype=np.float32)

        def par_tete():
            for h in range(h_q):
                O[:, h] = sm.attention_tile(
                    np.ascontiguousarray(Q[:, h]),
                    np.ascontiguousarray(K[:, h // rep]),
                    np.ascontiguousarray(V[:, h // rep]), lengths=lens)
            return O

        Kr = np.repeat(K, rep, axis=1)
        Vr = np.repeat(V, rep, axis=1)
        mask = np.arange(tk)[None, :] >= lens[:, None]

        def par_numpy():
            s = np.einsum("thd,khd->htk", Q, Kr) / np.float32(np.sqrt(d))
            s = np.where(mask[None], np.float32(-np.inf), s)
            s = s - s.max(axis=2, keepdims=True)
            e = np.exp(s)
            w = e / e.sum(axis=2, keepdims=True)
            return np.einsum("htk,khd->thd", w, Vr)

        got = sm.attention_mha(Q, K, V, lengths=lens)
        ref = ref_mha(Q, K, V, lens)
        err = float(np.abs(got - ref).max() / np.abs(ref).max())
        err_loop = float(np.abs(par_tete() - ref).max() / np.abs(ref).max())

        t = race({"boucle": par_tete,
                  "mha": lambda: sm.attention_mha(Q, K, V, lengths=lens, out=O),
                  "numpy": par_numpy}, repeats=6, block=3)
        flops = 4.0 * tq * tk * d * h_q
        row = {"tq": tq, "tk": tk, "d": d, "h_q": h_q, "h_kv": h_kv,
               "ms_boucle": round(t["boucle"], 5), "ms_mha": round(t["mha"], 5),
               "ms_numpy": round(t["numpy"], 5),
               "gain_vs_boucle": round(t["boucle"] / t["mha"], 3),
               "gain_vs_numpy": round(t["numpy"] / t["mha"], 3),
               "gf_boucle": round(flops / (t["boucle"] / 1e3) / 1e9, 1),
               "gf_mha": round(flops / (t["mha"] / 1e3) / 1e9, 1),
               "err": err, "err_boucle": err_loop}
        rows.append(row)
        print(f"{tq:>4}{tk:>6}{d:>5}{h_q:>5}/{h_kv:<2} {t['boucle']:9.4f} "
              f"{t['mha']:9.4f} {t['numpy']:9.4f} {row['gain_vs_boucle']:9.2f} "
              f"{row['gain_vs_numpy']:8.2f} {row['gf_boucle']:10.1f} {row['gf_mha']:8.1f}")

    g = [r["gain_vs_boucle"] for r in rows]
    gn = [r["gain_vs_numpy"] for r in rows]
    print(f"\n  vs boucle par tete : median x{sorted(g)[len(g)//2]:.2f} "
          f"(min x{min(g):.2f}, max x{max(g):.2f})")
    print(f"  vs numpy           : median x{sorted(gn)[len(gn)//2]:.2f} "
          f"(min x{min(gn):.2f}, max x{max(gn):.2f})")
    print(f"  GFLOPS petites tuiles (tq=4) : {rows[0]['gf_boucle']:.1f} -> "
          f"{rows[0]['gf_mha']:.1f}")
    print(f"  erreur relative max : {max(r['err'] for r in rows):.2e}")

    print("\n" + "=" * 84)
    print("Regime decodage : cache KV packe une fois, reutilise par N tuiles")
    print("=" * 84)
    print(f"\n{'tq':>4}{'tk':>7}{'Hq/Hkv':>8} {'complet':>10} {'cache':>9} {'gain':>6} "
          f"{'numpy':>10} {'x numpy':>8} {'GF cache':>9}")
    dec = []
    for tq, tk, d, h_q, h_kv in [(1, 1024, 64, 8, 2), (1, 4096, 64, 8, 2),
                                 (4, 2048, 64, 8, 2), (4, 8192, 64, 8, 2),
                                 (8, 4096, 128, 12, 4)]:
        rng = np.random.default_rng(3)
        Q = (rng.standard_normal((tq, h_q, d)) / np.sqrt(d)).astype(np.float32)
        K = (rng.standard_normal((tk, h_kv, d)) / np.sqrt(d)).astype(np.float32)
        V = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
        O = np.empty((tq, h_q, d), dtype=np.float32)
        cache = sm.KVCache(K, V)
        rep = h_q // h_kv
        Kr, Vr = np.repeat(K, rep, axis=1), np.repeat(V, rep, axis=1)

        def np_dec():
            s_ = np.einsum("thd,khd->htk", Q, Kr) / np.float32(np.sqrt(d))
            s_ = s_ - s_.max(axis=2, keepdims=True)
            e = np.exp(s_)
            return np.einsum("htk,khd->thd", e / e.sum(axis=2, keepdims=True), Vr)

        assert np.abs(cache.attend(Q) - sm.attention_mha(Q, K, V)).max() == 0.0
        t = race({"complet": lambda: sm.attention_mha(Q, K, V, out=O),
                  "cache": lambda: cache.attend(Q, out=O),
                  "numpy": np_dec}, repeats=8, block=4)
        flops = 4.0 * tq * tk * d * h_q
        row = {"regime": "decodage", "tq": tq, "tk": tk, "d": d, "h_q": h_q,
               "h_kv": h_kv, "ms_complet": round(t["complet"], 5),
               "ms_cache": round(t["cache"], 5), "ms_numpy": round(t["numpy"], 5),
               "gain_cache": round(t["complet"] / t["cache"], 3),
               "gain_vs_numpy": round(t["numpy"] / t["cache"], 3),
               "gf_cache": round(flops / (t["cache"] / 1e3) / 1e9, 1),
               "kv_Mo": round(cache.nbytes / 1e6, 2)}
        dec.append(row)
        print(f"{tq:>4}{tk:>7}{h_q:>5}/{h_kv:<2} {t['complet']:10.4f} {t['cache']:9.4f} "
              f"{row['gain_cache']:6.2f} {t['numpy']:10.4f} {row['gain_vs_numpy']:8.2f} "
              f"{row['gf_cache']:9.1f}")
    gd = [r["gain_cache"] for r in dec]
    print(f"\n  amortir le packing : median x{sorted(gd)[len(gd)//2]:.2f} "
          f"(min x{min(gd):.2f}, max x{max(gd):.2f})")
    (RESULTS / "mha_decode.json").write_text(json.dumps(dec, indent=2))

    with (RESULTS / "mha_bench.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "mha_bench.json").write_text(json.dumps(rows, indent=2))
    print("-> results/mha_bench.csv")


if __name__ == "__main__":
    main()
