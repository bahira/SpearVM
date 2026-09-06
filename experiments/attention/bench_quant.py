"""Poids quantifies : le seul levier quand le decodage est limite par la memoire.

A m petit, un GEMM ne fait que 2.m flops par poids lu. Le temps est alors
proportionnel au nombre d'octets par poids — pas au nombre d'operations. On
mesure donc trois formats sur les formes du FFN et des projections, puis sur le
bloc de decodeur complet.

Sortie : results/quant_weights.csv / .json
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
sys.path.insert(0, str(HERE))

import spur_math as sm  # noqa: E402
from bench_block import Block, best_ms, rmsnorm  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)


def bench_shapes():
    print("=" * 92)
    print("1. GEMM a m petit : trois formats de poids, memes formes")
    print("=" * 92)
    print(f"\n{'m':>4}{'k':>6}{'n':>6} {'f32 ms':>9}{'bf16 ms':>9}{'i8 ms':>9} "
          f"{'x bf16':>7}{'x i8':>6} {'Go/s f32':>9}{'Go/s i8':>8} "
          f"{'err bf16':>9}{'err i8':>9}")
    rows = []
    for k, n in [(768, 3072), (3072, 768), (512, 2048), (2048, 512), (512, 512)]:
        for m in (1, 2, 4, 16, 64):
            rng = np.random.default_rng(0)
            W = (rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)
            A = (rng.standard_normal((m, k)) / np.sqrt(k)).astype(np.float32)
            C = np.empty((m, n), dtype=np.float32)
            qb = sm.QuantizedWeight(W, dtype="bf16")
            qi = sm.QuantizedWeight(W, dtype="i8")
            ref = A.astype(np.float64) @ W.astype(np.float64).T
            eb = float(np.abs(qb.matmul(A) - ref).max() / np.abs(ref).max())
            ei = float(np.abs(qi.matmul(A) - ref).max() / np.abs(ref).max())

            t32 = best_ms(lambda: sm.matmul_nt(A, W, out=C), reps=10)
            tb = best_ms(lambda: qb.matmul(A, out=C), reps=10)
            ti = best_ms(lambda: qi.matmul(A, out=C), reps=10)
            rows.append({"m": m, "k": k, "n": n, "ms_f32": round(t32, 4),
                         "ms_bf16": round(tb, 4), "ms_i8": round(ti, 4),
                         "gain_bf16": round(t32 / tb, 3), "gain_i8": round(t32 / ti, 3),
                         "GBs_f32": round(n * k * 4 / (t32 / 1e3) / 1e9, 1),
                         "GBs_i8": round(n * k / (ti / 1e3) / 1e9, 1),
                         "err_bf16": eb, "err_i8": ei})
            r = rows[-1]
            print(f"{m:>4}{k:>6}{n:>6} {t32:9.4f}{tb:9.4f}{ti:9.4f} "
                  f"{r['gain_bf16']:7.2f}{r['gain_i8']:6.2f} {r['GBs_f32']:9.1f}"
                  f"{r['GBs_i8']:8.1f} {eb:9.1e}{ei:9.1e}")
    return rows


class QuantBlock(Block):
    """Meme bloc, mais poids du FFN et des projections quantifies."""

    def quantize(self, dtype):
        self.q = {n: sm.QuantizedWeight(getattr(self, n), dtype=dtype)
                  for n in ("wq", "wo", "w1", "w2")}
        return self

    def q_decode(self, x1, cache):
        h = rmsnorm(x1, self.n1)
        q = self.q["wq"].matmul(h).reshape(1, self.h_q, self.dh)
        a = cache.attend(q).reshape(1, -1)
        h = x1 + self.q["wo"].matmul(a)
        n = rmsnorm(h, self.n2)
        z = self.q["w1"].matmul(n)
        u = np.clip(0.306923 * z + 0.501, 0.0, 1.002).astype(np.float32)
        g = (0.997729 * (z * u) - 0.004004).astype(np.float32)
        return h + self.q["w2"].matmul(g)


def bench_block():
    print("\n" + "=" * 92)
    print("2. Bloc de decodeur, un token contre un cache KV")
    print("=" * 92)
    print(f"\n{'tk':>6}{'d_model':>9}{'d_ff':>6} {'numpy':>8}{'f32':>8}{'bf16':>8}{'i8':>8} "
          f"{'x f32':>7}{'x bf16':>8}{'x i8':>6} {'Mo poids':>10} {'err i8':>9}")
    rows = []
    for tk, dm, h_q, h_kv, d_ff in [(512, 512, 8, 2, 2048), (2048, 512, 8, 2, 2048),
                                    (512, 768, 12, 4, 3072), (4096, 768, 12, 4, 3072),
                                    (8192, 768, 12, 4, 3072)]:
        rng = np.random.default_rng(2)
        dh = dm // h_q
        blk = Block(dm, h_q, h_kv, d_ff)
        x1 = (rng.standard_normal((1, dm)) / np.sqrt(dm)).astype(np.float32)
        k = (rng.standard_normal((tk, h_kv, dh)) / np.sqrt(dh)).astype(np.float32)
        v = rng.standard_normal((tk, h_kv, dh)).astype(np.float32)
        cache = sm.KVCache(k, v)
        qb = QuantBlock(dm, h_q, h_kv, d_ff)
        qb.__dict__.update({a: getattr(blk, a) for a in
                            ("wq", "wk", "wv", "wo", "w1", "w2", "n1", "n2")}
                           ) if False else None
        for a in ("wq", "wk", "wv", "wo", "w1", "w2", "n1", "n2"):
            setattr(qb, a, getattr(blk, a))
        bf = QuantBlock(dm, h_q, h_kv, d_ff)
        for a in ("wq", "wk", "wv", "wo", "w1", "w2", "n1", "n2"):
            setattr(bf, a, getattr(blk, a))
        qb.quantize("i8")
        bf.quantize("bf16")

        ref = blk.np_decode(x1, k, v)
        r32 = blk.spur_decode(x1, cache)
        rbf = bf.q_decode(x1, cache)
        ri8 = qb.q_decode(x1, cache)
        err_i8 = float(np.abs(ri8 - ref).max() / np.abs(ref).max())
        err_bf = float(np.abs(rbf - ref).max() / np.abs(ref).max())
        assert np.abs(r32 - ref).max() / np.abs(ref).max() < 1e-4

        tn = best_ms(lambda: blk.np_decode(x1, k, v), reps=12)
        t32 = best_ms(lambda: blk.spur_decode(x1, cache), reps=12)
        tbf = best_ms(lambda: bf.q_decode(x1, cache), reps=12)
        ti8 = best_ms(lambda: qb.q_decode(x1, cache), reps=12)
        poids = sum(qb.q[n].nbytes for n in qb.q) / 1e6
        rows.append({"tk": tk, "d_model": dm, "d_ff": d_ff,
                     "ms_numpy": round(tn, 4), "ms_f32": round(t32, 4),
                     "ms_bf16": round(tbf, 4), "ms_i8": round(ti8, 4),
                     "gain_f32_vs_numpy": round(tn / t32, 3),
                     "gain_bf16_vs_numpy": round(tn / tbf, 3),
                     "gain_i8_vs_numpy": round(tn / ti8, 3),
                     "gain_i8_vs_f32": round(t32 / ti8, 3),
                     "Mo_poids_i8": round(poids, 2),
                     "err_bf16": err_bf, "err_i8": err_i8,
                     "tok_s_i8": round(1 / ti8 * 1e3)})
        r = rows[-1]
        print(f"{tk:>6}{dm:>9}{d_ff:>6} {tn:8.3f}{t32:8.3f}{tbf:8.3f}{ti8:8.3f} "
              f"{r['gain_f32_vs_numpy']:7.2f}{r['gain_bf16_vs_numpy']:8.2f}"
              f"{r['gain_i8_vs_numpy']:6.2f} {poids:10.2f} {err_i8:9.1e}")
    return rows


def main():
    a = bench_shapes()
    b = bench_block()
    print(f"\n  bloc : i8 contre f32  -> median x"
          f"{sorted(r['gain_i8_vs_f32'] for r in b)[len(b)//2]:.2f}")
    print(f"  bloc : i8 contre numpy -> median x"
          f"{sorted(r['gain_i8_vs_numpy'] for r in b)[len(b)//2]:.2f}, "
          f"jusqu'a {max(r['tok_s_i8'] for r in b)} tokens/s")
    print(f"  erreur i8 sur la sortie du bloc : "
          f"{min(r['err_i8'] for r in b):.1e} a {max(r['err_i8'] for r in b):.1e}")
    with (RESULTS / "quant_weights.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(a[0]))
        w.writeheader()
        w.writerows(a)
    (RESULTS / "quant_weights.json").write_text(json.dumps({"gemm": a, "bloc": b}, indent=2))
    print("-> results/quant_weights.csv / .json")


if __name__ == "__main__":
    main()
