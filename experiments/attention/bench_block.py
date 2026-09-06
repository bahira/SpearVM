"""Bloc de decodeur complet : ce que les noyaux donnent au niveau produit.

On assemble un bloc transformeur standard avec les briques du depot :

    h  = x + Attention(RMSNorm(x))          -> attention_mha / KVCache
    y  = h + W2 . GELU(W1 . RMSNorm(h))     -> matmul_nt_gelu + matmul_nt

et on le compare a la meme chose ecrite en numpy. Deux regimes :

  * **prefill**  : tq = tk = N tokens d'un coup (masque causal) ;
  * **decodage** : un token a la fois contre un cache KV de tk tokens.

Sortie : results/block.csv / .json
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


def rmsnorm(x, w, eps=1e-6):
    n = np.sqrt(np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return ((x / n) * w).astype(np.float32)


class Block:
    """Bloc de decodeur : d_model, H_q tetes de requetes, H_kv tetes de cles."""

    def __init__(self, d_model, h_q, h_kv, d_ff, seed=0):
        rng = np.random.default_rng(seed)
        self.d_model, self.h_q, self.h_kv = d_model, h_q, h_kv
        self.dh = d_model // h_q
        f = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(np.float32)
        self.wq = f(d_model, d_model)
        self.wk = f(h_kv * self.dh, d_model)
        self.wv = f(h_kv * self.dh, d_model)
        self.wo = f(d_model, d_model)
        self.w1 = f(d_ff, d_model)
        self.w2 = f(d_model, d_ff)
        self.n1 = np.ones(d_model, dtype=np.float32)
        self.n2 = np.ones(d_model, dtype=np.float32)

    # -- projections communes aux deux implementations ---------------------
    def project(self, x, mm):
        t = x.shape[0]
        q = mm(x, self.wq).reshape(t, self.h_q, self.dh)
        k = mm(x, self.wk).reshape(t, self.h_kv, self.dh)
        v = mm(x, self.wv).reshape(t, self.h_kv, self.dh)
        return q, k, v

    # -- SpearVM ------------------------------------------------------------
    def spur_prefill(self, x):
        h = rmsnorm(x, self.n1)
        q, k, v = self.project(h, sm.matmul_nt)
        lens = np.arange(1, x.shape[0] + 1, dtype=np.int32)
        a = sm.attention_mha(q, k, v, lengths=lens).reshape(x.shape[0], -1)
        h = x + sm.matmul_nt(a, self.wo)
        n = rmsnorm(h, self.n2)
        return h + sm.matmul_nt(sm.matmul_nt_gelu(n, self.w1), self.w2)

    def spur_decode(self, x1, cache):
        h = rmsnorm(x1, self.n1)
        q = sm.matmul_nt(h, self.wq).reshape(1, self.h_q, self.dh)
        a = cache.attend(q).reshape(1, -1)
        h = x1 + sm.matmul_nt(a, self.wo)
        n = rmsnorm(h, self.n2)
        return h + sm.matmul_nt(sm.matmul_nt_gelu(n, self.w1), self.w2)

    # -- numpy --------------------------------------------------------------
    @staticmethod
    def _np_mm(a, b):
        return np.ascontiguousarray(a @ b.T)

    def _np_gelu(self, z):
        u = np.clip(0.306923 * z + 0.501, 0.0, 1.002).astype(np.float32)
        return (0.997729 * (z * u) - 0.004004).astype(np.float32)

    def np_prefill(self, x):
        t = x.shape[0]
        h = rmsnorm(x, self.n1)
        q, k, v = self.project(h, self._np_mm)
        rep = self.h_q // self.h_kv
        kr, vr = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.einsum("thd,khd->htk", q, kr) / np.float32(np.sqrt(self.dh))
        s = np.where(np.triu(np.ones((t, t), bool), 1)[None], np.float32(-np.inf), s)
        s = s - s.max(axis=2, keepdims=True)
        e = np.exp(s)
        a = np.einsum("htk,khd->thd", e / e.sum(axis=2, keepdims=True), vr).reshape(t, -1)
        h = x + self._np_mm(a.astype(np.float32), self.wo)
        n = rmsnorm(h, self.n2)
        return h + self._np_mm(self._np_gelu(self._np_mm(n, self.w1)), self.w2)

    def np_decode(self, x1, k, v):
        h = rmsnorm(x1, self.n1)
        q = self._np_mm(h, self.wq).reshape(1, self.h_q, self.dh)
        rep = self.h_q // self.h_kv
        kr, vr = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
        s = np.einsum("thd,khd->htk", q, kr) / np.float32(np.sqrt(self.dh))
        s = s - s.max(axis=2, keepdims=True)
        e = np.exp(s)
        a = np.einsum("htk,khd->thd", e / e.sum(axis=2, keepdims=True), vr).reshape(1, -1)
        h = x1 + self._np_mm(a.astype(np.float32), self.wo)
        n = rmsnorm(h, self.n2)
        return h + self._np_mm(self._np_gelu(self._np_mm(n, self.w1)), self.w2)


def best_ms(fn, reps=6):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def main():
    print("=" * 80)
    print("Bloc de decodeur complet : SpearVM contre numpy")
    print("=" * 80)
    rows = []

    print(f"\n### Prefill (tout le contexte d'un coup)\n")
    print(f"{'N':>6}{'d_model':>9}{'Hq/Hkv':>8}{'d_ff':>7} {'spur ms':>9}{'numpy ms':>10} "
          f"{'gain':>6} {'tok/s spur':>11} {'ecart':>9}")
    for N, dm, h_q, h_kv, d_ff in [(256, 512, 8, 2, 2048), (512, 512, 8, 2, 2048),
                                   (1024, 768, 12, 4, 3072), (2048, 768, 12, 4, 3072)]:
        blk = Block(dm, h_q, h_kv, d_ff)
        rng = np.random.default_rng(1)
        x = (rng.standard_normal((N, dm)) / np.sqrt(dm)).astype(np.float32)
        a, b = blk.spur_prefill(x), blk.np_prefill(x)
        err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))
        ts, tn = best_ms(lambda: blk.spur_prefill(x)), best_ms(lambda: blk.np_prefill(x))
        rows.append({"regime": "prefill", "N": N, "d_model": dm, "h_q": h_q,
                     "h_kv": h_kv, "d_ff": d_ff, "ms_spur": round(ts, 3),
                     "ms_numpy": round(tn, 3), "gain": round(tn / ts, 3),
                     "tok_s": round(N / ts * 1e3), "ecart": err})
        r = rows[-1]
        print(f"{N:>6}{dm:>9}{h_q:>5}/{h_kv:<2}{d_ff:>7} {ts:9.3f}{tn:10.3f} "
              f"{r['gain']:6.2f} {r['tok_s']:>11,} {err:9.1e}")

    print(f"\n### Decodage (un token contre un cache de tk)\n")
    print(f"{'tk':>6}{'d_model':>9}{'Hq/Hkv':>8} {'spur ms':>9}{'numpy ms':>10} "
          f"{'gain':>6} {'tok/s spur':>11} {'ecart':>9}")
    for tk, dm, h_q, h_kv, d_ff in [(512, 512, 8, 2, 2048), (2048, 512, 8, 2, 2048),
                                    (4096, 768, 12, 4, 3072), (8192, 768, 12, 4, 3072)]:
        blk = Block(dm, h_q, h_kv, d_ff)
        rng = np.random.default_rng(2)
        dh = dm // h_q
        x1 = (rng.standard_normal((1, dm)) / np.sqrt(dm)).astype(np.float32)
        k = (rng.standard_normal((tk, h_kv, dh)) / np.sqrt(dh)).astype(np.float32)
        v = rng.standard_normal((tk, h_kv, dh)).astype(np.float32)
        cache = sm.KVCache(k, v)
        a, b = blk.spur_decode(x1, cache), blk.np_decode(x1, k, v)
        err = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))
        ts = best_ms(lambda: blk.spur_decode(x1, cache), reps=12)
        tn = best_ms(lambda: blk.np_decode(x1, k, v), reps=12)
        rows.append({"regime": "decodage", "N": 1, "tk": tk, "d_model": dm,
                     "h_q": h_q, "h_kv": h_kv, "d_ff": d_ff,
                     "ms_spur": round(ts, 4), "ms_numpy": round(tn, 4),
                     "gain": round(tn / ts, 3), "tok_s": round(1 / ts * 1e3),
                     "ecart": err})
        r = rows[-1]
        print(f"{tk:>6}{dm:>9}{h_q:>5}/{h_kv:<2} {ts:9.4f}{tn:10.4f} {r['gain']:6.2f} "
              f"{r['tok_s']:>11,} {err:9.1e}")

    g = [r["gain"] for r in rows]
    print(f"\n  gain median x{sorted(g)[len(g)//2]:.2f} (min x{min(g):.2f}, max x{max(g):.2f})")
    print(f"  ecart max avec la reference numpy : {max(r['ecart'] for r in rows):.1e}")
    print("  (les deux implementations partagent la meme gelu SPEAR ; l'ecart")
    print("   restant vient de l'ordre des sommations en float32)")

    with (RESULTS / "block.csv").open("w", newline="") as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "block.json").write_text(json.dumps(rows, indent=2))
    print("-> results/block.csv")


if __name__ == "__main__":
    main()
