"""Un petit transformeur entraine avec les noyaux SpearVM.

Objectif : disposer de poids **appris** pour mesurer honnetement la fidelite de
l'attention creuse. Sur des poids aleatoires, les tetes d'un groupe GQA sont
independantes et aucune selection partagee ne peut marcher — c'est la limite
identifiee dans docs/ATTENTION_AUDIT.md section 8, et elle n'est levable que
par l'entrainement.

Modele : transformeur causal caractere par caractere, RMSNorm, GQA, FFN GELU,
embeddings de sortie lies. Retropropagation ecrite a la main (aucun framework
n'est disponible ici) et **verifiee par differences finies**.

Les produits matriciels passent par `spur_math.matmul_nt` / `matmul_backward` :
le modele est donc entraine par les noyaux que l'on cherche a evaluer.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import spur_math as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
def load_corpus(limit: int = 900_000) -> str:
    """Corpus : les sources et la documentation du depot lui-meme."""
    pats = ("docs/*.md", "*.md", "src/*.c", "spur_math/*.py", "experiments/*.py",
            "experiments/attention/*.py", "web/server/spearvm_sim/*.py",
            "web/server/spearvm_sim/sims/*.py", "web/client/src/*.ts",
            "web/client/src/*/*.ts", "tests/*.py")
    parts = []
    for pat in pats:
        for f in sorted(ROOT.glob(pat)):
            try:
                parts.append(f.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return ("\n\n".join(parts))[:limit]


def rmsnorm(x, g, eps=1e-5):
    inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x * inv) * g, inv


def rmsnorm_back(dy, x, g, inv):
    d = x.shape[-1]
    xg = x * inv
    dg = (dy * xg).sum(axis=0)
    t = dy * g
    dx = (t - xg * (np.sum(t * xg, axis=-1, keepdims=True) / d)) * inv
    return dx, dg


def gelu(x):
    u = np.clip(0.306923 * x + 0.501, 0.0, 1.002)
    return 0.997729 * (x * u) - 0.004004


def gelu_back(dy, x):
    u = 0.306923 * x + 0.501
    inside = (u > 0.0) & (u < 1.002)
    g = np.where(inside, 0.997729 * (u + 0.306923 * x),
                 np.where(u <= 0.0, 0.0, 0.997729 * 1.002))
    return dy * g


# ---------------------------------------------------------------------------
class TinyLM:
    def __init__(self, vocab, d_model=128, n_layer=2, h_q=4, h_kv=2, d_ff=384,
                 ctx=128, seed=0, dtype=np.float32):
        # dtype configurable : la verification par differences finies exige du
        # float64 (en float32 le bruit d'arrondi de la perte, ~1e-7 relatif,
        # est du meme ordre que la difference que l'on cherche a mesurer).
        self.dt = dtype
        rng = np.random.default_rng(seed)
        self.V, self.d, self.L = vocab, d_model, n_layer
        self.h_q, self.h_kv, self.ctx = h_q, h_kv, ctx
        self.dh = d_model // h_q
        self.rep = h_q // h_kv
        f = lambda *s: (rng.standard_normal(s) * (1.0 / np.sqrt(s[-1]))).astype(self.dt)
        self.p = {
            "emb": (rng.standard_normal((vocab, d_model)) * 0.02).astype(self.dt),
            "pos": (rng.standard_normal((ctx, d_model)) * 0.01).astype(self.dt),
            "gf": np.ones(d_model, dtype=self.dt),
        }
        for l in range(n_layer):
            self.p[f"g1_{l}"] = np.ones(d_model, dtype=self.dt)
            self.p[f"g2_{l}"] = np.ones(d_model, dtype=self.dt)
            self.p[f"wq_{l}"] = f(h_q * self.dh, d_model)
            self.p[f"wk_{l}"] = f(h_kv * self.dh, d_model)
            self.p[f"wv_{l}"] = f(h_kv * self.dh, d_model)
            self.p[f"wo_{l}"] = f(d_model, h_q * self.dh)
            self.p[f"w1_{l}"] = f(d_ff, d_model)
            self.p[f"w2_{l}"] = f(d_model, d_ff)
        self.mask = np.triu(np.ones((ctx, ctx), dtype=bool), 1)

    # -- passe avant ; `keep` collecte les poids d'attention pour l'analyse --
    def forward(self, idx, keep=None):
        B, T = idx.shape
        d = self.d
        x = (self.p["emb"][idx.reshape(-1)] + np.tile(self.p["pos"][:T], (B, 1)))
        x = np.ascontiguousarray(x, dtype=self.dt)
        cache = {"idx": idx, "B": B, "T": T, "x0": x}
        mask = self.mask[:T, :T]

        for l in range(self.L):
            h, inv1 = rmsnorm(x, self.p[f"g1_{l}"])
            q = sm.matmul_nt(h, self.p[f"wq_{l}"])
            k = sm.matmul_nt(h, self.p[f"wk_{l}"])
            v = sm.matmul_nt(h, self.p[f"wv_{l}"])
            qs = q.reshape(B, T, self.h_q, self.dh)
            ks = k.reshape(B, T, self.h_kv, self.dh)
            vs = v.reshape(B, T, self.h_kv, self.dh)
            P = np.zeros((B, self.h_q, T, T), dtype=self.dt)
            O = np.zeros((B, T, self.h_q, self.dh), dtype=self.dt)
            for hh in range(self.h_q):
                g = hh // self.rep
                s = np.einsum("btd,bsd->bts", qs[:, :, hh], ks[:, :, g]) / np.sqrt(self.dh)
                s = np.where(mask[None], -np.inf, s)
                s -= s.max(axis=-1, keepdims=True)
                e = np.exp(s)
                p = e / e.sum(axis=-1, keepdims=True)
                P[:, hh] = p
                O[:, :, hh] = np.einsum("bts,bsd->btd", p, vs[:, :, g])
            a = np.ascontiguousarray(O.reshape(B * T, -1), dtype=self.dt)
            ao = sm.matmul_nt(a, self.p[f"wo_{l}"])
            x1 = x + ao
            h2, inv2 = rmsnorm(x1, self.p[f"g2_{l}"])
            z = sm.matmul_nt(h2, self.p[f"w1_{l}"])
            gz = gelu(z)
            x2 = x1 + sm.matmul_nt(gz, self.p[f"w2_{l}"])
            cache[l] = dict(xin=x, h=h, inv1=inv1, qs=qs, ks=ks, vs=vs, P=P, a=a, x1=x1,
                            h2=h2, inv2=inv2, z=z, gz=gz)
            if keep is not None:
                keep.append({"layer": l, "P": P.copy(), "q": qs.copy(), "k": ks.copy()})
            x = x2

        hf, invf = rmsnorm(x, self.p["gf"])
        logits = sm.matmul_nt(hf, self.p["emb"])
        cache.update(hf=hf, invf=invf, xL=x)
        return logits, cache

    def loss_and_grads(self, idx, tgt):
        logits, c = self.forward(idx)
        B, T = idx.shape
        n = B * T
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        flat = tgt.reshape(-1)
        loss = float(-np.log(p[np.arange(n), flat] + 1e-12).mean())

        d = {k: np.zeros_like(v) for k, v in self.p.items()}
        dl = p
        dl[np.arange(n), flat] -= 1.0
        dl = np.ascontiguousarray(dl / n, dtype=self.dt)
        dhf, demb = sm.matmul_backward(dl, c["hf"], self.p["emb"])
        d["emb"] += demb
        dx, dgf = rmsnorm_back(dhf, c["xL"], self.p["gf"], c["invf"])
        d["gf"] += dgf

        for l in range(self.L - 1, -1, -1):
            cc = c[l]
            dgz, dw2 = sm.matmul_backward(dx, cc["gz"], self.p[f"w2_{l}"])
            d[f"w2_{l}"] += dw2
            dz = gelu_back(dgz, cc["z"])
            dh2, dw1 = sm.matmul_backward(dz, cc["h2"], self.p[f"w1_{l}"])
            d[f"w1_{l}"] += dw1
            dx1, dg2 = rmsnorm_back(dh2, cc["x1"], self.p[f"g2_{l}"], cc["inv2"])
            d[f"g2_{l}"] += dg2
            dx1 = dx1 + dx

            da, dwo = sm.matmul_backward(dx1, cc["a"], self.p[f"wo_{l}"])
            d[f"wo_{l}"] += dwo
            dO = da.reshape(B, T, self.h_q, self.dh)
            dq = np.zeros_like(cc["qs"])
            dk = np.zeros_like(cc["ks"])
            dv = np.zeros_like(cc["vs"])
            for hh in range(self.h_q):
                g = hh // self.rep
                P = cc["P"][:, hh]
                dv[:, :, g] += np.einsum("bts,btd->bsd", P, dO[:, :, hh])
                dP = np.einsum("btd,bsd->bts", dO[:, :, hh], cc["vs"][:, :, g])
                ds = P * (dP - (dP * P).sum(axis=-1, keepdims=True))
                ds = ds.astype(self.dt) / np.sqrt(self.dh)
                dq[:, :, hh] += np.einsum("bts,bsd->btd", ds, cc["ks"][:, :, g])
                dk[:, :, g] += np.einsum("bts,btd->bsd", ds, cc["qs"][:, :, hh])
            dhq, dwq = sm.matmul_backward(
                np.ascontiguousarray(dq.reshape(B * T, -1)), cc["h"], self.p[f"wq_{l}"])
            dhk, dwk = sm.matmul_backward(
                np.ascontiguousarray(dk.reshape(B * T, -1)), cc["h"], self.p[f"wk_{l}"])
            dhv, dwv = sm.matmul_backward(
                np.ascontiguousarray(dv.reshape(B * T, -1)), cc["h"], self.p[f"wv_{l}"])
            d[f"wq_{l}"] += dwq
            d[f"wk_{l}"] += dwk
            d[f"wv_{l}"] += dwv
            dh, dg1 = rmsnorm_back(dhq + dhk + dhv, cc["xin"],
                                   self.p[f"g1_{l}"], cc["inv1"])
            d[f"g1_{l}"] += dg1
            dx = dx1 + dh

        np.add.at(d["emb"], idx.reshape(-1), dx)
        d["pos"][:T] += dx.reshape(B, T, -1).sum(axis=0)
        return loss, d


# ---------------------------------------------------------------------------
def gradcheck():
    """Differences finies sur un modele minuscule : la retropropagation est-elle juste ?"""
    rng = np.random.default_rng(0)
    m = TinyLM(vocab=11, d_model=16, n_layer=2, h_q=4, h_kv=2, d_ff=24, ctx=8,
               seed=1, dtype=np.float64)
    idx = rng.integers(0, 11, size=(2, 8))
    tgt = rng.integers(0, 11, size=(2, 8))
    loss, g = m.loss_and_grads(idx, tgt)
    worst = 0.0
    for name in ("wq_0", "wk_0", "wv_0", "wo_0", "w1_1", "w2_1", "g1_0", "g2_1",
                 "gf", "pos", "emb"):
        P = m.p[name]
        flat = P.reshape(-1)
        for _ in range(6):
            i = int(rng.integers(0, flat.size))
            old = float(flat[i])
            h = max(1e-6 * abs(old), 1e-6)
            flat[i] = old + h
            lp, _ = m.loss_and_grads(idx, tgt)
            flat[i] = old - h
            lm, _ = m.loss_and_grads(idx, tgt)
            flat[i] = old
            num = (lp - lm) / (2 * h)
            ana = float(g[name].reshape(-1)[i])
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-6)
            worst = max(worst, rel)
    return worst, loss


def main():
    if "--gradcheck" in sys.argv:
        worst, loss = gradcheck()
        print(f"gradcheck : ecart relatif max {worst:.2e} (perte {loss:.4f})")
        print("OK" if worst < 2e-2 else "ECHEC")
        return

    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int32)
    print(f"corpus : {len(text):,} caracteres, vocabulaire {len(chars)}")

    ctx, B = 128, 16
    m = TinyLM(vocab=len(chars), d_model=128, n_layer=2, h_q=4, h_kv=2, d_ff=384,
               ctx=ctx, seed=7)
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 1500
    lr, b1, b2, eps = 3e-3, 0.9, 0.95, 1e-8
    mom = {k: np.zeros_like(v) for k, v in m.p.items()}
    vel = {k: np.zeros_like(v) for k, v in m.p.items()}
    rng = np.random.default_rng(0)
    split = int(0.95 * len(data))
    t0 = time.perf_counter()
    hist = []

    for step in range(1, steps + 1):
        i = rng.integers(0, split - ctx - 1, size=B)
        idx = np.stack([data[j:j + ctx] for j in i])
        tgt = np.stack([data[j + 1:j + ctx + 1] for j in i])
        loss, g = m.loss_and_grads(idx, tgt)
        warm = min(1.0, step / 100.0)
        cur = lr * warm * (0.5 * (1 + np.cos(np.pi * step / steps)) * 0.9 + 0.1)
        for k in m.p:
            mom[k] = b1 * mom[k] + (1 - b1) * g[k]
            vel[k] = b2 * vel[k] + (1 - b2) * g[k] * g[k]
            mh = mom[k] / (1 - b1 ** step)
            vh = vel[k] / (1 - b2 ** step)
            m.p[k] -= (cur * mh / (np.sqrt(vh) + eps)).astype(np.float32)
        if step % 50 == 0 or step == 1:
            hist.append({"step": step, "loss": round(loss, 4)})
            print(f"  step {step:5d}  perte {loss:.4f}  "
                  f"({(time.perf_counter()-t0)/step*1e3:.0f} ms/pas)", flush=True)

    # perte de validation
    vl = []
    for j in range(split, min(len(data) - ctx - 1, split + 40 * ctx), ctx):
        idx = data[j:j + ctx][None]
        tgt = data[j + 1:j + ctx + 1][None]
        vl.append(m.loss_and_grads(idx, tgt)[0])
    print(f"\nperte finale : entrainement {hist[-1]['loss']:.4f}, "
          f"validation {np.mean(vl):.4f}  (hasard = {np.log(len(chars)):.4f})")

    out = RESULTS / "tiny_lm.pkl"
    with out.open("wb") as fh:
        pickle.dump({"params": m.p, "chars": chars,
                     "cfg": dict(vocab=len(chars), d_model=128, n_layer=2, h_q=4,
                                 h_kv=2, d_ff=384, ctx=ctx)}, fh)
    (RESULTS / "tiny_lm_train.json").write_text(json.dumps(
        {"hist": hist, "val": float(np.mean(vl)), "hasard": float(np.log(len(chars))),
         "vocab": len(chars), "corpus": len(text)}, indent=2))
    print(f"-> {out.name} ({out.stat().st_size/1e6:.1f} Mo)")


if __name__ == "__main__":
    main()
