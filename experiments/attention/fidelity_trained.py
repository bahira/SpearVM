"""Fidelite de l'attention creuse sur un modele ENTRAINE.

C'est la mesure qui manquait. Sur poids aleatoires, les tetes d'un groupe GQA
sont independantes et aucune selection partagee ne peut marcher (docs/
ATTENTION_AUDIT.md section 8). La question est de savoir si l'entrainement change
cela — et elle ne se tranche que sur des poids appris.

Quatre mesures, de la plus mecanique a la plus decisive :

  1. concentration : entropie et masse top-k de l'attention apprise ;
  2. accord entre tetes d'un meme groupe GQA — le facteur limitant identifie ;
  3. structure par blocs : la masse se regroupe-t-elle en zones contigues ?
  4. **cout en perplexite** : on remplace l'attention dense par le noyau creux
     dans le modele et on mesure la perte sur du texte de validation.

Sortie : results/fidelity_trained.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import spur_math as sm  # noqa: E402
from tiny_lm import TinyLM, load_corpus, gelu, rmsnorm  # noqa: E402

RESULTS = HERE.parent / "results"


def load_model():
    with (RESULTS / "tiny_lm.pkl").open("rb") as fh:
        blob = pickle.load(fh)
    cfg = blob["cfg"]
    m = TinyLM(**cfg, seed=0)
    m.p = blob["params"]
    return m, blob["chars"], cfg


def batches(data, ctx, n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        j = int(rng.integers(0, len(data) - ctx - 1))
        yield data[j:j + ctx][None], data[j + 1:j + ctx + 1][None]


# ---------------------------------------------------------------------------
def mesure_concentration(m, data, n_batch=8):
    """Entropie, masse top-k, accord inter-tetes, structure par blocs."""
    ent, topk, agree, blocky = [], [], [], []
    T = m.ctx
    for idx, _ in batches(data, T, n_batch, seed=1):
        keep = []
        m.forward(idx, keep=keep)
        for rec in keep:
            P = rec["P"][0]                      # (h_q, T, T)
            for t in range(T // 2, T, 8):        # positions avec assez de passe
                w = P[:, t, :t + 1]
                w = w / w.sum(axis=1, keepdims=True)
                e = -(w * np.log(w + 1e-12)).sum(axis=1) / np.log(max(t + 1, 2))
                ent.extend(e.tolist())
                kk = max(1, (t + 1) // 4)        # budget = 25 % du passe
                topk.extend(np.sort(w, axis=1)[:, -kk:].sum(axis=1).tolist())
                tops = [set(np.argsort(w[h])[-kk:].tolist()) for h in range(m.h_q)]
                for g in range(m.h_kv):
                    hs = range(g * m.rep, (g + 1) * m.rep)
                    for a in hs:
                        for b in hs:
                            if a < b:
                                agree.append(len(tops[a] & tops[b]) / kk)
                # structure par blocs de 4 : part de la masse dans les blocs
                # les mieux notes, contre une redistribution aleatoire
                nb = (t + 1) // 4
                if nb >= 8:
                    bm = w[:, :nb * 4].reshape(m.h_q, nb, 4).sum(axis=2)
                    top_b = np.sort(bm, axis=1)[:, -max(1, nb // 4):].sum(axis=1)
                    blocky.extend(top_b.tolist())
    return {"entropie_moy": float(np.mean(ent)),
            "entropie_min": float(np.min(ent)),
            "masse_top25pct": float(np.mean(topk)),
            "accord_intra_groupe": float(np.mean(agree)) if agree else float("nan"),
            "masse_blocs_top25pct": float(np.mean(blocky)) if blocky else float("nan"),
            "n_mesures": len(ent)}


# ---------------------------------------------------------------------------
def forward_creux(m, idx, budget, block=4):
    """Meme passe avant, mais l'attention passe par le noyau creux du depot."""
    B, T = idx.shape
    assert B == 1
    x = np.ascontiguousarray(
        m.p["emb"][idx.reshape(-1)] + m.p["pos"][:T], dtype=np.float32)
    tile = 8
    for l in range(m.L):
        h, _ = rmsnorm(x, m.p[f"g1_{l}"])
        h = np.ascontiguousarray(h, dtype=np.float32)
        q = sm.matmul_nt(h, m.p[f"wq_{l}"]).reshape(T, m.h_q, m.dh)
        k = np.ascontiguousarray(
            sm.matmul_nt(h, m.p[f"wk_{l}"]).reshape(T, m.h_kv, m.dh))
        v = np.ascontiguousarray(
            sm.matmul_nt(h, m.p[f"wv_{l}"]).reshape(T, m.h_kv, m.dh))
        cache = sm.KVCache(k, v).build_index(block=block, factor=8)
        out = np.empty((T, m.h_q, m.dh), dtype=np.float32)
        for p0 in range(0, T, tile):
            n = min(tile, T - p0)
            qt = np.ascontiguousarray(q[p0:p0 + n])
            if budget <= 0 or p0 == 0:
                lens = np.arange(p0 + 1, p0 + n + 1, dtype=np.int32)
                out[p0:p0 + n] = cache.attend(qt, lengths=lens)
            else:
                out[p0:p0 + n] = cache.attend_sparse(qt, pos=p0, budget=budget)[0]
        a = np.ascontiguousarray(out.reshape(T, -1))
        x = x + sm.matmul_nt(a, m.p[f"wo_{l}"])
        h2, _ = rmsnorm(x, m.p[f"g2_{l}"])
        h2 = np.ascontiguousarray(h2, dtype=np.float32)
        x = x + sm.matmul_nt(gelu(sm.matmul_nt(h2, m.p[f"w1_{l}"])), m.p[f"w2_{l}"])
    hf, _ = rmsnorm(x, m.p["gf"])
    return sm.matmul_nt(np.ascontiguousarray(hf, dtype=np.float32), m.p["emb"])


def forward_reference(m, idx, budget, mode):
    """Memes budgets, mais selection triviale — pour savoir si le routage sert.

    mode="recent" : on garde les `budget` tokens les plus recents ;
    mode="hasard" : on tire `budget` tokens au hasard dans le passe (+ la
    fenetre locale de 8, comme le noyau creux, pour ne pas les desavantager).
    """
    T = idx.shape[1]
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(m.p["emb"][idx.reshape(-1)] + m.p["pos"][:T],
                             dtype=np.float32)
    for l in range(m.L):
        h, _ = rmsnorm(x, m.p[f"g1_{l}"])
        h = np.ascontiguousarray(h, dtype=np.float32)
        q = sm.matmul_nt(h, m.p[f"wq_{l}"]).reshape(T, m.h_q, m.dh)
        k = sm.matmul_nt(h, m.p[f"wk_{l}"]).reshape(T, m.h_kv, m.dh)
        v = sm.matmul_nt(h, m.p[f"wv_{l}"]).reshape(T, m.h_kv, m.dh)
        out = np.zeros((T, m.h_q, m.dh), dtype=np.float64)
        for t in range(T):
            lo = max(0, t + 1 - budget) if mode == "recent" else 0
            if mode == "recent":
                keep = np.arange(lo, t + 1)
            else:
                loc = np.arange(max(0, t - 7), t + 1)
                pool = np.arange(0, max(0, t - 7))
                extra = rng.choice(pool, size=min(len(pool), max(budget - 8, 0)),
                                   replace=False) if len(pool) else np.array([], int)
                keep = np.union1d(loc, extra)
            for hh in range(m.h_q):
                g = hh // m.rep
                sc = (q[t, hh].astype(np.float64) @ k[keep, g].astype(np.float64).T)
                sc /= np.sqrt(m.dh)
                sc -= sc.max()
                w = np.exp(sc)
                out[t, hh] = (w / w.sum()) @ v[keep, g].astype(np.float64)
        a = np.ascontiguousarray(out.reshape(T, -1), dtype=np.float32)
        x = x + sm.matmul_nt(a, m.p[f"wo_{l}"])
        h2, _ = rmsnorm(x, m.p[f"g2_{l}"])
        h2 = np.ascontiguousarray(h2, dtype=np.float32)
        x = x + sm.matmul_nt(gelu(sm.matmul_nt(h2, m.p[f"w1_{l}"])), m.p[f"w2_{l}"])
    hf, _ = rmsnorm(x, m.p["gf"])
    return sm.matmul_nt(np.ascontiguousarray(hf, dtype=np.float32), m.p["emb"])


def perte(logits, tgt):
    z = logits.astype(np.float64)
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    f = tgt.reshape(-1)
    return float(-np.log(p[np.arange(len(f)), f] + 1e-12).mean())


def mesure_perplexite(m, data, budgets, n_batch=12):
    print(f"\n{'budget':>8}{'% du contexte':>15} {'perte':>9}{'vs dense':>10} "
          f"{'perplexite':>12}{'surcout':>9}")
    ref = []
    rows = []
    for idx, tgt in batches(data, m.ctx, n_batch, seed=2):
        ref.append(perte(forward_creux(m, idx, budget=0), tgt))
    base = float(np.mean(ref))
    rows.append({"budget": "dense", "pct": 100.0, "perte": base, "delta": 0.0,
                 "ppl": float(np.exp(base)), "surcout_ppl": 0.0})
    print(f"{'dense':>8}{100.0:>14.0f} % {base:9.4f}{0.0:>10.4f} "
          f"{np.exp(base):>12.3f}{0.0:>8.1f} %")
    for b in budgets:
        ls = []
        for idx, tgt in batches(data, m.ctx, n_batch, seed=2):
            ls.append(perte(forward_creux(m, idx, budget=b), tgt))
        lo = float(np.mean(ls))
        rows.append({"budget": b, "pct": 100.0 * b / m.ctx, "perte": lo,
                     "delta": lo - base, "ppl": float(np.exp(lo)),
                     "surcout_ppl": 100.0 * (np.exp(lo) / np.exp(base) - 1)})
        r = rows[-1]
        print(f"{b:>8}{r['pct']:>14.0f} % {lo:9.4f}{r['delta']:>10.4f} "
              f"{r['ppl']:>12.3f}{r['surcout_ppl']:>8.1f} %")
    return rows


def main():
    m, chars, cfg = load_model()
    text = load_corpus()
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi.get(c, 0) for c in text], dtype=np.int32)
    split = int(0.95 * len(data))
    val = data[split:]
    print("=" * 82)
    print(f"Fidelite sur poids APPRIS — {cfg['n_layer']} couches, "
          f"{cfg['h_q']} tetes Q / {cfg['h_kv']} KV, contexte {cfg['ctx']}")
    print("=" * 82)

    print("\n### 1-3. Ce que l'entrainement a produit")
    c = mesure_concentration(m, val)
    print(f"  entropie moyenne de l'attention        : {c['entropie_moy']:.3f} "
          f"(1.0 = uniforme)")
    print(f"  masse dans le quart le mieux note      : {c['masse_top25pct']:.3f}")
    print(f"  masse dans le quart des BLOCS de 4     : {c['masse_blocs_top25pct']:.3f}")
    print(f"  accord des tops entre tetes d'un groupe: "
          f"{c['accord_intra_groupe']*100:.1f} %")
    print(f"  ({c['n_mesures']} mesures)")

    print("\n### 4. Cout en perplexite du noyau creux (mesure decisive)")
    rows = mesure_perplexite(m, val, budgets=[64, 32, 16, 8])

    print("\n### 5. Le routage sert-il, face aux selections triviales ?")
    base = rows[0]["perte"]
    comp = []
    print(f"{'budget':>8} {'index (C)':>11}{'recents':>10}{'hasard':>10}"
          f"   surcout de perplexite")
    for b in (32, 16, 8):
        ours = [r for r in rows if r["budget"] == b][0]["surcout_ppl"]
        vals = {}
        for mode in ("recent", "hasard"):
            ls = [perte(forward_reference(m, idx, b, mode), tgt)
                  for idx, tgt in batches(val, m.ctx, 6, seed=2)]
            vals[mode] = 100.0 * (np.exp(np.mean(ls)) / np.exp(base) - 1)
        comp.append({"budget": b, "index": ours, **vals})
        print(f"{b:>8} {ours:>10.1f} %{vals['recent']:>9.1f} %"
              f"{vals['hasard']:>9.1f} %")

    (RESULTS / "fidelity_trained.json").write_text(json.dumps(
        {"config": cfg, "concentration": c, "perplexite": rows,
         "references": comp}, indent=2))
    print("\n-> results/fidelity_trained.json")


if __name__ == "__main__":
    main()
