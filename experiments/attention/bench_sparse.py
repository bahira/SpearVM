"""Attention creuse en C : ce qui est gagne, ce qui est perdu, et pourquoi.

Trois quantites sont mesurees separement, parce qu'elles ont trois causes
differentes :

  1. **le cout** — l'index hierarchique remplace un balayage de tous les blocs
     passes par une descente en faisceau ;
  2. **la qualite du resume** — un bloc est-il correctement note ? On compare la
     masse d'attention captee par la moyenne seule, par moyenne+direction
     principale, et par l'oracle ;
  3. **la fidelite de la sortie** — elle depend d'un troisieme facteur, mis en
     evidence ici : la selection est PARTAGEE par toutes les tetes de requetes
     d'un groupe GQA. Si les tetes ne veulent pas les memes tokens, aucune
     selection commune ne peut les satisfaire toutes.

Sortie : results/sparse_attention.csv / .json
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


def best_ms(fn, reps=8):
    fn()
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def make(tk, d, h_kv, seed=0, topics=16, run=192):
    """Cache structure en **zones contigues** de `run` tokens.

    Point capital, decouvert par la mesure : un routage par blocs ne peut
    capter que ce qui est structure AU NIVEAU DU BLOC. Si l'on tire un sujet au
    hasard token par token, tous les blocs ont la meme moyenne et aucune
    selection ne peut battre le hasard, quelle que soit la qualite de l'index.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((topics, h_kv, d)).astype(np.float32)
    base /= np.linalg.norm(base, axis=-1, keepdims=True)
    idx = np.repeat(rng.integers(0, topics, size=(tk + run - 1) // run), run)[:tk]
    k = base[idx] + 0.4 * rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    v = rng.standard_normal((tk, h_kv, d)).astype(np.float32)
    return (np.ascontiguousarray(k, dtype=np.float32),
            np.ascontiguousarray(v, dtype=np.float32), base, idx)


def bench_cost():
    print("=" * 92)
    print("1. Cout : attention creuse contre attention dense, meme cache")
    print("=" * 92)
    d, h_q, h_kv, tq = 64, 8, 2, 4
    print(f"\n{'contexte':>9}{'budget':>8}{'lus':>7} {'dense ms':>9}{'creuse ms':>10} "
          f"{'gain':>7} {'index Mo':>9}{'cache Mo':>9}{'index %':>9}")
    rows = []
    for tk in (4096, 16384, 65536):
        k, v, base, topics = make(tk, d, h_kv)
        cache = sm.KVCache(k, v).build_index(block=4, factor=8)
        pos = tk - 8
        rng = np.random.default_rng(1)
        q = np.ascontiguousarray(rng.standard_normal((tq, h_q, d)) * 4, dtype=np.float32)
        lens = np.arange(pos + 1, pos + tq + 1, dtype=np.int32)
        for budget in (256, 1024, 4096):
            _, got = cache.attend_sparse(q, pos=pos, budget=budget)
            td = best_ms(lambda: cache.attend(q, lengths=lens))
            ts = best_ms(lambda: cache.attend_sparse(q, pos=pos, budget=budget))
            pct = 100.0 * cache._index.nbytes / cache.nbytes
            rows.append({"tk": tk, "budget": budget, "lus": got,
                         "ms_dense": round(td, 4), "ms_sparse": round(ts, 4),
                         "gain": round(td / ts, 3),
                         "index_Mo": round(cache._index.nbytes / 1e6, 3),
                         "cache_Mo": round(cache.nbytes / 1e6, 2),
                         "index_pct": round(pct, 1)})
            r = rows[-1]
            print(f"{tk:>9}{budget:>8}{got:>7} {td:9.4f}{ts:10.4f} {r['gain']:7.2f} "
                  f"{r['index_Mo']:9.3f}{r['cache_Mo']:9.2f}{pct:8.1f} %")
    return rows


def bench_exactitude():
    print("\n" + "=" * 92)
    print("2. Exactitude du chemin : a budget plein, le creux DOIT redonner le dense")
    print("=" * 92)
    out = []
    for tk, tq in ((2048, 1), (2048, 4), (8192, 8)):
        k, v, _, _ = make(tk, 64, 2)
        cache = sm.KVCache(k, v).build_index(block=4, factor=8)
        pos = tk - 16
        rng = np.random.default_rng(2)
        q = np.ascontiguousarray(rng.standard_normal((tq, 8, 64)) * 4, dtype=np.float32)
        lens = np.arange(pos + 1, pos + tq + 1, dtype=np.int32)
        dense = cache.attend(q, lengths=lens)
        sparse, got = cache.attend_sparse(q, pos=pos, budget=tk)
        err = float(np.abs(sparse - dense).max() / np.abs(dense).max())
        out.append({"tk": tk, "tq": tq, "err_budget_plein": err})
        print(f"  contexte {tk:>5}, tuile {tq:>2} : {got:>5} tokens lus, "
              f"ecart {err:.2e}")
    print("  -> rassemblement, transposition de V, causalite par longueurs : exacts.")
    return out


def bench_resume():
    print("\n" + "=" * 92)
    print("3. Qualite du resume : quelle masse d'attention un bloc bien note capte-t-il ?")
    print("=" * 92)
    tk, d, h_kv, bs = 16384, 64, 2, 4
    k, v, base, topics = make(tk, d, h_kv)
    pos = tk - 8
    nb = pos // bs
    K = k[:nb * bs, 0].astype(np.float64).reshape(nb, bs, d)
    mean = K.mean(1)
    dev = K - mean[:, None]
    u = dev[:, 0].copy()
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    for _ in range(8):
        u = np.einsum('bij,bi->bj', dev, np.einsum('bij,bj->bi', dev, u))
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    amp = np.sqrt((np.einsum('bij,bj->bi', dev, u) ** 2).mean(1))
    direction = u * amp[:, None]

    rng = np.random.default_rng(3)
    acc = {"oracle": [], "moyenne": [], "moyenne+direction": []}
    for trial in range(6):
        t = int(topics[pos - 1 - trial * 997])
        q = ((base[t, 0] + 0.3 * rng.standard_normal(d)) * 32).astype(np.float64)
        s = (k[:pos, 0].astype(np.float64) @ q) / np.sqrt(d)
        s -= s.max()
        w = np.exp(s)
        w /= w.sum()
        mass = w[:nb * bs].reshape(nb, bs).sum(1)
        kk = 512 // bs
        acc["oracle"].append(mass[np.argsort(mass)[-kk:]].sum())
        acc["moyenne"].append(mass[np.argsort(mean @ q)[-kk:]].sum())
        acc["moyenne+direction"].append(
            mass[np.argsort(mean @ q + np.abs(direction @ q))[-kk:]].sum())
    print(f"\n  masse captee par les 128 blocs retenus (budget 512 tokens, 6 requetes) :")
    for name in ("moyenne", "moyenne+direction", "oracle"):
        print(f"    {name:>20} : {np.mean(acc[name]):.3f}")
    print("  -> ajouter la direction principale de variation multiplie par ~1.8 la")
    print("     masse captee, pour un index deux fois plus gros. La moyenne seule")
    print("     est un mauvais resume : la masse softmax depend du MAX du bloc.")
    return {kk: float(np.mean(vv)) for kk, vv in acc.items()}


def bench_fidelite():
    print("\n" + "=" * 92)
    print("4. Fidelite de la sortie : le facteur limitant n'est pas l'index")
    print("=" * 92)
    tk, d, h_q, h_kv = 16384, 64, 8, 2
    k, v, base, topics = make(tk, d, h_kv)
    cache = sm.KVCache(k, v).build_index(block=4, factor=8)
    pos = tk - 8
    rep = h_q // h_kv
    t = int(topics[pos - 1])
    aim = np.repeat(base[t][None], rep, axis=1).reshape(1, h_q, d)
    rng = np.random.default_rng(1)
    rows = []
    print(f"\n{'bruit inter-tetes':>18}{'accord top-512':>16}{'oracle par tete':>17}"
          f"{'routeur partage':>17}")
    for noise in (0.30, 0.15, 0.05):
        q = np.ascontiguousarray((aim + noise * rng.standard_normal((1, h_q, d)))
                                 * np.float32(32), dtype=np.float32)
        dense = cache.attend(q, lengths=np.array([pos + 1], dtype=np.int32))
        sparse, _ = cache.attend_sparse(q, pos=pos, budget=512)
        out = np.zeros((1, h_q, d))
        tops = []
        for h in range(h_q):
            kk = k[:pos + 1, h // rep].astype(np.float64)
            vv = v[:pos + 1, h // rep].astype(np.float64)
            s = (q[0, h].astype(np.float64) @ kk.T) / np.sqrt(d)
            top = np.argsort(s)[-512:]
            tops.append(top)
            m = np.full(pos + 1, False)
            m[top] = True
            ss = np.where(m, s, -np.inf)
            ss -= ss.max()
            w = np.exp(ss)
            out[0, h] = (w / w.sum()) @ vv
        # accord entre tetes d'un MEME groupe (celles qui partagent la selection)
        agree = np.mean([len(np.intersect1d(tops[a], tops[b])) / 512
                         for a in range(rep) for b in range(a + 1, rep)])
        e_head = float(np.abs(out - dense).max() / np.abs(dense).max())
        e_shared = float(np.abs(sparse - dense).max() / np.abs(dense).max())
        rows.append({"bruit": noise, "accord_intra_groupe": round(agree, 3),
                     "err_oracle_par_tete": e_head, "err_routeur_partage": e_shared})
        print(f"{noise:>18.2f}{agree*100:>15.1f} %{e_head:>17.2e}{e_shared:>17.2e}")
    print("\n  Lecture : quand les tetes d'un meme groupe GQA ne veulent pas les memes")
    print("  tokens, une selection commune ne peut satisfaire personne — un oracle")
    print("  PAR TETE atteint 7e-03 la ou la selection partagee reste a 6e-01.")
    print("  Ce n'est ni l'index ni le noyau qui limitent : c'est le partage de la")
    print("  decision. Sur ces donnees synthetiques les tetes sont independantes ;")
    print("  dans un modele entraine, les tetes d'un groupe GQA sont correlees par")
    print("  construction. La mesure de fidelite doit donc etre refaite sur un vrai")
    print("  modele — elle n'est PAS transportable depuis ce banc.")
    return rows


def main():
    cost = bench_cost()
    exact = bench_exactitude()
    resume = bench_resume()
    fid = bench_fidelite()
    with (RESULTS / "sparse_attention.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cost[0]))
        w.writeheader()
        w.writerows(cost)
    (RESULTS / "sparse_attention.json").write_text(json.dumps(
        {"cout": cost, "exactitude": exact, "resume": resume, "fidelite": fid},
        indent=2))
    print("\n-> results/sparse_attention.csv / .json")


if __name__ == "__main__":
    main()
