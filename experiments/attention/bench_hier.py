"""Le routage hierarchique tient-il ses promesses ? (cout ET fidelite)

Deux questions, deux mesures, aucune concession :

  1. **Cout** : le nombre de produits scalaires de routage suit-il O(N log N)
     au lieu de O(N^2) ? On compte les operations exactement (pas de chronometre :
     le surcout Python par bloc masquerait la loi d'echelle).

  2. **Fidelite** : un faisceau grossier-vers-fin rate forcement des blocs que
     le score exhaustif aurait retenus. Combien ? On mesure l'accord avec le
     top-k exact ET la masse d'attention reellement capturee — la metrique que
     le memoire audite ne donne jamais.

Sortie : ../results/hier_routing.csv + ../results/hier_routing.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modules as M  # noqa: E402
from hierarchical import HierarchicalIndex, flat_route, qsa_routed  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)


def make_case(N, H_q=8, H_kv=2, d=64, seed=0):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((N, H_q, d)) / np.sqrt(d)).astype(np.float32)
    K = (rng.standard_normal((N, H_kv, d)) / np.sqrt(d)).astype(np.float32)
    V = rng.standard_normal((N, H_kv, d)).astype(np.float32)
    return Q, K, V


def cost_scaling(budget=512, factor=8, beam=None):
    """Comptage exact des operations de routage, plat contre hierarchique."""
    print(f"\n### 1. Cout du routage (budget {budget} tokens, facteur {factor}, "
          f"faisceau {beam or 'minimal'})")
    print(f"{'N':>8} {'plat':>12} {'hier':>12} {'ratio':>8} {'plat/req':>9} {'hier/req':>9}")
    rows = []
    for N in (1024, 2048, 4096, 8192, 16384, 32768):
        Q, K, V = make_case(N)
        _, _, c_flat = qsa_routed(Q, K, V, budget_tokens=budget, router="flat")
        _, _, c_hier = qsa_routed(Q, K, V, budget_tokens=budget, router="hier",
                                  factor=factor, beam=beam)
        nb = N // 4
        rows.append({"N": N, "route_plat": c_flat["route_ops"],
                     "route_hier": c_hier["route_ops"],
                     "attn_ops": c_flat["attn_ops"],
                     "plat_par_requete": c_flat["route_ops"] / nb,
                     "hier_par_requete": c_hier["route_ops"] / nb,
                     "ratio": c_flat["route_ops"] / max(c_hier["route_ops"], 1)})
        r = rows[-1]
        print(f"{N:>8} {r['route_plat']:>12,} {r['route_hier']:>12,} "
              f"{r['ratio']:>8.2f} {r['plat_par_requete']:>9.0f} "
              f"{r['hier_par_requete']:>9.0f}")

    ns = np.array([r["N"] for r in rows], dtype=float)
    for key in ("route_plat", "route_hier"):
        y = np.array([r[key] for r in rows], dtype=float)
        slope = float(np.polyfit(np.log(ns), np.log(y), 1)[0])
        print(f"  exposant empirique {key:11s} : N^{slope:.2f}")
        for r in rows:
            r[f"exp_{key}"] = round(slope, 3)
    return rows


def extrapolate(rows):
    """Extrapolation a 1M tokens des lois d'echelle mesurees."""
    print("\n### 2. Extrapolation a N = 1 000 000 (budget 2048 tokens)")
    ns = np.array([r["N"] for r in rows], dtype=float)
    out = {}
    for key in ("route_plat", "route_hier", "attn_ops"):
        y = np.array([r[key] for r in rows], dtype=float)
        p = np.polyfit(np.log(ns), np.log(y), 1)
        val = float(np.exp(np.polyval(p, np.log(1e6))))
        out[key] = val
        print(f"  {key:11s} : {val:.3e} produits scalaires  (N^{p[0]:.2f})")
    print(f"  -> le routage plat coute {out['route_plat']/out['attn_ops']:.1f}x "
          f"l'attention elle-meme ; le hierarchique {out['route_hier']/out['attn_ops']:.3f}x")
    return out


def fidelity():
    """Ce que la descente en faisceau fait perdre, en accord et en masse."""
    print("\n### 3. Fidelite du routage (N=2048, budget 512 tokens = 128 blocs)")
    N, budget = 2048, 512
    Q, K, V = make_case(N)
    rows = []
    ref_sel = None
    for label, kwargs in [("plat (exhaustif)", {"router": "flat"}),
                          ("hier f=8 beam min", {"router": "hier", "factor": 8}),
                          ("hier f=8 beam 32", {"router": "hier", "factor": 8, "beam": 32}),
                          ("hier f=8 beam 64", {"router": "hier", "factor": 8, "beam": 64}),
                          ("hier f=8 max", {"router": "hier", "factor": 8, "how": "max"}),
                          ("hier f=4 beam min", {"router": "hier", "factor": 4}),
                          ("hier f=16 beam min", {"router": "hier", "factor": 16})]:
        t0 = time.perf_counter()
        _, sel, cost = qsa_routed(Q, K, V, budget_tokens=budget, **kwargs)
        dt = time.perf_counter() - t0
        mean_rec, min_rec = M.attention_mass_recall(Q, K, sel[:256], 4)
        if ref_sel is None:
            ref_sel = sel
            agree = 1.0
        else:
            agree = float(np.mean([
                len(np.intersect1d(a, b)) / max(len(b), 1)
                for a, b in zip(sel, ref_sel)]))
        rows.append({"routeur": label, "ops_routage": cost["route_ops"],
                     "accord_top_k": round(agree, 4),
                     "masse_moyenne": round(mean_rec, 4),
                     "masse_min": round(min_rec, 4), "secondes": round(dt, 2)})
        print(f"  {label:20s} ops={cost['route_ops']:>9,}  accord={agree*100:5.1f} %  "
              f"masse={mean_rec*100:5.1f} % (min {min_rec*100:4.1f} %)")
    return rows


def sanity():
    """Le routeur hierarchique reste-t-il causal et coherent ?"""
    print("\n### 4. Verifications")
    N = 1024
    Q, K, V = make_case(N, seed=3)
    out_h, sel_h, _ = qsa_routed(Q, K, V, budget_tokens=256, router="hier")
    ok_causal = all(int(s.max()) <= b for b, s in enumerate(sel_h) if s.size)
    ok_budget = all(s.size <= 64 for s in sel_h)
    ok_finite = bool(np.isfinite(out_h).all())

    # budget total = tout le contexte -> doit redonner l'attention dense exacte
    out_full, sel_full, _ = qsa_routed(Q, K, V, budget_tokens=N, router="hier")
    ref = M.dense_causal_attention(Q, K, V)
    err = float(np.abs(out_full - ref).max() / np.abs(ref).max())
    print(f"  causalite stricte           : {ok_causal}")
    print(f"  budget respecte             : {ok_budget}")
    print(f"  sortie finie                : {ok_finite}")
    print(f"  budget = N -> attention dense : ecart relatif {err:.2e}")
    return {"causal": ok_causal, "budget": ok_budget, "fini": ok_finite,
            "ecart_dense_budget_total": err}


def main():
    print("=" * 78)
    print("ROUTAGE HIERARCHIQUE — la selection peut-elle etre sous-quadratique ?")
    print("=" * 78)
    cost = cost_scaling(budget=512)
    extra = extrapolate(cost)
    fid = fidelity()
    checks = sanity()

    with (RESULTS / "hier_routing.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cost[0]))
        w.writeheader()
        w.writerows(cost)
    (RESULTS / "hier_routing.json").write_text(json.dumps(
        {"cout": cost, "extrapolation_1M": extra, "fidelite": fid,
         "verifications": checks}, indent=2, default=float))
    print("\n-> results/hier_routing.csv / .json")


if __name__ == "__main__":
    main()
