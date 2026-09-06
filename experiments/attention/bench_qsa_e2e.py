"""QSA bout-en-bout : le listing du memoire contre la version corrigee + noyaux.

Quatre variantes, memes entrees, meme budget :

  1. `memoire`      : le listing tel qu'ecrit (routage plat, bug causal inclus) ;
  2. `corrige`      : routage hierarchique + masque intra-bloc, attention numpy ;
  3. `spur`         : idem + `spur_math.attention_tile` (GEMM NT + softmax C) ;
  4. `spur_groupe`  : idem, mais G blocs de requetes consecutifs partagent une
                      seule decision de routage — le tile passe de 4 a 4G lignes,
                      ce qui sort le noyau du regime ou l'appel domine.

La variante 4 echange de la qualite contre du debit : on mesure les deux.

Sortie : results/qsa_e2e.csv / .json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import modules as M  # noqa: E402
from hierarchical import HierarchicalIndex, flat_route  # noqa: E402

import spur_math as sm  # noqa: E402

RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
B = 4  # taille du micro-bloc


def make_case(N, H_q=8, H_kv=2, d=64, seed=0):
    rng = np.random.default_rng(seed)
    Q = (rng.standard_normal((N, H_q, d)) / np.sqrt(d)).astype(np.float32)
    K = (rng.standard_normal((N, H_kv, d)) / np.sqrt(d)).astype(np.float32)
    V = rng.standard_normal((N, H_kv, d)).astype(np.float32)
    return Q, K, V


def _index_projections(Q, K, seed=42, d_idx=128):
    N, H_q, d_h = Q.shape
    H_kv = K.shape[1]
    rng = np.random.default_rng(seed)
    W_k = (rng.standard_normal((H_kv * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)
    W_q = (rng.standard_normal((H_q * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)
    nb = N // B
    K_idx = (K.reshape(N, -1) @ W_k).reshape(nb, B, d_idx).mean(1)
    Q_idx = (Q.reshape(N, -1) @ W_q).reshape(nb, B, d_idx).mean(1)
    return K_idx, Q_idx


def run_variant(Q, K, V, budget_tokens, variant, group=1):
    """Renvoie (sortie, selections, secondes)."""
    N, H_q, d = Q.shape
    H_kv = K.shape[1]
    rep = H_q // H_kv
    nb = N // B
    k_blocks = min(budget_tokens // B, nb)

    t0 = time.perf_counter()
    if variant == "memoire":
        out, sel = M.qsa_microblock_attention(Q, K, V, budget_tokens=budget_tokens)
        return out, sel, time.perf_counter() - t0

    K_idx, Q_idx = _index_projections(Q, K)
    index = HierarchicalIndex.build(K_idx, factor=8)
    out = np.zeros_like(Q)
    sel_all: list[np.ndarray] = []
    step = group

    for g0 in range(0, nb, step):
        g1 = min(g0 + step, nb)
        last = g1 - 1
        qsig = Q_idx[g0:g1].mean(0) if step > 1 else Q_idx[g0]
        sel = index.route(qsig, last, k_blocks)
        for _ in range(g0, g1):
            sel_all.append(sel)
        if sel.size == 0:
            continue
        idx = (sel[:, None] * B + np.arange(B)).ravel()
        rows = slice(g0 * B, g1 * B)
        q_sub = Q[rows]
        tq = q_sub.shape[0]
        pos = np.arange(g0 * B, g1 * B)
        # nombre de cles visibles : celles dont la position <= celle de la requete
        lens = np.searchsorted(idx, pos, side="right").astype(np.int32)

        if variant == "corrige":
            Ks = np.repeat(K[idx], rep, axis=1)
            Vs = np.repeat(V[idx], rep, axis=1)
            s = np.einsum("thd,khd->htk", q_sub, Ks) / np.sqrt(d)
            m = idx[None, None, :] > pos[None, :, None]
            s = np.where(m, -np.inf, s)
            s -= s.max(axis=2, keepdims=True)
            e = np.exp(s)
            w = e / (e.sum(axis=2, keepdims=True) + 1e-30)
            out[rows] = np.einsum("htk,khd->thd", w, Vs).astype(np.float32)
        else:  # noyaux SpearVM, une tuile par tete
            for h in range(H_q):
                kk = np.ascontiguousarray(K[idx, h // rep])
                vv = np.ascontiguousarray(V[idx, h // rep])
                out[rows, h] = sm.attention_tile(
                    np.ascontiguousarray(q_sub[:, h]), kk, vv, lengths=lens)
    return out, sel_all, time.perf_counter() - t0


def main():
    print("=" * 78)
    print("QSA bout-en-bout : listing du memoire contre version corrigee + noyaux")
    print("=" * 78)
    rows = []
    budget = 512
    for N in (2048, 4096, 8192):
        Q, K, V = make_case(N)
        base = None
        print(f"\n--- N = {N}, budget {budget} tokens")
        for label, variant, group in [("memoire", "memoire", 1),
                                      ("corrige (numpy)", "corrige", 1),
                                      ("spur", "spur", 1),
                                      ("spur groupe 4", "spur", 4),
                                      ("spur groupe 16", "spur", 16)]:
            out, sel, dt = run_variant(Q, K, V, budget, variant, group)
            if base is None:
                base = dt
            # sur les DERNIERS blocs : les premiers tiennent dans le budget,
            # leur masse vaut 100 % par construction et ne mesure rien.
            tail = 48
            mean_rec, min_rec = M.attention_mass_recall(
                Q, K, sel[-tail:], B, offset=len(sel) - tail)
            rows.append({"N": N, "variante": label, "secondes": round(dt, 4),
                         "tokens_par_s": round(N / dt, 1),
                         "gain_vs_memoire": round(base / dt, 3),
                         "masse_moyenne": round(mean_rec, 4),
                         "masse_min": round(min_rec, 4)})
            r = rows[-1]
            print(f"  {label:16s} {dt*1e3:8.1f} ms  {r['tokens_par_s']:9.0f} tok/s  "
                  f"x{r['gain_vs_memoire']:5.2f}  masse {mean_rec*100:5.1f} %")

    print("\n### Debit mesure (et pourquoi on n'extrapole pas ici)")
    big = max(r["N"] for r in rows)
    for label in ("memoire", "spur", "spur groupe 16"):
        sub = [r for r in rows if r["variante"] == label]
        tail = [r for r in sub if r["N"] == big][0]
        serie = " -> ".join(f"{r['tokens_par_s']:.0f}" for r in sub)
        print(f"  {label:16s} : {serie} tok/s (N croissant), "
              f"soit {tail['tokens_par_s']:.0f} tok/s a N={big}")
    print("  Le debit MONTE avec N pour les variantes groupees : le surcout")
    print("  Python par tuile s'amortit. Un ajustement en loi de puissance")
    print("  donnerait un exposant < 1, ce qui n'a aucun sens physique — le")
    print("  terme de routage finira par dominer. On ne l'extrapole donc pas :")
    print("  la loi d'echelle rigoureuse est celle du COMPTAGE d'operations")
    print("  (bench_hier.py), pas celle du temps de paroi a ces tailles.")

    with (RESULTS / "qsa_e2e.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (RESULTS / "qsa_e2e.json").write_text(json.dumps(rows, indent=2))
    print("\n-> results/qsa_e2e.csv")


if __name__ == "__main__":
    main()
