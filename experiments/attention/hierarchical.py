"""Index hierarchique pour le routage QSA — contre-proposition mesuree.

Constat de l'audit (docs/ATTENTION_AUDIT.md §1) : borner le budget d'attention
ne borne pas le cout de la **selection**. Dans le listing du memoire, chaque
bloc de requetes score tous les blocs passes :

    cout_routage(N) = sum_b (b) = O((N/B)^2)   ->  mesure en N^1.92

L'attention elle-meme est bien bornee (budget k blocs), mais le routage ne
l'est pas. On propose ici l'index manquant : un arbre de resumes construit par
pooling successif, parcouru en faisceau du grossier vers le fin.

    niveau 0 : N/B blocs      (resumes IndexPool 4:1 du memoire)
    niveau 1 : N/(B*f) noeuds (pooling de f enfants)
    ...
    cout_routage = O( (N/B) * beam * f * profondeur ) = O(N log N)

Deux fonctions de resume sont implementees (moyenne et maximum par
coordonnee) : laquelle preserve le mieux le classement est une question de
mesure, pas de gout — `bench_hier.py` tranche.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _pool(x: np.ndarray, factor: int, how: str) -> np.ndarray:
    """Regroupe `factor` lignes consecutives en une seule (dernier groupe partiel
    autorise)."""
    n, d = x.shape
    pad = (-n) % factor
    if pad:
        fill = -np.inf if how == "max" else 0.0
        x = np.concatenate([x, np.full((pad, d), fill, dtype=x.dtype)], axis=0)
        if how != "max":                       # moyenne : ne pas diluer avec des zeros
            counts = np.full((x.shape[0] // factor, 1), factor, dtype=x.dtype)
            counts[-1] = factor - pad
            return x.reshape(-1, factor, d).sum(1) / counts
    g = x.reshape(-1, factor, d)
    return g.max(1) if how == "max" else g.mean(1)


@dataclass
class HierarchicalIndex:
    """Arbre de resumes sur les cles compressees (niveau 0 = blocs du memoire)."""

    levels: list[np.ndarray]
    factor: int
    how: str
    ops: int = field(default=0)      # nombre de produits scalaires effectues

    @classmethod
    def build(cls, block_keys: np.ndarray, factor: int = 8, how: str = "mean"):
        levels = [block_keys]
        while levels[-1].shape[0] > factor:
            levels.append(_pool(levels[-1], factor, how))
        return cls(levels=levels, factor=factor, how=how)

    # -- routage -----------------------------------------------------------
    def route(self, q: np.ndarray, last_block: int, k_blocks: int,
              beam: int | None = None) -> np.ndarray:
        """Renvoie <= k_blocks indices de blocs fins, causaux (<= last_block).

        Descente en faisceau : a chaque niveau on garde `beam` noeuds, on les
        developpe, et on ne score que leurs enfants.
        """
        # Faisceau minimal viable : il faut que beam * factor >= k_blocks pour
        # pouvoir encore restituer k blocs fins apres le dernier developpement.
        # Prendre beam = k_blocks (premier reflexe) fait scorer tout l'index :
        # mesure a l'appui, cela coute plus cher que le routage plat.
        if beam is None:
            beam = max(-(-k_blocks // self.factor), 8)
        top = len(self.levels) - 1
        n_top = self.levels[top].shape[0]
        # noeuds du niveau le plus grossier couvrant [0, last_block]
        span_top = self.factor ** top
        cand = np.arange(min(n_top, last_block // span_top + 1))

        for lvl in range(top, -1, -1):
            nodes = self.levels[lvl]
            cand = cand[cand < nodes.shape[0]]
            if cand.size == 0:
                return np.zeros(0, dtype=np.int64)
            scores = nodes[cand] @ q
            self.ops += cand.size
            keep = beam if lvl > 0 else k_blocks
            if cand.size > keep:
                cand = cand[np.argpartition(scores, -keep)[-keep:]]
                scores = nodes[cand] @ q
                self.ops += cand.size
            if lvl == 0:
                order = np.argsort(scores)
                return np.sort(cand[order[-k_blocks:]])
            # developpement vers les enfants, borne par la causalite
            child = (cand[:, None] * self.factor + np.arange(self.factor)).ravel()
            span_child = self.factor ** (lvl - 1)
            cand = np.unique(child[child * span_child <= last_block])
        return np.zeros(0, dtype=np.int64)


def flat_route(block_keys: np.ndarray, q: np.ndarray, last_block: int,
               k_blocks: int) -> tuple[np.ndarray, int]:
    """Routage du memoire : score exhaustif de tous les blocs passes."""
    scores = block_keys[:last_block + 1] @ q
    k = min(k_blocks, scores.size)
    sel = np.argpartition(scores, -k)[-k:]
    return np.sort(sel), int(scores.size)


# ---------------------------------------------------------------------------
def qsa_routed(Q, K, V, micro_block_size=4, budget_tokens=2048, seed=42,
               router="flat", factor=8, how="mean", beam=None):
    """QSA causale (bug intra-bloc corrige) avec routage au choix.

    Renvoie (sortie, blocs_selectionnes, statistiques de cout).
    """
    N, H_q, d_h = Q.shape
    H_kv = K.shape[1]
    nb = N // micro_block_size
    budget_blocks = min(budget_tokens // micro_block_size, nb)
    d_idx = 128

    rng = np.random.default_rng(seed)
    W_k = (rng.standard_normal((H_kv * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)
    W_q = (rng.standard_normal((H_q * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)
    K_idx = (K.reshape(N, -1) @ W_k).reshape(nb, micro_block_size, d_idx).mean(1)
    Q_idx = (Q.reshape(N, -1) @ W_q).reshape(nb, micro_block_size, d_idx).mean(1)

    index = HierarchicalIndex.build(K_idx, factor=factor, how=how) if router == "hier" else None
    rep = H_q // H_kv
    out = np.zeros_like(Q)
    selected: list[np.ndarray] = []
    route_ops = 0
    attn_ops = 0

    for b in range(nb):
        if router == "hier":
            before = index.ops
            sel = index.route(Q_idx[b], b, budget_blocks, beam=beam)
            route_ops += index.ops - before
        else:
            sel, used = flat_route(K_idx, Q_idx[b], b, budget_blocks)
            route_ops += used
        selected.append(sel)

        idx = (sel[:, None] * micro_block_size + np.arange(micro_block_size)).ravel()
        q_sub = Q[b * micro_block_size:(b + 1) * micro_block_size]
        K_sel = np.repeat(K[idx], rep, axis=1)
        V_sel = np.repeat(V[idx], rep, axis=1)
        attn = np.einsum("thd,khd->hkt", q_sub, K_sel) / np.sqrt(d_h)

        q_pos = b * micro_block_size + np.arange(micro_block_size)[None, :]
        attn = np.where(idx[:, None] <= q_pos, attn, -np.inf)   # causalite stricte
        m = np.max(attn, axis=1, keepdims=True)
        m = np.where(np.isfinite(m), m, 0.0)
        e = np.exp(attn - m)
        w = e / (np.sum(e, axis=1, keepdims=True) + 1e-8)
        out[b * micro_block_size:(b + 1) * micro_block_size] = np.einsum(
            "hkt,khd->thd", w, V_sel)
        attn_ops += idx.size * micro_block_size * H_q

    return out, selected, {"route_ops": route_ops, "attn_ops": attn_ops}
