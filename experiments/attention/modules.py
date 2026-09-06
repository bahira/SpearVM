"""Reimplementation fidele des modules du memoire "Theorie unifiee des
architectures d'attention hybrides" (R. Abdel-Aal, 27 aout 2026).

Regle du dossier : on code **ce que le texte dit**, y compris ses equations
telles qu'ecrites, sans les corriger silencieusement. Les variantes corrigees
sont fournies a cote, suffixees `_fixed`, pour pouvoir mesurer l'ecart entre ce
qui est affirme et ce qui tient.

Rien ici n'est un jugement : `audit.py` mesure, ce fichier ne fait qu'exposer
les operateurs sous une forme testable.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 2.1 Gated DeltaNet
# ---------------------------------------------------------------------------


def gated_deltanet_forward(Q, K, V, Beta, G, return_states=False, variant="code"):
    """Les deux lectures possibles du memoire — elles ne coincident pas.

    variant="code" (listing section 3) : la porte est appliquee AVANT de
        calculer l'erreur, donc l'erreur se mesure contre l'etat deja amorti
            S_t = g S_{t-1} + beta (v_t - g S_{t-1} k~) k~^T
                = g S_{t-1} (I - beta k~ k~^T) + beta v_t k~^T
        operateur homogene : g (I - beta k k^T), rayon g*max(1, |1-beta|).

    variant="text" (equation de la section 2.1) : e_t = v_t - S_{t-1} k~_t est
        defini avec l'etat NON amorti
            S_t = g S_{t-1} + beta (v_t - S_{t-1} k~) k~^T
                = S_{t-1} (g I - beta k~ k~^T) + beta v_t k~^T
        operateur homogene : g I - beta k k^T, rayon max(|g|, |g - beta|).

    Q:(T,H_qk,d) K:(T,H_qk,d) V:(T,H_v,d) Beta:(T,H_v) G:(T,H_v)
    """
    T, H_v, d_h = V.shape
    H_qk = Q.shape[1]
    ratio = H_v // H_qk

    K_rep = np.repeat(K, ratio, axis=1)
    Q_rep = np.repeat(Q, ratio, axis=1)
    K_norm = K_rep / (np.linalg.norm(K_rep, axis=-1, keepdims=True) + 1e-7)

    S = np.zeros((H_v, d_h, d_h), dtype=np.float32)
    Out = np.zeros((T, H_v, d_h), dtype=np.float32)
    norms = np.zeros((T, H_v), dtype=np.float64) if return_states else None

    for t in range(T):
        if variant == "code":
            S = G[t, :, None, None] * S
            retrieved = np.einsum("hij,hj->hi", S, K_norm[t])
            delta_v = V[t] - retrieved
        else:  # "text"
            retrieved = np.einsum("hij,hj->hi", S, K_norm[t])
            delta_v = V[t] - retrieved
            S = G[t, :, None, None] * S
        S = S + Beta[t, :, None, None] * np.einsum("hi,hj->hij", delta_v, K_norm[t])
        Out[t] = np.einsum("hij,hj->hi", S, Q_rep[t])
        if return_states:
            norms[t] = np.linalg.norm(S.reshape(H_v, -1), axis=1)

    return (Out, S, norms) if return_states else (Out, S)


def deltanet_state_operator(g: float, beta: float, variant: str = "code") -> float:
    """Rayon spectral de la recurrence homogene (||k|| = 1).

    "code" : g (I - beta k k^T)  -> valeurs propres g et g(1 - beta)
    "text" : g I - beta k k^T    -> valeurs propres g et g - beta
    """
    if variant == "code":
        return max(abs(g), abs(g * (1.0 - beta)))
    return max(abs(g), abs(g - beta))


# ---------------------------------------------------------------------------
# 1.2 mHC : projection sur {X : X^T X = I}
# ---------------------------------------------------------------------------


def project_stiefel_manifold(W: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(W, full_matrices=False)
    return U @ Vt


def mhc_propagate(x, Hs, residual_fn=None, scale=1.0 / np.sqrt(2.0)):
    """Propagation **telle qu'ecrite** en section 1.2 :

        x_{l+1} = scale * ( Pi(H_l) x_l + F(Pi(H_l) x_l) )

    `residual_fn=None` correspond a F == 0 (le cas ou la borne spectrale du
    memoire est censee etre exacte).
    """
    traj = [np.linalg.norm(x)]
    for H in Hs:
        y = project_stiefel_manifold(H) @ x
        x = scale * (y + (residual_fn(y) if residual_fn is not None else 0.0))
        traj.append(np.linalg.norm(x))
    return x, np.asarray(traj)


def jacobian_product_condition(Hs) -> tuple[float, float, float]:
    """kappa du produit des seules matrices de melange projetees."""
    P = np.eye(Hs[0].shape[0])
    for H in Hs:
        P = project_stiefel_manifold(H) @ P
    s = np.linalg.svd(P, compute_uv=False)
    return float(s.max() / s.min()), float(s.min()), float(s.max())


def jacobian_product_condition_with_block(Hs, Jf, scale=1.0 / np.sqrt(2.0)):
    """kappa du produit des **vraies** jacobiennes de couche.

    J_l = scale * (I + F'(.)) Pi(H_l) : c'est la jacobienne de la propagation
    ecrite en 1.2, terme residuel inclus.
    """
    M = Hs[0].shape[0]
    P = np.eye(M)
    for H, Jl in zip(Hs, Jf):
        P = (scale * (np.eye(M) + Jl) @ project_stiefel_manifold(H)) @ P
    s = np.linalg.svd(P, compute_uv=False)
    return float(s.max() / max(s.min(), 1e-300)), float(s.min()), float(s.max())


# ---------------------------------------------------------------------------
# 2.2 QSA / IndexPool
# ---------------------------------------------------------------------------


def qsa_microblock_attention(Q, K, V, micro_block_size=4, budget_tokens=2048,
                             seed=42, causal_within_block=False):
    """Version du memoire. `causal_within_block=True` ajoute le masque causal
    intra-bloc qui manque dans le listing original."""
    N, H_q, d_h = Q.shape
    H_kv = K.shape[1]
    num_blocks = N // micro_block_size
    budget_blocks = min(budget_tokens // micro_block_size, num_blocks)
    d_idx = 128

    rng = np.random.default_rng(seed)
    W_idx_k = (rng.standard_normal((H_kv * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)
    W_idx_q = (rng.standard_normal((H_q * d_h, d_idx)) / np.sqrt(d_h)).astype(np.float32)

    K_idx = (K.reshape(N, -1) @ W_idx_k).reshape(num_blocks, micro_block_size, d_idx).mean(1)
    Q_idx = (Q.reshape(N, -1) @ W_idx_q).reshape(num_blocks, micro_block_size, d_idx).mean(1)

    rep = H_q // H_kv
    out = np.zeros_like(Q)
    selected = []
    for b in range(num_blocks):
        q_sub = Q[b * micro_block_size:(b + 1) * micro_block_size]
        scores = K_idx[:b + 1] @ Q_idx[b]
        k_blocks = min(budget_blocks, b + 1)
        sel = np.argsort(scores)[-k_blocks:]
        selected.append(sel)
        idx = (sel[:, None] * micro_block_size + np.arange(micro_block_size)).ravel()

        K_sel = np.repeat(K[idx], rep, axis=1)
        V_sel = np.repeat(V[idx], rep, axis=1)
        attn = np.einsum("thd,khd->hkt", q_sub, K_sel) / np.sqrt(d_h)

        if causal_within_block:
            # position absolue de chaque cle selectionnee vs position de la requete
            key_pos = idx[:, None]                                  # (k,1)
            q_pos = b * micro_block_size + np.arange(micro_block_size)[None, :]
            attn = np.where(key_pos <= q_pos, attn, -np.inf)

        m = np.max(attn, axis=1, keepdims=True)
        e = np.exp(attn - m)
        w = e / (np.sum(e, axis=1, keepdims=True) + 1e-8)
        out[b * micro_block_size:(b + 1) * micro_block_size] = np.einsum("hkt,khd->thd", w, V_sel)
    return out, selected


def dense_causal_attention(Q, K, V):
    """Reference exacte (softmax causale, GQA), en float64."""
    N, H_q, d_h = Q.shape
    rep = H_q // K.shape[1]
    Kr = np.repeat(K, rep, axis=1).astype(np.float64)
    Vr = np.repeat(V, rep, axis=1).astype(np.float64)
    Qd = Q.astype(np.float64)
    out = np.zeros((N, H_q, d_h), dtype=np.float64)
    mask = np.triu(np.full((N, N), -np.inf), 1)
    for h in range(H_q):
        s = (Qd[:, h] @ Kr[:, h].T) / np.sqrt(d_h) + mask
        s -= s.max(axis=1, keepdims=True)
        w = np.exp(s)
        w /= w.sum(axis=1, keepdims=True)
        out[:, h] = w @ Vr[:, h]
    return out


def attention_mass_recall(Q, K, selected, micro_block_size, causal=True, offset=0):
    """Fraction de la masse d'attention exacte capturee par les blocs choisis.

    C'est la metrique qui dit si le routage IndexPool est fidele : le memoire
    ne la mesure jamais.

    `offset` : indice du premier bloc de `selected`. Mesurer sur les premiers
    blocs d'une sequence n'a aucun sens — ils tiennent entierement dans le
    budget, donc la masse vaut trivialement 100 %.
    """
    N, H_q, d_h = Q.shape
    rep = H_q // K.shape[1]
    Kr = np.repeat(K, rep, axis=1).astype(np.float64)
    Qd = Q.astype(np.float64)
    recalls = []
    for b0, sel in enumerate(selected):
        b = b0 + offset          # indice reel du bloc de requetes
        keep = np.zeros(N, dtype=bool)
        idx = (np.asarray(sel)[:, None] * micro_block_size
               + np.arange(micro_block_size)).ravel()
        keep[idx] = True
        for t in range(b * micro_block_size, (b + 1) * micro_block_size):
            lim = t + 1 if causal else N
            for h in range(H_q):                      # masse par tete, pas agregee
                s = (Kr[:lim, h] @ Qd[t, h]) / np.sqrt(d_h)
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                recalls.append(float(w[keep[:lim]].sum()))
    return float(np.mean(recalls)), float(np.min(recalls))


# ---------------------------------------------------------------------------
# 2.5 Muon / Newton-Schulz
# ---------------------------------------------------------------------------


def muon_orthogonalize(G: np.ndarray, iters: int = 6) -> np.ndarray:
    rows, cols = G.shape
    transposed = False
    if rows < cols:
        G, rows, cols = G.T, cols, rows
        transposed = True
    X = G / (np.linalg.norm(G) + 1e-7)
    Id = np.eye(cols, dtype=G.dtype)
    for _ in range(iters):
        X = X @ (1.5 * Id - 0.5 * (X.T @ X))
    return X.T if transposed else X


def orthogonality_defect(X: np.ndarray) -> float:
    """||X^T X - I||_F / sqrt(p) : 0 <=> colonnes orthonormees."""
    p = X.shape[1]
    return float(np.linalg.norm(X.T @ X - np.eye(p)) / np.sqrt(p))


# ---------------------------------------------------------------------------
# 4.2 ReplaySSM (recalcul d'etat par checkpoints)
# ---------------------------------------------------------------------------


def replay_ssm(Q, K, V, Beta, G, chunk: int):
    """Rejoue la recurrence a partir de checkpoints espaces de `chunk` pas.

    On stocke S tous les `chunk` pas, puis on reconstruit chaque S_t manquant en
    rejouant les pas depuis le checkpoint le plus proche, dans le meme ordre.
    """
    T = V.shape[0]
    ckpts = {}
    S = np.zeros((V.shape[1], V.shape[2], V.shape[2]), dtype=np.float32)
    ratio = V.shape[1] // Q.shape[1]
    K_rep = np.repeat(K, ratio, axis=1)
    K_norm = K_rep / (np.linalg.norm(K_rep, axis=-1, keepdims=True) + 1e-7)

    def advance(S, t):
        S = G[t, :, None, None] * S
        r = np.einsum("hij,hj->hi", S, K_norm[t])
        return S + Beta[t, :, None, None] * np.einsum("hi,hj->hij", V[t] - r, K_norm[t])

    for t in range(T):
        if t % chunk == 0:
            ckpts[t] = S.copy()
        S = advance(S, t)
    final_direct = S

    # reconstruction du dernier etat depuis le checkpoint le plus proche
    start = ((T - 1) // chunk) * chunk
    S_r = ckpts[start].copy()
    for t in range(start, T):
        S_r = advance(S_r, t)
    return final_direct, S_r, len(ckpts)


# ---------------------------------------------------------------------------
# 4.2 Quantisation W8A8
# ---------------------------------------------------------------------------


def quantize_w8a8(X: np.ndarray, W: np.ndarray, per_channel: bool = False):
    """GEMM Y = X W en int8 symetrique, accumulation int32 -> dequantification."""
    def qsym(A, axis=None):
        amax = np.max(np.abs(A), axis=axis, keepdims=axis is not None)
        scale = np.maximum(amax, 1e-12) / 127.0
        q = np.clip(np.rint(A / scale), -127, 127).astype(np.int8)
        return q, scale

    xq, sx = qsym(X, axis=1 if per_channel else None)
    wq, sw = qsym(W, axis=0 if per_channel else None)
    acc = xq.astype(np.int32) @ wq.astype(np.int32)
    return acc.astype(np.float64) * (sx * sw)


def sqnr_db(ref: np.ndarray, got: np.ndarray) -> float:
    err = np.linalg.norm(got - ref)
    return float(20.0 * np.log10(np.linalg.norm(ref) / max(err, 1e-300)))


# ---------------------------------------------------------------------------
# 2.4 Table N-grammes
# ---------------------------------------------------------------------------


def ngram_hash(w2, w1, w0, p=1_000_003, mod=20_000_000):
    return (w2.astype(np.int64) * p * p + w1.astype(np.int64) * p + w0) % mod


def muon_quintic(G: np.ndarray, iters: int = 5) -> np.ndarray:
    """Variante quintique reellement utilisee par Muon (Jordan et al.).

    Coefficients (3.4445, -4.7750, 2.0315) : ils sacrifient la convergence
    asymptotique pour amplifier tres vite les petites valeurs singulieres, ce
    que l'iteration cubique du memoire ne fait pas.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    rows, cols = G.shape
    transposed = rows < cols
    if transposed:
        G = G.T
        rows, cols = cols, rows
    X = G / (np.linalg.norm(G) + 1e-7)
    for _ in range(iters):
        A = X.T @ X
        B = b * A + c * (A @ A)
        X = a * X + X @ B
    return X.T if transposed else X
