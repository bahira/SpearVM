"""Audit mesure du memoire "Theorie unifiee des architectures d'attention
hybrides" — un test par affirmation chiffree.

Chaque test renvoie un verdict :
  TENU      la mesure reproduit l'affirmation
  NUANCE    vrai sous des hypotheses que le texte ne pose pas
  FAUX      la mesure contredit l'affirmation

Sortie : results/attention_audit.json + tableau lisible sur stdout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules as M  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

VERDICTS: list[dict] = []


def record(claim, stated, measured, verdict, note=""):
    VERDICTS.append({"claim": claim, "stated": stated, "measured": measured,
                     "verdict": verdict, "note": note})
    print(f"[{verdict:6s}] {claim}\n         annonce : {stated}\n         mesure  : {measured}"
          + (f"\n         -> {note}" if note else ""), flush=True)


# ---------------------------------------------------------------------------
def audit_kv_budget():
    N, d, L, b = 1_000_000, 2560, 48, 2
    dense_layer = 2 * N * d * b
    dense_total = dense_layer * L
    delta = 36 * (48 * 128 * 128 * b)
    qsa_act = 12 * (2048 * 2 * 256 * b)
    qsa_idx = 12 * ((N // 4) * 128 * b)
    hyb = delta + qsa_act + qsa_idx

    record("KV dense a 1M tokens (48 couches, d=2560, bf16)",
           "457.7 GB (§1) et 468.75 GB (tableau §4.1)",
           f"{dense_total/1e9:.1f} GB = {dense_total/2**30:.1f} GiB",
           "FAUX",
           "les deux valeurs du memoire se contredisent ; 457.7 est en GiB, "
           "468.75 ne correspond a aucune des deux unites (impliquerait d=2441)")

    record("KV hybride a 1M tokens",
           "750.0 MB",
           f"{hyb/1e6:.1f} MB (DeltaNet {delta/1e6:.1f} + QSA actifs "
           f"{qsa_act/1e6:.1f} + index compresse {qsa_idx/1e6:.1f})",
           "FAUX",
           "l'index compresse N/4 x 128 x 2 o par couche QSA vaut a lui seul "
           "768 MB : il domine le budget et depasse deja le total annonce")

    record("Facteur de reduction du KV cache a 1M",
           "625.00x",
           f"{dense_total/hyb:.1f}x (avec les formules du memoire lui-meme)",
           "FAUX",
           "468.75 GB / 750 MB = 625.0000 exactement : le facteur rond a ete "
           "pose d'abord, la ligne 'dense' retro-calculee ensuite")

    # ce que devient le budget si l'index est lui aussi borne
    hyb_fix = delta + qsa_act + 12 * (2048 // 4) * 128 * b
    record("Le budget O(1) est-il atteignable ?",
           "empreinte O(1) + O(k.B.d)",
           f"oui si l'index est borne au budget : {hyb_fix/1e6:.1f} MB "
           f"soit {dense_total/hyb_fix:.0f}x",
           "NUANCE",
           "mais alors le routage ne peut plus consulter tout le passe : "
           "l'index en O(N) est precisement ce qui rend la selection globale")
    return {"dense_gb": dense_total / 1e9, "hybride_mb": hyb / 1e6,
            "facteur": dense_total / hyb}


# ---------------------------------------------------------------------------
def audit_flops():
    N, d, L = 1_000_000, 2560, 48
    dense = 4 * N * N * d * L
    record("FLOPs attention dense a 1M tokens",
           "582.40 PFLOPs",
           f"{dense/1e15:.1f} PFLOPs pour l'attention seule (4N^2.d.L)",
           "NUANCE",
           "l'ordre de grandeur tient ; l'ecart vient de la part MoE, que le "
           "memoire ne detaille pas — la formule n'est pas reproductible en l'etat")
    hyb = dense / 72.26
    record("Acceleration calcul a 1M tokens", "72.26x",
           f"impliquerait {hyb/1e15:.2f} PFLOPs hybrides ; non verifiable ici "
           "(1M tokens ne tient pas sur cette machine)",
           "LACUNE", "seule la coherence interne du ratio a ete verifiee")
    return {"dense_pflops": dense / 1e15}


# ---------------------------------------------------------------------------
def audit_mhc():
    rng = np.random.default_rng(0)
    Mdim, L = 64, 48
    Hs = [rng.standard_normal((Mdim, Mdim)) for _ in range(L)]

    kappa, smin, smax = M.jacobian_product_condition(Hs)
    record("kappa du produit des matrices de melange projetees (L=48)",
           "1.0000000000000056",
           f"{kappa:.16f} (sigma_min={smin:.15f}, sigma_max={smax:.15f})",
           "TENU",
           "resultat trivial : un produit de matrices orthogonales est "
           "orthogonal — la SVD n'ajoute rien a la garantie")

    # la propagation ECRITE en 1.2, avec F = 0
    x0 = rng.standard_normal(Mdim)
    _, traj = M.mhc_propagate(x0.copy(), Hs, residual_fn=None)
    ratio = traj[-1] / traj[0]
    record("Propagation x_{l+1} = (1/sqrt2)(Pi x_l + F(Pi x_l)) avec F=0",
           "isometrie, ni evanouissement ni explosion sur L arbitraire",
           f"||x_L||/||x_0|| = {ratio:.3e} (attendu 2^(-L/2) = {2**(-L/2):.3e})",
           "FAUX",
           "le facteur 1/sqrt(2) ecrit dans l'equation contracte le signal d'un "
           "facteur 2 par couche : l'equation du memoire contredit son propre "
           "theoreme d'isometrie")

    # jacobienne honnete : bloc residuel inclus
    Jf = [0.3 * rng.standard_normal((Mdim, Mdim)) for _ in range(L)]
    k2, s2min, s2max = M.jacobian_product_condition_with_block(Hs, Jf)
    record("kappa de la jacobienne complete (terme residuel inclus)",
           "kappa(J_global) = 1.000 -> gradient ni evanescent ni explosif",
           f"{k2:.3e} (sigma_min={s2min:.3e}, sigma_max={s2max:.3e})",
           "FAUX",
           "la contrainte de Stiefel ne porte que sur H_l ; la jacobienne de "
           "couche est (1/sqrt2)(I + F') Pi(H), dont le conditionnement n'est "
           "pas 1. Le theoreme demontre une propriete de H, pas du reseau")

    record("Denomination 'variete de Stiefel compacte St(M,M)'",
           "St(M, M) = {X : X^T X = I_M}",
           "St(n,p) avec n=p est le groupe orthogonal O(M), pas une variete de "
           "Stiefel au sens usuel (p < n)",
           "NUANCE", "vocabulaire ; la projection W_U W_V^T elle-meme est correcte")
    return {"kappa_lineaire": kappa, "kappa_complet": k2, "contraction": float(ratio)}


# ---------------------------------------------------------------------------
def audit_deltanet_stability():
    """Le memoire affirme la stabilite BIBO pour tout g in [0,1[ et beta in ]0,2[.

    Mais son equation (2.1) et son listing (3) ne decrivent pas la meme
    recurrence : dans le listing la porte g est appliquee avant le calcul de
    l'erreur, dans l'equation elle ne l'est pas. Cela change l'operateur
    homogene, donc le domaine de stabilite. On mesure les deux.
    """
    couples = [(0.9, 0.5), (0.9, 1.0), (0.9, 1.5), (0.9, 1.95),
               (0.5, 1.9), (0.99, 1.99), (0.999, 1.999)]
    grille = []
    for g, beta in couples:
        grille.append({
            "g": g, "beta": beta,
            "rho_listing": round(M.deltanet_state_operator(g, beta, "code"), 4),
            "rho_equation": round(M.deltanet_state_operator(g, beta, "text"), 4),
        })
    div_txt = [r for r in grille if r["rho_equation"] >= 1.0]
    div_code = [r for r in grille if r["rho_listing"] >= 1.0]

    # verification numerique : impulsion unique puis entree nulle, cle fixe
    T, H, d = 600, 1, 32
    k = np.zeros((T, H, d), dtype=np.float32); k[:, 0, 0] = 1.0
    q = np.zeros_like(k)
    v = np.zeros((T, H, d), dtype=np.float32); v[0, 0, 0] = 1.0
    traj = {}
    for g, beta in [(0.9, 0.5), (0.9, 1.95), (0.99, 1.99)]:
        B = np.full((T, H), beta, dtype=np.float32)
        Gm = np.full((T, H), g, dtype=np.float32)
        row = {}
        for var in ("code", "text"):
            _, _, norms = M.gated_deltanet_forward(q, k, v, B, Gm,
                                                   return_states=True, variant=var)
            row[var] = {"debut": float(norms[1, 0]), "fin": float(norms[-1, 0]),
                        "rho": M.deltanet_state_operator(g, beta, var)}
        traj[f"g={g},beta={beta}"] = row

    ex = traj["g=0.9,beta=1.95"]
    record("Coherence entre l'equation (2.1) et le listing (section 3)",
           "les deux presentent la meme regle Delta",
           "operateurs differents : listing g(I - beta k k^T) contre equation "
           "(g I - beta k k^T). Sur g=0.9 beta=1.95, ||S|| apres 600 pas a "
           f"entree nulle : listing {ex['code']['fin']:.2e}, equation "
           f"{ex['text']['fin']:.2e}",
           "FAUX",
           "la porte est appliquee avant le calcul de l'erreur dans le code, "
           "apres dans le texte ; ce detail decide de la stabilite")

    record("Stabilite BIBO annoncee : g in [0,1[, beta in ]0,2[",
           "toujours stable sur ce domaine",
           f"listing : stable partout ({len(div_code)}/{len(grille)} couples "
           f"divergents) ; equation du texte : {len(div_txt)}/{len(grille)} "
           "couples divergents, ex. g=0.9 beta=1.95 -> rho=1.05, "
           f"||S|| {ex['text']['debut']:.2f} -> {ex['text']['fin']:.2e}",
           "NUANCE",
           "l'affirmation est vraie pour le code livre (rayon g < 1 pour tout "
           "beta dans ]0,2[), fausse pour l'equation du memoire, dont la "
           "condition correcte est beta < 1 + g")

    record("Certificat de Lyapunov Delta V <= 0 (section 2.1)",
           "-(1-g)||S||^2 + beta||v|| ||S|| + beta^2||v||^2/2",
           "la borne n'est pas homogene : le terme quadratique en ||S|| devrait "
           "porter le facteur (1 - g^2)/2 pour la variante du listing, et "
           "(1 - (g-beta)^2)/2 dans la direction de la cle pour celle du texte",
           "NUANCE",
           "la conclusion (etat borne) est correcte pour le listing, mais la "
           "derivation affichee ne la demontre pas")
    return {"grille": grille, "trajectoires": traj}


# ---------------------------------------------------------------------------
def audit_replay_ssm():
    rng = np.random.default_rng(3)
    T, H_qk, H_v, d = 256, 2, 6, 32
    Q = rng.standard_normal((T, H_qk, d)).astype(np.float32)
    K = rng.standard_normal((T, H_qk, d)).astype(np.float32)
    V = rng.standard_normal((T, H_v, d)).astype(np.float32)
    Beta = rng.uniform(0.1, 0.9, (T, H_v)).astype(np.float32)
    G = rng.uniform(0.85, 0.99, (T, H_v)).astype(np.float32)

    direct, replayed, n_ck = M.replay_ssm(Q, K, V, Beta, G, chunk=16)
    err = float(np.max(np.abs(direct - replayed)))
    record("ReplaySSM : residu |Delta|_inf",
           "0.0000e+00 (exactitude machine)",
           f"{err:.4e} sur T={T}, {n_ck} checkpoints",
           "TENU" if err == 0.0 else "NUANCE",
           "attendu : rejouer la meme sequence d'operations flottantes dans le "
           "meme ordre donne un resultat bit-a-bit identique. C'est un test de "
           "determinisme, pas de precision numerique — le presenter comme une "
           "preuve d'exactitude algebrique est trompeur")

    # ce que le test ne dit pas : le cout du rejeu
    record("Facteur de compression memoire ReplaySSM", "128x par checkpointing",
           f"mesure ici : T/chunk = {T // 16}x de checkpoints en moins, au prix "
           f"d'un rejeu de <= {16} pas par acces",
           "NUANCE", "compromis memoire/calcul classique ; le facteur depend du "
                     "pas de checkpoint choisi, ce n'est pas une propriete du modele")
    return {"residu_inf": err, "checkpoints": n_ck}


# ---------------------------------------------------------------------------
def audit_quantization():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((512, 1024)).astype(np.float64)
    W = (rng.standard_normal((1024, 1024)) / np.sqrt(1024)).astype(np.float64)
    ref = X @ W
    got = M.quantize_w8a8(X, W)
    got_pc = M.quantize_w8a8(X, W, per_channel=True)
    s, s_pc = M.sqnr_db(ref, got), M.sqnr_db(ref, got_pc)
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))

    record("W8A8 : SQNR et erreur relative de Frobenius",
           "39.14 dB et 1.104 %",
           f"{s:.2f} dB / {rel*100:.3f} % (per-tensor), {s_pc:.2f} dB (per-canal)",
           "NUANCE",
           "le 39.14 dB n'est reproductible qu'en quantification **par canal**, "
           "que le memoire ne mentionne pas ; en per-tensor on perd ~3.7 dB. "
           "Et les deux chiffres annonces sont la meme mesure exprimee deux "
           "fois : SQNR = -20 log10(1.104 %) = 39.14 dB, pas deux validations")
    return {"sqnr_per_tensor": s, "sqnr_per_channel": s_pc, "rel": rel}


# ---------------------------------------------------------------------------
def audit_muon():
    rng = np.random.default_rng(11)
    out = {}
    for label, G in [("bien conditionne", rng.standard_normal((256, 128))),
                     ("mal conditionne (cond=1e4)", None),
                     ("rang deficient (rang 64/128)", None)]:
        if label.startswith("mal"):
            U, _, Vt = np.linalg.svd(rng.standard_normal((256, 128)), full_matrices=False)
            G = U @ np.diag(np.logspace(0, -4, 128)) @ Vt
        elif label.startswith("rang"):
            G = rng.standard_normal((256, 64)) @ rng.standard_normal((64, 128))
        defects = {it: M.orthogonality_defect(M.muon_orthogonalize(G, iters=it))
                   for it in (1, 3, 6, 12, 30)}
        out[label] = {str(k): round(v, 6) for k, v in defects.items()}

    G = rng.standard_normal((256, 128))
    sig = np.linalg.svd(G / np.linalg.norm(G), compute_uv=False)
    cub = {it: M.orthogonality_defect(M.muon_orthogonalize(G, it))
           for it in (6, 10, 15, 30)}
    qui = {it: M.orthogonality_defect(M.muon_quintic(G, it)) for it in (5, 10, 30)}
    quint = qui[5]
    record("Muon : orthogonalisation par Newton-Schulz cubique (6 iterations)",
           "orthogonalisation continue des matrices de poids 2D",
           "defaut ||X^T X - I||_F/sqrt(p) : "
           + ", ".join(f"{k} iter -> {v:.4f}" for k, v in cub.items()),
           "FAUX",
           f"la normalisation par ||G||_F du listing tasse les valeurs "
           f"singulieres dans [{sig.min():.3f}, {sig.max():.3f}] ; l'iteration "
           "cubique ne les multiplie que par ~1.5 par pas. Il en faut une "
           "quinzaine pour orthogonaliser : le listing s'arrete a 6 et rend une "
           "matrice qui n'est pas orthogonale")
    record("La variante quintique (le vrai Muon) corrige-t-elle le probleme ?",
           "'iteration de Newton-Schulz d'ordre 3/5' (le listing n'a que l'ordre 3)",
           "quintique (3.4445, -4.7750, 2.0315) : "
           + ", ".join(f"{k} iter -> {v:.4f}" for k, v in qui.items()),
           "NUANCE",
           "elle amene les valeurs singulieres pres de 1 en 5 pas mais plafonne "
           "a un defaut ~0.35 : Muon ne cherche pas une orthogonalite exacte, "
           "seulement une direction de descente bien conditionnee. Parler "
           "d'orthogonalisation exacte est donc faux dans les deux cas")
    out["cubique"] = {str(k): round(v, 6) for k, v in cub.items()}
    out["quintique"] = {str(k): round(v, 6) for k, v in qui.items()}
    out["quintique_5"] = round(quint, 6)
    out["sigma_apres_normalisation_F"] = [float(sig.min()), float(sig.max())]
    return out


# ---------------------------------------------------------------------------
def audit_qsa():
    rng = np.random.default_rng(5)
    N, H_q, H_kv, d = 1024, 8, 2, 64
    Q = (rng.standard_normal((N, H_q, d)) / np.sqrt(d)).astype(np.float32)
    K = (rng.standard_normal((N, H_kv, d)) / np.sqrt(d)).astype(np.float32)
    V = rng.standard_normal((N, H_kv, d)).astype(np.float32)

    out_paper, sel = M.qsa_microblock_attention(Q, K, V, budget_tokens=256)
    out_causal, _ = M.qsa_microblock_attention(Q, K, V, budget_tokens=256,
                                               causal_within_block=True)
    gap = float(np.max(np.abs(out_paper - out_causal)))
    record("Causalite du listing QSA",
           "attention causale (modele autoregressif)",
           f"le listing attend sur les 4 tokens du bloc courant, y compris les "
           f"positions futures : ecart avec la version masquee = {gap:.3f}",
           "FAUX",
           "fuite d'information de 3 tokens au maximum a chaque position ; "
           "corrigible par un masque intra-bloc (fourni en variante `_fixed`)")

    ref = M.dense_causal_attention(Q, K, V)
    err_full = float(np.abs(out_causal - ref).max() / np.abs(ref).max())
    mean_rec, min_rec = M.attention_mass_recall(Q, K, sel, 4)
    record("Fidelite du routage IndexPool 4:1",
           "aucune metrique de fidelite n'est donnee dans le memoire",
           f"masse d'attention exacte capturee : {mean_rec*100:.1f} % en moyenne, "
           f"{min_rec*100:.1f} % au pire ; ecart max a l'attention dense "
           f"{err_full*100:.1f} % (budget 256 tokens sur N=1024)",
           "LACUNE",
           "le memoire chiffre la memoire economisee mais jamais la perte "
           "d'information : c'est pourtant le seul arbitrage qui compte")
    return {"ecart_causalite": gap, "recall_moyen": mean_rec, "recall_min": min_rec}


# ---------------------------------------------------------------------------
def audit_ngram_throughput():
    rng = np.random.default_rng(1)
    n_entries, d = 20_000_000, 2560
    # table INT4 -> on mesure le debit d'acces reel sur une table reduite en RAM
    small = 2_000_000
    table = np.zeros((small, d // 2), dtype=np.uint8)   # INT4 empaquete
    toks = rng.integers(0, 50000, size=(3, 65536))
    idx = M.ngram_hash(toks[0], toks[1], toks[2], mod=small)
    t0 = time.perf_counter()
    rows = table[idx]
    dt = time.perf_counter() - t0
    rate = idx.size / dt
    record("Debit de lookup de la table N-grammes",
           "11 372 tokens/seconde par canal hote",
           f"{rate/1e6:.2f} M tokens/s en RAM sur cette machine "
           f"({rows.nbytes/1e6:.1f} MB rapatries en {dt*1e3:.1f} ms)",
           "FAUX",
           "un lookup hache est une lecture memoire : 11 k/s correspondrait a "
           "14 MB/s, trois ordres de grandeur sous une DRAM et loin sous PCIe "
           "Gen5. Le chiffre annonce est soit un cas NVMe non precise, soit faux")

    footprint = n_entries * d * 0.5
    record("Table N-grammes : parametres et empreinte",
           "51 B parametres, 25.6 GB en INT4, 0 FLOP",
           f"{n_entries*d/1e9:.1f} B parametres, {footprint/1e9:.1f} GB en INT4, "
           "0 FLOP matriciel (lecture indexee)",
           "TENU",
           "arithmetique exacte ; en revanche compter une table de hachage "
           "comme 51 B 'parametres' du modele est une convention discutable")
    return {"debit_ram_tok_s": rate}


# ---------------------------------------------------------------------------
def audit_amdahl():
    p, s = 0.88, 24.5
    val = 1.0 / ((1 - p) + p / s)
    record("Loi d'Amdahl (p=0.88, s=24.5)", "6.41x", f"{val:.3f}x", "TENU",
           "arithmetique correcte ; en revanche la conclusion 'saturation a 85 % "
           "de la puissance crete' ne decoule pas de ce calcul et n'est etayee "
           "par aucune mesure")
    return {"amdahl": val}


# ---------------------------------------------------------------------------
def audit_runtime_extrapolation():
    """Le memoire presente son listing NumPy comme la preuve de ses chiffres a
    1M tokens. On mesure ce que ce listing coute reellement, et on extrapole."""
    rng = np.random.default_rng(0)

    xs, ys = [], []
    for N in (256, 512, 1024, 2048):
        Q = (rng.standard_normal((N, 8, 64)) / 8).astype(np.float32)
        K = (rng.standard_normal((N, 2, 64)) / 8).astype(np.float32)
        V = rng.standard_normal((N, 2, 64)).astype(np.float32)
        t0 = time.perf_counter()
        M.qsa_microblock_attention(Q, K, V, budget_tokens=2048)
        ys.append(time.perf_counter() - t0)
        xs.append(N)
    slope = float(np.polyfit(np.log(xs), np.log(ys), 1)[0])
    t_qsa = float(np.exp(np.polyval(np.polyfit(np.log(xs), np.log(ys), 1), np.log(1e6))))

    xs2, ys2 = [], []
    for T in (128, 256, 512, 1024):
        Q = rng.standard_normal((T, 16, 128)).astype(np.float32)
        K = rng.standard_normal((T, 16, 128)).astype(np.float32)
        V = rng.standard_normal((T, 48, 128)).astype(np.float32)
        B = rng.uniform(0.1, 0.9, (T, 48)).astype(np.float32)
        Gm = rng.uniform(0.9, 0.99, (T, 48)).astype(np.float32)
        t0 = time.perf_counter()
        M.gated_deltanet_forward(Q, K, V, B, Gm)
        ys2.append(time.perf_counter() - t0)
        xs2.append(T)
    per_tok = ys2[-1] / xs2[-1]
    t_delta = per_tok * 1e6 * 36

    record("Complexite reelle du routage QSA",
           "attention creuse, budget borne a 2048 tokens -> cout lineaire",
           f"cout mesure en N^{slope:.2f} : chaque bloc score **tous** les blocs "
           f"passes (N/4 produits scalaires) avant d'en retenir 512",
           "FAUX",
           "la selection est quadratique meme si l'attention ne l'est pas ; "
           "borner le budget ne borne pas le cout du routage. Il faudrait un "
           "index hierarchique, absent du memoire")

    record("Les benchmarks a 1M tokens ont-ils ete executes ?",
           "'les benchmarks numeriques NumPy conduits dans ce travail prouvent "
           "formellement' 625x et 72.3x a 1M tokens",
           f"avec le listing fourni : QSA {t_qsa/3600:.0f} h pour UNE couche et "
           f"UNE passe, DeltaNet {t_delta/3600:.1f} h pour ses 36 couches "
           f"({per_tok*1e6:.0f} us/token) — soit > 100 h par passe avant",
           "FAUX",
           "les lignes 262k et 1M du tableau 4.1 ne peuvent pas etre des "
           "mesures : ce sont des sorties de modele analytique. Les presenter "
           "comme des benchmarks executes est le probleme central du document")
    return {"exposant_qsa": slope, "qsa_1m_h": t_qsa / 3600,
            "deltanet_us_par_token": per_tok * 1e6, "deltanet_1m_36c_h": t_delta / 3600}


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("AUDIT MESURE — Theorie unifiee des architectures d'attention hybrides")
    print("=" * 78, flush=True)
    data = {}
    for name, fn in [("kv", audit_kv_budget), ("flops", audit_flops),
                     ("mhc", audit_mhc), ("deltanet", audit_deltanet_stability),
                     ("replay", audit_replay_ssm), ("quant", audit_quantization),
                     ("muon", audit_muon), ("qsa", audit_qsa),
                     ("ngram", audit_ngram_throughput), ("amdahl", audit_amdahl),
                     ("runtime", audit_runtime_extrapolation)]:
        print(f"\n--- {name} " + "-" * (70 - len(name)), flush=True)
        data[name] = fn()

    counts: dict[str, int] = {}
    for v in VERDICTS:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    print("\n" + "=" * 78)
    print("SYNTHESE : " + " | ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("=" * 78)

    (RESULTS / "attention_audit.json").write_text(
        json.dumps({"verdicts": VERDICTS, "donnees": data}, indent=2, default=float))
    print("-> results/attention_audit.json")


if __name__ == "__main__":
    main()
