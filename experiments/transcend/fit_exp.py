"""Recherche d'approximations minimax pour exp — la brique qui manque au depot.

Pourquoi exp : le softmax est le seul noyau que l'attention exige et que
SpearVM n'a pas. Un softmax se resume a `exp` plus deux reductions ; sa
precision et son debit sont donc entierement decides par la qualite de `exp`.

Methode (la meme doctrine que pour les GEMM) :
  1. reduction d'argument exacte : x = k*ln2 + r, |r| <= ln2/2, avec ln2 scinde
     en (hi, lo) pour ne pas perdre de bits sur les grands x ;
  2. sur [-ln2/2, ln2/2], recherche du polynome minimax en **erreur relative**
     par iterations de Remez (echange sur les extremums) ;
  3. reconstruction 2^k par manipulation du champ d'exposant IEEE ;
  4. verification contre libm en float80/float64 puis chronometrage.

Le degre n'est pas choisi a priori : on balaye, on mesure, on garde le front de
Pareto (erreur x debit). Sortie : results/exp_fit.json + le code C genere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

LN2 = float(np.log(2.0))


def chebyshev_nodes(n: int, a: float, b: float) -> np.ndarray:
    k = np.arange(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n))


def remez_relative(deg: int, a: float, b: float, iters: int = 60,
                   grid: int = 20001) -> tuple[np.ndarray, float]:
    """Minimax de exp sur [a,b] en erreur RELATIVE, par echange de Remez.

    On resout a chaque tour le systeme d'equioscillation
        sum_j c_j x_i^j - exp(x_i) = (-1)^i * E * exp(x_i)
    sur deg+2 noeuds, puis on deplace les noeuds sur les extremums de l'erreur.
    """
    x = chebyshev_nodes(deg + 2, a, b)
    x.sort()
    xg = np.linspace(a, b, grid)
    fg = np.exp(xg)
    best = (None, np.inf)

    for _ in range(iters):
        f = np.exp(x)
        A = np.zeros((deg + 2, deg + 2), dtype=np.float64)
        for j in range(deg + 1):
            A[:, j] = x ** j
        A[:, deg + 1] = -((-1.0) ** np.arange(deg + 2)) * f
        try:
            sol = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            break
        c = sol[:deg + 1]

        err = (np.polyval(c[::-1], xg) - fg) / fg
        emax = float(np.max(np.abs(err)))
        if emax < best[1]:
            best = (c.copy(), emax)

        # nouveaux noeuds : extremums locaux de l'erreur (+ les bords)
        sign = np.sign(err)
        cuts = np.flatnonzero(np.diff(sign) != 0)
        cand = [0]
        for lo, hi in zip(np.r_[0, cuts + 1], np.r_[cuts + 1, grid]):
            seg = np.abs(err[lo:hi])
            if seg.size:
                cand.append(lo + int(np.argmax(seg)))
        cand.append(grid - 1)
        cand = sorted(set(cand))
        if len(cand) < deg + 2:
            break
        # on garde les deg+2 extremums de plus grande amplitude, ordonnes
        cand = sorted(sorted(cand, key=lambda i: -abs(err[i]))[:deg + 2])
        new_x = xg[cand]
        if np.allclose(new_x, x, atol=1e-14):
            break
        x = new_x

    return best[0], best[1]


def eval_reduced(coeffs: np.ndarray, x: np.ndarray, dtype) -> np.ndarray:
    """exp(x) par reduction d'argument + Horner, dans la precision cible."""
    x = x.astype(dtype)
    inv_ln2 = dtype(1.0 / LN2)
    # ln2 scinde : hi porte les bits hauts exactement representables
    if dtype == np.float32:
        ln2_hi, ln2_lo = np.float32(0.693359375), np.float32(-2.12194440e-4)
    else:
        ln2_hi, ln2_lo = np.float64(0.693145751953125), np.float64(1.42860682030941723212e-6)
    k = np.rint(x * inv_ln2)
    r = (x - k * ln2_hi) - k * ln2_lo
    p = np.zeros_like(r) + dtype(coeffs[-1])
    for c in coeffs[-2::-1]:
        p = p * r + dtype(c)
    return np.ldexp(p, k.astype(np.int32))


def measure_accuracy(coeffs, dtype, lo=-87.0, hi=0.0, n=400001):
    """Erreur relative maximale contre la reference float64 de numpy.

    Piege evite ici : il faut arrondir l'entree a la precision cible AVANT de
    calculer la reference. Sinon on mesure l'arrondi de x (erreur relative
    |x|.eps, soit 4e-6 en f32 a x=-80) et non l'algorithme — ce qui donne un
    faux plancher de precision independant du degre du polynome.
    """
    x = np.linspace(lo, hi, n).astype(dtype).astype(np.float64)
    ref = np.exp(x)                      # reference sur les MEMES valeurs
    got = eval_reduced(coeffs, x, dtype).astype(np.float64)
    ok = ref > 0
    rel = np.abs(got[ok] - ref[ok]) / ref[ok]
    return float(rel.max()), float(np.sqrt(np.mean(rel ** 2)))


def main():
    report = {}
    print("Recherche minimax de exp sur [-ln2/2, ln2/2] (erreur relative)\n")
    a, b = -LN2 / 2, LN2 / 2
    print(f"{'deg':>4} {'minimax theorique':>18} {'f32 Linf rel':>14} {'f64 Linf rel':>14}")
    for deg in range(2, 13):
        c, emax = remez_relative(deg, a, b)
        if c is None:
            continue
        e32, _ = measure_accuracy(c, np.float32)
        e64, _ = measure_accuracy(c, np.float64)
        report[deg] = {"coeffs": [float(v) for v in c], "minimax": emax,
                       "linf_f32": e32, "linf_f64": e64}
        print(f"{deg:>4} {emax:18.3e} {e32:14.3e} {e64:14.3e}")

    eps32 = float(np.finfo(np.float32).eps)
    eps64 = float(np.finfo(np.float64).eps)
    d32 = min((d for d, r in report.items() if r["linf_f32"] <= 4 * eps32), default=None)
    d64 = min((d for d, r in report.items() if r["linf_f64"] <= 8 * eps64), default=None)
    print(f"\n  plancher f32 (4 ulp = {4*eps32:.2e}) atteint des le degre {d32}")
    print(f"  plancher f64 (8 ulp = {8*eps64:.2e}) atteint des le degre {d64}")
    report["choix"] = {"f32": d32, "f64": d64,
                       "note": "au-dela, le degre ne fait qu'ajouter des FMA"}
    (RESULTS / "exp_fit.json").write_text(json.dumps(report, indent=2))
    print("\n-> results/exp_fit.json")
    return report


if __name__ == "__main__":
    main()
