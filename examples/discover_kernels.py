# -*- coding: utf-8 -*-
"""SPEAR Auto-Kernel Factory v2 — Découverte hybride polyfit+GP.
Phase 1 : polyfit pour l'initialisation (optimal global)
Phase 2 : GP raffine les coefficients pour minimiser l'erreur MAX
C'est plus rapide et plus fiable que GP seul."""
import numpy as np
import math, json, time

def discover(name, exact_vec, lo, hi, degree, n_evals=2000, gp_gens=100):
    """Découvre un polynôme optimal par polyfit + raffinement GP."""
    x = np.linspace(lo, hi, n_evals)
    y_exact = exact_vec(x)

    # Phase 1 : least-squares fit initial
    coeffs = np.polyfit(x, y_exact, degree)  # ordre décroissant

    def eval_poly(c, xv):
        return np.polyval(c, xv)

    # Phase 2 : raffinement par évolution locale
    rng = np.random.RandomState(42)
    best_err = float(np.max(np.abs(eval_poly(coeffs,x)-y_exact)))
    best_c = coeffs.copy()

    perturbation_scale = 0.001 * (np.max(np.abs(coeffs))+1)
    for gen in range(gp_gens):
        # génère un candidat muté
        candidate = best_c.copy()
        idx = rng.randint(0, len(candidate))
        candidate[idx] += rng.normal(0, perturbation_scale)

        # évalue
        y_app = eval_poly(candidate, x)
        err = np.max(np.abs(y_app - y_exact))

        if err < best_err:
            best_err = err
            best_c = candidate.copy()
            perturbation_scale *= 0.99  # réduit progressivement

    return best_c, best_err


def main():
    print("="*60)
    print("  SPEAR Auto-Kernel Factory v2")
    print("  Polyfit initial + raffinement évolutionnaire")
    print("="*60)

    kernels = [
        ("sigmoid", lambda x: 0.5*(1+np.tanh(0.5*x)), -6, 6, 9),
        ("gelu",    lambda x: 0.997729*(x*np.minimum(1.002,np.maximum(0,0.306923*x+0.501)))-0.004004,
                    -0.6, 1.8, 7),
        ("erf",     lambda x: np.vectorize(math.erf)(x), -2, 2, 8),
        ("tanh",    np.tanh, -3, 3, 7),
    ]

    results = []
    for name, fn, lo, hi, deg in kernels:
        print(f"\n[{name}] découverte...")
        t0 = time.time()

        x_eval = np.linspace(lo, hi, 500)
        y_exact = fn(x_eval)

        # phase 1 : polyfit
        coeffs = np.polyfit(x_eval, y_exact, deg)
        err_init = np.max(np.abs(np.polyval(coeffs, x_eval) - y_exact))

        # phase 2 : raffinement évolutionnaire
        rng = np.random.RandomState(42)
        best_c = coeffs.copy()
        best_err = err_init

        for gen in range(200):
            c_test = best_c.copy()
            # mutation gaussienne sur un coefficient aléatoire
            idx = rng.randint(0, len(c_test))
            c_test[idx] += rng.normal(0, 0.01)

            y_app = np.polyval(c_test, x_eval)
            err = np.max(np.abs(y_app - y_exact))

            if err < best_err:
                best_err = err
                best_c = c_test.copy()

        elapsed = time.time() - t0

        # certification fine
        x_fine = np.linspace(lo,hi,5000)
        cert_err = np.max(np.abs(np.polyval(best_c,x_fine) - fn(x_fine)))

        print(f"  init_err={err_init:.2e} -> refined={cert_err:.2e} ({elapsed:.1f}s)")
        print(f"  degré={deg}, ops={deg+1}")

        results.append({
            "name":name, "degree":deg, "err_max":float(cert_err),
            "coeffs":[float(c) for c in best_c],
            "domain":[lo,hi]
        })

    # export C
    print("\n=== Kernels découverts ===")
    c_code = "/* Auto-discovered by SPEAR Kernel Factory */\n"
    for r in results:
        coeffs_str = ",".join(f"{c:.15f}" for c in r["coeffs"])
        fn_name = f"spur_gp_{r['name']}"
        c_code += f"\ndouble {fn_name}(double x) {{\n"
        c_code += f"    double c[] = {{{coeffs}}};\n"
        c_code += f"    double r = c[{len(r['coeffs'])-1}];\n"
        for i in range(len(r["coeffs"])-2,-1,-1):
            c_code += f"    r = r*x + c[{i}];\n"
        c_code += "    return r;\n}\n"

    with open("discovered_kernels.c","w") as f: f.write(c_code)
    with open("discovered_kernels.json","w") as f:
        import json; json.dump(results,f,indent=2,default=str)

    print(f"\nExportés: discovered_kernels.c + .json")

if __name__=="__main__":
    main()
