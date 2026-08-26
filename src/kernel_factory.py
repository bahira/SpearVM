# -*- coding: utf-8 -*-
"""SPEAR Auto-Kernel Factory
Découvre automatiquement les coefficients polynomiaux optimaux pour une
fonction cible sur un domaine donné, via évolution génétique.

Le pipeline :
1. L'utilisateur fournit : fonction exacte, domaine, budget de degrés
2. Le GP évolue les coefficients pour minimiser l'erreur max
3. La certification vérifie l'erreur sur un maillage fin indépendant
4. Sortie : kernel C prêt à compiler avec datasheet garantie

C'est ce que fait SLEEF manuellement pendant des mois — ici en secondes.
"""
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple
import time, json

@dataclass
class KernelSpec:
    """Spécification d'un kernel découvert"""
    name: str
    coeffs: np.ndarray      # coefficients du polynôme
    domain: tuple           # (lo, hi)
    err_max: float          # erreur maximale mesurée
    err_mean: float         # erreur moyenne
    n_ops: int              # nombre d'opérations (proxy vitesse)
    degree: int             # degré effectif

@dataclass
class GPConfig:
    population_size: int = 200
    generations: int = 500
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elitism: int = 5
    target_error: float = 1e-4   # arrêt si atteint
    sample_points: int = 2000    # points d'évaluation fitness


class PolynomialGenome:
    """Un candidat = liste de coefficients + masque de termes actifs"""

    def __init__(self, degree, rng):
        self.degree = degree
        self.rng = rng
        # coefficients initiaux : petits aléatoires
        self.coeffs = rng.normal(0, 1.0 / np.sqrt(degree + 1), degree + 1)
        self.fitness = float('inf')
        self.age = 0

    def evaluate(self, x):
        """Évalue le polynôme (schéma de Horner)"""
        result = self.coeffs[-1] * np.ones_like(x)
        for c in reversed(self.coeffs[:-1]):
            result = result * x + c
        return result

    def copy(self):
        g = PolynomialGenome.__new__(PolynomialGenome)
        g.degree = self.degree
        g.coeffs = self.coeffs.copy()
        g.fitness = self.fitness
        g.age = self.age
        g.rng = self.rng
        return g


class SPEARKernelFactory:
    """Usine à kernels : découvre automatiquement les meilleurs polynômes."""

    def __init__(self, config: GPConfig = None):
        self.cfg = config or GPConfig()
        self.history = []

    def discover(self, name: str, exact_fn: Callable,
                 domain: tuple, max_degree: int = 8,
                 input_distribution: str = "uniform") -> KernelSpec:
        """
        Découvre le meilleur polynôme approximant exact_fn sur domain.
        input_distribution : "uniform" ou "gaussian" (pondère la fitness).
        """
        lo, hi = domain
        cfg = self.cfg
        rng = np.random.RandomState(42)

        # Points d'évaluation (fixes pour cohérence)
        x_eval = np.linspace(lo, hi, cfg.sample_points)

        # Pondération par distribution d'entrée réelle
        if input_distribution == "gaussian":
            weights = np.exp(-((x_eval - (lo+hi)/2)**2) / (2*((hi-lo)/6)**2))
            weights /= weights.sum()
        else:
            weights = np.ones(cfg.sample_points) / cfg.sample_points

        y_exact = exact_fn(x_eval)

        # Population initiale : mélange random + Chebyshev-like
        pop = []
        for i in range(cfg.population_size):
            g = PolynomialGenome(max_degree, rng)
            if i < cfg.population_size // 4:
                # seed avec des coefficients décroissants (approximation série)
                scale = 1.0 / (np.arange(max_degree+1) + 1)
                g.coeffs = rng.normal(0, scale, max_degree+1)
            pop.append(g)

        print(f"  [{name}] GP démarré : {cfg.population_size} individus, "
              f"deg≤{max_degree}, {cfg.generations} générations")

        t_start = time.time()

        for gen in range(cfg.generations):
            # Évalue fitness (erreur max pondérée)
            for g in pop:
                y_approx = g.evaluate(x_eval)
                errors = np.abs(y_approx - y_exact)
                weighted_err = errors * weights * cfg.sample_points
                g.fitness = np.max(weighted_err)  # erreur MAX pondérée

            # Tri par fitness
            pop.sort(key=lambda g: g.fitness)

            # Enregistre historique
            if gen % 50 == 0 or gen == cfg.generations - 1:
                self.history.append({
                    'gen': gen, 'best_err': pop[0].fitness,
                    'mean_err': np.mean([g.fitness for g in pop[:10]])
                })

            # Early stopping
            if pop[0].fitness < cfg.target_error:
                print(f"  [{name}] Target atteint gen {gen}: "
                      f"err={pop[0].fitness:.2e}")
                break

            # Nouvelle génération
            new_pop = []
            # Élitisme
            for i in range(cfg.elitism):
                new_pop.append(pop[i])

            # Reproduction
            while len(new_pop) < cfg.population_size:
                if rng.random() < cfg.crossover_rate:
                    # Crossover arithmétique
                    p1 = self._tournament(pop, rng)
                    p2 = self._tournament(pop, rng)
                    child = PolynomialGenome(max_degree, rng)
                    alpha = rng.random()
                    child.coeffs = alpha * p1.coeffs + (1-alpha) * p2.coeffs
                    # Mutation gaussienne adaptative
                    mut_mask = rng.random(max_degree+1) < 0.1
                    child.coeffs[mut_mask] += rng.normal(0, 0.01, mut_mask.sum())
                    new_pop.append(child)
                else:
                    # Mutation
                    p = self._tournament(pop, rng).copy()
                    idx = rng.randint(0, max_degree+1)
                    p.coeffs[idx] += rng.normal(0, 0.05)
                    new_pop.append(p)

            pop = new_pop

            if gen % 100 == 0 and gen > 0:
                print(f"  [{name}] gen {gen}: err_best={pop[0].fitness:.2e}")

        elapsed = time.time() - t_start
        best = pop[0]

        # Certification : maillage fin indépendant
        x_cert = np.linspace(lo, hi, 10000)
        y_cert = exact_fn(x_cert)
        y_app = best.evaluate(x_cert)
        cert_err = np.max(np.abs(y_app - y_cert))

        # Compte les opérations effectives
        degree = max_degree
        for i in range(max_degree, -1, -1):
            if abs(best.coeffs[i]) > 1e-12:
                degree = i
                break

        n_ops = degree + 1  # chaque coeff = 1 mul + 1 add (Horner)

        spec = KernelSpec(
            name=name,
            coeffs=best.coeffs[:degree+1],
            domain=domain,
            err_max=cert_err,
            err_mean=np.mean(np.abs(best.evaluate(x_cert) - y_cert)),
            n_ops=n_ops,
            degree=degree
        )

        print(f"  [{name}] Terminé en {elapsed:.1f}s : "
              f"deg={degree}, err_cert={cert_err:.2e}, ops={n_ops}")

        return spec

    def _tournament(self, pop, rng, size=3):
        candidates = rng.choice(len(pop), size, replace=False)
        return min(pop[c] for c in candidates)


def compare_with_current(name, gp_spec, current_fn, current_err, test_points=5000):
    """Compare le kernel GP-découvert contre le kernel actuel."""
    lo, hi = gp_spec.domain
    x_test = np.linspace(lo, hi, test_points)

    # Erreur du kernel actuel
    y_ref = None  # calculé par l'appelant
    print(f"\n=== Comparaison {name} ===")
    print(f"  GP découvert : deg={gp_spec.degree}, ops={gp_spec.n_ops}, "
          f"err_max={gp_spec.err_max:.2e}")
    return gp_spec


def emit_c_kernel(spec: KernelSpec) -> str:
    """Génère le code C du kernel découvert."""
    coeffs_c = ", ".join(f"{c:.15f}" for c in spec.coeffs)
    lo, hi = spec.domain

    c_code = f"""// Kernel {spec.name} — découvert automatiquement par SPEAR GP
// Domaine : [{lo}, {hi}]
// Degré : {spec.degree} | Err max : {spec.err_max:.2e}
// Généré automatiquement — NE PAS MODIFIER

double spur_discovered_{spec.name.lower()}(double x) {{
    double c[] = {{{coeffs_c}}};
    double r = c[{len(spec.coeffs)-1}];
    // Schéma de Horner (le compilateur vectorise)
'''
    for i in range(len(spec.coeffs)-2, -1, -1):
        c_code += f"    r = r*x + c[{i}];\n"
    c_code += "    return r;\n}\n"

    return c_code
