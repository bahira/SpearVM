"""Cas d'usage 4 — Banc d'essai des noyaux (donnees du "Kernel Lab").

Mesure a la demande, sur la machine qui heberge le serveur :
  * debit des noyaux element-par-element (SpearVM AVX2 vs baseline numpy),
  * erreur vs reference IEEE (`math.erf`) — verification du datasheet,
  * GFLOPS matmul NT et gain du FFN fusionne `gelu(A.B^T)`.

Le tout est borne en temps (quelques centaines de ms) et serialise en JSON
pour etre rendu en 3D par le client.
"""

from __future__ import annotations

import math
import os
import platform
import threading
import time
from typing import Any, Callable

import numpy as np

from .kernels import get_kernels

_LOCK = threading.Lock()
_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

# Filet de securite : petite pause avant chaque bloc pour laisser les pools de
# threads se garer (le reglage de fond est fait dans spearvm_sim/__init__.py).
QUIESCE_S = 0.005

_erf_vec = np.vectorize(math.erf, otypes=[np.float64])


# ---------------------------------------------------------------------------
# References numpy (baselines honnetes, toutes vectorisees sauf erf/libm)
# ---------------------------------------------------------------------------
def np_gelu_tanh(x: np.ndarray) -> np.ndarray:
    """Approximation tanh de GELU (la baseline usuelle PyTorch)."""
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


def np_gelu_exact(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf_vec(x / _SQRT_2))


def _timeit(fn: Callable[[], Any], repeats: int) -> float:
    """min-of-N en ms (le min filtre le bruit d'ordonnancement des VMs)."""
    return _race({"x": fn}, repeats)["x"]


def _race(variants: dict[str, Callable[[], Any]], repeats: int,
          block: int = 3) -> dict[str, float]:
    """Chronometre plusieurs variantes par **blocs alternes**, min global.

    Deux pieges sur une machine 2 vCPU :
      * mesurer A puis B en un seul bloc expose la variante malchanceuse a une
        fenetre de contention entiere -> on alterne les blocs ;
      * juste apres un appel BLAS, les threads OpenBLAS *spinnent* encore et
        affament la region OpenMP suivante -> on ne mesure pas une variante sur
        la premiere iteration qui suit le changement, d'ou des blocs de
        `block` iterations dont on garde le minimum.
    """
    for fn in variants.values():
        fn()  # warmup / premieres fautes de page
    best = {key: math.inf for key in variants}
    for _ in range(max(1, repeats)):
        for key, fn in variants.items():
            time.sleep(QUIESCE_S)  # laisse les pools de threads (BLAS/OpenMP) se garer
            for _ in range(max(1, block)):
                t0 = time.perf_counter()
                fn()
                best[key] = min(best[key], time.perf_counter() - t0)
    return {key: value * 1e3 for key, value in best.items()}


def _accuracy(approx: Callable[[np.ndarray], np.ndarray],
              exact: Callable[[np.ndarray], np.ndarray],
              lo: float, hi: float, n: int = 20000) -> dict[str, float]:
    xs = np.linspace(lo, hi, n)
    a = np.asarray(approx(xs), dtype=np.float64)
    e = np.asarray(exact(xs), dtype=np.float64)
    d = np.abs(a - e)
    return {
        "linf": float(d.max()),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "domain": [lo, hi],
    }


def _curve(fn: Callable[[np.ndarray], np.ndarray], exact: Callable[[np.ndarray], np.ndarray],
           lo: float = -4.0, hi: float = 4.0, n: int = 192) -> dict[str, list[float]]:
    xs = np.linspace(lo, hi, n)
    y = np.asarray(fn(xs), dtype=np.float64)
    ref = np.asarray(exact(xs), dtype=np.float64)
    err = np.abs(y - ref)
    return {
        "x": [round(v, 4) for v in xs.tolist()],
        "y": [round(v, 5) for v in y.tolist()],
        "err": [round(v, 9) for v in err.tolist()],
    }


# ---------------------------------------------------------------------------
def elementwise_report(n: int = 1 << 20, repeats: int = 3) -> list[dict[str, Any]]:
    k = get_kernels()
    rng = np.random.default_rng(1234)
    x = rng.standard_normal(n) * 1.5

    specs: list[tuple[str, str, Callable, Callable, str, tuple[float, float]]] = [
        ("gelu", "GELU v1 (legacy)", k.gelu, np_gelu_tanh, "numpy tanh-GELU", (-2.0, 2.0)),
        ("gelu_quintic", "GELU v2 quintique", k.gelu_quintic, np_gelu_tanh, "numpy tanh-GELU", (-3.5, 3.5)),
        ("gelu_erf", "GELU erf (haute precision)", k.gelu_erf, np_gelu_tanh, "numpy tanh-GELU", (-3.5, 3.5)),
        ("erf", "erf", k.erf, _erf_vec, "libm scalaire via np.vectorize (surcout Python inclus)", (-2.0, 2.0)),
        ("tanh", "tanh", k.tanh, np.tanh, "numpy tanh", (-3.0, 3.0)),
    ]

    out: list[dict[str, Any]] = []
    for key, label, spear, base, base_label, (lo, hi) in specs:
        # erf scalaire : on reduit la taille, sinon np.vectorize plombe le bench
        nn = n if key != "erf" else min(n, 1 << 17)
        xv = np.ascontiguousarray(x[:nn])
        timings = _race(
            {"spear": lambda s=spear: s(xv), "base": lambda b=base: b(xv)},
            1 if key == "erf" else repeats,
        )
        ms_spear, ms_base = timings["spear"], timings["base"]
        exact = np_gelu_exact if key.startswith("gelu") else (_erf_vec if key == "erf" else np.tanh)
        out.append({
            "key": key,
            "label": label,
            "n": int(nn),
            "spear_ms": round(ms_spear, 4),
            "baseline_ms": round(ms_base, 4),
            "baseline_label": base_label,
            "speedup": round(ms_base / ms_spear, 2) if ms_spear > 0 else None,
            "mele_per_s": round(nn / 1e6 / (ms_spear / 1e3), 1),
            "accuracy": _accuracy(spear, exact, lo, hi),
            "curve": _curve(spear, exact),
        })
    return out


def matmul_report(sizes: tuple[int, ...] = (256, 512), repeats: int = 2) -> list[dict[str, Any]]:
    k = get_kernels()
    rng = np.random.default_rng(7)
    out: list[dict[str, Any]] = []
    for nsz in sizes:
        for dtype, tag in ((np.float32, "f32"), (np.float64, "f64")):
            A = np.ascontiguousarray(rng.standard_normal((nsz, nsz)), dtype=dtype)
            B = np.ascontiguousarray(rng.standard_normal((nsz, nsz)), dtype=dtype)
            flops = 2.0 * nsz ** 3
            timings = _race(
                {"spear": lambda: k.matmul_nt(A, B), "numpy": lambda: A @ B.T}, repeats
            )
            ms_spear, ms_np = timings["spear"], timings["numpy"]
            err = float(np.max(np.abs(k.matmul_nt(A, B) - A @ B.T)))
            out.append({
                "key": f"mm{nsz}_{tag}",
                "label": f"matmul NT {nsz}^3 {tag}",
                "n": nsz,
                "dtype": tag,
                "spear_ms": round(ms_spear, 3),
                "baseline_ms": round(ms_np, 3),
                "baseline_label": "numpy BLAS",
                "spear_gflops": round(flops / 1e9 / (ms_spear / 1e3), 2),
                "baseline_gflops": round(flops / 1e9 / (ms_np / 1e3), 2),
                "speedup": round(ms_np / ms_spear, 2) if ms_spear > 0 else None,
                "max_abs_err": err,
            })
    return out


def fused_ffn_report(m: int = 512, kdim: int = 256, n: int = 1024,
                     repeats: int = 2) -> dict[str, Any]:
    """`gelu(X.W^T)` fusionne (1 passage) vs matmul puis gelu (2 passages)."""
    k = get_kernels()
    rng = np.random.default_rng(11)
    X = np.ascontiguousarray(rng.standard_normal((m, kdim)), dtype=np.float32)
    W = np.ascontiguousarray(rng.standard_normal((n, kdim)) * 0.05, dtype=np.float32)
    flops = 2.0 * m * kdim * n

    timings = _race(
        {
            "fused": lambda: k.matmul_nt_gelu(X, W),
            "split": lambda: k.gelu(k.matmul_nt(X, W)),
            "numpy": lambda: np_gelu_tanh(X @ W.T),
        },
        max(repeats, 3),
    )
    ms_fused, ms_split, ms_numpy = timings["fused"], timings["split"], timings["numpy"]
    return {
        "shape": {"m": m, "k": kdim, "n": n},
        "flops": flops,
        "fused_ms": round(ms_fused, 3),
        "split_ms": round(ms_split, 3),
        "numpy_ms": round(ms_numpy, 3),
        "fusion_gain": round(ms_split / ms_fused, 2) if ms_fused > 0 else None,
        "vs_numpy": round(ms_numpy / ms_fused, 2) if ms_fused > 0 else None,
        "fused_gflops": round(flops / 1e9 / (ms_fused / 1e3), 2),
    }


def gradcheck_report(n: int = 4096, eps: float = 1e-4) -> dict[str, Any]:
    """Verifie `gelu_backward` par differences finies (contrat backprop)."""
    k = get_kernels()
    rng = np.random.default_rng(5)
    x = np.ascontiguousarray(rng.standard_normal(n) * 2.0)
    ones = np.ones_like(x)
    analytic = np.asarray(k.gelu_backward(ones, x), dtype=np.float64)
    numeric = (np.asarray(k.gelu(x + eps)) - np.asarray(k.gelu(x - eps))) / (2 * eps)
    d = np.abs(analytic - numeric)
    return {
        "n": n,
        "eps": eps,
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "passed": bool(d.max() < 5e-4),
    }


def full_report(quick: bool = False) -> dict[str, Any]:
    """Rapport complet, serialisable tel quel. Serialise par un verrou."""
    k = get_kernels()
    with _LOCK:
        t0 = time.perf_counter()
        elementwise = elementwise_report(n=1 << (18 if quick else 20), repeats=2 if quick else 3)
        matmuls = matmul_report(sizes=(256,) if quick else (256, 512))
        ffn = fused_ffn_report(m=256 if quick else 512, kdim=256, n=512 if quick else 1024)
        grad = gradcheck_report(n=2048 if quick else 4096)
        elapsed = (time.perf_counter() - t0) * 1e3
    threads = os.cpu_count() or 1
    return {
        "generated_at": time.time(),
        "elapsed_ms": round(elapsed, 1),
        "quick": quick,
        "method": (
            "min-of-N par blocs alternes (min 3 iterations consecutives par variante) ; "
            "OPENBLAS_THREAD_TIMEOUT=1 pour eviter que les threads BLAS en spin "
            "n'affament les regions OpenMP"
        ),
        "noisy_host": threads < 4,
        "host": {
            "backend": k.name,
            "native": k.native,
            "processor": platform.processor() or platform.machine(),
            **k.capabilities(),
        },
        "elementwise": elementwise,
        "matmul": matmuls,
        "fused_ffn": ffn,
        "gradcheck": grad,
    }
