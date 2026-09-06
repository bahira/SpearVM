"""Couche d'acces aux noyaux SpearVM.

Objectif : donner aux simulations une API stable (`Kernels`) qui utilise les
noyaux AVX2 de `spur_math` quand ils sont disponibles, et retombe sur une
implementation numpy de reference sinon. Le serveur demarre donc **toujours**,
meme sur une machine ARM ou sans compilateur, en signalant clairement le mode
degrade via `capabilities()`.

Ordre de resolution :
  1. `import spur_math` (paquet installe via `pip install spur-math`)
  2. sinon, racine du depot ajoutee au sys.path (dev in-tree) ; si le `.so`
     manque il est compile une fois depuis `src/spur_kernels.c`
  3. sinon, backend numpy (`SPEARVM_FORCE_FALLBACK=1` force ce mode)
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger("spearvm.kernels")

REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = REPO_ROOT / "src" / "spur_kernels.c"
_SO = REPO_ROOT / "spur_math" / "libspur_kernels.so"

_TRUE = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE


def _cpu_flags() -> list[str]:
    """Flags SIMD pertinents lus sur /proc/cpuinfo (Linux) — best effort."""
    wanted = ("avx2", "fma", "avx512f")
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
    except OSError:
        return []
    line = ""
    for raw in text.splitlines():
        if raw.startswith("flags") or raw.startswith("Features"):
            line = raw
            break
    present = set(line.split())
    return [f for f in wanted if f in present]


def _ensure_native_lib() -> bool:
    """Compile `libspur_kernels.so` in-tree si besoin. True si la lib existe."""
    if _SO.exists():
        return True
    if not _SRC.exists():
        return False
    cc = shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")
    if cc is None:
        log.warning("aucun compilateur C trouve, noyaux natifs indisponibles")
        return False
    cmd = [cc, "-O3", "-mavx2", "-mfma", "-fopenmp", "-shared", "-fPIC",
           "-o", str(_SO), str(_SRC), "-lm"]
    log.info("compilation des noyaux natifs: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd, timeout=180)
    except (subprocess.SubprocessError, OSError) as exc:  # pragma: no cover
        log.warning("compilation des noyaux echouee: %s", exc)
        return False
    return _SO.exists()


def _load_spur_math():
    """Retourne le module spur_math ou None."""
    if _env_flag("SPEARVM_FORCE_FALLBACK"):
        log.info("SPEARVM_FORCE_FALLBACK=1 -> backend numpy force")
        return None
    try:
        import spur_math  # type: ignore

        return spur_math
    except Exception:  # noqa: BLE001 - optional native backend
        log.debug("installed spur_math is unavailable", exc_info=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    _ensure_native_lib()
    try:
        import spur_math  # type: ignore

        return spur_math
    except Exception as exc:  # noqa: BLE001
        log.warning("spur_math indisponible (%s) -> backend numpy", exc)
        return None


# --------------------------------------------------------------------------
# Implementations de reference numpy (fallback + oracle des tests)
# --------------------------------------------------------------------------

SQRT_2 = float(np.sqrt(2.0))


def ref_gelu(x: np.ndarray) -> np.ndarray:
    from math import erf as _erf

    xv = np.asarray(x)
    vec = np.vectorize(_erf, otypes=[xv.dtype if xv.dtype == np.float32 else np.float64])
    return 0.5 * xv * (1.0 + vec(xv / SQRT_2))


def ref_erf(x: np.ndarray) -> np.ndarray:
    from math import erf as _erf

    return np.vectorize(_erf, otypes=[np.float64])(np.asarray(x, dtype=np.float64))


def ref_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(np.asarray(x, dtype=np.float64))


def ref_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _np_matmul_nt(a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    dt = np.float32 if np.asarray(a).dtype == np.float32 else np.float64
    res = np.ascontiguousarray(np.asarray(a, dtype=dt) @ np.asarray(b, dtype=dt).T)
    if out is None:
        return res
    out[...] = res           # meme signature que le backend natif (`out=`)
    return out


def _np_matmul_nt_gelu(a: np.ndarray, b: np.ndarray, bias: np.ndarray | None = None,
                       out: np.ndarray | None = None) -> np.ndarray:
    res = _np_matmul_nt(a, b)
    if bias is not None:
        res = res + np.asarray(bias, dtype=res.dtype)
    res = np.asarray(ref_gelu(res), dtype=res.dtype)
    if out is None:
        return res
    out[...] = res
    return out


def _np_gelu_backward(dY: np.ndarray, x: np.ndarray) -> np.ndarray:
    xv = np.asarray(x)
    dt = xv.dtype if xv.dtype == np.float32 else np.float64
    xd = np.asarray(xv, dtype=np.float64)
    cdf = 0.5 * (1.0 + ref_erf(xd / SQRT_2))
    pdf = np.exp(-0.5 * xd * xd) / np.sqrt(2.0 * np.pi)
    return np.asarray(np.asarray(dY, dtype=np.float64) * (cdf + xd * pdf), dtype=dt)


def _np_matmul_backward(dY: np.ndarray, A: np.ndarray, B: np.ndarray):
    dYv = np.asarray(dY)
    dt = np.float32 if dYv.dtype == np.float32 else np.float64
    dA = _np_matmul_nt(dYv, np.ascontiguousarray(np.asarray(B, dtype=dt).T))
    dB = _np_matmul_nt(np.ascontiguousarray(dYv.T), np.ascontiguousarray(np.asarray(A, dtype=dt).T))
    return dA, dB


@dataclass
class Kernels:
    """Facade appelee par les simulations."""

    name: str
    native: bool
    details: dict[str, Any] = field(default_factory=dict)

    gelu: Callable[..., np.ndarray] = ref_gelu
    gelu_quintic: Callable[..., np.ndarray] = ref_gelu
    gelu_erf: Callable[..., np.ndarray] = ref_gelu
    erf: Callable[..., np.ndarray] = ref_erf
    tanh: Callable[..., np.ndarray] = ref_tanh
    matmul_nt: Callable[..., np.ndarray] = _np_matmul_nt
    matmul_nt_gelu: Callable[..., np.ndarray] = _np_matmul_nt_gelu
    gelu_backward: Callable[..., np.ndarray] = _np_gelu_backward
    matmul_backward: Callable[..., Any] = _np_matmul_backward

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "native": self.native,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "cpu_flags": _cpu_flags(),
            "threads": os.cpu_count(),
            **self.details,
        }


def _build_kernels() -> Kernels:
    sm = _load_spur_math()
    if sm is None:
        return Kernels(
            name="numpy-reference",
            native=False,
            details={
                "library": None,
                "note": "noyaux AVX2 indisponibles - simulation exacte mais plus lente",
            },
        )
    lib = getattr(sm, "_dll_path", None)
    return Kernels(
        name="spearvm-avx2",
        native=True,
        details={
            "library": str(lib) if lib else None,
            "note": "noyaux SPEAR AVX2/FMA + OpenMP",
        },
        gelu=sm.gelu,
        gelu_quintic=sm.gelu_quintic,
        gelu_erf=sm.gelu_erf,
        erf=sm.erf,
        tanh=sm.tanh,
        matmul_nt=sm.matmul_nt,
        matmul_nt_gelu=sm.matmul_nt_gelu,
        gelu_backward=sm.gelu_backward,
        matmul_backward=sm.matmul_backward,
    )


_KERNELS: Kernels | None = None


def get_kernels() -> Kernels:
    """Singleton — la resolution/compilation n'a lieu qu'une fois."""
    global _KERNELS
    if _KERNELS is None:
        t0 = time.perf_counter()
        _KERNELS = _build_kernels()
        log.info(
            "backend noyaux: %s (%.0f ms)", _KERNELS.name, (time.perf_counter() - t0) * 1e3
        )
    return _KERNELS


def reset_kernels() -> None:
    """Utilise par les tests pour reevaluer l'environnement."""
    global _KERNELS
    _KERNELS = None
