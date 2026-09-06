"""Harnais de mesure : compilation, verification, chronometrage stable.

Regles de mesure (une VM 2 vCPU ment facilement) :
  * warmup obligatoire, puis min-of-N (le min est l'estimateur le moins
    pollue par l'ordonnanceur) ;
  * variantes mesurees par **blocs alternes** pour qu'aucune ne monopolise une
    fenetre de contention ;
  * pause de quiescence avant chaque bloc (threads BLAS/OpenMP qui spinnent) ;
  * chaque variante est d'abord **verifiee** contre numpy avant d'etre timee :
    une variante fausse est rejetee, jamais classee.
"""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

CFLAGS = ["-O3", "-mavx2", "-mfma", "-fno-math-errno"]
QUIESCE_S = 0.004


class CompileError(RuntimeError):
    pass


def compile_source(source: str, name: str, extra: Iterable[str] = ()) -> ctypes.CDLL:
    """Compile une source C en .so (cache par empreinte du contenu)."""
    digest = hashlib.sha1((source + " ".join(extra)).encode()).hexdigest()[:12]
    so = BUILD / f"{name}_{digest}.so"
    if not so.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as fh:
            fh.write(source)
            src = fh.name
        cmd = ["gcc", *CFLAGS, *extra, "-shared", "-fPIC", "-o", str(so), src, "-lm"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(src)
        if proc.returncode != 0:
            raise CompileError(proc.stderr[-2000:])
    return ctypes.CDLL(str(so))


def bind_gemm(lib: ctypes.CDLL, name: str, dtype: str):
    fn = getattr(lib, name)
    ptr = ctypes.POINTER(ctypes.c_double if dtype == "f64" else ctypes.c_float)
    fn.argtypes = [ptr, ptr, ptr, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong]
    fn.restype = None
    np_t = np.float64 if dtype == "f64" else np.float32

    def call(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> None:
        m, k = A.shape
        n = B.shape[0]
        fn(A.ctypes.data_as(ptr), B.ctypes.data_as(ptr), C.ctypes.data_as(ptr), m, k, n)

    call.np_t = np_t  # type: ignore[attr-defined]
    return call


@dataclass
class Case:
    m: int
    k: int
    n: int

    @property
    def flops(self) -> float:
        return 2.0 * self.m * self.k * self.n

    def label(self) -> str:
        return f"{self.m}x{self.k}x{self.n}"


def make_operands(case: Case, dtype: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    np_t = np.float64 if dtype == "f64" else np.float32
    A = np.ascontiguousarray(rng.standard_normal((case.m, case.k)), dtype=np_t)
    B = np.ascontiguousarray(rng.standard_normal((case.n, case.k)), dtype=np_t)
    C = np.zeros((case.m, case.n), dtype=np_t)
    return A, B, C


def verify(call, dtype: str, cases: Iterable[Case] = ()) -> float:
    """Erreur relative max sur des tailles volontairement non alignees."""
    checks = list(cases) or [Case(7, 5, 3), Case(16, 33, 17), Case(65, 8, 64),
                             Case(1, 129, 1), Case(37, 64, 129), Case(128, 96, 96)]
    worst = 0.0
    for case in checks:
        A, B, C = make_operands(case, dtype, seed=case.m + case.k + case.n)
        C[:] = np.nan  # detecte les cases jamais ecrites
        call(A, B, C)
        ref = A.astype(np.float64) @ B.astype(np.float64).T
        if not np.isfinite(C).all():
            return math.inf
        scale = max(float(np.abs(ref).max()), 1e-30)
        worst = max(worst, float(np.abs(C.astype(np.float64) - ref).max()) / scale)
    return worst


def race(variants: dict[str, Callable[[], None]], repeats: int = 3,
         block: int = 3) -> dict[str, float]:
    for fn in variants.values():
        fn()
    best = {key: math.inf for key in variants}
    for _ in range(max(1, repeats)):
        for key, fn in variants.items():
            time.sleep(QUIESCE_S)
            for _ in range(max(1, block)):
                t0 = time.perf_counter()
                fn()
                best[key] = min(best[key], time.perf_counter() - t0)
    return {key: value * 1e3 for key, value in best.items()}


def gflops(case: Case, ms: float) -> float:
    return case.flops / 1e9 / (ms / 1e3)
