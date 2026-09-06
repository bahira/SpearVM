"""Cas d'usage 2 — Membrane / surface d'eau non lineaire.

Equation des ondes 2D resolue en differences finies, avec **raidissement non
lineaire** applique par le noyau `tanh` certifie SPEAR : au lieu de laisser
l'amplitude diverger, la membrane sature en douceur (comportement d'une peau
tendue). Le champ de hauteur est quantifie en int16 et deplace un plan
Three.js dans le vertex shader.

Pourquoi SpearVM : le pas de temps applique `tanh` sur `M^2` cellules a chaque
sous-pas — c'est un noyau element-par-element, memory-bound, exactement la
cible des batchs AVX2 (~x26 vs libm d'apres la datasheet).
"""

from __future__ import annotations

import numpy as np

from ..protocol import quantize_i16
from .base import Frame, ParamSpec, Simulation


class WaveFieldSim(Simulation):
    sim_id = "wavefield"
    title = "Membrane non lineaire"
    subtitle = "ondes 2D + saturation tanh SPEAR"
    description = (
        "Equation des ondes en differences finies ; le noyau tanh AVX2 applique "
        "une saturation douce a chaque sous-pas (membrane qui se raidit). "
        "Cliquez la surface pour injecter une impulsion."
    )
    kernels_used = ("tanh", "erf")
    default_rate = 30.0
    structural_params = ("size",)

    params_spec = (
        ParamSpec("size", "Resolution", "select", 160, options=[96, 128, 160, 192],
                  help="Cote de la grille de simulation"),
        ParamSpec("courant", "Nombre de Courant", "range", 0.45, minimum=0.1, maximum=0.7,
                  step=0.01, help="c*dt/dx — au dela de 0.7 le schema explose"),
        ParamSpec("damping", "Amortissement", "range", 0.0025, minimum=0.0, maximum=0.02,
                  step=0.0005),
        ParamSpec("stiffness", "Raideur (tanh)", "range", 0.55, minimum=0.0, maximum=1.0,
                  step=0.01, help="0 = lineaire, 1 = saturation tanh forte"),
        ParamSpec("substeps", "Sous-pas / tick", "range", 2, minimum=1, maximum=4, step=1),
        ParamSpec("driver", "Source oscillante", "bool", True),
        ParamSpec("driver_freq", "Frequence source", "range", 1.6, minimum=0.2, maximum=6.0,
                  step=0.1),
    )

    # ------------------------------------------------------------------
    def setup(self) -> None:
        m = int(self.params["size"])
        self.m = m
        self.u = np.zeros((m, m), dtype=np.float64)
        self.u_prev = np.zeros((m, m), dtype=np.float64)
        self._mask = self._build_mask(m)
        self._pending: list[tuple[float, float, float]] = []
        self._drop(0.5, 0.5, 1.0, radius=0.05)

    @staticmethod
    def _build_mask(m: int) -> np.ndarray:
        """Bord absorbant doux : facteur multiplicatif < 1 pres des bords."""
        r = np.linspace(-1.0, 1.0, m)
        gx, gy = np.meshgrid(r, r, indexing="ij")
        d = np.maximum(np.abs(gx), np.abs(gy))
        return np.clip(1.0 - np.maximum(d - 0.88, 0.0) / 0.12 * 0.06, 0.0, 1.0)

    def _drop(self, cx: float, cy: float, amp: float, radius: float = 0.035) -> None:
        m = self.m
        r = (np.arange(m) + 0.5) / m
        gx, gy = np.meshgrid(r, r, indexing="ij")
        d2 = (gx - cx) ** 2 + (gy - cy) ** 2
        self.u += amp * np.exp(-d2 / (2.0 * radius * radius))

    # ------------------------------------------------------------------
    def command(self, message: dict) -> None:
        mtype = message.get("type")
        if mtype == "pulse":
            try:
                x = min(max(float(message.get("x", 0.5)), 0.0), 1.0)
                y = min(max(float(message.get("y", 0.5)), 0.0), 1.0)
                amp = min(max(float(message.get("amp", 1.0)), -3.0), 3.0)
            except (TypeError, ValueError):
                return
            self._pending.append((x, y, amp))
        elif mtype == "clear":
            self.u[:] = 0.0
            self.u_prev[:] = 0.0

    def step(self, dt: float) -> Frame:
        self.tick += 1
        self.sim_time += dt
        while self._pending:
            x, y, amp = self._pending.pop()
            self._drop(x, y, amp)

        c = float(self.params["courant"])
        c2 = c * c
        damp = float(self.params["damping"])
        stiff = float(self.params["stiffness"])
        substeps = int(self.params["substeps"])

        with self.timer():
            for _ in range(substeps):
                u = self.u
                lap = (
                    np.roll(u, 1, 0) + np.roll(u, -1, 0)
                    + np.roll(u, 1, 1) + np.roll(u, -1, 1)
                    - 4.0 * u
                )
                nxt = (2.0 - damp) * u - (1.0 - damp) * self.u_prev + c2 * lap
                if stiff > 0.0:
                    # saturation douce : u <- (1-s)*u + s*A*tanh(u/A), noyau AVX2
                    a = 1.0 / max(stiff, 1e-3)
                    sat = a * self.k.tanh(np.ascontiguousarray(nxt / a))
                    nxt = (1.0 - stiff) * nxt + stiff * sat
                nxt *= self._mask
                self.u_prev, self.u = u, nxt

            if self.params["driver"]:
                w = 2.0 * np.pi * float(self.params["driver_freq"])
                self.u[self.m // 2, self.m // 6] += 0.06 * np.sin(w * self.sim_time)

            height = np.asarray(self.u, dtype=np.float32)

        peak = float(np.max(np.abs(height))) if height.size else 0.0
        energy = float(np.mean(height * height))
        payload, scale = quantize_i16(height, amplitude=max(peak, 1e-3))
        cells = self.m * self.m * substeps
        return Frame(
            kind="height_grid",
            payload=payload.reshape(-1),
            shape=(self.m, self.m),
            scale=scale,
            stats={
                "size": self.m,
                "peak": round(peak, 4),
                "energy": round(energy, 6),
                "cells_per_tick": cells,
                "mcells_per_s": round(cells / 1e6 / max(self._compute_ms / 1e3, 1e-9), 1),
                "substeps": substeps,
            },
        )

    def metadata(self) -> dict:
        return {"size": self.m, "interactive": True}
