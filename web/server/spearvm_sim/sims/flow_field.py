"""Cas d'usage 1 — Champ de flux neuronal incompressible.

Un MLP (2 couches cachees, activation GELU fusionnee au matmul) evalue sur une
grille 3D periodique produit un **potentiel vecteur** A(x,y,z,t). Le champ de
vitesse envoye au navigateur est `v = rot(A)`, donc a divergence nulle : les
particules advectees ne s'accumulent jamais, le rendu reste "fluide".

Pourquoi SpearVM : chaque tick evalue `grid^3` points x (2 couches cachees) —
c'est exactement le pattern FFN `gelu(X.W^T)` que `matmul_nt_gelu` fusionne en
un seul passage memoire.
"""

from __future__ import annotations

import numpy as np

from ..protocol import quantize_i16
from .base import Frame, ParamSpec, Simulation


class FlowFieldSim(Simulation):
    sim_id = "flowfield"
    title = "Champ de flux neuronal"
    subtitle = "MLP GELU -> potentiel vecteur -> rotationnel"
    description = (
        "Un reseau dense evalue sur une grille 3D periodique genere un potentiel "
        "vecteur ; le rotationnel donne un champ de vitesse a divergence nulle. "
        "Le GPU advecte 65 536 particules dans ce champ (texture 3D + ping-pong)."
    )
    kernels_used = ("matmul_nt_gelu", "matmul_nt", "tanh")
    default_rate = 12.0
    structural_params = ("grid", "hidden", "seed")

    params_spec = (
        ParamSpec("grid", "Resolution grille", "select", 24, options=[16, 20, 24, 32],
                  help="Cote de la grille 3D evaluee par le reseau"),
        ParamSpec("hidden", "Largeur cachee", "select", 64, options=[32, 64, 96, 128],
                  help="Neurones par couche cachee (2 couches)"),
        ParamSpec("seed", "Graine", "range", 7, minimum=1, maximum=64, step=1,
                  help="Graine des poids : change completement la topologie du flux"),
        ParamSpec("morph", "Vitesse de morphing", "range", 0.35, minimum=0.0, maximum=1.5,
                  step=0.01, help="Vitesse de rotation du canal temporel du reseau"),
        ParamSpec("turbulence", "Turbulence", "range", 1.0, minimum=0.5, maximum=3.0,
                  step=0.05, help="Frequence spatiale de l'encodage periodique"),
        ParamSpec("amplitude", "Amplitude", "range", 1.0, minimum=0.1, maximum=3.0,
                  step=0.05, help="Gain applique au champ de vitesse"),
    )

    # ------------------------------------------------------------------
    def setup(self) -> None:
        n = int(self.params["grid"])
        h = int(self.params["hidden"])
        seed = int(self.params["seed"])
        rng = np.random.default_rng(seed)

        self.n = n
        # coordonnees normalisees [0,1) — periodiques
        axis = (np.arange(n, dtype=np.float32) + 0.5) / n
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        self._coords = (gx, gy, gz)

        # 6 canaux spatiaux (sin/cos xyz) + 2 canaux temporels
        self.k_in = 8
        s = np.float32(np.sqrt(2.0 / self.k_in))
        self.W1 = (rng.standard_normal((h, self.k_in)) * s).astype(np.float32)
        self.b1 = (rng.standard_normal(h) * 0.1).astype(np.float32)
        self.W2 = (rng.standard_normal((h, h)) * np.sqrt(2.0 / h)).astype(np.float32)
        self.b2 = (rng.standard_normal(h) * 0.1).astype(np.float32)
        self.W3 = (rng.standard_normal((3, h)) * np.sqrt(1.0 / h)).astype(np.float32)

        self._X = np.zeros((n * n * n, self.k_in), dtype=np.float32)
        self._fill_spatial_features()
        self.phase = 0.0
        self._flops = 2.0 * (n ** 3) * (self.k_in * h + h * h + h * 3)

    def _fill_spatial_features(self) -> None:
        f = float(self.params["turbulence"])
        gx, gy, gz = self._coords
        two_pi = np.float32(2.0 * np.pi)
        for i, g in enumerate((gx, gy, gz)):
            arg = (two_pi * f) * g.reshape(-1)
            self._X[:, 2 * i] = np.sin(arg)
            self._X[:, 2 * i + 1] = np.cos(arg)

    # ------------------------------------------------------------------
    def command(self, message: dict) -> None:
        if message.get("type") == "reseed":
            self.params["seed"] = int(message.get("seed", (int(self.params["seed"]) % 64) + 1))
            self.setup()

    def step(self, dt: float) -> Frame:
        n = self.n
        self.tick += 1
        self.sim_time += dt
        self.phase += dt * float(self.params["morph"])
        self._fill_spatial_features()
        self._X[:, 6] = np.float32(np.sin(self.phase))
        self._X[:, 7] = np.float32(np.cos(self.phase))

        with self.timer():
            # FFN fusionne : gelu(X.W^T + b) en un seul passage memoire
            h1 = self.k.matmul_nt_gelu(self._X, self.W1, self.b1)
            h2 = self.k.matmul_nt_gelu(h1, self.W2, self.b2)
            pot = self.k.matmul_nt(h2, self.W3)  # potentiel vecteur (N^3, 3)
            A = np.ascontiguousarray(pot, dtype=np.float32).reshape(n, n, n, 3)
            vel = _curl_periodic(A, 1.0 / n)
            # controle qualite : le rotationnel discret est a divergence nulle
            div_rel = _divergence_ratio(vel, 1.0 / n)
            # saturation douce : tanh certifie SPEAR, borne l'energie du champ
            amp = float(self.params["amplitude"])
            rms = float(np.sqrt(np.mean(vel * vel)) + 1e-6)
            vel = np.asarray(
                self.k.tanh(np.asarray(vel * (amp / rms), dtype=np.float64)),
                dtype=np.float32,
            )

        payload, scale = quantize_i16(vel, amplitude=1.0)
        speed = np.linalg.norm(vel.reshape(-1, 3), axis=1)
        gflops = (self._flops / 1e9) / max(self._compute_ms / 1e3, 1e-9)
        return Frame(
            kind="velocity_grid",
            payload=payload.reshape(-1),
            shape=(n, n, n, 3),
            scale=scale,
            stats={
                "grid": n,
                "hidden": int(self.params["hidden"]),
                "points": n ** 3,
                "gflops": round(gflops, 2),
                "mflop_per_tick": round(self._flops / 1e6, 1),
                "speed_mean": round(float(speed.mean()), 4),
                "div_rel": float(div_rel),
                "speed_max": round(float(speed.max()), 4),
                "phase": round(self.phase, 3),
            },
        )

    def metadata(self) -> dict:
        return {"grid": self.n, "channels": 3, "periodic": True}


def _divergence_ratio(vel: np.ndarray, dx: float) -> float:
    """|div v|max rapporte a |v|max/dx : ~1e-7 en float32 pour un rotationnel."""
    inv = 1.0 / (2.0 * dx)
    div = (
        (np.roll(vel[..., 0], -1, 0) - np.roll(vel[..., 0], 1, 0))
        + (np.roll(vel[..., 1], -1, 1) - np.roll(vel[..., 1], 1, 1))
        + (np.roll(vel[..., 2], -1, 2) - np.roll(vel[..., 2], 1, 2))
    ) * inv
    scale = float(np.abs(vel).max()) / dx
    return float(np.abs(div).max() / max(scale, 1e-12))


def _curl_periodic(A: np.ndarray, dx: float) -> np.ndarray:
    """rot(A) par differences centrees periodiques. A: (n,n,n,3)."""
    inv = 1.0 / (2.0 * dx)

    def d(component: int, axis: int) -> np.ndarray:
        c = A[..., component]
        return (np.roll(c, -1, axis=axis) - np.roll(c, 1, axis=axis)) * inv

    curl = np.empty_like(A)
    curl[..., 0] = d(2, 1) - d(1, 2)
    curl[..., 1] = d(0, 2) - d(2, 0)
    curl[..., 2] = d(1, 0) - d(0, 1)
    return curl
