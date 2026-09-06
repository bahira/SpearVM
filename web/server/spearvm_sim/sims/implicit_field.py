"""Champ implicite neuronal : un MLP definit une surface (SDF) sculptee en direct.

Pourquoi ce cas d'usage : c'est le profil de forme ou les noyaux SpearVM sont
les plus rentables. Evaluer un reseau **par point** sur une grille 3D donne des
GEMM `(N^3 lignes) x (k tres court)` — 14 canaux d'entree, 64 a 128 neurones
caches. OpenBLAS amortit mal son packing sur ces formes ; le noyau NT sans
copie, lui, y est chez lui (cf. `docs/EXPERIMENTS.md`, 512x16x512 : x1.49 vs
numpy).

Pipeline d'un tick :
  1. encodage periodique de (x,y,z) sur deux frequences + 2 canaux temporels ;
  2. MLP 2 couches GELU fusionnees (`matmul_nt_gelu`) -> deplacement scalaire ;
  3. distance signee = primitive analytique + amplitude * deplacement ;
  4. **renormalisation de Lipschitz** : on mesure max|grad d| sur la grille et
     on divise le champ par ce facteur s'il depasse 1. C'est la condition de
     validite du sphere tracing cote GPU : sans elle le rendu "perce" la
     surface. La valeur mesuree est renvoyee dans les stats, donc verifiable.
  5. quantification int16 -> le navigateur reconstruit un volume et le
     ray-marche entierement sur GPU.
"""

from __future__ import annotations

import numpy as np

from ..protocol import quantize_i16
from .base import Frame, ParamSpec, Simulation

SHAPES = ("sphere", "tore", "boite", "gyroide")


def _base_sdf(kind: str, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Primitive analytique, exprimee dans le domaine [-1,1]^3."""
    if kind == "tore":
        q = np.sqrt(gx * gx + gz * gz) - 0.52
        return np.sqrt(q * q + gy * gy) - 0.20
    if kind == "boite":
        b = 0.46
        dx, dy, dz = np.abs(gx) - b, np.abs(gy) - b, np.abs(gz) - b
        outside = np.sqrt(np.maximum(dx, 0.0) ** 2 + np.maximum(dy, 0.0) ** 2
                          + np.maximum(dz, 0.0) ** 2)
        inside = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
        return outside + inside - 0.06
    if kind == "gyroide":
        # surface triplement periodique : |gyroide| borne, echelle ~1/(2*pi*f)
        f = 2.0
        s = (np.sin(f * np.pi * gx) * np.cos(f * np.pi * gy)
             + np.sin(f * np.pi * gy) * np.cos(f * np.pi * gz)
             + np.sin(f * np.pi * gz) * np.cos(f * np.pi * gx))
        return (s * 0.16 - 0.02).astype(np.float32)
    return np.sqrt(gx * gx + gy * gy + gz * gz) - 0.62  # sphere


class ImplicitFieldSim(Simulation):
    sim_id = "implicit"
    title = "Champ implicite neuronal"
    subtitle = "MLP par point -> SDF -> sphere tracing GPU"
    description = (
        "Un MLP evalue sur chaque voxel d'une grille 3D deforme une primitive "
        "signee ; le navigateur ray-marche le volume. Profil de forme ideal "
        "pour les noyaux SpearVM : beaucoup de lignes, k tres court."
    )
    kernels_used = ("matmul_nt_gelu", "matmul_nt")
    default_rate = 8.0
    structural_params = ("grid", "hidden", "seed", "freq")

    params_spec = (
        ParamSpec("grid", "Resolution volume", "select", 48, options=[32, 40, 48, 56],
                  help="Cote de la grille 3D : le MLP est evalue sur grid^3 points"),
        ParamSpec("hidden", "Largeur cachee", "select", 64, options=[32, 64, 96, 128],
                  help="Neurones par couche cachee (2 couches GELU fusionnees)"),
        ParamSpec("shape", "Primitive", "select", "sphere", options=list(SHAPES),
                  help="Distance signee de base que le reseau vient sculpter"),
        ParamSpec("amplitude", "Amplitude sculpture", "range", 0.55, minimum=0.0,
                  maximum=1.2, step=0.01,
                  help="Poids du deplacement neuronal ajoute a la primitive"),
        ParamSpec("morph", "Vitesse de morphing", "range", 0.30, minimum=0.0,
                  maximum=1.5, step=0.01,
                  help="Rotation du canal temporel : la surface se deforme"),
        ParamSpec("freq", "Frequence spatiale", "range", 2.1, minimum=0.5, maximum=4.0,
                  step=0.05, help="Frequence de l'encodage periodique (detail)"),
        ParamSpec("seed", "Graine", "range", 11, minimum=1, maximum=64, step=1,
                  help="Graine des poids : change completement la sculpture"),
    )

    # ------------------------------------------------------------------
    def setup(self) -> None:
        n = int(self.params["grid"])
        h = int(self.params["hidden"])
        rng = np.random.default_rng(int(self.params["seed"]))

        self.n = n
        axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        self._coords = (gx, gy, gz)
        self._voxel = float(2.0 / (n - 1))

        # 2 frequences x sin/cos x 3 axes = 12 canaux + 2 canaux temporels
        self.k_in = 14
        s = np.float32(np.sqrt(2.0 / self.k_in))
        self.W1 = (rng.standard_normal((h, self.k_in)) * s).astype(np.float32)
        self.b1 = (rng.standard_normal(h) * 0.1).astype(np.float32)
        self.W2 = (rng.standard_normal((h, h)) * np.sqrt(2.0 / h)).astype(np.float32)
        self.b2 = (rng.standard_normal(h) * 0.1).astype(np.float32)
        self.W3 = (rng.standard_normal((1, h)) * np.sqrt(1.0 / h)).astype(np.float32)

        self._X = np.zeros((n * n * n, self.k_in), dtype=np.float32)
        self._fill_spatial_features()
        self._base = np.ascontiguousarray(
            _base_sdf(str(self.params["shape"]), gx, gy, gz), dtype=np.float32)
        # gradient de la primitive : constant, calcule une fois (sert au
        # controle de Lipschitz sans refaire np.gradient sur le champ complet)
        self._base_grad = [np.ascontiguousarray(g, dtype=np.float32)
                           for g in np.gradient(self._base, self._voxel)]
        self._shape_key = str(self.params["shape"])
        self.phase = 0.0
        self._flops = 2.0 * (n ** 3) * (self.k_in * h + h * h + h)

        # Evaluation par paquets de lignes : les activations intermediaires
        # (jusqu'a 16 Mo pour 56^3 points) restent en cache au lieu de faire
        # deux aller-retours en RAM. Mesure : 12.8 ms -> 8.6 ms sur 64 000
        # points (experiments/sweep_tall.py). Tampons reutilises via `out=` :
        # l'allocation seule coutait plus cher que le GEMM.
        self._chunk = min(8192, n * n * n)
        self._h1 = np.empty((self._chunk, h), dtype=np.float32)
        self._h2 = np.empty((self._chunk, h), dtype=np.float32)
        self._disp = np.empty((n * n * n, 1), dtype=np.float32)

    def _fill_spatial_features(self) -> None:
        f = float(self.params["freq"])
        pi = np.float32(np.pi)
        for i, g in enumerate(self._coords):
            flat = g.reshape(-1)
            for j, mult in enumerate((1.0, 2.0)):
                arg = (pi * f * mult) * flat
                self._X[:, 4 * i + 2 * j] = np.sin(arg)
                self._X[:, 4 * i + 2 * j + 1] = np.cos(arg)

    # ------------------------------------------------------------------
    def command(self, message: dict) -> None:
        if message.get("type") == "reseed":
            self.params["seed"] = int(message.get("seed",
                                                  (int(self.params["seed"]) % 64) + 1))
            self.setup()

    def step(self, dt: float) -> Frame:
        n = self.n
        self.tick += 1
        self.sim_time += dt
        self.phase += dt * float(self.params["morph"])

        if str(self.params["shape"]) != self._shape_key:   # changement non structurel
            gx, gy, gz = self._coords
            self._base = np.ascontiguousarray(
                _base_sdf(str(self.params["shape"]), gx, gy, gz), dtype=np.float32)
            self._shape_key = str(self.params["shape"])

        self._X[:, 12] = np.float32(np.sin(self.phase))
        self._X[:, 13] = np.float32(np.cos(self.phase))

        with self.timer():
            # MLP par point, par paquets : (c,14) -> (c,h) -> (c,h) -> (c,1)
            step_rows = self._chunk
            total = n * n * n
            for s0 in range(0, total, step_rows):
                s1 = min(s0 + step_rows, total)
                rows = s1 - s0
                h1 = self._h1[:rows]
                h2 = self._h2[:rows]
                self.k.matmul_nt_gelu(self._X[s0:s1], self.W1, self.b1, out=h1)
                self.k.matmul_nt_gelu(h1, self.W2, self.b2, out=h2)
                self.k.matmul_nt(h2, self.W3, out=self._disp[s0:s1])
            disp = self._disp.reshape(n, n, n)

            rms = float(np.sqrt(np.mean(disp * disp)) + 1e-6)
            amp = float(self.params["amplitude"])

            # Sphere tracing : le rendu GPU n'est correct que si |grad d| <= 1.
            # On borne d'abord la pente du deplacement, puis on renormalise le
            # champ complet si besoin. grad(base) est precalcule.
            dgx, dgy, dgz = np.gradient(disp, self._voxel)
            dnorm = np.sqrt(dgx * dgx + dgy * dgy + dgz * dgz)
            # quantile et non maximum : quelques pics isoles ne doivent pas
            # ecraser toute la sculpture (le garde-fou final reste exact)
            dgq = float(np.quantile(dnorm, 0.995)) + 1e-9
            gain = np.float32(min(amp / rms, amp / dgq))
            gx_t = self._base_grad[0] + gain * dgx
            gy_t = self._base_grad[1] + gain * dgy
            gz_t = self._base_grad[2] + gain * dgz
            grad_max = float(np.sqrt(gx_t * gx_t + gy_t * gy_t + gz_t * gz_t).max())
            lip = max(grad_max, 1.0)
            sdf = (self._base + gain * disp) / np.float32(lip)
            sdf = np.ascontiguousarray(sdf, dtype=np.float32)

        near = float(np.mean(np.abs(sdf) < (1.5 * self._voxel)))
        payload, scale = quantize_i16(sdf)
        gflops = (self._flops / 1e9) / max(self._compute_ms / 1e3, 1e-9)
        return Frame(
            kind="sdf_grid",
            payload=payload.reshape(-1),
            shape=(n, n, n),
            scale=scale,
            stats={
                "grid": n,
                "hidden": int(self.params["hidden"]),
                "points": n ** 3,
                "k_in": self.k_in,
                "gflops": round(gflops, 2),
                "mflop_per_tick": round(self._flops / 1e6, 1),
                "grad_max": round(grad_max, 4),
                "lipschitz": round(grad_max / lip, 4),   # <= 1 apres correction
                "surface_frac": round(near, 5),
                "voxel": round(self._voxel, 5),
                "shape": self._shape_key,
                "phase": round(self.phase, 3),
            },
        )

    def metadata(self) -> dict:
        return {"domain": [-1.0, 1.0], "encoding": "sin/cos x2 frequences",
                "layers": [self.k_in, int(self.params["hidden"]),
                           int(self.params["hidden"]), 1]}
