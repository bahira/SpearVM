"""Cas d'usage 3 — Entrainement live d'un reseau sur une surface cible.

Un MLP (2 couches cachees GELU) apprend une hauteur `z = f(x, y)`. Tout le pas
d'entrainement passe par SpearVM :

  forward   : `matmul_nt` + `gelu`   (et `matmul_nt_gelu` fusionne pour l'eval)
  backward  : `gelu_backward` puis `matmul_backward` (dX, dW en une passe NT)
  optimiseur: Adam numpy (peu de FLOPs, garde le code lisible)

Le navigateur affiche la surface cible (fantome) et la surface predite qui
converge, plus la courbe de perte — la visualisation *est* le gradcheck.
"""

from __future__ import annotations

import numpy as np

from .base import Frame, ParamSpec, Simulation

TARGETS = ("ridges", "islands", "crater", "saddle")


def target_field(name: str, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Surfaces cibles, coordonnees dans [0,1]."""
    x = 2.0 * gx - 1.0
    y = 2.0 * gy - 1.0
    if name == "ridges":
        z = 0.6 * np.sin(3.2 * np.pi * x) * np.cos(2.4 * np.pi * y) * np.exp(-1.1 * (x * x + y * y))
    elif name == "islands":
        z = np.zeros_like(x)
        for cx, cy, a, s in ((-0.45, -0.3, 0.9, 0.16), (0.4, 0.35, 0.75, 0.10),
                             (0.15, -0.5, -0.6, 0.07), (-0.3, 0.55, 0.5, 0.05)):
            z = z + a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * s))
    elif name == "crater":
        r = np.sqrt(x * x + y * y)
        z = np.exp(-((r - 0.45) ** 2) / 0.02) - 0.75 * np.exp(-(r * r) / 0.05)
    else:  # saddle
        z = 0.8 * (x * x - y * y) * np.exp(-0.8 * (x * x + y * y))
    return np.asarray(z, dtype=np.float32)


def features(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Encodage de Fourier 2 frequences -> 8 canaux (K=8, multiple de 4)."""
    x = gx.reshape(-1).astype(np.float32)
    y = gy.reshape(-1).astype(np.float32)
    pi = np.float32(np.pi)
    cols = [
        np.sin(2 * pi * x), np.cos(2 * pi * x),
        np.sin(2 * pi * y), np.cos(2 * pi * y),
        np.sin(4 * pi * x), np.cos(4 * pi * x),
        np.sin(4 * pi * y), np.cos(4 * pi * y),
    ]
    return np.ascontiguousarray(np.stack(cols, axis=1), dtype=np.float32)


class TrainerSim(Simulation):
    sim_id = "trainer"
    title = "Entrainement live"
    subtitle = "forward fusionne + backprop SpearVM"
    description = (
        "Un MLP apprend une surface cible en direct : forward `matmul_nt_gelu`, "
        "backward `gelu_backward` + `matmul_backward`, optimiseur Adam. "
        "La surface predite converge vers le fantome cible sous vos yeux."
    )
    kernels_used = ("matmul_nt", "matmul_nt_gelu", "gelu", "gelu_backward", "matmul_backward")
    default_rate = 20.0
    structural_params = ("hidden", "target", "seed", "eval_grid")

    params_spec = (
        ParamSpec("target", "Surface cible", "select", "ridges", options=list(TARGETS)),
        ParamSpec("hidden", "Largeur cachee", "select", 96, options=[32, 64, 96, 128]),
        ParamSpec("lr", "Learning rate", "range", 0.012, minimum=0.0005, maximum=0.05, step=0.0005),
        ParamSpec("batch", "Taille de batch", "select", 2048, options=[512, 1024, 2048, 4096]),
        ParamSpec("steps", "Pas SGD / tick", "range", 4, minimum=1, maximum=12, step=1),
        ParamSpec("eval_grid", "Grille d'affichage", "select", 72, options=[48, 64, 72, 96]),
        ParamSpec("seed", "Graine", "range", 3, minimum=1, maximum=64, step=1),
    )

    # ------------------------------------------------------------------
    def setup(self) -> None:
        g = int(self.params["eval_grid"])
        h = int(self.params["hidden"])
        rng = np.random.default_rng(int(self.params["seed"]))
        self.g = g
        self.h = h
        self.rng = rng

        axis = (np.arange(g, dtype=np.float32) + 0.5) / g
        ex, ey = np.meshgrid(axis, axis, indexing="ij")
        self.eval_X = features(ex, ey)
        self.target_eval = target_field(str(self.params["target"]), ex, ey)

        # jeu d'entrainement : grille plus fine, echantillonnee en minibatchs
        t = 128
        taxis = (np.arange(t, dtype=np.float32) + 0.5) / t
        tx, ty = np.meshgrid(taxis, taxis, indexing="ij")
        self.train_X = features(tx, ty)
        self.train_y = target_field(str(self.params["target"]), tx, ty).reshape(-1, 1)

        k_in = self.train_X.shape[1]
        self.W1 = (rng.standard_normal((h, k_in)) * np.sqrt(2.0 / k_in)).astype(np.float32)
        self.b1 = np.zeros(h, dtype=np.float32)
        self.W2 = (rng.standard_normal((h, h)) * np.sqrt(2.0 / h)).astype(np.float32)
        self.b2 = np.zeros(h, dtype=np.float32)
        self.W3 = (rng.standard_normal((1, h)) * np.sqrt(1.0 / h)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

        self._adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in self._params().items()}
        self._adam_t = 0
        self.step_count = 0
        self.loss_hist: list[float] = []
        self.loss0: float | None = None

    def _params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    # ------------------------------------------------------------------
    def command(self, message: dict) -> None:
        if message.get("type") in ("reset", "reseed"):
            if "seed" in message:
                self.params["seed"] = int(message["seed"])
            else:
                self.params["seed"] = (int(self.params["seed"]) % 64) + 1
            self.setup()

    def initial_frames(self) -> list[Frame]:
        return [
            Frame(
                kind="target_grid",
                payload=np.ascontiguousarray(self.target_eval.reshape(-1), dtype=np.float32),
                shape=(self.g, self.g),
                stats={"target": self.params["target"]},
            )
        ]

    # ------------------------------------------------------------------
    def _train_step(self, lr: float, batch: int) -> float:
        idx = self.rng.integers(0, self.train_X.shape[0], size=batch)
        X = np.ascontiguousarray(self.train_X[idx])
        y = np.ascontiguousarray(self.train_y[idx])

        # ---- forward (on garde les pre-activations pour la backprop) ----
        z1 = self.k.matmul_nt(X, self.W1) + self.b1
        a1 = np.ascontiguousarray(self.k.gelu(z1), dtype=np.float32)
        z2 = self.k.matmul_nt(a1, self.W2) + self.b2
        a2 = np.ascontiguousarray(self.k.gelu(z2), dtype=np.float32)
        pred = self.k.matmul_nt(a2, self.W3) + self.b3

        diff = (pred - y).astype(np.float32)
        loss = float(np.mean(diff * diff))

        # ---- backward ----
        dpred = np.ascontiguousarray((2.0 / batch) * diff, dtype=np.float32)
        da2, dW3 = self.k.matmul_backward(dpred, a2, self.W3)
        db3 = dpred.sum(axis=0, dtype=np.float32)

        dz2 = np.ascontiguousarray(self.k.gelu_backward(da2, z2), dtype=np.float32)
        da1, dW2 = self.k.matmul_backward(dz2, a1, self.W2)
        db2 = dz2.sum(axis=0, dtype=np.float32)

        dz1 = np.ascontiguousarray(self.k.gelu_backward(da1, z1), dtype=np.float32)
        _, dW1 = self.k.matmul_backward(dz1, X, self.W1)
        db1 = dz1.sum(axis=0, dtype=np.float32)

        self._adam_step({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                         "W3": dW3, "b3": db3}, lr)
        self.step_count += 1
        return loss

    def _adam_step(self, grads: dict[str, np.ndarray], lr: float,
                   b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> None:
        self._adam_t += 1
        t = self._adam_t
        for name, param in self._params().items():
            g = np.asarray(grads[name], dtype=np.float32).reshape(param.shape)
            m, v = self._adam[name]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            mhat = m / (1.0 - b1 ** t)
            vhat = v / (1.0 - b2 ** t)
            param -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

    def step(self, dt: float) -> Frame:
        self.tick += 1
        self.sim_time += dt
        lr = float(self.params["lr"])
        batch = int(self.params["batch"])
        n_steps = int(self.params["steps"])

        with self.timer():
            loss = 0.0
            for _ in range(n_steps):
                loss = self._train_step(lr, batch)
            # eval : forward fusionne (gelu + matmul en un passage)
            h1 = self.k.matmul_nt_gelu(self.eval_X, self.W1, self.b1)
            h2 = self.k.matmul_nt_gelu(h1, self.W2, self.b2)
            pred = (self.k.matmul_nt(h2, self.W3) + self.b3).reshape(self.g, self.g)
            pred = np.ascontiguousarray(pred, dtype=np.float32)

        if self.loss0 is None:
            self.loss0 = loss
        self.loss_hist.append(loss)
        if len(self.loss_hist) > 512:
            self.loss_hist = self.loss_hist[-512:]

        err = np.abs(pred - self.target_eval)
        k_in = self.train_X.shape[1]
        flops_step = 2.0 * batch * (k_in * self.h + self.h * self.h + self.h) * 3.0
        gflops = (flops_step * n_steps / 1e9) / max(self._compute_ms / 1e3, 1e-9)

        return Frame(
            kind="prediction_grid",
            payload=pred.reshape(-1),
            shape=(self.g, self.g),
            stats={
                "loss": float(loss),
                "loss0": float(self.loss0),
                "improvement": round(float(self.loss0 / max(loss, 1e-12)), 2),
                "steps": self.step_count,
                "samples": self.step_count * batch,
                "linf": round(float(err.max()), 4),
                "rmse": round(float(np.sqrt(np.mean(err * err))), 5),
                "gflops": round(gflops, 2),
                "history": [round(v, 6) for v in self.loss_hist[-160:]],
                "target": self.params["target"],
                "hidden": self.h,
            },
        )

    def metadata(self) -> dict:
        return {"grid": self.g, "hidden": self.h, "target": self.params["target"]}
