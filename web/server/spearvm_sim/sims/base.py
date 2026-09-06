"""Contrat commun a toutes les simulations exposees par le serveur."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..kernels import Kernels, get_kernels
from ..protocol import encode_frame


@dataclass
class Frame:
    """Resultat d'un pas de simulation, pret a etre serialise."""

    kind: str
    payload: np.ndarray | None
    shape: tuple[int, ...] = ()
    scale: float = 1.0
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamSpec:
    """Description d'un parametre pilotable depuis l'UI."""

    key: str
    label: str
    kind: str  # "range" | "select" | "bool"
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: list[Any] | None = None
    help: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "help": self.help,
        }
        if self.kind == "range":
            out.update({"min": self.minimum, "max": self.maximum, "step": self.step})
        if self.kind == "select":
            out["options"] = self.options
        return out


class Simulation:
    """Simulation temps reel pilotee par le serveur.

    Cycle de vie : `__init__(params)` -> `step(dt)` en boucle -> `command(...)`
    a la demande du client. Une instance appartient a une seule connexion
    WebSocket : pas d'etat partage, pas de verrou necessaire.
    """

    sim_id: str = "base"
    title: str = ""
    subtitle: str = ""
    description: str = ""
    kernels_used: tuple[str, ...] = ()
    default_rate: float = 30.0
    params_spec: tuple[ParamSpec, ...] = ()

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.k: Kernels = get_kernels()
        self.params: dict[str, Any] = {p.key: p.default for p in self.params_spec}
        self.tick = 0
        self.sim_time = 0.0
        self._compute_ms = 0.0
        self._ms_ema = 0.0
        if params:
            self.apply_params(params)
        self.setup()

    # -- surcharges ---------------------------------------------------------
    def setup(self) -> None:
        """Alloue l'etat. Appele apres chaque changement structurel."""

    def step(self, dt: float) -> Frame:  # pragma: no cover - abstrait
        raise NotImplementedError

    def command(self, message: dict[str, Any]) -> None:
        """Commande client (clic, impulsion, reset...). Silencieux par defaut."""

    def initial_frames(self) -> list["Frame"]:
        """Frames envoyees une fois a la connexion (ex : la surface cible)."""
        return []

    def metadata(self) -> dict[str, Any]:
        return {}

    # -- utilitaires --------------------------------------------------------
    def apply_params(self, incoming: dict[str, Any]) -> bool:
        """Valide/clampe les parametres. True si un `setup()` est necessaire."""
        structural = False
        for spec in self.params_spec:
            if spec.key not in incoming:
                continue
            value = incoming[spec.key]
            try:
                if spec.kind == "range":
                    value = float(value)
                    if spec.minimum is not None:
                        value = max(spec.minimum, value)
                    if spec.maximum is not None:
                        value = min(spec.maximum, value)
                    if spec.step is not None and float(spec.step).is_integer() and spec.step >= 1:
                        value = float(int(round(value / spec.step) * spec.step))
                elif spec.kind == "bool":
                    value = bool(value)
                elif spec.kind == "select":
                    if spec.options and value not in spec.options:
                        continue
            except (TypeError, ValueError):
                continue
            if self.params.get(spec.key) != value:
                self.params[spec.key] = value
                if spec.key in self.structural_params:
                    structural = True
        return structural

    structural_params: tuple[str, ...] = ()

    def reconfigure(self, incoming: dict[str, Any]) -> None:
        if self.apply_params(incoming):
            self.setup()

    def timer(self):
        return _Timer(self)

    def encode(self, frame: Frame) -> bytes:
        stats = dict(frame.stats)
        stats.setdefault("backend", self.k.name)
        stats.setdefault("compute_ms_avg", round(self._ms_ema, 3))
        return encode_frame(
            frame.kind,
            frame.payload,
            tick=self.tick,
            sim_time=self.sim_time,
            compute_ms=self._compute_ms,
            shape=frame.shape,
            scale=frame.scale,
            stats=stats,
        )

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "id": cls.sim_id,
            "title": cls.title,
            "subtitle": cls.subtitle,
            "description": cls.description,
            "kernels": list(cls.kernels_used),
            "rate": cls.default_rate,
            "params": [p.to_json() for p in cls.params_spec],
        }


class _Timer:
    """Chronometre le coeur de calcul d'un pas (hors serialisation)."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        ms = (time.perf_counter() - self._t0) * 1e3
        self.sim._compute_ms = ms
        self.sim._ms_ema = ms if self.sim._ms_ema == 0.0 else 0.9 * self.sim._ms_ema + 0.1 * ms


def periodic_features(coords: Iterable[np.ndarray], freq: float = 1.0) -> np.ndarray:
    """Encodage sin/cos periodique -> le champ appris se raccorde aux bords."""
    parts: list[np.ndarray] = []
    for c in coords:
        parts.append(np.sin(2.0 * np.pi * freq * c))
        parts.append(np.cos(2.0 * np.pi * freq * c))
    return np.stack(parts, axis=-1).astype(np.float32)
