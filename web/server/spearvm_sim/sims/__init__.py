"""Registre des simulations exposees par le serveur."""

from __future__ import annotations

from typing import Any

from .base import Frame, ParamSpec, Simulation
from .flow_field import FlowFieldSim
from .trainer import TrainerSim
from .wave_field import WaveFieldSim

REGISTRY: dict[str, type[Simulation]] = {
    cls.sim_id: cls for cls in (FlowFieldSim, WaveFieldSim, TrainerSim)
}

__all__ = [
    "Frame",
    "ParamSpec",
    "Simulation",
    "FlowFieldSim",
    "WaveFieldSim",
    "TrainerSim",
    "REGISTRY",
    "describe_all",
    "create",
]


def describe_all() -> list[dict[str, Any]]:
    return [cls.describe() for cls in REGISTRY.values()]


def create(sim_id: str, params: dict[str, Any] | None = None) -> Simulation:
    if sim_id not in REGISTRY:
        raise KeyError(sim_id)
    return REGISTRY[sim_id](params)
