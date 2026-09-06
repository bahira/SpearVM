"""Configuration du serveur — 100 % variables d'environnement (12-factor)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = SERVER_DIR.parent
DEFAULT_STATIC = WEB_DIR / "client" / "dist"


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # 0.0.0.0 is intentional for containers; bind to 127.0.0.1 when local.
    host: str = os.environ.get("SPEARVM_HOST", "0.0.0.0")  # nosec B104
    port: int = _int("SPEARVM_PORT", 8000)
    log_level: str = os.environ.get("SPEARVM_LOG_LEVEL", "info")

    max_clients: int = _int("SPEARVM_MAX_CLIENTS", 8)
    max_rate: float = _float("SPEARVM_MAX_RATE", 60.0)
    min_rate: float = _float("SPEARVM_MIN_RATE", 1.0)
    idle_timeout_s: float = _float("SPEARVM_IDLE_TIMEOUT", 900.0)
    auth_required: bool = os.environ.get("SPEARVM_AUTH_REQUIRED", "0").lower() in {"1", "true", "yes"}
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            entry.strip() for entry in os.environ.get("SPEARVM_API_KEYS", "").split(",") if entry.strip()
        )
    )
    max_clients_per_tenant: int = _int("SPEARVM_MAX_CLIENTS_PER_TENANT", 4)

    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.environ.get("SPEARVM_CORS", "*").split(",") if o.strip()
        )
    )
    static_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("SPEARVM_STATIC", str(DEFAULT_STATIC)))
    )

    @property
    def serve_static(self) -> bool:
        return (self.static_dir / "index.html").exists()


settings = Settings()
