"""Point d'entree : `python -m spearvm_sim`."""

from __future__ import annotations

import uvicorn

from .config import settings


def main() -> None:
    uvicorn.run(
        "spearvm_sim.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )


if __name__ == "__main__":
    main()
