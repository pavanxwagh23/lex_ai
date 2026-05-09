from __future__ import annotations

import logging
from pathlib import Path

import uvicorn


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "backend-launcher.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


if __name__ == "__main__":
    try:
        uvicorn.run(
            "backend.main:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
        )
    except Exception:
        logging.exception("Backend failed to start")
        raise
