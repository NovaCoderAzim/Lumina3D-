"""
Logging helpers.

Guide section 11 requires that backend logs capture, per processing step:
timestamp, project_id, module, status, error, processing_time.
`log_module_event` emits exactly that as a single structured line.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from ..config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    # Windows consoles often default to cp1252, which can't encode the
    # non-ASCII characters that show up in third-party error messages.
    # Force UTF-8 on the stream where possible so logging never crashes.
    try:  # pragma: no cover - platform dependent
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    # Avoid duplicate handlers if uvicorn already attached one.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_module_event(
    logger: logging.Logger,
    *,
    project_id: str,
    module: str,
    status: str,
    processing_time: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Emit the structured per-module event required by guide section 11."""
    parts = [
        f"project_id={project_id}",
        f"module={module}",
        f"status={status}",
    ]
    if processing_time is not None:
        parts.append(f"processing_time={processing_time:.2f}s")
    if error:
        parts.append(f"error={error!r}")
    line = " ".join(parts)
    if status.upper() in {"FAILED", "ERROR"}:
        logger.error(line)
    else:
        logger.info(line)
