"""Logging setup: one configured root handler, module-scoped child loggers."""

from __future__ import annotations

import logging
import sys

from src.utils.config import settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("credit_risk")
    root.setLevel(settings.log_level.upper())
    root.handlers = [handler]
    root.propagate = False
    # Third-party noise we never want in the demo output.
    for noisy in ("httpx", "anthropic", "matplotlib", "shap", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, e.g. get_logger(__name__)."""
    _configure_root()
    short = name.replace("src.", "").replace("__main__", "main")
    return logging.getLogger(f"credit_risk.{short}")
