"""Small shared utilities: JSON IO, formatting, timing, deterministic seeding."""

from __future__ import annotations

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# The Home Credit sentinel: 1000 years of "employment" encodes "not employed".
DAYS_EMPLOYED_SENTINEL = 365243


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_json(payload: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote %s", path.name)
    return path


def load_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    logger.info("%s took %.1fs", label, time.perf_counter() - start)


def fmt_money(value: float) -> str:
    """Compact currency for UI/report text: 1_234_567 -> '$1.23M'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= cutoff:
            return f"{sign}${value / cutoff:,.2f}{suffix}"
    return f"{sign}${value:,.0f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def humanise_feature(name: str) -> str:
    """Raw column name -> readable label for non-technical users."""
    from src.data.feature_dictionary import FEATURE_LABELS

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    return name.replace("_", " ").title()
