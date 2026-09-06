"""Runtime/environment helpers: container detection, data & artifact readiness.

The Streamlit app and the entrypoint script both need to answer "is this thing
built yet?" without importing the heavy ML stack, so the checks live here.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.utils.config import settings

REQUIRED_RAW_FILES = ("application_train.csv",)
OPTIONAL_RAW_FILES = ("bureau.csv", "previous_application.csv", "application_test.csv")


def in_docker() -> bool:
    """True when running inside a container (cgroup marker or explicit env flag)."""
    if os.environ.get("RUNNING_IN_DOCKER") == "1":
        return True
    if Path("/.dockerenv").exists():
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text()
    except (OSError, FileNotFoundError):
        return False


def raw_data_status() -> dict[str, bool]:
    """Which raw CSVs are present, keyed by filename."""
    return {
        name: (settings.raw_dir / name).exists()
        for name in REQUIRED_RAW_FILES + OPTIONAL_RAW_FILES
    }


def raw_data_available() -> bool:
    return all((settings.raw_dir / name).exists() for name in REQUIRED_RAW_FILES)


def database_ready() -> bool:
    return settings.sqlite_path.exists() and settings.sqlite_path.stat().st_size > 0


def model_ready() -> bool:
    return settings.model_path.exists() and settings.preprocessor_path.exists()


def rules_ready() -> bool:
    return settings.rules_path.exists()


def artifact_status() -> dict[str, bool]:
    return {
        "raw_data": raw_data_available(),
        "database": database_ready(),
        "model": model_ready(),
        "rules": rules_ready(),
        "holdout": settings.holdout_path.exists(),
    }


def missing_data_message() -> str:
    """Actionable instructions shown in the UI when the dataset is absent."""
    return (
        f"Home Credit CSVs were not found in `{settings.raw_dir}`.\n\n"
        "Download them from "
        "https://www.kaggle.com/competitions/home-credit-default-risk/data "
        "and unzip `application_train.csv`, `application_test.csv`, `bureau.csv` and "
        f"`previous_application.csv` into `{settings.raw_dir}` "
        "(or mount that folder into the container)."
    )


def disk_free_gb(path: Path | None = None) -> float:
    target = path or settings.data_dir
    target = target if target.exists() else target.parent
    return shutil.disk_usage(target).free / 1e9
