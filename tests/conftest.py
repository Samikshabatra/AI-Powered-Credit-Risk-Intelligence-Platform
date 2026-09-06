"""Shared fixtures.

Tests split into two families:

* **Always-on** - guardrails, semantic-layer SQL, preprocessing, evaluation
  maths, reason-code generation. These need no artifacts beyond the database (or
  nothing at all) and run in CI on every commit.
* **Artifact-dependent** - anything needing the trained model. Skipped with a
  clear message rather than failing when the pipeline has not been run, so a
  fresh clone gets a green suite instead of a wall of red.

No test calls the Anthropic API. LLM behaviour is covered by the eval harness,
which is a measurement tool, not a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import settings  # noqa: E402
from src.utils.docker_utils import database_ready, model_ready  # noqa: E402

requires_database = pytest.mark.skipif(
    not database_ready(),
    reason="No SQLite database. Run `python -m src.data.build_database`.",
)
requires_model = pytest.mark.skipif(
    not model_ready(),
    reason="No trained model. Run `python -m src.ml.train`.",
)


@pytest.fixture(scope="session")
def scorer():
    from src.ml.predict import get_scorer

    if not model_ready():
        pytest.skip("No trained model.")
    return get_scorer()


@pytest.fixture(scope="session")
def holdout():
    from src.ml.predict import load_holdout

    if not settings.holdout_path.exists():
        pytest.skip("No holdout snapshot.")
    return load_holdout()


@pytest.fixture(scope="session")
def sample_applicant(holdout):
    """One high-risk applicant - the interesting path through every module."""
    high = holdout[holdout["RISK_BAND"] == "High"]
    pool = high if not high.empty else holdout
    return pool.iloc[0]


@pytest.fixture
def raw_frame():
    """A small slice of the real joined dataset for preprocessing tests."""
    from src.data.loader import load_dataset

    try:
        return load_dataset(nrows=None).head(3000)
    except FileNotFoundError:
        pytest.skip("Raw dataset not available.")
