"""Every canonical metric must be executable SQL that returns a sane value.

A semantic layer that is only a prompt string is a promise. Executing each
formula against the real database is what turns it into a guarantee - and it
catches the exact failure it exists to prevent, which is the `DAYS_EMPLOYED`
sentinel leaking back into an average.
"""

from __future__ import annotations

import pytest

from src.talk_to_data.query_runner import run_sql
from src.talk_to_data.semantic_layer import (
    METRICS, METRICS_BY_NAME, get_metric, render_for_prompt, render_markdown,
)
from tests.conftest import requires_database

# Metrics computed over the predictions table rather than applications.
_PREDICTION_METRICS = {"approval_rate", "predicted_default_rate"}
# Row-level expressions: they need a row context, not an aggregate wrapper.
_ROW_LEVEL = {"age", "employment_years", "credit_income_ratio", "debt_service_ratio",
              "external_score_mean", "age_band", "income_bracket"}


def _table_for(name: str) -> str:
    return "predictions" if name in _PREDICTION_METRICS else "applications"


@requires_database
@pytest.mark.parametrize("metric", METRICS, ids=[m.name for m in METRICS])
def test_metric_expression_executes(metric) -> None:
    """Each formula runs without error against the live schema."""
    table = _table_for(metric.name)
    if metric.name in _ROW_LEVEL:
        sql = f"SELECT {metric.expression} AS value FROM {table} LIMIT 5"
    else:
        sql = f"SELECT {metric.expression} AS value FROM {table}"
    result = run_sql(sql, validate=False)
    assert result.row_count >= 1


@requires_database
def test_default_rate_matches_the_known_base_rate() -> None:
    result = run_sql(f"SELECT {get_metric('default_rate').expression} FROM applications")
    assert 0.080 < float(result.rows.iloc[0, 0]) < 0.081


@requires_database
def test_default_rate_formula_survives_integer_division() -> None:
    """The trap this metric exists to avoid: COUNT/COUNT truncates to 0 in SQLite."""
    naive = run_sql(
        "SELECT COUNT(CASE WHEN TARGET = 1 THEN 1 END) / COUNT(*) AS rate FROM applications",
        validate=False,
    )
    assert float(naive.rows.iloc[0, 0]) == 0.0  # the wrong answer, demonstrated

    canonical = run_sql(
        f"SELECT {get_metric('default_rate').expression} AS rate FROM applications"
    )
    assert float(canonical.rows.iloc[0, 0]) > 0.08  # the right one


@requires_database
def test_employment_years_excludes_the_sentinel() -> None:
    """The whole point of the canonical definition."""
    canonical = run_sql(
        f"SELECT AVG({get_metric('employment_years').expression}) AS years FROM applications",
        validate=False,
    )
    years = float(canonical.rows.iloc[0, 0])
    assert 5 < years < 8, f"Employment years should be single digits, got {years}"

    naive = run_sql(
        "SELECT AVG(-DAYS_EMPLOYED / 365.0) AS years FROM applications", validate=False
    )
    assert float(naive.rows.iloc[0, 0]) < -100  # the 1000-year career, demonstrated


@requires_database
def test_age_is_positive_and_plausible() -> None:
    result = run_sql(
        f"SELECT MIN({get_metric('age').expression}) AS lo, "
        f"MAX({get_metric('age').expression}) AS hi FROM applications",
        validate=False,
    )
    low, high = float(result.rows.iloc[0, 0]), float(result.rows.iloc[0, 1])
    assert 18 <= low < 30 and 60 < high < 80


@requires_database
def test_external_score_mean_ignores_missing_scores() -> None:
    """Averaging only the present scores, not treating NULL as zero."""
    expression = get_metric("external_score_mean").expression
    result = run_sql(
        f"SELECT {expression} AS score FROM applications "
        "WHERE EXT_SOURCE_1 IS NULL AND EXT_SOURCE_2 IS NOT NULL LIMIT 200",
        validate=False,
    )
    scores = result.rows["score"].dropna()
    assert not scores.empty
    assert scores.between(0, 1).all()


@requires_database
def test_age_band_partitions_the_whole_population() -> None:
    expression = get_metric("age_band").expression
    result = run_sql(
        f"SELECT {expression} AS band, COUNT(*) AS n FROM applications GROUP BY band",
        validate=False,
    )
    assert result.rows["n"].sum() == 307_511
    assert result.rows["band"].notna().all()


def test_prompt_block_contains_every_metric() -> None:
    block = render_for_prompt()
    for metric in METRICS:
        assert metric.name in block
        assert metric.expression in block


def test_prompt_block_stays_compact() -> None:
    """It sits in the cached prefix, but it is still read on every turn."""
    block = render_for_prompt()
    assert len(block) < 4000, f"Semantic block grew to {len(block)} chars"


def test_markdown_rendering_is_a_table() -> None:
    markdown = render_markdown()
    assert markdown.startswith("| Metric |")
    assert len(markdown.splitlines()) == len(METRICS) + 2


def test_metric_names_are_unique() -> None:
    assert len(METRICS_BY_NAME) == len(METRICS)
