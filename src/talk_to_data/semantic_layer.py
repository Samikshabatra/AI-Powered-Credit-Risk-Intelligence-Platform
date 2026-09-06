"""Canonical metric definitions - one agreed formula per business concept.

The problem this solves is specific. Ask an LLM for "the default rate" three
times and you can get `AVG(TARGET)`, `SUM(TARGET)*1.0/COUNT(*)` and
`COUNT(CASE WHEN TARGET=1 THEN 1 END)/COUNT(*)`. All three are arguably right;
the third silently returns integer 0 in SQLite. Ask for "employment length" and
you get `-DAYS_EMPLOYED/365`, which for 18% of rows reports a 1000-year career
because of the 365243 sentinel.

So the formulas are defined here, once, and injected into every NL->SQL prompt.
Three things follow:

* **Reliability** - the same question yields the same SQL across turns and users.
* **Hallucination control** - the model reuses a pinned formula instead of
  inventing one, and the sentinel bug is baked into `employment_years` so the
  metric cannot lie even when the model forgets the caveat.
* **Token discipline** - a ~350-token canonical block replaces the model
  re-deriving business logic in reasoning tokens on every single turn.

This module is pure data plus formatting. It has no dependency on the LLM, so
it is directly unit-testable: `test_semantic_layer.py` executes every formula
against the real database.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Metric:
    """One canonical metric: name, SQL expression, and why it is written that way."""

    name: str
    expression: str
    description: str
    note: str | None = None


METRICS: tuple[Metric, ...] = (
    Metric(
        name="default_rate",
        expression="AVG(TARGET)",
        description="Share of applicants who defaulted. Multiply by 100 for a percentage.",
        note="AVG over the 0/1 TARGET. Never COUNT/COUNT - SQLite integer division "
             "returns 0.",
    ),
    Metric(
        name="default_count",
        expression="SUM(TARGET)",
        description="Number of applicants who defaulted.",
    ),
    Metric(
        name="applicant_count",
        expression="COUNT(*)",
        description="Number of applicants in the group.",
    ),
    Metric(
        name="age",
        expression="(-DAYS_BIRTH / 365.0)",
        description="Applicant age in years.",
        note="DAYS_BIRTH is stored negative, so it must be negated before dividing.",
    ),
    Metric(
        name="employment_years",
        expression=(
            "CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL "
            "ELSE -DAYS_EMPLOYED / 365.0 END"
        ),
        description="Years in current employment; NULL when the applicant is not employed.",
        note="365243 is Home Credit's 'not employed' sentinel. Without the CASE it "
             "reads as 1000 years and destroys every average built on it.",
    ),
    Metric(
        name="avg_income",
        expression="AVG(AMT_INCOME_TOTAL)",
        description="Average declared annual income.",
    ),
    Metric(
        name="avg_credit",
        expression="AVG(AMT_CREDIT)",
        description="Average loan amount requested.",
    ),
    Metric(
        name="avg_annuity",
        expression="AVG(AMT_ANNUITY)",
        description="Average yearly repayment amount.",
    ),
    Metric(
        name="credit_income_ratio",
        expression="(AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0))",
        description="Loan-to-income ratio for one applicant.",
        note="NULLIF guards against a zero-income row.",
    ),
    Metric(
        name="debt_service_ratio",
        expression="(AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0))",
        description="Share of annual income absorbed by the loan repayment.",
    ),
    Metric(
        name="external_score_mean",
        expression=(
            "((COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) "
            "+ COALESCE(EXT_SOURCE_3, 0)) / NULLIF("
            "(EXT_SOURCE_1 IS NOT NULL) + (EXT_SOURCE_2 IS NOT NULL) "
            "+ (EXT_SOURCE_3 IS NOT NULL), 0))"
        ),
        description="Average of whichever external credit scores are present (0-1, "
                    "higher is safer).",
        note="Averages only the non-NULL scores; EXT_SOURCE_1 is missing for 56% of rows.",
    ),
    Metric(
        name="age_band",
        expression=(
            "CASE WHEN -DAYS_BIRTH / 365.0 < 25 THEN 'Under 25' "
            "WHEN -DAYS_BIRTH / 365.0 < 35 THEN '25-34' "
            "WHEN -DAYS_BIRTH / 365.0 < 45 THEN '35-44' "
            "WHEN -DAYS_BIRTH / 365.0 < 60 THEN '45-59' "
            "ELSE '60+' END"
        ),
        description="Standard reporting age bands.",
    ),
    Metric(
        name="income_bracket",
        expression=(
            "CASE WHEN AMT_INCOME_TOTAL < 100000 THEN 'Under 100k' "
            "WHEN AMT_INCOME_TOTAL < 150000 THEN '100k-150k' "
            "WHEN AMT_INCOME_TOTAL < 200000 THEN '150k-200k' "
            "WHEN AMT_INCOME_TOTAL < 300000 THEN '200k-300k' "
            "ELSE '300k+' END"
        ),
        description="Standard reporting income brackets.",
    ),
    Metric(
        name="approval_rate",
        expression="AVG(CASE WHEN DECISION = 'Approve' THEN 1.0 ELSE 0.0 END)",
        description="Share of applicants the model approves at the cost-optimal threshold.",
        note="Requires the `predictions` table, written by src/ml/train.py.",
    ),
    Metric(
        name="predicted_default_rate",
        expression="AVG(DEFAULT_PROBABILITY)",
        description="Average model-predicted probability of default (calibrated).",
        note="From `predictions`. Compare against default_rate on `applications` to "
             "check calibration for any slice.",
    ),
)

METRICS_BY_NAME: dict[str, Metric] = {metric.name: metric for metric in METRICS}


def get_metric(name: str) -> Metric | None:
    return METRICS_BY_NAME.get(name)


def render_for_prompt() -> str:
    """The canonical-metric block injected into the NL->SQL system prompt.

    Kept terse on purpose: it sits inside the cached prefix, but every token
    here is still a token the model reads on every turn.
    """
    lines = [
        "CANONICAL METRIC DEFINITIONS - use these exact expressions. "
        "Do not invent an alternative formula for a metric defined here.",
    ]
    for metric in METRICS:
        lines.append(f"- {metric.name}: {metric.expression}")
        lines.append(f"    {metric.description}")
        if metric.note:
            lines.append(f"    NOTE: {metric.note}")
    return "\n".join(lines)


def render_markdown() -> str:
    """The same definitions as a table, for the UI and the README."""
    rows = ["| Metric | SQL definition | Meaning |", "|---|---|---|"]
    for metric in METRICS:
        expression = metric.expression.replace("|", "\\|")
        rows.append(f"| `{metric.name}` | `{expression}` | {metric.description} |")
    return "\n".join(rows)
