"""NL->SQL evaluation harness.

    python -m src.talk_to_data.eval_harness              # all cases
    python -m src.talk_to_data.eval_harness --ids q01 q08

"Five query patterns work" is a claim. This turns it into a number.

Twenty-five hand-labelled cases in `evaluation/nl_sql_questions.jsonl` cover
every required pattern plus the ones that actually break systems: the
`DAYS_EMPLOYED` sentinel, a multi-turn follow-up that only resolves against
conversation memory, three questions the warehouse genuinely cannot answer, and
one prompt-injection attempt.

Each case carries a **ground-truth SQL query, not a hardcoded number**, so the
expected answer is recomputed from the live database on every run. A rebuilt
database or a resampled split cannot silently invalidate the labels.

Reported metrics, in decreasing order of how much they mean:

    answer_accuracy      the value returned matches ground truth  <- headline
    execution_accuracy   the generated SQL ran at all
    fallback_accuracy    unanswerable questions were correctly refused
    retry_rate           share of cases needing self-correction
    sql_similarity       normalised string overlap with the reference SQL
                         (soft signal only: many correct queries look nothing
                         like the reference, so a low score here means little)
    latency / tokens     median per question, with cache hit rate
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.talk_to_data.nl_to_sql import NLQueryResult, NLToSQLAgent
from src.talk_to_data.prompt_templates import PROMPT_VERSION
from src.talk_to_data.query_runner import run_sql
from src.utils.config import PROJECT_ROOT, settings
from src.utils.helpers import save_json
from src.utils.llm import LEDGER, LLMUnavailable
from src.utils.logger import get_logger

logger = get_logger(__name__)

CASES_PATH = PROJECT_ROOT / "evaluation" / "nl_sql_questions.jsonl"


@dataclass
class EvalCase:
    """One labelled question."""

    id: str
    pattern: str
    question: str
    expected_kind: str
    expected_sql: str | None = None
    key_column: str | None = None
    k: int | None = None
    min_rows: int | None = None
    tolerance: float = 0.01
    depends_on: str | None = None


@dataclass
class CaseOutcome:
    """What the agent did on one case, and whether it was right."""

    id: str
    pattern: str
    question: str
    executed: bool
    correct: bool
    fallback_expected: bool
    fallback_taken: bool
    attempts: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    sql: str | None = None
    expected: Any = None
    actual: Any = None
    sql_similarity: float = 0.0
    failure_reason: str | None = None
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_cases(path: Path | None = None) -> list[EvalCase]:
    path = path or CASES_PATH
    cases: list[EvalCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(EvalCase(**{k: v for k, v in payload.items() if k in EvalCase.__annotations__}))
    logger.info("Loaded %d evaluation cases from %s", len(cases), path.name)
    return cases


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _numbers_match(expected: Any, actual: Any, tolerance: float) -> bool:
    """Relative comparison for numbers, exact for everything else."""
    if expected is None or actual is None:
        return False
    try:
        expected_value, actual_value = float(expected), float(actual)
    except (TypeError, ValueError):
        return str(expected).strip().lower() == str(actual).strip().lower()
    if np.isnan(expected_value) and np.isnan(actual_value):
        return True
    if tolerance == 0:
        return expected_value == actual_value
    denominator = max(abs(expected_value), 1e-9)
    return abs(expected_value - actual_value) / denominator <= tolerance


def _first_numeric_column(frame: pd.DataFrame, exclude: str | None = None) -> str | None:
    for column in frame.columns:
        if column == exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            return column
    return None


def _key_value_map(frame: pd.DataFrame, key_column: str | None) -> dict[str, float] | None:
    """Reduce a result table to {group key: metric} for comparison.

    The agent is free to name its columns differently from the reference, so
    matching is positional-by-type: the first non-numeric column is the key and
    the first numeric column is the value.
    """
    if frame.empty:
        return None
    key = key_column if key_column in frame.columns else None
    if key is None:
        candidates = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        key = candidates[0] if candidates else frame.columns[0]
    value_column = _first_numeric_column(frame, exclude=key)
    if value_column is None:
        return None
    return {
        str(k).strip().lower(): float(v)
        for k, v in zip(frame[key], frame[value_column])
        if pd.notna(v)
    }


def _normalise_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower().replace('"', "").replace("`", "")).strip()


def _sql_similarity(expected: str | None, actual: str | None) -> float:
    """Token-level Jaccard overlap. A soft signal, reported but never gating."""
    if not expected or not actual:
        return 0.0
    expected_tokens = set(re.findall(r"[a-z_0-9.]+", _normalise_sql(expected)))
    actual_tokens = set(re.findall(r"[a-z_0-9.]+", _normalise_sql(actual)))
    if not expected_tokens or not actual_tokens:
        return 0.0
    return round(
        len(expected_tokens & actual_tokens) / len(expected_tokens | actual_tokens), 3
    )


def grade(case: EvalCase, result: NLQueryResult) -> CaseOutcome:
    """Compare one agent result against its ground truth."""
    outcome = CaseOutcome(
        id=case.id,
        pattern=case.pattern,
        question=case.question,
        executed=result.success,
        correct=False,
        fallback_expected=case.expected_kind == "fallback",
        fallback_taken=result.fallback,
        attempts=result.attempts,
        latency_ms=round(result.elapsed_ms, 1),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        sql=result.sql,
        errors=result.errors,
    )

    # --- questions the warehouse cannot answer: refusing IS the right answer ---
    if case.expected_kind == "fallback":
        outcome.correct = result.fallback and not result.success
        outcome.expected = "refusal"
        outcome.actual = "refused" if outcome.correct else "answered anyway"
        if not outcome.correct:
            outcome.failure_reason = "Answered a question it should have refused."
        return outcome

    if not result.success:
        outcome.failure_reason = (
            result.errors[-1] if result.errors else "No query was executed."
        )
        return outcome

    truth = run_sql(case.expected_sql, validate=False)
    outcome.sql_similarity = _sql_similarity(case.expected_sql, result.sql)

    if case.expected_kind == "scalar":
        expected = truth.rows.iloc[0, 0]
        actual = result.scalar()
        if actual is None:
            # A single-row, multi-column answer still counts if one cell matches.
            numeric = result.rows.select_dtypes("number")
            actual = numeric.iloc[0, 0] if len(numeric) == 1 and not numeric.empty else None
        outcome.expected, outcome.actual = expected, actual
        outcome.correct = _numbers_match(expected, actual, case.tolerance)

    elif case.expected_kind in {"table", "top_k"}:
        expected_map = _key_value_map(truth.rows, case.key_column)
        actual_map = _key_value_map(result.rows, case.key_column)
        outcome.expected, outcome.actual = expected_map, actual_map
        if not expected_map or not actual_map:
            outcome.failure_reason = "Result could not be reduced to key/value pairs."
        elif case.expected_kind == "top_k":
            k = case.k or len(expected_map)
            expected_keys = list(expected_map)[:k]
            actual_keys = list(actual_map)[:k]
            outcome.correct = expected_keys == actual_keys
            if not outcome.correct:
                outcome.failure_reason = (
                    f"Ranking differs: expected {expected_keys}, got {actual_keys}."
                )
        else:
            missing = set(expected_map) - set(actual_map)
            if missing:
                outcome.failure_reason = f"Missing groups: {sorted(missing)}"
            else:
                mismatched = [
                    key for key, value in expected_map.items()
                    if not _numbers_match(value, actual_map[key], case.tolerance)
                ]
                outcome.correct = not mismatched
                if mismatched:
                    outcome.failure_reason = f"Values differ for: {mismatched[:3]}"

    elif case.expected_kind == "row_count":
        minimum = case.min_rows or 1
        outcome.expected, outcome.actual = f">= {minimum} rows", f"{result.row_count} rows"
        outcome.correct = result.row_count >= minimum
        if not outcome.correct:
            outcome.failure_reason = f"Returned {result.row_count} rows, expected >= {minimum}."

    return outcome


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_evaluation(
    case_ids: list[str] | None = None,
    path: Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Run every case through a single agent instance and aggregate the results.

    One agent for the whole suite is intentional - it exercises the prompt cache
    the way a real session does - but **conversation memory is reset before each
    case**. The agent keeps a rolling four-turn window, so running 25 unrelated
    questions back to back makes it read an independent question as a follow-up
    to whatever happened to come before it, and decline. That is a property of
    the harness, not of the agent: measured on the first full run, q15, q17 and
    q25 were refused in sequence and all three answer correctly in isolation.

    A case with `depends_on` is the one exception: its antecedent is replayed
    immediately before it, so the follow-up is graded against exactly the
    context a real user would have given it - which is the point of that case.
    The antecedent turn is not graded twice.
    """
    cases = load_cases(path)
    if case_ids:
        wanted = set(case_ids)
        # Pull in any case a selected follow-up depends on.
        wanted |= {c.depends_on for c in cases if c.id in wanted and c.depends_on}
        cases = [c for c in cases if c.id in wanted]
    by_id = {case.id: case for case in cases}

    LEDGER.reset()
    agent = NLToSQLAgent()
    outcomes: list[CaseOutcome] = []

    for case in cases:
        logger.info("[%s] %s", case.id, case.question)
        agent.reset_memory()
        antecedent = by_id.get(case.depends_on) if case.depends_on else None
        if antecedent is not None:
            logger.info("  context turn <- [%s] %s", antecedent.id, antecedent.question)
            agent.ask(antecedent.question)
        result = agent.ask(case.question)
        outcome = grade(case, result)
        outcomes.append(outcome)
        logger.info(
            "  %s | %d attempt(s) | %.0fms%s",
            "PASS" if outcome.correct else "FAIL",
            outcome.attempts, outcome.latency_ms,
            "" if outcome.correct else f" | {outcome.failure_reason}",
        )

    report = summarise(outcomes)
    if save:
        save_json(report, settings.eval_results_path)
    _log_report(report)
    return report


def summarise(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    """Aggregate case outcomes into the metrics table the README publishes."""
    if not outcomes:
        return {"cases": 0}

    answerable = [o for o in outcomes if not o.fallback_expected]
    unanswerable = [o for o in outcomes if o.fallback_expected]
    latencies = sorted(o.latency_ms for o in outcomes)

    def rate(numerator: int, denominator: int) -> float | None:
        # None, not 0.0: an empty denominator means "not measured in this run",
        # and reporting that as 0% would look like a total failure.
        return round(numerator / denominator, 4) if denominator else None

    by_pattern: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        bucket = by_pattern.setdefault(outcome.pattern, {"cases": 0, "correct": 0})
        bucket["cases"] += 1
        bucket["correct"] += int(outcome.correct)
    for bucket in by_pattern.values():
        bucket["accuracy"] = rate(bucket["correct"], bucket["cases"])

    return {
        "prompt_version": PROMPT_VERSION,
        "cases": len(outcomes),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "headline": {
            "answer_accuracy": rate(sum(o.correct for o in answerable), len(answerable)),
            "execution_accuracy": rate(sum(o.executed for o in answerable), len(answerable)),
            "fallback_accuracy": rate(sum(o.correct for o in unanswerable), len(unanswerable)),
        },
        "reliability": {
            "retry_rate": rate(sum(o.attempts > 1 for o in outcomes), len(outcomes)),
            "mean_attempts": round(float(np.mean([o.attempts for o in outcomes])), 2),
            "false_fallback_rate": rate(
                sum(o.fallback_taken for o in answerable), len(answerable)
            ),
        },
        "sql_similarity_median": round(
            float(np.median([o.sql_similarity for o in answerable])), 3
        ) if answerable else 0.0,
        "latency_ms": {
            "median": latencies[len(latencies) // 2],
            "p90": latencies[min(int(len(latencies) * 0.9), len(latencies) - 1)],
            "max": latencies[-1],
        },
        "tokens": {
            "median_input_per_question": int(
                np.median([o.input_tokens for o in outcomes])
            ),
            "median_output_per_question": int(
                np.median([o.output_tokens for o in outcomes])
            ),
            "total_cache_read": sum(o.cache_read_tokens for o in outcomes),
            "ledger": LEDGER.summary(),
        },
        "by_pattern": by_pattern,
        "failures": [
            {"id": o.id, "question": o.question, "reason": o.failure_reason, "sql": o.sql}
            for o in outcomes if not o.correct
        ],
        "cases_detail": [asdict(o) for o in outcomes],
    }


def _log_report(report: dict[str, Any]) -> None:
    headline = report.get("headline", {})
    logger.info("=" * 68)
    logger.info("NL->SQL evaluation (%d cases, prompt %s)",
                report["cases"], report["prompt_version"])
    def pct(value: float | None) -> str:
        return "not measured" if value is None else f"{100 * value:.1f}%"

    logger.info("  Answer accuracy     %s", pct(headline.get("answer_accuracy")))
    logger.info("  Execution accuracy  %s", pct(headline.get("execution_accuracy")))
    logger.info("  Fallback accuracy   %s", pct(headline.get("fallback_accuracy")))
    logger.info("  Retry rate          %s", pct(report["reliability"]["retry_rate"]))
    logger.info("  Median latency      %.0f ms", report["latency_ms"]["median"])
    ledger = report["tokens"]["ledger"]
    if ledger.get("calls"):
        logger.info("  Cache hit rate      %.1f%% (input token saving %.1f%%)",
                    100 * ledger.get("cache_hit_rate", 0),
                    100 * ledger.get("input_token_saving_pct", 0))
    logger.info("=" * 68)


def results_table(report: dict[str, Any]) -> pd.DataFrame:
    """Per-case results as a table for the UI and the README."""
    return pd.DataFrame([
        {
            "ID": case["id"],
            "Pattern": case["pattern"],
            "Question": case["question"],
            "Result": "PASS" if case["correct"] else "FAIL",
            "Attempts": case["attempts"],
            "Latency (ms)": case["latency_ms"],
        }
        for case in report.get("cases_detail", [])
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NL->SQL evaluation suite.")
    parser.add_argument("--ids", nargs="*", default=None, help="run only these case ids")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    try:
        report = run_evaluation(case_ids=args.ids, save=not args.no_save)
    except LLMUnavailable as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    print(json.dumps(report["headline"], indent=2))


if __name__ == "__main__":
    main()
