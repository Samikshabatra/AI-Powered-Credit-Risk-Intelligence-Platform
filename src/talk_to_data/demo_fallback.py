"""Cached Talk-to-Data answers for the public demo.

The deployed app runs on a public URL with one API key behind it. Two states
have to be survivable without the page looking broken:

* no key configured at all, and
* `DEMO_MODE` on with the per-session call budget already spent.

In both cases the assistant falls back to here. The fallback is not a canned
string: `reports/nl_sql_eval.json` records, for all 25 labelled evaluation
cases, the exact SQL the model wrote during the measured run. That SQL is
**replayed against the live database** and the rows are formatted
deterministically, so a cached answer is a real query result computed now -
against whichever warehouse is mounted - rather than a number frozen at eval
time. On the sampled deployment the figures are therefore the sample's own, and
they stay correct if the database is rebuilt.

Every answer produced here is labelled "cached demo output" by the caller. It
must never be presented as the agent having run: the routing, the retry loop and
the self-correction are exactly what the cache cannot reproduce.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.utils.config import settings
from src.utils.helpers import load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Words that carry no discriminating signal when matching a typed question
# against the recorded ones. Deliberately small: over-stripping makes "how many
# applicants" and "how many occupations" collide.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "in", "on", "at",
    "to", "and", "or", "by", "with", "what", "which", "show", "list", "give",
    "me", "please", "do", "does", "did", "have", "has", "that", "this", "it",
    "be", "as", "from", "there", "their", "our", "us", "you", "i",
})

# Below this overlap the recorded question is a different question, and pretending
# otherwise would be worse than saying the cache has no answer. Tuned against the
# recorded set: plain Jaccard puts "which occupations have the highest default
# rate" and "what is the overall default rate" at 0.40, because three of the four
# content words are shared boilerplate. Weighting by inverse document frequency
# separates them - "occupations" is rare across the corpus, "rate" is not.
_MATCH_FLOOR = 0.34

_REFUSAL_TEXT = (
    "That question cannot be answered from this warehouse. The tables hold loan "
    "applications, credit-bureau summaries, prior applications and model "
    "predictions - no cardholder data, no branch or loan-officer records, and no "
    "forward-looking series. Refusing is the recorded behaviour for this question "
    "in the evaluation set, not a failure to understand it."
)


@dataclass
class DemoAnswer:
    """One replayed answer, plus everything the UI needs to label it honestly."""

    answer: str
    matched_question: str | None = None
    case_id: str | None = None
    sql: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    elapsed_ms: float = 0.0
    match_score: float = 0.0
    replayed: bool = False
    matched: bool = True

    def as_message(self) -> dict[str, Any]:
        """Shaped like the SQL-evidence payload the chat transcript already stores."""
        payload: dict[str, Any] = {"role": "assistant", "content": self.answer}
        if self.sql:
            payload |= {
                "sql": self.sql,
                "rows": self.rows,
                "row_count": self.row_count,
                "elapsed_ms": self.elapsed_ms,
            }
        return payload


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in _STOPWORDS}


@lru_cache(maxsize=1)
def _cases() -> list[dict[str, Any]]:
    """Recorded evaluation cases, newest report wins. Empty list if none shipped."""
    report = load_json(settings.eval_results_path)
    if not report:
        return []
    return [case for case in report.get("cases_detail", []) if case.get("question")]


@lru_cache(maxsize=1)
def _idf() -> dict[str, float]:
    """Inverse document frequency of each token over the recorded questions."""
    cases = _cases()
    document_frequency: dict[str, int] = {}
    for case in cases:
        for token in _tokens(case["question"]):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    total = max(len(cases), 1)
    return {
        token: math.log(1 + total / count)
        for token, count in document_frequency.items()
    }


def _weight(tokens: set[str]) -> float:
    """Total IDF mass of a token set. Unseen tokens get the maximum weight."""
    table = _idf()
    unseen = math.log(1 + max(len(_cases()), 1))
    return sum(table.get(token, unseen) for token in tokens)


def cached_questions() -> list[str]:
    """The questions the cache can answer, for the UI to offer as buttons."""
    return [case["question"] for case in _cases() if not case.get("fallback_expected")]


def available() -> bool:
    return bool(_cases())


def _best_match(question: str) -> tuple[dict[str, Any] | None, float]:
    asked = _tokens(question)
    if not asked:
        return None, 0.0

    best: dict[str, Any] | None = None
    best_score = 0.0
    for case in _cases():
        recorded = _tokens(case["question"])
        if not recorded:
            continue
        # IDF-weighted Jaccard over content words: symmetric, so neither a terse
        # nor a verbose phrasing is favoured, but shared boilerplate ("default",
        # "rate") cannot carry a match on its own.
        union = _weight(asked | recorded)
        score = _weight(asked & recorded) / union if union else 0.0
        if score > best_score:
            best, best_score = case, score
    return best, best_score


def _format_rows(rows: list[dict[str, Any]], row_count: int) -> str:
    """Query rows -> a sentence or a short markdown table, deterministically."""
    if not rows:
        return "The query returned no rows."

    # A single scalar - "what is the default rate" - reads as a sentence.
    if len(rows) == 1 and len(rows[0]) == 1:
        (label, value), = rows[0].items()
        return f"**{_humanise(label)}:** {_format_value(value)}"

    header = list(rows[0].keys())
    lines = [
        "| " + " | ".join(_humanise(column) for column in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows[:10]:
        lines.append("| " + " | ".join(_format_value(row[c]) for c in header) + " |")
    table = "\n".join(lines)
    if row_count > 10:
        table += f"\n\nShowing 10 of {row_count:,} rows."
    return table


def _humanise(column: str) -> str:
    return column.replace("_", " ").strip().capitalize()


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        # Rates in this schema are 0-1 averages of TARGET; anything larger is a
        # money amount or a count, which reads better with separators.
        if 0.0 <= value <= 1.0:
            return f"{value:.4f}"
        return f"{value:,.2f}"
    return str(value)


def demo_answer(question: str) -> DemoAnswer:
    """Best cached answer for `question`. Always returns something renderable."""
    if not available():
        return DemoAnswer(
            answer=(
                "The AI assistant is unavailable in this build and no recorded "
                "evaluation run was published with it, so there is nothing to "
                "replay. Every other section - scoring, explanation, policy rules, "
                "EDA and fairness - runs without a model call."
            ),
            matched=False,
        )

    case, score = _best_match(question)
    if case is None or score < _MATCH_FLOOR:
        catalogue = "\n".join(f"- {q}" for q in cached_questions()[:8])
        return DemoAnswer(
            answer=(
                "No recorded answer is close enough to that question to replay "
                "honestly. The cached demo can answer these:\n\n" + catalogue
            ),
            match_score=round(score, 3),
            matched=False,
        )

    if case.get("fallback_expected") or not case.get("sql"):
        return DemoAnswer(
            answer=_REFUSAL_TEXT,
            matched_question=case["question"],
            case_id=case.get("id"),
            match_score=round(score, 3),
        )

    sql = case["sql"]
    try:
        from src.talk_to_data.query_runner import run_sql

        result = run_sql(sql)
        body = _format_rows(result.rows.head(25).to_dict(orient="records"),
                            result.row_count)
        return DemoAnswer(
            answer=body,
            matched_question=case["question"],
            case_id=case.get("id"),
            sql=result.executed_sql,
            rows=result.rows.head(25).to_dict(orient="records"),
            row_count=result.row_count,
            elapsed_ms=result.elapsed_ms,
            match_score=round(score, 3),
            replayed=True,
        )
    except Exception as exc:  # the database may be absent or rebuilt differently
        logger.warning("Cached demo replay failed for %s: %s", case.get("id"), exc)
        return DemoAnswer(
            answer=(
                "The recorded query for this question could not be replayed against "
                f"the database ({type(exc).__name__}). The SQL it would have run is "
                "shown below."
            ),
            matched_question=case["question"],
            case_id=case.get("id"),
            sql=sql,
            match_score=round(score, 3),
        )
