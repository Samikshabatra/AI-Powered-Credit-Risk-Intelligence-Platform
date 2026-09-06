"""Self-correcting NL -> SQL agent.

The loop, once per question:

    Sonnet (schema + metrics + exemplars, cached)
        -> tool call run_sql(sql)
            -> guardrails reject it, or SQLite errors  -> error text goes back
               to Sonnet, which rewrites the query (up to SQL_MAX_RETRIES times)
            -> success -> rows go to **Haiku**, which writes the business answer

Handing the result rows to the cheap model rather than back to Sonnet is a
deliberate token decision. Summarising a result set is the easiest step in the
pipeline and the one with the largest payload, so it runs on the cheapest model;
Sonnet's context never grows by the rows at all, which keeps the cached prefix
intact for the next turn.

Multi-turn memory keeps the previous questions and the SQL that answered them -
but never the result rows - so "and for women only?" resolves against the last
query while the conversation stays cheap to carry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.talk_to_data.prompt_templates import (
    PROMPT_VERSION, SQL_TOOLS, SUMMARY_SYSTEM, build_retry_message,
    build_summary_prompt, build_system_prompt,
)
from src.talk_to_data.query_runner import (
    QueryResult, SQLValidationError, get_schema, run_sql,
)
from src.utils.config import settings
from src.utils.llm import LLMUnavailable, get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

FALLBACK_MARKER = "CANNOT_ANSWER"
MAX_SUMMARY_ROWS = 25


@dataclass
class NLQueryResult:
    """Everything one natural-language question produced, including its failures."""

    question: str
    answer: str
    sql: str | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    row_count: int = 0
    attempts: int = 0
    errors: list[str] = field(default_factory=list)
    fallback: bool = False
    success: bool = False
    elapsed_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sql": self.sql,
            "row_count": self.row_count,
            "attempts": self.attempts,
            "errors": self.errors,
            "fallback": self.fallback,
            "success": self.success,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "cache_read": self.cache_read_tokens,
            },
            "prompt_version": self.prompt_version,
        }

    def scalar(self) -> float | str | None:
        """The single value, when the query returned exactly one - used by the evals."""
        if self.rows.shape == (1, 1):
            return self.rows.iloc[0, 0]
        return None


class NLToSQLAgent:
    """Turns questions into answers, with the SQL kept visible at every step."""

    def __init__(self, max_retries: int | None = None, use_memory: bool = True) -> None:
        self.client = get_client()  # raises LLMUnavailable if no API key
        self.max_retries = settings.sql_max_retries if max_retries is None else max_retries
        self.use_memory = use_memory
        self.schema_ddl = get_schema()
        self.system_prompt = build_system_prompt(self.schema_ddl)
        self.history: list[dict[str, str]] = []
        # The last successful execution, kept so a caller (the UI) can show the
        # SQL and the rows behind an answer. History deliberately stores only the
        # question and the SQL, so the rows would otherwise be unrecoverable.
        self.last_result: QueryResult | None = None
        logger.info(
            "NL->SQL agent ready (prompt %s, %d chars of cached prefix, %d retries)",
            PROMPT_VERSION, len(self.system_prompt), self.max_retries,
        )

    # ------------------------------------------------------------------ #
    def reset_memory(self) -> None:
        self.history.clear()
        self.last_result = None

    def _messages_for(self, question: str) -> list[dict[str, Any]]:
        """Conversation so far (questions + the SQL that answered them) + the new one."""
        messages: list[dict[str, Any]] = []
        if self.use_memory:
            for turn in self.history[-4:]:  # 4 turns is enough for follow-ups
                messages.append({"role": "user", "content": turn["question"]})
                messages.append({
                    "role": "assistant",
                    "content": f"I ran:\n{turn['sql']}\n\n{turn['answer']}",
                })
        messages.append({"role": "user", "content": question})
        return messages

    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> NLQueryResult:
        """Answer one natural-language question end to end."""
        started = time.perf_counter()
        result = NLQueryResult(question=question, answer="")
        messages = self._messages_for(question)
        query_result: QueryResult | None = None

        for attempt in range(1, self.max_retries + 2):  # first try + retries
            result.attempts = attempt
            response = self.client.complete(
                system=self.system_prompt,
                messages=messages,
                model=settings.sql_model,
                tools=SQL_TOOLS,
                max_tokens=1200,
                cache_system=True,
                label=f"nl_to_sql:attempt{attempt}",
            )
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            result.cache_read_tokens += response.usage.cache_read_tokens

            # --- the model declined to answer: the designed fallback path ---
            if not response.tool_calls:
                text = response.text.strip()
                if FALLBACK_MARKER in text:
                    result.fallback = True
                    result.answer = text.split(FALLBACK_MARKER, 1)[1].lstrip(": ").strip() \
                        or "That cannot be answered from this dataset."
                else:
                    result.answer = text or "No answer was produced."
                    result.fallback = True
                break

            tool_call = response.tool_calls[0]

            # --- schema request: cheap, answer it and let the model continue ---
            if tool_call["name"] == "get_schema":
                messages.append({"role": "assistant", "content": response.raw.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": self.schema_ddl,
                    }],
                })
                continue

            sql = str(tool_call["input"].get("sql", "")).strip()
            result.sql = sql

            # --- run it through the guardrails ---
            try:
                query_result = run_sql(sql)
                self.last_result = query_result
                result.success = True
                break
            except (SQLValidationError, FileNotFoundError) as exc:
                error = str(exc)
                result.errors.append(error)
                logger.info("Attempt %d rejected: %s", attempt, error[:120])
                if attempt > self.max_retries:
                    break
                messages.append({"role": "assistant", "content": response.raw.content})
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": build_retry_message(sql, error),
                        "is_error": True,
                    }],
                })

        # --- summarise, on the cheap model ---
        if result.success and query_result is not None:
            result.rows = query_result.rows
            result.row_count = query_result.row_count
            result.sql = query_result.executed_sql
            result.answer = self._summarise(question, query_result)
            if self.use_memory:
                self.history.append({
                    "question": question, "sql": result.sql, "answer": result.answer,
                })
        elif not result.answer:
            result.answer = (
                "I could not produce a valid query for that question after "
                f"{result.attempts} attempts. Last error: "
                f"{result.errors[-1] if result.errors else 'unknown'}"
            )
            result.fallback = True

        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    # ------------------------------------------------------------------ #
    def _summarise(self, question: str, query_result: QueryResult) -> str:
        """Haiku turns result rows into prose. Falls back to the raw table."""
        try:
            response = self.client.complete(
                system=SUMMARY_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": build_summary_prompt(
                        question,
                        query_result.executed_sql,
                        query_result.to_compact_text(MAX_SUMMARY_ROWS),
                    ),
                }],
                model=settings.summary_model,
                max_tokens=350,
                cache_system=True,
                label="nl_to_sql:summary",
            )
            return response.text or query_result.to_markdown()
        except Exception as exc:  # a summarisation failure must not lose the data
            logger.warning("Summarisation failed (%s); returning the raw table.", exc)
            return query_result.to_markdown()


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #
_AGENT: NLToSQLAgent | None = None


def get_agent(reset: bool = False) -> NLToSQLAgent:
    """Shared agent instance, so the cached prefix and memory persist across calls."""
    global _AGENT
    if _AGENT is None or reset:
        _AGENT = NLToSQLAgent()
    return _AGENT


def ask(question: str) -> NLQueryResult:
    """Answer one question. Raises LLMUnavailable when no API key is configured."""
    return get_agent().ask(question)


def ask_safely(question: str) -> NLQueryResult:
    """Same, but degrades to an explanatory result instead of raising."""
    try:
        return ask(question)
    except LLMUnavailable as exc:
        return NLQueryResult(
            question=question,
            answer=f"Talk-to-Data needs an AI API key. {exc}",
            fallback=True,
        )
