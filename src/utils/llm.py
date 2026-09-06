"""Shared Anthropic client: prompt caching, token accounting, graceful degradation.

Every model call in the platform goes through `LLMClient` so that three
cross-cutting concerns are handled once rather than three times:

* **Prompt caching.** The NL->SQL system prompt (schema DDL + semantic layer +
  few-shot exemplars) is ~2.5k stable tokens resent on every turn. Marking the
  final system block with `cache_control: ephemeral` turns those into cache
  reads at 10% of the input price from the second call onwards. `usage`
  reports `cache_read_input_tokens`, so the saving is measured, not asserted.
* **Token accounting.** A process-wide `TokenLedger` records every call, which
  is what the README's before/after token table and the eval harness report.
* **Graceful degradation.** No API key is a normal state (the evaluator may run
  the container without one). Callers catch `LLMUnavailable` and fall back to
  deterministic behaviour rather than crashing the app.

Model split is deliberate: Sonnet writes SQL and routes the agent, Haiku turns
result rows into prose. Summarisation is the highest-volume, lowest-difficulty
call in the system, so it runs on the cheap model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached: no API key, or the SDK is absent."""


@dataclass
class CallRecord:
    """One model request, for the token report."""

    label: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    latency_ms: float
    stop_reason: str | None = None

    @property
    def billable_input(self) -> float:
        """Cache reads bill at 10% of base input; cache writes at 125%."""
        return (
            self.input_tokens
            + 0.1 * self.cache_read_tokens
            + 1.25 * self.cache_creation_tokens
        )


@dataclass
class TokenLedger:
    """Process-wide usage accumulator behind the README's token numbers."""

    records: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.records.append(record)

    def reset(self) -> None:
        self.records.clear()

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"calls": 0}
        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        cache_reads = sum(r.cache_read_tokens for r in self.records)
        cache_writes = sum(r.cache_creation_tokens for r in self.records)
        billable = sum(r.billable_input for r in self.records)
        # What the same traffic would have cost with caching switched off: every
        # cached token would have been a full-price input token instead.
        uncached_equivalent = total_input + cache_reads + cache_writes

        return {
            "calls": len(self.records),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": cache_reads,
            "cache_creation_tokens": cache_writes,
            "billable_input_tokens": round(billable, 1),
            "uncached_equivalent_input_tokens": uncached_equivalent,
            "input_token_saving_pct": (
                round(1 - billable / uncached_equivalent, 4) if uncached_equivalent else 0.0
            ),
            "cache_hit_rate": (
                round(cache_reads / uncached_equivalent, 4) if uncached_equivalent else 0.0
            ),
            "median_latency_ms": round(
                sorted(r.latency_ms for r in self.records)[len(self.records) // 2], 1
            ),
            "by_model": _group_by_model(self.records),
        }


def _group_by_model(records: Iterable[CallRecord]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = grouped.setdefault(
            record.model, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        bucket["calls"] += 1
        bucket["input_tokens"] += record.input_tokens
        bucket["output_tokens"] += record.output_tokens
    return grouped


LEDGER = TokenLedger()


@dataclass
class LLMResponse:
    """Normalised model response: text, tool calls and the raw SDK message."""

    text: str
    tool_calls: list[dict[str, Any]]
    stop_reason: str | None
    usage: CallRecord
    raw: Any = None


class LLMClient:
    """Thin wrapper over `anthropic.Anthropic` with caching and accounting built in."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key if api_key is not None else settings.anthropic_api_key
        if not key:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a key "
                "to enable Talk-to-Data, the agent and LLM-phrased reason codes."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise LLMUnavailable(f"anthropic SDK not installed: {exc}") from exc

        self._client = Anthropic(api_key=key)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _system_blocks(system: str, cache: bool) -> list[dict[str, Any]]:
        """System prompt as blocks, with a cache breakpoint on the stable prefix."""
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache and settings.enable_prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        cache_system: bool = True,
        label: str = "call",
    ) -> LLMResponse:
        """Send one request and return the normalised response.

        No `temperature`: the Messages API in anthropic SDK 1.x does not accept
        it (it survives only on the legacy completions endpoint), and passing it
        raises TypeError. Determinism for SQL generation therefore comes from
        the prompt - a pinned semantic layer and worked exemplars - rather than
        from a sampling parameter.
        """
        model = model or settings.sql_model
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or settings.max_tokens,
            "system": self._system_blocks(system, cache_system),
            "messages": messages,
        }
        if tools:
            request["tools"] = tools

        started = time.perf_counter()
        message = self._client.messages.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

        usage = CallRecord(
            label=label,
            model=model,
            input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
            output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
            cache_creation_tokens=getattr(message.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(message.usage, "cache_read_input_tokens", 0) or 0,
            latency_ms=round(latency_ms, 1),
            stop_reason=message.stop_reason,
        )
        LEDGER.add(usage)
        logger.debug(
            "%s | %s | in=%d cached=%d out=%d | %.0fms",
            label, model, usage.input_tokens, usage.cache_read_tokens,
            usage.output_tokens, latency_ms,
        )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=message.stop_reason,
            usage=usage,
            raw=message,
        )

    def count_tokens(self, system: str, messages: list[dict[str, Any]],
                     model: str | None = None) -> int:
        """Exact input-token count without spending an inference call."""
        result = self._client.messages.count_tokens(
            model=model or settings.sql_model,
            system=system,
            messages=messages,
        )
        return int(result.input_tokens)


@lru_cache(maxsize=1)
def get_client() -> LLMClient:
    """Cached client. Raises LLMUnavailable if no key is configured."""
    return LLMClient()


def llm_available() -> bool:
    """True when AI features can run. Used to gate the UI, never to crash it."""
    try:
        get_client()
        return True
    except LLMUnavailable:
        return False
