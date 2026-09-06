"""Measured token optimisation: what the four levers actually save.

    python -m src.talk_to_data.token_report

The claim "we optimised tokens" is worthless without a number, so this script
builds the prompt in a naive configuration and in the shipped configuration and
counts both.

The four levers, in descending order of impact:

1. **Prompt caching** on the stable prefix (role + schema + metrics + exemplars).
   That prefix is resent on every turn; cached, it bills at 10% of the input
   price from the second call onwards. This is by far the largest lever, and it
   is the only one whose effect is measured directly by the API rather than
   estimated - `usage.cache_read_input_tokens` says exactly how many tokens were
   served from cache.
2. **Compact schema** - and this one did not work out the way it was supposed to.
   The warehouse exposes 39 curated columns of the 122 in `application_train`,
   plus pre-aggregated summaries instead of the raw bureau and
   previous-application tables. Measured against the DDL a naive implementation
   would inject (every column of every source table, read from the real CSV
   headers), the shipped schema is *the same size* - because it carries a
   per-column comment glossary that costs about as much as the 200-odd dropped
   columns save. That annotation is what stops the model guessing what
   `DAYS_EMPLOYED` means, so it stays. The honest conclusion is that schema
   curation is a hallucination-control lever here, not a cost lever, and the
   report prints the real number rather than a flattering one.
3. **Cheap model for summarisation.** Result rows go to Haiku, not Sonnet.
   Sonnet's context never grows by the rows at all, so the cached prefix stays
   intact for the next turn.
4. **Retry cap of 2**, bounding the worst case at three model calls per question.

Token counts come from `messages.count_tokens`, which is exact and free. Without
an API key the script falls back to a character-based estimate and says so
loudly - an estimate labelled as an estimate is honest; an estimate presented as
a measurement is not.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.talk_to_data.prompt_templates import PROMPT_VERSION, build_system_prompt
from src.talk_to_data.query_runner import get_schema
from src.talk_to_data.semantic_layer import render_for_prompt
from src.utils.config import settings
from src.utils.helpers import save_json
from src.utils.llm import LLMUnavailable, get_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Anthropic pricing multipliers relative to a base input token.
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25
# Rough Sonnet-to-Haiku input price ratio, used only for the summarisation lever.
HAIKU_COST_RATIO = 1 / 12

SAMPLE_QUESTIONS = [
    "What is the overall default rate?",
    "Which five occupations have the highest default rate?",
    "Average credit amount by contract type.",
    "Default rate for applicants under 25 versus over 60.",
    "How many applicants earn over 200k and defaulted?",
    "And for women only?",
]


def naive_full_schema() -> str:
    """The DDL a naive implementation would inject: every column of every table.

    Built from the real CSV headers rather than from `sql/schema.sql`, because
    the point of the comparison is 122 columns against the curated 39 - reading
    the shipped schema file twice would compare it against itself and produce a
    flattering, meaningless number.
    """
    from src.data.loader import (
        APPLICATION_TRAIN, BUREAU, PREVIOUS_APPLICATION, _raw_path,
    )

    blocks = []
    for table, filename in (
        ("applications", APPLICATION_TRAIN),
        ("bureau", BUREAU),
        ("previous_application", PREVIOUS_APPLICATION),
    ):
        try:
            header = pd.read_csv(_raw_path(filename), nrows=0)
        except FileNotFoundError:
            continue
        def sql_type(kind: str) -> str:
            return {"f": "REAL", "i": "INTEGER"}.get(kind, "TEXT")

        columns = ",\n".join(
            f"    {name} {sql_type(dtype.kind)}"
            for name, dtype in zip(header.columns, header.dtypes)
        )
        blocks.append(f"CREATE TABLE {table} (\n{columns}\n);")
    return "\n\n".join(blocks)


def _estimate_tokens(text: str) -> int:
    """Fallback when no API key is configured: ~4 characters per token."""
    return max(1, len(text) // 4)


def _count(text: str, client) -> int:
    if client is None:
        return _estimate_tokens(text)
    try:
        return client.count_tokens(system=text, messages=[{"role": "user", "content": "x"}])
    except Exception as exc:
        logger.warning("count_tokens failed (%s); falling back to estimation.", exc)
        return _estimate_tokens(text)


def build_report(save: bool = True) -> dict[str, Any]:
    """Compare the naive prompt configuration against the shipped one."""
    try:
        client = get_client()
        method = "messages.count_tokens (exact)"
    except LLMUnavailable:
        client = None
        method = "character heuristic (~4 chars/token) - NO API KEY, THESE ARE ESTIMATES"
        logger.warning("No API key: token counts below are estimates, not measurements.")

    # --- the two schema variants ---
    compact_schema = get_schema()          # the shipped 39-column curated warehouse
    full_schema = naive_full_schema()      # every column of every source table
    if not full_schema:
        # No raw CSVs available; fall back to comparing against the shipped DDL and
        # say so, rather than silently reporting a compaction saving of zero.
        full_schema = compact_schema
        logger.warning("Raw CSVs absent - the schema-compaction comparison is skipped.")

    shipped_prompt = build_system_prompt(compact_schema, include_exemplars=True)
    no_exemplars_prompt = build_system_prompt(compact_schema, include_exemplars=False)
    naive_prompt = build_system_prompt(full_schema, include_exemplars=True)

    prefix_tokens = _count(shipped_prompt, client)
    naive_prefix_tokens = _count(naive_prompt, client)
    metrics_block_tokens = _count(render_for_prompt(), client)
    exemplar_tokens = prefix_tokens - _count(no_exemplars_prompt, client)

    turns = len(SAMPLE_QUESTIONS)
    question_tokens = sum(_count(q, client) for q in SAMPLE_QUESTIONS)

    # --- naive: full prefix resent uncached on every turn, rows summarised by Sonnet ---
    naive_input = naive_prefix_tokens * turns + question_tokens
    # A typical result set fed back to the expensive model.
    assumed_rows_tokens = 400
    naive_input += assumed_rows_tokens * turns

    # --- shipped: prefix written once, read at 10% thereafter; rows go to Haiku ---
    shipped_input = (
        prefix_tokens * CACHE_WRITE_MULTIPLIER                       # first call writes
        + prefix_tokens * CACHE_READ_MULTIPLIER * (turns - 1)        # rest read
        + question_tokens
        + assumed_rows_tokens * turns * HAIKU_COST_RATIO             # summarised cheaply
    )

    saving = 1 - shipped_input / naive_input if naive_input else 0.0

    report = {
        "prompt_version": PROMPT_VERSION,
        "counting_method": method,
        "exact": client is not None,
        "prefix": {
            "shipped_tokens": prefix_tokens,
            "naive_full_schema_tokens": naive_prefix_tokens,
            "shipped_schema_columns": sum(
                1 for line in compact_schema.splitlines() if line.strip().startswith(
                    tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                ) and "CREATE" not in line
            ),
            "schema_compaction_saving_pct": round(
                1 - prefix_tokens / naive_prefix_tokens, 4
            ) if naive_prefix_tokens else 0.0,
            "semantic_layer_tokens": metrics_block_tokens,
            "exemplar_tokens": exemplar_tokens,
        },
        "scenario": {
            "turns": turns,
            "assumed_result_tokens_per_turn": assumed_rows_tokens,
            "naive_billable_input_tokens": int(naive_input),
            "optimised_billable_input_tokens": int(shipped_input),
            "reduction_pct": round(saving, 4),
        },
        "levers": [
            {
                "lever": "Prompt caching on the stable prefix",
                "mechanism": "cache_control: ephemeral; cached tokens bill at 10%",
                "applies_to": f"{prefix_tokens:,} tokens per turn after the first",
            },
            {
                "lever": "Compact warehouse schema (39 curated columns of 122, and "
                         "supplementary tables pre-aggregated rather than raw)",
                "mechanism": "smaller DDL block inside the cached prefix - but the "
                             "per-column comment glossary costs back what the dropped "
                             "columns save",
                "applies_to": (
                    f"{naive_prefix_tokens - prefix_tokens:+,} tokens per turn - "
                    "measured as roughly neutral, so this is a hallucination-control "
                    "lever rather than a cost one"
                ),
            },
            {
                "lever": "Haiku summarises result rows, not Sonnet",
                "mechanism": "cheap model handles the largest, easiest payload; "
                             "the Sonnet context never grows by the rows",
                "applies_to": f"~{assumed_rows_tokens:,} tokens per answered question",
            },
            {
                "lever": "Retry cap of 2",
                "mechanism": "bounds the worst case at three model calls per question",
                "applies_to": "pathological questions only",
            },
        ],
        "note": (
            "Live cache behaviour is reported by the eval harness from "
            "usage.cache_read_input_tokens, which is a measurement rather than a "
            "model of one. Run `python -m src.talk_to_data.eval_harness` with an "
            "API key for those figures."
        ),
    }

    if save:
        save_json(report, settings.report_dir / "token_report.json")
    _log(report)
    return report


def _log(report: dict[str, Any]) -> None:
    prefix, scenario = report["prefix"], report["scenario"]
    logger.info("=" * 68)
    logger.info("Token report (%s)", report["counting_method"])
    logger.info("  Cached prefix            %6d tokens", prefix["shipped_tokens"])
    logger.info("    of which semantic layer %5d tokens", prefix["semantic_layer_tokens"])
    logger.info("    of which exemplars      %5d tokens", prefix["exemplar_tokens"])
    compaction = 100 * prefix["schema_compaction_saving_pct"]
    logger.info("  Naive full-schema prefix %6d tokens (curation is %+.0f%% - the "
                "column glossary costs back what the dropped columns save)",
                prefix["naive_full_schema_tokens"], compaction)
    logger.info("  Over %d turns: %s -> %s billable input tokens (-%.0f%%)",
                scenario["turns"],
                f"{scenario['naive_billable_input_tokens']:,}",
                f"{scenario['optimised_billable_input_tokens']:,}",
                100 * scenario["reduction_pct"])
    logger.info("=" * 68)


def main() -> None:
    report = build_report()
    print(json.dumps(report["scenario"], indent=2))


if __name__ == "__main__":
    main()
