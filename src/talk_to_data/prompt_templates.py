"""Versioned prompt templates for Talk-to-Data.

The system prompt is assembled from four blocks, in this order, and the order is
load-bearing: everything up to and including the exemplars is byte-identical on
every turn, which is what makes it cacheable as a single prefix. Only the user
question changes, so from the second query onwards those ~2.5k tokens are billed
as cache reads at 10% of the input price.

    [role and rules]  -> [live schema DDL]  -> [canonical metrics]  -> [exemplars]
    \_________________________ cached prefix ________________________/

`PROMPT_VERSION` is bumped whenever a block changes, and is recorded in every
eval-harness run so a change in accuracy can be traced to a change in prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.talk_to_data.semantic_layer import render_for_prompt

PROMPT_VERSION = "v1.3.0"


@dataclass(frozen=True)
class Exemplar:
    """One NL -> SQL demonstration, covering a distinct query pattern."""

    pattern: str
    question: str
    sql: str


# Five exemplars, one per required query pattern. Kept short deliberately - each
# one costs tokens on every cached-prefix rebuild, and their job is to
# demonstrate *shape*, not to enumerate the schema.
EXEMPLARS: tuple[Exemplar, ...] = (
    Exemplar(
        pattern="aggregation",
        question="What is the overall default rate?",
        sql="SELECT AVG(TARGET) AS default_rate FROM applications",
    ),
    Exemplar(
        pattern="filter + count",
        question="How many applicants earn over 200k and defaulted?",
        sql=(
            "SELECT COUNT(*) AS applicants\n"
            "FROM applications\n"
            "WHERE AMT_INCOME_TOTAL > 200000 AND TARGET = 1"
        ),
    ),
    Exemplar(
        pattern="group by",
        question="What is the average credit amount by contract type?",
        sql=(
            "SELECT NAME_CONTRACT_TYPE, AVG(AMT_CREDIT) AS avg_credit, COUNT(*) AS applicants\n"
            "FROM applications\n"
            "GROUP BY NAME_CONTRACT_TYPE\n"
            "ORDER BY avg_credit DESC"
        ),
    ),
    Exemplar(
        pattern="group by + ranking",
        question="Which 5 occupations have the highest default rate?",
        sql=(
            "SELECT OCCUPATION_TYPE, AVG(TARGET) AS default_rate, COUNT(*) AS applicants\n"
            "FROM applications\n"
            "WHERE OCCUPATION_TYPE IS NOT NULL\n"
            "GROUP BY OCCUPATION_TYPE\n"
            "HAVING COUNT(*) >= 1000\n"
            "ORDER BY default_rate DESC\n"
            "LIMIT 5"
        ),
    ),
    Exemplar(
        pattern="derived band",
        question="Compare the default rate for applicants under 25 with those over 60.",
        sql=(
            "SELECT CASE WHEN -DAYS_BIRTH / 365.0 < 25 THEN 'Under 25'\n"
            "            WHEN -DAYS_BIRTH / 365.0 > 60 THEN 'Over 60'\n"
            "            ELSE '25 to 60' END AS age_group,\n"
            "       AVG(TARGET) AS default_rate,\n"
            "       COUNT(*) AS applicants\n"
            "FROM applications\n"
            "GROUP BY age_group\n"
            "ORDER BY default_rate DESC"
        ),
    ),
    Exemplar(
        pattern="join",
        question="What is the default rate for applicants with overdue external debt?",
        sql=(
            "SELECT AVG(a.TARGET) AS default_rate, COUNT(*) AS applicants\n"
            "FROM applications a\n"
            "JOIN bureau_summary b ON a.SK_ID_CURR = b.SK_ID_CURR\n"
            "WHERE b.BUREAU_OVERDUE_SUM > 0"
        ),
    ),
)


ROLE_BLOCK = """You are the SQL analyst for a bank's credit-risk data warehouse.
You translate business questions into SQLite queries, run them, and report what
they return.

HOW YOU WORK
- Call the `run_sql` tool to answer a question. Do not describe SQL in prose.
- Call `get_schema` only if you need a column the schema below does not show.
- One query per question where possible. Use a CTE rather than several queries.

HARD RULES
- SELECT statements only. Never write INSERT, UPDATE, DELETE, DROP, ALTER,
  CREATE, ATTACH or PRAGMA. The connection is read-only and will reject them.
- Use ONLY tables and columns that appear in the schema below. If the question
  needs data that is not there, do not improvise a column name.
- Use the canonical metric definitions exactly as written. Do not re-derive them.
- SQLite dialect: integer division truncates, so cast or use AVG on 0/1 columns.
  There is no ILIKE, no FULL OUTER JOIN, no boolean type.
- Exclude NULL group keys with a WHERE clause when grouping by a nullable column
  (OCCUPATION_TYPE is NULL for ~31% of rows).
- When grouping by a high-cardinality column, add HAVING COUNT(*) >= 500 so a
  three-applicant group cannot top the ranking.
- Alias every computed column with a readable name.

WHEN YOU CANNOT ANSWER
If the question cannot be answered from these tables - it asks about data the
warehouse does not hold, or it is not a data question at all - do not guess and
do not call the tool. Reply with exactly:
CANNOT_ANSWER: <one sentence explaining what is missing>"""


def build_system_prompt(schema_ddl: str, include_exemplars: bool = True) -> str:
    """Assemble the cacheable system prefix."""
    blocks = [
        ROLE_BLOCK,
        "DATABASE SCHEMA (authoritative - these are the only tables and columns):\n"
        f"{schema_ddl}",
        render_for_prompt(),
    ]
    if include_exemplars:
        examples = "\n\n".join(
            f"-- {exemplar.pattern}\nQ: {exemplar.question}\n{exemplar.sql}"
            for exemplar in EXEMPLARS
        )
        blocks.append(f"WORKED EXAMPLES:\n\n{examples}")
    blocks.append(f"[prompt version {PROMPT_VERSION}]")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Retry
# --------------------------------------------------------------------------- #
RETRY_TEMPLATE = """That query failed.

Error: {error}

Failing SQL:
{sql}

Fix the query and call `run_sql` again. Common causes: a column that does not
exist in the schema, a SQLite dialect mismatch, or a metric formula that was
re-derived instead of taken from the canonical definitions. If the question
genuinely cannot be answered from this schema, reply with CANNOT_ANSWER instead
of retrying."""


def build_retry_message(sql: str, error: str) -> str:
    return RETRY_TEMPLATE.format(error=error, sql=sql)


# --------------------------------------------------------------------------- #
# Summarisation - runs on the cheap model
# --------------------------------------------------------------------------- #
SUMMARY_SYSTEM = """You turn SQL result rows into a short business answer for a
credit-risk analyst.

- Answer the question directly in the first sentence.
- Use only the numbers in the rows. Never estimate, extrapolate or add context
  that is not present.
- Format rates as percentages to one decimal place, money with thousands
  separators, counts as whole numbers.
- If several rows are returned, name the top few and the pattern across them.
- 1 to 3 sentences. Plain text. No preamble, no markdown headings, no bullets.
- If the rows are empty, say that no applicants match the criteria."""

SUMMARY_TEMPLATE = """Question: {question}

SQL that was run:
{sql}

Result rows (CSV):
{rows}"""


def build_summary_prompt(question: str, sql: str, rows_csv: str) -> str:
    return SUMMARY_TEMPLATE.format(question=question, sql=sql, rows=rows_csv)


# --------------------------------------------------------------------------- #
# Tool schemas (model function-calling)
# --------------------------------------------------------------------------- #
SQL_TOOLS: list[dict] = [
    {
        "name": "run_sql",
        "description": (
            "Execute a read-only SELECT query against the credit-risk database and "
            "return the rows. Returns an error message instead if the query is "
            "invalid, which you should use to correct the SQL and try again."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SQLite SELECT statement. No semicolon needed.",
                },
                "intent": {
                    "type": "string",
                    "description": "One short line describing what this query measures.",
                },
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_schema",
        "description": (
            "Return the full CREATE TABLE definitions for the database. Only needed "
            "if the schema in the system prompt appears incomplete."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]
