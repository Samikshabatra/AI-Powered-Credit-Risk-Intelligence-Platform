"""Guardrail tests: the security boundary between the LLM and the database.

These are the tests that would matter in a real deployment. Every one of them is
a query a prompt-injected model could plausibly emit.
"""

from __future__ import annotations

import pytest

from src.talk_to_data.query_runner import (
    ALLOWED_TABLES, SQLValidationError, enforce_limit, run_sql, validate_sql,
)
from tests.conftest import requires_database

# --------------------------------------------------------------------------- #
# Rejected
# --------------------------------------------------------------------------- #
BLOCKED = [
    ("drop", "DROP TABLE applications"),
    ("delete", "DELETE FROM applications WHERE TARGET = 1"),
    ("update", "UPDATE applications SET TARGET = 0"),
    ("insert", "INSERT INTO applications (SK_ID_CURR) VALUES (1)"),
    ("alter", "ALTER TABLE applications ADD COLUMN evil TEXT"),
    ("create", "CREATE TABLE evil (id INTEGER)"),
    ("attach", "ATTACH DATABASE '/etc/passwd' AS leak"),
    ("pragma", "PRAGMA table_info(applications)"),
    ("vacuum", "VACUUM"),
    ("stacked", "SELECT 1 FROM applications; DROP TABLE applications"),
    ("stacked_delete", "SELECT * FROM applications; DELETE FROM predictions"),
    ("unknown_table", "SELECT * FROM customers"),
    ("hallucinated_table", "SELECT * FROM loan_officers"),
    ("empty", "   "),
]


@pytest.mark.parametrize("label,sql", BLOCKED, ids=[label for label, _ in BLOCKED])
def test_dangerous_sql_is_rejected(label: str, sql: str) -> None:
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_injection_attempt_via_comment_is_rejected() -> None:
    """Comments are stripped before validation, so nothing can hide behind one."""
    with pytest.raises(SQLValidationError):
        validate_sql("SELECT 1 FROM applications -- ok\n; DROP TABLE applications")


# --------------------------------------------------------------------------- #
# Allowed
# --------------------------------------------------------------------------- #
ALLOWED = [
    ("simple", "SELECT AVG(TARGET) FROM applications"),
    ("group_by", "SELECT CODE_GENDER, AVG(TARGET) FROM applications GROUP BY CODE_GENDER"),
    ("join", "SELECT AVG(a.TARGET) FROM applications a "
             "JOIN bureau_summary b ON a.SK_ID_CURR = b.SK_ID_CURR"),
    ("cte", "WITH scored AS (SELECT TARGET FROM applications) SELECT AVG(TARGET) FROM scored"),
    ("trailing_semicolon", "SELECT COUNT(*) FROM applications;"),
    ("code_fence", "```sql\nSELECT COUNT(*) FROM applications\n```"),
    ("case_expression", "SELECT CASE WHEN TARGET = 1 THEN 'bad' ELSE 'good' END AS flag, "
                        "COUNT(*) FROM applications GROUP BY flag"),
]


@pytest.mark.parametrize("label,sql", ALLOWED, ids=[label for label, _ in ALLOWED])
def test_legitimate_sql_is_allowed(label: str, sql: str) -> None:
    assert validate_sql(sql)


def test_column_named_like_a_keyword_is_not_a_false_positive() -> None:
    """The denylist matches whole tokens, so 'updated' must not trip 'UPDATE'."""
    assert validate_sql("SELECT SK_ID_CURR AS updated_id FROM applications")


def test_cte_name_is_not_treated_as_an_unknown_table() -> None:
    sql = ("WITH bureau_summary_cte AS (SELECT SK_ID_CURR FROM applications) "
           "SELECT COUNT(*) FROM bureau_summary_cte")
    assert validate_sql(sql)


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
def test_limit_is_injected_on_row_returning_queries() -> None:
    assert "LIMIT" in enforce_limit("SELECT SK_ID_CURR FROM applications").upper()


def test_limit_is_not_injected_on_aggregates() -> None:
    """One row out of an aggregate needs no bound; a LIMIT there is just noise."""
    assert "LIMIT" not in enforce_limit("SELECT AVG(TARGET) FROM applications").upper()


def test_existing_limit_is_respected() -> None:
    sql = enforce_limit("SELECT SK_ID_CURR FROM applications LIMIT 5")
    assert sql.upper().count("LIMIT") == 1


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
@requires_database
def test_read_only_connection_blocks_writes_at_the_engine() -> None:
    """Layer 4: even bypassing validation, the connection itself refuses writes."""
    with pytest.raises(SQLValidationError, match="readonly|attempt to write|SQLite error"):
        run_sql("DELETE FROM predictions", validate=False)


@requires_database
def test_query_returns_expected_shape() -> None:
    result = run_sql("SELECT AVG(TARGET) AS default_rate FROM applications")
    assert result.row_count == 1
    assert 0.07 < float(result.rows.iloc[0, 0]) < 0.09


@requires_database
def test_row_cap_bounds_a_runaway_query() -> None:
    result = run_sql("SELECT SK_ID_CURR FROM applications")
    assert result.row_count <= 200
    assert result.truncated


@requires_database
def test_allowed_tables_all_exist() -> None:
    from src.talk_to_data.query_runner import table_summary

    summary = table_summary()
    assert set(summary) == ALLOWED_TABLES
    assert summary["applications"]["rows"] > 300_000
