"""Materialise the analytical SQLite database that Talk-to-Data queries.

Run directly:  python -m src.data.build_database

The DDL lives in `sql/schema.sql` rather than in Python string literals for one
specific reason: the NL->SQL prompt injects that same file verbatim, so the
schema the model sees is provably the schema that exists.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.data.loader import (
    _raw_path, bureau_aggregates, load_application, previous_application_aggregates,
)
from src.utils.config import PROJECT_ROOT, settings
from src.utils.helpers import timed
from src.utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"

APPLICATION_COLUMNS = [
    "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "OWN_CAR_AGE", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "ORGANIZATION_TYPE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", "AMT_REQ_CREDIT_BUREAU_YEAR",
    "FLAG_DOCUMENT_3", "FLAG_EMAIL", "FLAG_PHONE", "REG_CITY_NOT_WORK_CITY",
]

BUREAU_SUMMARY_COLUMNS = [
    "BUREAU_LOAN_COUNT", "BUREAU_ACTIVE_COUNT", "BUREAU_CLOSED_COUNT",
    "BUREAU_CREDIT_SUM", "BUREAU_DEBT_SUM", "BUREAU_OVERDUE_SUM",
    "BUREAU_DAYS_OVERDUE_MAX", "BUREAU_DAYS_CREDIT_MEAN", "BUREAU_DEBT_CREDIT_RATIO",
    "BUREAU_CREDIT_TYPE_NUNIQUE",
]

PREVIOUS_SUMMARY_COLUMNS = [
    "PREV_APP_COUNT", "PREV_APPROVED_COUNT", "PREV_REFUSED_COUNT", "PREV_REFUSAL_RATE",
    "PREV_AMT_CREDIT_MEAN", "PREV_CNT_PAYMENT_MEAN", "PREV_DAYS_DECISION_MAX",
]


def read_schema_sql() -> str:
    """The DDL text, also used as the grounding block in the NL->SQL prompt."""
    return SCHEMA_PATH.read_text(encoding="utf-8")


def build_database(db_path: Path | None = None, force: bool = False) -> Path:
    """Create the SQLite database from the raw CSVs. Idempotent unless force=True."""
    db_path = Path(db_path or settings.sqlite_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists() and not force:
        logger.info("Database already exists at %s (use --force to rebuild)", db_path)
        return db_path

    if db_path.exists():
        db_path.unlink()

    with timed("build SQLite database"), sqlite3.connect(db_path) as connection:
        connection.executescript(read_schema_sql())

        applications = load_application("train")[APPLICATION_COLUMNS]
        applications.to_sql("applications", connection, if_exists="append", index=False)
        logger.info("applications -> %s rows", f"{len(applications):,}")

        bureau = bureau_aggregates()[BUREAU_SUMMARY_COLUMNS].reset_index()
        bureau.to_sql("bureau_summary", connection, if_exists="append", index=False)
        logger.info("bureau_summary -> %s rows", f"{len(bureau):,}")

        previous = previous_application_aggregates()[PREVIOUS_SUMMARY_COLUMNS].reset_index()
        previous.to_sql("previous_summary", connection, if_exists="append", index=False)
        logger.info("previous_summary -> %s rows", f"{len(previous):,}")

        connection.commit()

    logger.info("Database built: %s (%.1f MB)", db_path, db_path.stat().st_size / 1e6)
    return db_path


def write_predictions(predictions: pd.DataFrame, db_path: Path | None = None) -> int:
    """Replace the `predictions` table with fresh model output.

    Expects columns: SK_ID_CURR, DEFAULT_PROBABILITY, RISK_BAND, DECISION, DATA_SPLIT.
    """
    db_path = Path(db_path or settings.sqlite_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No database at {db_path}; run build_database() first.")

    expected = ["SK_ID_CURR", "DEFAULT_PROBABILITY", "RISK_BAND", "DECISION", "DATA_SPLIT"]
    missing = [column for column in expected if column not in predictions.columns]
    if missing:
        raise ValueError(f"predictions frame is missing columns: {missing}")

    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM predictions")
        predictions[expected].to_sql(
            "predictions", connection, if_exists="append", index=False
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_pred_band ON predictions (RISK_BAND)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_pred_split ON predictions (DATA_SPLIT)")
        connection.commit()

    logger.info("predictions -> %s rows", f"{len(predictions):,}")
    return len(predictions)


def table_row_counts(db_path: Path | None = None) -> dict[str, int]:
    """Row count per table - used by the UI status panel and the tests."""
    db_path = Path(db_path or settings.sqlite_path)
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as connection:
        tables = [
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in tables
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the analytical SQLite database.")
    parser.add_argument("--force", action="store_true", help="rebuild even if it exists")
    args = parser.parse_args()

    settings.ensure_dirs()
    build_database(force=args.force)
    for table, count in table_row_counts().items():
        logger.info("  %-20s %s rows", table, f"{count:,}")


if __name__ == "__main__":
    main()
