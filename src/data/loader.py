"""Dataset loading and cross-table feature aggregation.

Scope decision (see spec section 2): the model is trained on `application_train`
plus a curated set of aggregates from `bureau` and `previous_application`. The
four remaining Home Credit tables (bureau_balance, installments_payments,
credit_card_balance, POS_CASH_balance) are deliberately out of scope - they add
~1.9 GB of IO for a marginal AUC gain and would dominate the build time.

The joined frame is cached as parquet so EDA, training and the app all read the
same materialised table instead of re-aggregating 575 MB of CSV on every run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import settings
from src.utils.helpers import timed
from src.utils.logger import get_logger

logger = get_logger(__name__)

APPLICATION_TRAIN = "application_train.csv"
APPLICATION_TEST = "application_test.csv"
BUREAU = "bureau.csv"
PREVIOUS_APPLICATION = "previous_application.csv"

# Columns actually read from the supplementary tables - reading only these keeps
# bureau + previous_application parsing under ~1 GB instead of ~4 GB.
_BUREAU_COLUMNS = [
    "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE", "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
]
_PREVIOUS_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_APPLICATION",
    "AMT_CREDIT", "AMT_DOWN_PAYMENT", "RATE_DOWN_PAYMENT", "CNT_PAYMENT",
    "DAYS_DECISION", "NAME_YIELD_GROUP",
]


class RawDataNotFound(FileNotFoundError):
    """Raised when a required Home Credit CSV is missing from the raw directory."""


def _raw_path(filename: str) -> Path:
    path = settings.raw_dir / filename
    if not path.exists():
        raise RawDataNotFound(
            f"{filename} not found in {settings.raw_dir}. Download the Home Credit "
            "Default Risk dataset from Kaggle and unzip it there (see README)."
        )
    return path


# --------------------------------------------------------------------------- #
# Base tables
# --------------------------------------------------------------------------- #
def load_application(split: str = "train", nrows: int | None = None) -> pd.DataFrame:
    """Load application_train.csv (has TARGET) or application_test.csv (no TARGET)."""
    filename = APPLICATION_TRAIN if split == "train" else APPLICATION_TEST
    with timed(f"load {filename}"):
        frame = pd.read_csv(_raw_path(filename), nrows=nrows)
    logger.info("%s -> %s rows x %s cols", filename, f"{len(frame):,}", frame.shape[1])
    return frame


def load_columns_description() -> pd.DataFrame:
    """The Kaggle column glossary, used to annotate the EDA. Optional file."""
    path = settings.raw_dir / "HomeCredit_columns_description.csv"
    if not path.exists():
        return pd.DataFrame(columns=["Table", "Row", "Description"])
    return pd.read_csv(path, encoding="latin-1", index_col=0)


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division where a zero denominator yields NaN, not inf."""
    return numerator / denominator.mask(denominator == 0)


def bureau_aggregates(nrows: int | None = None) -> pd.DataFrame:
    """Per-applicant summary of credit-bureau history, indexed by SK_ID_CURR.

    Captures the three things a credit officer asks about external debt: how much
    of it there is, how much is still open, and whether any of it went bad.
    """
    with timed("aggregate bureau.csv"):
        bureau = pd.read_csv(_raw_path(BUREAU), usecols=_BUREAU_COLUMNS, nrows=nrows)

        bureau["_is_active"] = (bureau["CREDIT_ACTIVE"] == "Active").astype("int8")
        bureau["_is_closed"] = (bureau["CREDIT_ACTIVE"] == "Closed").astype("int8")

        grouped = bureau.groupby("SK_ID_CURR")
        agg = pd.DataFrame({
            "BUREAU_LOAN_COUNT": grouped["SK_ID_BUREAU"].count(),
            "BUREAU_ACTIVE_COUNT": grouped["_is_active"].sum(),
            "BUREAU_CLOSED_COUNT": grouped["_is_closed"].sum(),
            "BUREAU_CREDIT_SUM": grouped["AMT_CREDIT_SUM"].sum(),
            "BUREAU_CREDIT_MEAN": grouped["AMT_CREDIT_SUM"].mean(),
            "BUREAU_CREDIT_MAX": grouped["AMT_CREDIT_SUM"].max(),
            "BUREAU_DEBT_SUM": grouped["AMT_CREDIT_SUM_DEBT"].sum(),
            "BUREAU_OVERDUE_SUM": grouped["AMT_CREDIT_SUM_OVERDUE"].sum(),
            "BUREAU_OVERDUE_MAX": grouped["AMT_CREDIT_MAX_OVERDUE"].max(),
            "BUREAU_DAYS_OVERDUE_MAX": grouped["CREDIT_DAY_OVERDUE"].max(),
            "BUREAU_DAYS_CREDIT_MEAN": grouped["DAYS_CREDIT"].mean(),
            "BUREAU_DAYS_CREDIT_MIN": grouped["DAYS_CREDIT"].min(),
            "BUREAU_DAYS_CREDIT_MAX": grouped["DAYS_CREDIT"].max(),
            "BUREAU_DAYS_ENDDATE_MAX": grouped["DAYS_CREDIT_ENDDATE"].max(),
            "BUREAU_PROLONG_SUM": grouped["CNT_CREDIT_PROLONG"].sum(),
            "BUREAU_CREDIT_TYPE_NUNIQUE": grouped["CREDIT_TYPE"].nunique(),
        })

        agg["BUREAU_ACTIVE_RATIO"] = agg["BUREAU_ACTIVE_COUNT"] / agg["BUREAU_LOAN_COUNT"]
        agg["BUREAU_DEBT_CREDIT_RATIO"] = _safe_divide(
            agg["BUREAU_DEBT_SUM"], agg["BUREAU_CREDIT_SUM"]
        )
        agg["BUREAU_HAS_OVERDUE"] = (agg["BUREAU_OVERDUE_SUM"] > 0).astype("int8")

    logger.info("bureau aggregates -> %s applicants x %s features",
                f"{len(agg):,}", agg.shape[1])
    return agg


def previous_application_aggregates(nrows: int | None = None) -> pd.DataFrame:
    """Per-applicant summary of prior Home Credit applications, indexed by SK_ID_CURR.

    Prior refusals are the sharpest signal in this table: a lender that already
    said no once holds private information the credit bureau does not carry.
    """
    with timed("aggregate previous_application.csv"):
        prev = pd.read_csv(
            _raw_path(PREVIOUS_APPLICATION), usecols=_PREVIOUS_COLUMNS, nrows=nrows
        )

        prev["_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype("int8")
        prev["_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
        prev["_cancelled"] = (prev["NAME_CONTRACT_STATUS"] == "Canceled").astype("int8")
        # 365243 is the same "no date recorded" sentinel the application table uses.
        prev["DAYS_DECISION"] = prev["DAYS_DECISION"].mask(prev["DAYS_DECISION"] == 365243)

        grouped = prev.groupby("SK_ID_CURR")
        agg = pd.DataFrame({
            "PREV_APP_COUNT": grouped["SK_ID_PREV"].count(),
            "PREV_APPROVED_COUNT": grouped["_approved"].sum(),
            "PREV_REFUSED_COUNT": grouped["_refused"].sum(),
            "PREV_CANCELLED_COUNT": grouped["_cancelled"].sum(),
            "PREV_AMT_APPLICATION_MEAN": grouped["AMT_APPLICATION"].mean(),
            "PREV_AMT_CREDIT_MEAN": grouped["AMT_CREDIT"].mean(),
            "PREV_AMT_CREDIT_MAX": grouped["AMT_CREDIT"].max(),
            "PREV_DOWN_PAYMENT_RATE_MEAN": grouped["RATE_DOWN_PAYMENT"].mean(),
            "PREV_CNT_PAYMENT_MEAN": grouped["CNT_PAYMENT"].mean(),
            "PREV_DAYS_DECISION_MAX": grouped["DAYS_DECISION"].max(),
            "PREV_DAYS_DECISION_MIN": grouped["DAYS_DECISION"].min(),
        })

        agg["PREV_REFUSAL_RATE"] = agg["PREV_REFUSED_COUNT"] / agg["PREV_APP_COUNT"]
        agg["PREV_APPROVAL_RATE"] = agg["PREV_APPROVED_COUNT"] / agg["PREV_APP_COUNT"]
        agg["PREV_CREDIT_TO_APPLICATION"] = _safe_divide(
            agg["PREV_AMT_CREDIT_MEAN"], agg["PREV_AMT_APPLICATION_MEAN"]
        )

    logger.info("previous_application aggregates -> %s applicants x %s features",
                f"{len(agg):,}", agg.shape[1])
    return agg


# --------------------------------------------------------------------------- #
# Joined dataset
# --------------------------------------------------------------------------- #
def _cache_path(split: str) -> Path:
    return settings.processed_dir / f"application_{split}_joined.parquet"


def load_dataset(
    split: str = "train",
    with_aggregates: bool = True,
    use_cache: bool = True,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Application table left-joined with bureau + previous-application aggregates.

    The left join is intentional: an applicant with no bureau history is a real
    and common case, and the resulting NaNs are informative to LightGBM rather
    than something to impute away.
    """
    cache = _cache_path(split)
    if use_cache and nrows is None and cache.exists():
        logger.info("Loading cached join from %s", cache.name)
        return pd.read_parquet(cache)

    frame = load_application(split=split, nrows=nrows)

    if with_aggregates:
        for name, aggregator in (
            ("bureau", bureau_aggregates),
            ("previous_application", previous_application_aggregates),
        ):
            try:
                agg = aggregator()
                before = frame.shape[1]
                frame = frame.merge(agg, how="left", left_on="SK_ID_CURR", right_index=True)
                logger.info("Joined %s (+%s features)", name, frame.shape[1] - before)
            except RawDataNotFound as exc:
                logger.warning("Skipping %s aggregates: %s", name, exc)

    if use_cache and nrows is None:
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
        logger.info("Cached joined dataset -> %s", cache.name)

    return frame


def dataset_profile(frame: pd.DataFrame) -> dict:
    """Row/column/memory/missing summary, reused by the EDA notebook and the UI."""
    missing = frame.isna().mean()
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "memory_mb": round(frame.memory_usage(deep=True).sum() / 1e6, 1),
        "dtypes": {str(k): int(v) for k, v in frame.dtypes.value_counts().items()},
        "duplicate_ids": (
            int(frame["SK_ID_CURR"].duplicated().sum()) if "SK_ID_CURR" in frame else 0
        ),
        "columns_over_40pct_missing": int((missing > 0.4).sum()),
        "columns_fully_populated": int((missing == 0).sum()),
        "mean_missing_rate": round(float(missing.mean()), 4),
    }
