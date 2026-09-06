"""Data cleaning and feature engineering.

The tests that matter here are the ones about the dataset's traps: the
`DAYS_EMPLOYED` sentinel, division by zero income, and train/serve skew in the
income clip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_dictionary import feature_group
from src.data.preprocessor import CreditRiskPreprocessor, missing_report
from src.utils.helpers import DAYS_EMPLOYED_SENTINEL


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """A tiny frame carrying every anomaly the cleaner is supposed to fix."""
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "TARGET": [0, 1, 0, 1],
        "DAYS_EMPLOYED": [-2000, DAYS_EMPLOYED_SENTINEL, -300, DAYS_EMPLOYED_SENTINEL],
        "DAYS_BIRTH": [-12000, -18000, -9000, 500],           # last one is corrupt
        "CODE_GENDER": ["M", "F", "XNA", "F"],
        "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 0.0, 117_000_000.0],
        "AMT_CREDIT": [300_000.0, 500_000.0, 250_000.0, 900_000.0],
        "AMT_ANNUITY": [20_000.0, 30_000.0, 15_000.0, 45_000.0],
        "AMT_GOODS_PRICE": [280_000.0, 480_000.0, 240_000.0, 850_000.0],
        "CNT_FAM_MEMBERS": [2.0, 0.0, 3.0, 1.0],              # a zero household
        "EXT_SOURCE_1": [0.5, np.nan, 0.2, np.nan],
        "EXT_SOURCE_2": [0.6, 0.4, np.nan, 0.7],
        "EXT_SOURCE_3": [np.nan, 0.3, 0.1, 0.8],
        "OCCUPATION_TYPE": ["Laborers", None, "Drivers", "Managers"],
        "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans", "Cash loans"],
    })


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_sentinel_becomes_nan_and_raises_a_flag(toy_frame) -> None:
    cleaned = CreditRiskPreprocessor.clean(toy_frame)
    sentinel_rows = [1, 3]
    assert cleaned.loc[sentinel_rows, "DAYS_EMPLOYED"].isna().all()
    assert cleaned.loc[sentinel_rows, "IS_UNEMPLOYED"].eq(1).all()
    assert cleaned.loc[[0, 2], "IS_UNEMPLOYED"].eq(0).all()


def test_sentinel_never_reaches_employment_years(toy_frame) -> None:
    """The failure this guards: 365243 / 365 = a 1000-year career."""
    engineered, _ = CreditRiskPreprocessor.engineer(CreditRiskPreprocessor.clean(toy_frame))
    years = engineered["EMPLOYMENT_YEARS"].dropna()
    assert (years.abs() < 100).all()


def test_unknown_gender_becomes_missing(toy_frame) -> None:
    cleaned = CreditRiskPreprocessor.clean(toy_frame)
    assert pd.isna(cleaned.loc[2, "CODE_GENDER"])


def test_positive_day_counts_are_treated_as_corrupt(toy_frame) -> None:
    """DAYS_* are negative by convention; a positive value is a data error."""
    cleaned = CreditRiskPreprocessor.clean(toy_frame)
    assert pd.isna(cleaned.loc[3, "DAYS_BIRTH"])


def test_income_clip_uses_the_fitted_cap_not_the_batch(toy_frame) -> None:
    """Train/serve skew guard: a single-row request must be clipped against the
    training distribution, not against itself."""
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit_transform(toy_frame)
    cap = preprocessor.state.income_cap
    assert cap is not None

    single = preprocessor.transform(toy_frame.iloc[[3]])
    assert float(single["AMT_INCOME_TOTAL"].iloc[0]) <= cap


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_division_by_zero_yields_nan_not_infinity(toy_frame) -> None:
    """Row 2 has zero income; row 1 has zero household members."""
    engineered, _ = CreditRiskPreprocessor.engineer(CreditRiskPreprocessor.clean(toy_frame))
    for column in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "INCOME_PER_PERSON"):
        assert not np.isinf(engineered[column].to_numpy(dtype=float)).any()


def test_external_score_aggregates_ignore_missing(toy_frame) -> None:
    engineered, _ = CreditRiskPreprocessor.engineer(toy_frame)
    # Row 0 has scores 0.5 and 0.6 present, 0.3 missing.
    assert engineered.loc[0, "EXT_SOURCE_MEAN"] == pytest.approx(0.55)
    assert engineered.loc[0, "EXT_SOURCE_COUNT"] == 2
    assert engineered.loc[0, "EXT_SOURCE_MIN"] == pytest.approx(0.5)


def test_engineered_features_are_created(toy_frame) -> None:
    _, created = CreditRiskPreprocessor.engineer(toy_frame)
    for expected in ("CREDIT_INCOME_RATIO", "AGE_YEARS", "EXT_SOURCE_MEAN", "CREDIT_TERM"):
        assert expected in created


# --------------------------------------------------------------------------- #
# Encoding and contract
# --------------------------------------------------------------------------- #
def test_output_is_all_numeric(toy_frame) -> None:
    features, _ = CreditRiskPreprocessor().fit_transform(toy_frame)
    assert features.select_dtypes(exclude="number").empty


def test_missing_category_encodes_as_negative_one(toy_frame) -> None:
    """LightGBM reads a negative categorical as missing, keeping it a separate branch."""
    preprocessor = CreditRiskPreprocessor()
    features, _ = preprocessor.fit_transform(toy_frame)
    assert features.loc[1, "OCCUPATION_TYPE"] == -1
    assert preprocessor.decode_category("OCCUPATION_TYPE", -1.0) is None
    assert preprocessor.decode_category("OCCUPATION_TYPE", features.loc[0, "OCCUPATION_TYPE"]) \
        == "Laborers"


def test_transform_reproduces_the_training_column_order(toy_frame) -> None:
    preprocessor = CreditRiskPreprocessor()
    features, _ = preprocessor.fit_transform(toy_frame)
    again = preprocessor.transform(toy_frame)
    assert list(again.columns) == list(features.columns)
    pd.testing.assert_frame_equal(features, again)


def test_unseen_category_does_not_crash(toy_frame) -> None:
    preprocessor = CreditRiskPreprocessor()
    preprocessor.fit_transform(toy_frame)
    novel = toy_frame.iloc[[0]].copy()
    novel["OCCUPATION_TYPE"] = "Astronaut"
    assert preprocessor.transform(novel)["OCCUPATION_TYPE"].iloc[0] == -1


def test_transform_before_fit_is_an_error(toy_frame) -> None:
    with pytest.raises(RuntimeError):
        CreditRiskPreprocessor().transform(toy_frame)


def test_target_and_id_are_excluded_from_features(toy_frame) -> None:
    features, target = CreditRiskPreprocessor().fit_transform(toy_frame)
    assert "TARGET" not in features.columns
    assert "SK_ID_CURR" not in features.columns
    assert target is not None and len(target) == len(toy_frame)


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def test_real_slice_preprocesses_cleanly(raw_frame) -> None:
    preprocessor = CreditRiskPreprocessor()
    features, target = preprocessor.fit_transform(raw_frame)
    assert len(features) == len(raw_frame)
    assert features.shape[1] > 120
    assert not np.isinf(features.to_numpy(dtype="float32", na_value=0.0)).any()
    assert target.isin([0, 1]).all()


def test_missing_report_is_sorted_worst_first(raw_frame) -> None:
    report = missing_report(raw_frame)
    assert report["missing_rate"].is_monotonic_decreasing


def test_feature_grouping_covers_the_known_columns() -> None:
    assert feature_group("EXT_SOURCE_2") == "external_source"
    assert feature_group("BUREAU_DEBT_SUM") == "credit_history_bureau"
    assert feature_group("FLAG_DOCUMENT_3") == "document"
    assert feature_group("AMT_CREDIT") == "financial"
    assert feature_group("SOME_UNKNOWN_COLUMN") == "other"
