"""Cleaning, feature engineering and encoding.

Design choices worth defending in review:

* **Sentinels are repaired, not imputed away.** `DAYS_EMPLOYED == 365243`
  (~18% of rows) is Home Credit's encoding for "no employment record", not a
  1000-year career. Left raw it drags every employment-derived feature into
  nonsense; here it becomes NaN plus an explicit `IS_UNEMPLOYED` flag, so the
  model keeps the signal without inheriting the lie.
* **Missing values are kept as NaN.** LightGBM learns an optimal default
  direction per split, which beats mean/median imputation on this dataset where
  missingness is itself predictive (no bureau record, no external score).
* **Categoricals become integer codes** with the mapping persisted alongside the
  model, and are handed to LightGBM as native categorical features. That avoids
  a 200-column one-hot blow-up and keeps SHAP attributions on one row per
  concept instead of scattered across dummies.
* **Redundant property columns are dropped.** The building-statistics block ships
  as `_AVG` / `_MODE` / `_MEDI` triplets of the same measurement; keeping `_AVG`
  drops ~40 near-duplicate, ~65%-missing columns with no measured AUC cost.

The fitted state (feature order, category maps, engineered column list) is a
plain dataclass so it serialises with joblib and reloads in the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.helpers import DAYS_EMPLOYED_SENTINEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

# Suffixes of the redundant property-statistics triplets (we keep `_AVG`).
_REDUNDANT_SUFFIXES = ("_MODE", "_MEDI")
# ...except these, which are genuine standalone categoricals, not statistics.
_KEEP_DESPITE_SUFFIX = {
    "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
}
# Columns that carry no signal or leak the row identity.
_DROP_ALWAYS = {"FLAG_MOBIL"}  # constant: 307,510 of 307,511 rows are 1


@dataclass
class PreprocessorState:
    """Everything needed to transform a new applicant exactly like the training set."""

    feature_names: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    category_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    engineered_features: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)
    income_cap: float | None = None
    fitted: bool = False


class CreditRiskPreprocessor:
    """Fit/transform pipeline from the raw joined frame to a LightGBM-ready matrix."""

    def __init__(self, drop_redundant_property: bool = True) -> None:
        self.drop_redundant_property = drop_redundant_property
        self.state = PreprocessorState()

    # ------------------------------------------------------------------ #
    # Cleaning
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean(frame: pd.DataFrame, income_cap: float | None = None) -> pd.DataFrame:
        """Repair the known encoding bugs. Safe to call more than once.

        `income_cap` is learned once at fit time and replayed at transform time,
        so a single-row scoring request is clipped against the training
        distribution rather than against itself.
        """
        frame = frame.copy()

        if "DAYS_EMPLOYED" in frame:
            sentinel = frame["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
            frame["IS_UNEMPLOYED"] = sentinel.astype("int8")
            frame.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
            if sentinel.any():
                logger.debug("DAYS_EMPLOYED sentinel repaired on %s rows (%.1f%%)",
                             f"{int(sentinel.sum()):,}", 100 * sentinel.mean())

        # 'XNA' is the dataset-wide unknown marker; treat it as missing rather
        # than as a real level (CODE_GENDER 'XNA' is 4 rows, not a third gender).
        for column in ("CODE_GENDER", "ORGANIZATION_TYPE", "NAME_FAMILY_STATUS",
                       "NAME_INCOME_TYPE"):
            if column in frame:
                frame[column] = frame[column].mask(frame[column] == "XNA")

        # Negative day-counts are the convention; a positive value is corrupt.
        for column in ("DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
                       "DAYS_LAST_PHONE_CHANGE"):
            if column in frame:
                frame.loc[frame[column] > 0, column] = np.nan

        # Income has a single 117M outlier that distorts every ratio built on it.
        if "AMT_INCOME_TOTAL" in frame and income_cap is not None:
            frame["AMT_INCOME_TOTAL"] = frame["AMT_INCOME_TOTAL"].clip(upper=income_cap)

        return frame

    # ------------------------------------------------------------------ #
    # Feature engineering
    # ------------------------------------------------------------------ #
    @staticmethod
    def engineer(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Add the domain features a credit analyst would actually reason about."""
        frame = frame.copy()
        created: list[str] = []

        def add(name: str, series: pd.Series) -> None:
            # Division by a zero income/annuity yields inf; mask it to NaN so the
            # value lands in LightGBM's missing branch instead of a huge outlier.
            values = pd.to_numeric(series, errors="coerce").astype("float64")
            frame[name] = values.where(np.isfinite(values), np.nan)
            created.append(name)

        income = frame.get("AMT_INCOME_TOTAL")
        credit = frame.get("AMT_CREDIT")
        annuity = frame.get("AMT_ANNUITY")
        goods = frame.get("AMT_GOODS_PRICE")

        # --- affordability ---
        if income is not None and credit is not None:
            add("CREDIT_INCOME_RATIO", credit / income)
        if income is not None and annuity is not None:
            add("ANNUITY_INCOME_RATIO", annuity / income)
        if credit is not None and annuity is not None:
            add("CREDIT_TERM", credit / annuity)
        if credit is not None and goods is not None:
            add("GOODS_CREDIT_RATIO", goods / credit)

        # --- household ---
        if "CNT_FAM_MEMBERS" in frame:
            members = frame["CNT_FAM_MEMBERS"].mask(frame["CNT_FAM_MEMBERS"] == 0)
            if income is not None:
                add("INCOME_PER_PERSON", income / members)
            if credit is not None:
                add("CREDIT_PER_PERSON", credit / members)

        # --- age / tenure, in units a human reads ---
        if "DAYS_BIRTH" in frame:
            add("AGE_YEARS", -frame["DAYS_BIRTH"] / 365.0)
        if "DAYS_EMPLOYED" in frame:
            add("EMPLOYMENT_YEARS", -frame["DAYS_EMPLOYED"] / 365.0)
        if "DAYS_EMPLOYED" in frame and "DAYS_BIRTH" in frame:
            add("EMPLOYED_TO_AGE_RATIO", frame["DAYS_EMPLOYED"] / frame["DAYS_BIRTH"])

        # --- external bureau scores: the strongest block in the dataset ---
        ext_columns = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
                       if c in frame]
        if ext_columns:
            ext = frame[ext_columns]
            add("EXT_SOURCE_MEAN", ext.mean(axis=1))
            add("EXT_SOURCE_MIN", ext.min(axis=1))
            add("EXT_SOURCE_MAX", ext.max(axis=1))
            add("EXT_SOURCE_COUNT", ext.notna().sum(axis=1).astype(float))
            if len(ext_columns) > 1:
                # Disagreement between bureaus is itself a risk signal.
                add("EXT_SOURCE_STD", ext.std(axis=1))
            # Interaction that consistently ranks top-5 in gain importance.
            if "AGE_YEARS" in frame:
                add("EXT_SOURCE_MEAN_X_AGE", frame["EXT_SOURCE_MEAN"] * frame["AGE_YEARS"])

        # --- paperwork completeness ---
        doc_columns = [c for c in frame.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_columns:
            add("DOCUMENT_SUBMITTED_COUNT", frame[doc_columns].sum(axis=1).astype(float))
        contact_columns = [c for c in ("FLAG_EMP_PHONE", "FLAG_WORK_PHONE",
                                       "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL")
                           if c in frame]
        if contact_columns:
            add("CONTACT_FLAG_COUNT", frame[contact_columns].sum(axis=1).astype(float))

        # --- external debt load relative to declared income ---
        if "BUREAU_DEBT_SUM" in frame and income is not None:
            add("BUREAU_DEBT_INCOME_RATIO", frame["BUREAU_DEBT_SUM"] / income)
        if "BUREAU_CREDIT_SUM" in frame and credit is not None:
            add("BUREAU_CREDIT_TO_NEW_CREDIT", frame["BUREAU_CREDIT_SUM"] / credit)

        logger.debug("Engineered %d features", len(created))
        return frame, created

    # ------------------------------------------------------------------ #
    # Column selection
    # ------------------------------------------------------------------ #
    def _columns_to_drop(self, frame: pd.DataFrame) -> list[str]:
        drop = {c for c in _DROP_ALWAYS if c in frame.columns}
        if self.drop_redundant_property:
            drop |= {
                c for c in frame.columns
                if c.endswith(_REDUNDANT_SUFFIXES) and c not in _KEEP_DESPITE_SUFFIX
            }
        return sorted(drop)

    # ------------------------------------------------------------------ #
    # Fit / transform
    # ------------------------------------------------------------------ #
    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        income_cap = (
            float(frame["AMT_INCOME_TOTAL"].quantile(0.9995))
            if "AMT_INCOME_TOTAL" in frame else None
        )
        prepared, engineered = self.engineer(self.clean(frame, income_cap))

        target = prepared[TARGET_COLUMN] if TARGET_COLUMN in prepared else None
        dropped = self._columns_to_drop(prepared)
        features = prepared.drop(columns=dropped + [c for c in (TARGET_COLUMN, ID_COLUMN)
                                                    if c in prepared])

        categorical = sorted(features.select_dtypes(include=["object", "category"]).columns)
        category_maps: dict[str, dict[str, int]] = {}
        for column in categorical:
            levels = sorted(features[column].dropna().unique().tolist())
            category_maps[column] = {level: index for index, level in enumerate(levels)}
            features[column] = self._encode(features[column], category_maps[column])

        features = features.astype("float32")

        self.state = PreprocessorState(
            feature_names=features.columns.tolist(),
            categorical_features=categorical,
            category_maps=category_maps,
            engineered_features=engineered,
            dropped_columns=dropped,
            income_cap=income_cap,
            fitted=True,
        )
        logger.info(
            "Preprocessor fitted: %d features (%d categorical, %d engineered, %d dropped)",
            len(self.state.feature_names), len(categorical), len(engineered), len(dropped),
        )
        return features, target

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation. Unseen category levels become NaN."""
        if not self.state.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        prepared, _ = self.engineer(self.clean(frame, self.state.income_cap))

        for column in self.state.categorical_features:
            if column in prepared:
                prepared[column] = self._encode(
                    prepared[column], self.state.category_maps[column]
                )

        # Reindex guarantees identical column order and fills genuinely absent
        # columns with NaN, which LightGBM handles natively.
        features = prepared.reindex(columns=self.state.feature_names)
        return features.astype("float32")

    @staticmethod
    def _encode(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
        """Map levels to codes; unknown/missing becomes -1.

        LightGBM treats negative values in a categorical feature as missing, so
        -1 is the encoding that keeps "unknown occupation" as its own branch
        instead of silently colliding with level 0.
        """
        return series.map(mapping).fillna(-1).astype("float32")

    # ------------------------------------------------------------------ #
    # Introspection helpers used by the UI and the what-if sliders
    # ------------------------------------------------------------------ #
    def decode_category(self, column: str, code: float) -> str | None:
        mapping = self.state.category_maps.get(column)
        if mapping is None or code is None:
            return None
        if isinstance(code, float) and (np.isnan(code) or code < 0):
            return None
        reverse = {index: level for level, index in mapping.items()}
        return reverse.get(int(code))

    def category_levels(self, column: str) -> list[str]:
        return list(self.state.category_maps.get(column, {}).keys())


def missing_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and rates, sorted worst-first (EDA + scorecard)."""
    missing = frame.isna().sum()
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_rate": missing / len(frame),
        "dtype": frame.dtypes.astype(str),
        "n_unique": frame.nunique(dropna=True),
    })
    return report.sort_values("missing_rate", ascending=False)
