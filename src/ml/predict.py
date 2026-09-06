"""Inference: score an applicant, band them, and decide.

Everything downstream - the UI, the SHAP explainer, the reason codes, the agent
tools - goes through `RiskScorer`, so a prediction is produced exactly one way.
The artifacts are loaded once and cached at module level, because the Streamlit
app re-imports on every interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessor import ID_COLUMN, TARGET_COLUMN, CreditRiskPreprocessor
from src.ml.evaluate import assign_risk_band
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

BAND_ACTIONS = {
    "Low": "Auto-approve. Straight-through processing, standard pricing.",
    "Medium": "Refer to manual underwriting. Request income verification "
              "and re-price for risk.",
    "High": "Decline, or approve only with a reduced limit and collateral.",
}


class ModelNotTrained(FileNotFoundError):
    """Raised when scoring is attempted before `python -m src.ml.train` has run."""


@dataclass
class RiskAssessment:
    """One applicant's decision, in the order the Decision Trace renders it."""

    applicant_id: int | None
    probability: float
    risk_band: str
    decision: str
    threshold: float
    band_edges: dict[str, float]
    recommended_action: str
    raw_score: float | None = None
    actual_target: int | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "applicant_id": self.applicant_id,
            "probability": round(self.probability, 4),
            "probability_pct": f"{self.probability * 100:.1f}%",
            "risk_band": self.risk_band,
            "decision": self.decision,
            "threshold": self.threshold,
            "band_edges": self.band_edges,
            "recommended_action": self.recommended_action,
        }
        if self.actual_target is not None:
            payload["actual_outcome"] = "defaulted" if self.actual_target else "repaid"
        return payload


class RiskScorer:
    """Loads the trained artifacts and turns raw application rows into decisions."""

    def __init__(self) -> None:
        if not settings.model_path.exists() or not settings.preprocessor_path.exists():
            raise ModelNotTrained(
                f"No trained model in {settings.model_dir}. "
                "Run `python -m src.ml.train` first."
            )
        bundle = joblib.load(settings.model_path)
        self.model = bundle["model"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.categorical_features: list[str] = bundle["categorical_features"]
        self.threshold: float = float(bundle["threshold"])
        self.band_edges: dict[str, float] = bundle["risk_band_edges"]
        self.cost_model: dict[str, float] = bundle["cost_model"]
        self.trained_at: str = bundle["trained_at"]

        self.preprocessor: CreditRiskPreprocessor = joblib.load(settings.preprocessor_path)
        self.calibrator = (
            joblib.load(settings.calibrator_path)
            if settings.calibrator_path.exists() else None
        )
        logger.info("Loaded model trained at %s (threshold %.3f)",
                    self.trained_at, self.threshold)

    # ------------------------------------------------------------------ #
    # Core scoring
    # ------------------------------------------------------------------ #
    def build_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Raw application rows -> the exact matrix the booster was trained on."""
        return self.preprocessor.transform(frame)

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(features)[:, 1]

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration. Without it, scores are inflated ~11x by
        the imbalance reweighting and cannot be read as probabilities."""
        if self.calibrator is None:
            return raw
        return np.asarray(self.calibrator.predict(raw), dtype=float)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default for a raw application frame."""
        return self.calibrate(self.raw_scores(self.build_features(frame)))

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring: probability, band and decision per row."""
        probabilities = self.predict_proba(frame)
        result = pd.DataFrame({
            "DEFAULT_PROBABILITY": probabilities,
            "RISK_BAND": assign_risk_band(probabilities, self.band_edges),
            "DECISION": np.where(probabilities >= self.threshold, "Decline", "Approve"),
        }, index=frame.index)
        if ID_COLUMN in frame:
            result.insert(0, ID_COLUMN, frame[ID_COLUMN].to_numpy())
        return result

    def assess(self, row: pd.DataFrame | pd.Series) -> RiskAssessment:
        """Score a single applicant and package the full decision context."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        if len(frame) != 1:
            raise ValueError(f"assess() expects exactly one row, got {len(frame)}")

        features = self.build_features(frame)
        raw = float(self.raw_scores(features)[0])
        probability = float(self.calibrate(np.array([raw]))[0])
        band = str(assign_risk_band(probability, self.band_edges))

        return RiskAssessment(
            applicant_id=(
                int(frame[ID_COLUMN].iloc[0])
                if ID_COLUMN in frame and pd.notna(frame[ID_COLUMN].iloc[0])
                else None
            ),
            probability=probability,
            raw_score=raw,
            risk_band=band,
            decision="Decline" if probability >= self.threshold else "Approve",
            threshold=self.threshold,
            band_edges=self.band_edges,
            recommended_action=BAND_ACTIONS[band],
            actual_target=(
                int(frame[TARGET_COLUMN].iloc[0])
                if TARGET_COLUMN in frame and pd.notna(frame[TARGET_COLUMN].iloc[0])
                else None
            ),
            features=features.iloc[0].to_dict(),
        )

    # ------------------------------------------------------------------ #
    # What-if / counterfactual
    # ------------------------------------------------------------------ #
    def what_if(self, row: pd.Series, overrides: dict[str, Any]) -> RiskAssessment:
        """Re-score an applicant with edited raw inputs.

        Overrides are applied to the *raw* columns, so every engineered feature
        (loan-to-income, debt service, employment years) is recomputed from the
        edited values rather than left stale.
        """
        edited = row.copy()
        for column, value in overrides.items():
            if column not in edited.index:
                raise KeyError(f"Unknown column for what-if: {column}")
            edited[column] = value
        return self.assess(edited)


# --------------------------------------------------------------------------- #
# Applicant pool (the held-out set the UI browses)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_scorer() -> RiskScorer:
    """Process-wide singleton; safe under Streamlit's re-run model."""
    return RiskScorer()


@lru_cache(maxsize=1)
def load_holdout() -> pd.DataFrame:
    """The scored holdout snapshot written by training, indexed by applicant id."""
    if not settings.holdout_path.exists():
        raise ModelNotTrained(
            f"No holdout snapshot at {settings.holdout_path}. Run `python -m src.ml.train`."
        )
    frame = pd.read_parquet(settings.holdout_path)
    return frame.set_index(ID_COLUMN, drop=False)


def get_applicant(applicant_id: int) -> pd.Series:
    """Fetch one raw applicant row from the holdout pool."""
    holdout = load_holdout()
    if applicant_id not in holdout.index:
        raise KeyError(
            f"Applicant {applicant_id} is not in the holdout set "
            f"({len(holdout):,} available; try list_applicants())."
        )
    return holdout.loc[applicant_id]


def list_applicants(
    limit: int = 50, band: str | None = None, seed: int = 0
) -> pd.DataFrame:
    """A browsable sample of holdout applicants for the UI selector."""
    holdout = load_holdout()
    pool = holdout if band is None else holdout[holdout["RISK_BAND"] == band]
    if pool.empty:
        return pd.DataFrame(columns=[ID_COLUMN, "DEFAULT_PROBABILITY", "RISK_BAND"])
    sample = pool.sample(n=min(limit, len(pool)), random_state=seed)
    columns = [
        ID_COLUMN, "DEFAULT_PROBABILITY", "RISK_BAND", "TARGET", "AMT_CREDIT",
        "AMT_INCOME_TOTAL", "CODE_GENDER", "NAME_CONTRACT_TYPE", "OCCUPATION_TYPE",
    ]
    available = [column for column in columns if column in sample.columns]
    return sample[available].sort_values("DEFAULT_PROBABILITY", ascending=False)


def score_applicant(applicant_id: int) -> RiskAssessment:
    """Convenience entry point used by the agent's `predict_risk` tool."""
    return get_scorer().assess(get_applicant(applicant_id))
