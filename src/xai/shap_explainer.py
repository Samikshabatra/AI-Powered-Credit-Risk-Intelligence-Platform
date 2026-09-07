"""SHAP explainability for the credit-risk model.

One honest caveat, stated up front because it matters for a lending decision:
SHAP explains the **raw booster output in log-odds space**, not the calibrated
probability. Isotonic calibration is monotone, so the *ranking and sign* of every
contribution carry over unchanged - "low external score pushed this applicant
riskier" stays true - but the magnitudes are log-odds, not percentage points.
The UI therefore shows contributions as a share of total push, never as "+7% of
default probability", which would be a number the method cannot support.

TreeExplainer is exact for tree ensembles (no sampling), so a per-applicant
explanation costs milliseconds and can run inline in the app.

`shap` is imported lazily, inside the functions that need it. It pulls in numba
and llvmlite - around 250 MB resident - and the deployed app has ~1 GB to work
with, so a visitor who never opens the Explainability page never pays for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.ml.predict import RiskScorer, get_scorer  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.helpers import humanise_feature  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _shap():
    """Import `shap` on first use. See the module docstring for why."""
    import shap

    return shap


# Features that carry no explanatory value for a human reader even when the
# model leans on them - process artefacts rather than applicant characteristics.
_UNINFORMATIVE = {
    "HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START",
    "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20",
    "FLAG_DOCUMENT_21",
}


@dataclass
class ShapExplanation:
    """A single applicant's attribution, ready for the UI and for reason codes."""

    applicant_id: int | None
    base_value: float
    raw_score_logit: float
    contributions: pd.DataFrame  # feature, label, value, shap, direction, share

    def top_risk_drivers(self, n: int = 5) -> pd.DataFrame:
        """Features pushing the applicant towards default, strongest first."""
        increasing = self.contributions[self.contributions["shap"] > 0]
        return increasing.head(n)

    def top_protective(self, n: int = 3) -> pd.DataFrame:
        """Features pushing the applicant towards repayment, strongest first."""
        decreasing = self.contributions[self.contributions["shap"] < 0]
        return decreasing.reindex(
            decreasing["shap"].abs().sort_values(ascending=False).index
        ).head(n)

    def to_dict(self, n: int = 5) -> dict:
        return {
            "applicant_id": self.applicant_id,
            "risk_drivers": self.top_risk_drivers(n).to_dict(orient="records"),
            "protective_factors": self.top_protective(3).to_dict(orient="records"),
        }


class ShapExplainer:
    """Wraps `shap.TreeExplainer` with the feature naming and framing the UI needs."""

    def __init__(self, scorer: RiskScorer | None = None) -> None:
        self.scorer = scorer or get_scorer()
        self.explainer = _shap().TreeExplainer(self.scorer.model)
        self.feature_names = self.scorer.feature_names
        logger.info("TreeExplainer ready over %d features", len(self.feature_names))

    # ------------------------------------------------------------------ #
    def _shap_matrix(self, features: pd.DataFrame) -> tuple[np.ndarray, float]:
        """SHAP values for the positive class plus its base value.

        Different SHAP/LightGBM version pairs return either a single matrix or a
        per-class list, so both shapes are normalised here rather than at every
        call site.
        """
        values = self.explainer.shap_values(features)
        expected = self.explainer.expected_value

        if isinstance(values, list):  # [class_0, class_1]
            values = values[1]
            expected = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
        elif isinstance(values, np.ndarray) and values.ndim == 3:  # (n, features, classes)
            values = values[:, :, 1]
            expected = expected[1] if np.ndim(expected) else expected

        return np.asarray(values), float(np.ravel(expected)[0])

    # ------------------------------------------------------------------ #
    def explain(self, row: pd.DataFrame | pd.Series) -> ShapExplanation:
        """Per-applicant attribution over the raw application row."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row
        features = self.scorer.build_features(frame)
        values, base_value = self._shap_matrix(features)
        shap_row = values[0]

        contributions = pd.DataFrame({
            "feature": self.feature_names,
            "label": [humanise_feature(name) for name in self.feature_names],
            "value": features.iloc[0].to_numpy(),
            "shap": shap_row,
        })
        contributions["display_value"] = [
            self._display_value(feature, value)
            for feature, value in zip(contributions["feature"], contributions["value"])
        ]
        contributions["direction"] = np.where(
            contributions["shap"] > 0, "increases risk", "reduces risk"
        )
        total_push = np.abs(shap_row).sum()
        contributions["share"] = (
            np.abs(contributions["shap"]) / total_push if total_push else 0.0
        )
        contributions = contributions[~contributions["feature"].isin(_UNINFORMATIVE)]
        contributions = contributions.sort_values("shap", ascending=False).reset_index(drop=True)

        # A manually-entered applicant has no id, so the column can be NaN.
        applicant_id = (
            int(frame["SK_ID_CURR"].iloc[0])
            if "SK_ID_CURR" in frame and pd.notna(frame["SK_ID_CURR"].iloc[0])
            else None
        )
        return ShapExplanation(
            applicant_id=applicant_id,
            base_value=base_value,
            raw_score_logit=float(base_value + shap_row.sum()),
            contributions=contributions,
        )

    def _display_value(self, feature: str, value: float) -> str:
        """Render a feature value the way a human reads it, decoding categoricals."""
        if feature in self.scorer.preprocessor.state.categorical_features:
            decoded = self.scorer.preprocessor.decode_category(feature, value)
            return decoded if decoded is not None else "not provided"
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "not provided"
        if abs(value) >= 10_000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:,.1f}"
        return f"{value:.3f}"

    # ------------------------------------------------------------------ #
    def global_importance(
        self, frame: pd.DataFrame, sample: int = 2000, seed: int = 42
    ) -> pd.DataFrame:
        """Mean |SHAP| per feature over a sample - the global driver ranking."""
        if len(frame) > sample:
            frame = frame.sample(n=sample, random_state=seed)
        features = self.scorer.build_features(frame)
        values, _ = self._shap_matrix(features)

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "label": [humanise_feature(name) for name in self.feature_names],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            # Sign of the average signed contribution: does a high value of this
            # feature usually push risk up or down across the portfolio?
            "mean_shap": values.mean(axis=0),
        })
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def plot_global_summary(
        self, frame: pd.DataFrame, sample: int = 2000, filename: str = "shap_summary.png"
    ) -> str:
        """Beeswarm summary plot - the one global SHAP figure the deck uses."""
        if len(frame) > sample:
            frame = frame.sample(n=sample, random_state=42)
        features = self.scorer.build_features(frame)
        values, _ = self._shap_matrix(features)

        labelled = features.copy()
        labelled.columns = [humanise_feature(c) for c in labelled.columns]

        settings.figures_dir.mkdir(parents=True, exist_ok=True)
        path = settings.figures_dir / filename
        plt.figure(figsize=(8, 7))
        _shap().summary_plot(values, labelled, max_display=18, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
        logger.info("Wrote %s", path.name)
        return path.name

    def plot_applicant_waterfall(
        self, row: pd.Series, filename: str | None = None, max_display: int = 12
    ) -> str:
        """Per-applicant waterfall: base rate -> this applicant's score."""
        frame = row.to_frame().T
        features = self.scorer.build_features(frame)
        values, base_value = self._shap_matrix(features)

        explanation = _shap().Explanation(
            values=values[0],
            base_values=base_value,
            data=features.iloc[0].to_numpy(),
            feature_names=[humanise_feature(c) for c in features.columns],
        )
        applicant_id = int(frame["SK_ID_CURR"].iloc[0]) if "SK_ID_CURR" in frame else 0
        filename = filename or f"shap_waterfall_{applicant_id}.png"

        settings.figures_dir.mkdir(parents=True, exist_ok=True)
        path = settings.figures_dir / filename
        plt.figure(figsize=(8, 6))
        _shap().plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
        return path.name


@lru_cache(maxsize=1)
def get_explainer() -> ShapExplainer:
    """Process-wide singleton, so Streamlit does not rebuild the explainer per click."""
    return ShapExplainer()


def explain_applicant(applicant_id: int) -> ShapExplanation:
    """Convenience entry point used by the agent's `explain_prediction` tool."""
    from src.ml.predict import get_applicant

    return get_explainer().explain(get_applicant(applicant_id))
