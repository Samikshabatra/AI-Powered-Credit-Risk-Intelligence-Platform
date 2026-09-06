"""Model evaluation: discrimination, calibration and expected loss in currency.

Three layers, in increasing order of how much a credit officer cares:

1. **Discrimination** - ROC-AUC, PR-AUC, KS. Can the model rank applicants?
   PR-AUC is reported alongside ROC-AUC because at an 8% base rate ROC-AUC
   flatters a model that is useless at the top of the ranking.
2. **Calibration** - Brier score and a reliability curve. Ranking is not enough:
   a portfolio is priced off the probability itself, so "0.30" has to mean
   30 defaults per 100.
3. **Expected loss** - the actual decision metric. Every threshold implies a
   portfolio; each portfolio has a cost in currency. The threshold that
   minimises that cost is almost never 0.5.

The cost model is deliberately per-applicant rather than a flat unit cost:
declining a 2M loan forgoes far more margin than declining a 50k one, and
approving a 2M loan that defaults hurts proportionally more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score, roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Discrimination
# --------------------------------------------------------------------------- #
def discrimination_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """ROC-AUC, PR-AUC, KS and Gini for a scored population."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    roc_auc = round(float(roc_auc_score(y_true, y_score)), 4)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    return {
        "roc_auc": roc_auc,
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        # Derived from the rounded AUC so the identity holds exactly in the
        # published metrics rather than differing in the fourth decimal.
        "gini": round(2 * roc_auc - 1, 4),
        "ks_statistic": round(float(np.max(true_positive_rate - false_positive_rate)), 4),
        "base_rate": round(float(y_true.mean()), 4),
        "n": int(len(y_true)),
    }


def threshold_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Confusion matrix and the rates derived from it at one decision threshold."""
    y_true = np.asarray(y_true)
    predicted = (np.asarray(y_score) >= threshold).astype(int)
    true_neg, false_pos, false_neg, true_pos = confusion_matrix(
        y_true, predicted, labels=[0, 1]
    ).ravel()

    def safe(numerator: float, denominator: float) -> float:
        return round(float(numerator / denominator), 4) if denominator else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "true_negatives": int(true_neg),
        "false_positives": int(false_pos),
        "false_negatives": int(false_neg),
        "true_positives": int(true_pos),
        "precision": safe(true_pos, true_pos + false_pos),
        "recall": safe(true_pos, true_pos + false_neg),
        "specificity": safe(true_neg, true_neg + false_pos),
        "f1": safe(2 * true_pos, 2 * true_pos + false_pos + false_neg),
        "approval_rate": safe(true_neg + false_neg, len(y_true)),
        "bad_rate_in_approved": safe(false_neg, true_neg + false_neg),
    }


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Brier score plus expected/maximum calibration error over 10 equal-count bins."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    brier = float(brier_score_loss(y_true, y_prob))
    curve = calibration_curve_data(y_true, y_prob, n_bins=10)
    weights = np.asarray(curve["bin_counts"], dtype=float)
    gaps = np.abs(np.asarray(curve["observed"]) - np.asarray(curve["predicted"]))
    weights = weights / weights.sum() if weights.sum() else weights

    return {
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(float(np.sum(weights * gaps)), 4),
        "max_calibration_error": round(float(np.max(gaps)) if len(gaps) else 0.0, 4),
        "mean_predicted": round(float(y_prob.mean()), 4),
        "mean_observed": round(float(y_true.mean()), 4),
    }


def calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    """Reliability-curve points using equal-count bins (quantile binning).

    Equal-width bins are useless here: with an 8% base rate almost every
    prediction lands in the first two bins and the curve carries no information.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    grouped = frame.groupby("bin")
    return {
        "predicted": [round(float(v), 4) for v in grouped["p"].mean()],
        "observed": [round(float(v), 4) for v in grouped["y"].mean()],
        "bin_counts": [int(v) for v in grouped.size()],
    }


# --------------------------------------------------------------------------- #
# Cost-sensitive decisioning
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    """Currency cost of each decision error, as a fraction of the loan amount.

    lgd_rate    - loss given default: the share of AMT_CREDIT written off when a
                  defaulting applicant is approved (a false negative).
    margin_rate - lifetime margin forgone when a good applicant is declined
                  (a false positive).
    """

    lgd_rate: float = 0.45
    margin_rate: float = 0.08

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost_model: CostModel,
) -> dict[str, float]:
    """Total portfolio cost at one threshold, split into its two error sources."""
    y_true = np.asarray(y_true)
    declined = np.asarray(y_prob) >= threshold
    amounts = np.asarray(amounts, dtype=float)

    # Approved but defaulted -> we lose LGD of the loan.
    default_loss = float((amounts[(~declined) & (y_true == 1)]).sum() * cost_model.lgd_rate)
    # Declined but would have repaid -> we lose the margin we would have earned.
    opportunity_loss = float(
        (amounts[declined & (y_true == 0)]).sum() * cost_model.margin_rate
    )

    return {
        "threshold": round(float(threshold), 4),
        "default_loss": round(default_loss, 2),
        "opportunity_loss": round(opportunity_loss, 2),
        "total_cost": round(default_loss + opportunity_loss, 2),
        "approval_rate": round(float((~declined).mean()), 4),
    }


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    cost_model: CostModel,
    n_points: int = 99,
) -> pd.DataFrame:
    """Portfolio cost across the whole threshold range - the curve the UI plots."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    return pd.DataFrame(
        [expected_cost(y_true, y_prob, amounts, t, cost_model) for t in thresholds]
    )


def optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    cost_model: CostModel,
    n_points: int = 199,
) -> tuple[float, pd.DataFrame]:
    """The threshold minimising expected portfolio cost, plus the full curve.

    Fit this on a split that is neither the training data nor the final holdout,
    otherwise the reported saving is just the threshold overfitting its own
    evaluation set.
    """
    curve = cost_curve(y_true, y_prob, amounts, cost_model, n_points=n_points)
    best = curve.loc[curve["total_cost"].idxmin()]
    threshold = float(best["threshold"])
    logger.info(
        "Cost-optimal threshold = %.3f (total cost %.0f, approval rate %.1f%%)",
        threshold, best["total_cost"], 100 * best["approval_rate"],
    )
    return threshold, curve


def cost_comparison(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    optimal: float,
    cost_model: CostModel,
) -> dict[str, dict[str, float]]:
    """Cost-optimal threshold against the two policies a bank would otherwise run."""
    scenarios = {
        "approve_everyone": 1.01,   # no model at all
        "naive_0.5": 0.5,           # textbook default threshold
        "cost_optimal": optimal,
    }
    results = {
        name: expected_cost(y_true, y_prob, amounts, t, cost_model)
        for name, t in scenarios.items()
    }
    optimal_cost = results["cost_optimal"]["total_cost"]
    for name, result in results.items():
        baseline = result["total_cost"]
        result["saving_vs_this"] = round(baseline - optimal_cost, 2)
        result["saving_pct"] = round(
            (baseline - optimal_cost) / baseline, 4
        ) if baseline else 0.0
    return results


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
def risk_band_edges(threshold: float) -> dict[str, float]:
    """Band cut-offs anchored on the cost-optimal threshold, not on round numbers.

    High starts exactly where the economics say decline; Medium is the review
    zone at half that probability; everything below is straight-through approve.
    """
    return {"low_max": round(threshold / 2, 4), "high_min": round(threshold, 4)}


def assign_risk_band(probability: float | np.ndarray, edges: dict[str, float]):
    """Map probability to 'Low' / 'Medium' / 'High'. Scalar or vectorised."""
    if np.isscalar(probability):
        if probability < edges["low_max"]:
            return "Low"
        return "Medium" if probability < edges["high_min"] else "High"
    probability = np.asarray(probability)
    bands = np.full(probability.shape, "Medium", dtype=object)
    bands[probability < edges["low_max"]] = "Low"
    bands[probability >= edges["high_min"]] = "High"
    return bands


def band_summary(
    y_true: np.ndarray, y_prob: np.ndarray, edges: dict[str, float]
) -> pd.DataFrame:
    """Population share and realised default rate per band - the band's own proof."""
    frame = pd.DataFrame({
        "band": assign_risk_band(y_prob, edges),
        "y": np.asarray(y_true),
        "p": np.asarray(y_prob),
    })
    summary = frame.groupby("band").agg(
        applicants=("y", "size"),
        observed_default_rate=("y", "mean"),
        mean_predicted=("p", "mean"),
    )
    summary["population_share"] = summary["applicants"] / len(frame)
    order = [b for b in ("Low", "Medium", "High") if b in summary.index]
    return summary.loc[order].round(4).reset_index()


# --------------------------------------------------------------------------- #
# Gains / lift
# --------------------------------------------------------------------------- #
def decile_lift(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Default rate per risk decile - the table a credit committee reads first."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), q=n_bins, labels=False)
    frame["decile"] = n_bins - frame["decile"]  # decile 1 = riskiest

    summary = frame.groupby("decile").agg(
        applicants=("y", "size"),
        defaults=("y", "sum"),
        default_rate=("y", "mean"),
        mean_score=("p", "mean"),
    )
    base_rate = frame["y"].mean()
    summary["lift"] = summary["default_rate"] / base_rate if base_rate else np.nan
    summary["cumulative_defaults_captured"] = (
        summary["defaults"].cumsum() / summary["defaults"].sum()
    )
    return summary.round(4).reset_index()
