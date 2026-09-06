"""Group performance and distribution shift.

    python -m src.monitoring.fairness

Two honest measurements, and one honest refusal.

**Group performance.** Discrimination, approval rate and error rates sliced by
gender and age band. This is where a risk officer looks first, because a model
can hold 0.78 AUC overall while approving one group at a materially different
rate. Reported alongside three standard fairness gaps:

    demographic parity gap   difference in approval rate between groups
    equal opportunity gap    difference in recall among applicants who defaulted
    predictive parity gap    difference in precision - are declines equally justified

None of these are "fixed" here. Which one a lender should equalise is a policy
and legal question, not a modelling one; the platform's job is to measure them
and put them on screen.

**Distribution shift, not drift.** Train versus holdout population stability
index. This is deliberately *not* called drift: `application_train.csv` has no
time axis, both splits are random samples of one snapshot, and any real drift
monitor needs data collected later than the training set. Calling a random-split
PSI "drift" would be measurement theatre. What it does prove is that the split
is unbiased and the PSI machinery is wired correctly for when time-ordered data
does arrive - so the numbers here should all be near zero, and a number that is
not near zero is a genuine bug worth chasing.
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loader import load_dataset  # noqa: E402
from src.ml.evaluate import threshold_metrics  # noqa: E402
from src.ml.predict import get_scorer, load_holdout  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.helpers import save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

MIN_GROUP_SIZE = 500  # below this, group metrics are noise, not signal

SHIFT_FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_BIRTH", "CNT_CHILDREN", "REGION_POPULATION_RELATIVE",
]


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def age_band(days_birth: pd.Series) -> pd.Series:
    years = -days_birth / 365.0
    return pd.cut(
        years,
        bins=[0, 25, 35, 45, 60, 200],
        labels=["Under 25", "25-34", "35-44", "45-59", "60+"],
        right=False,
    )


def _group_metrics(
    frame: pd.DataFrame, group_column: str, threshold: float
) -> list[dict[str, Any]]:
    """Per-group discrimination, approval and error rates."""
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(group_column, observed=True):
        if len(group) < MIN_GROUP_SIZE or group["TARGET"].nunique() < 2:
            continue
        y_true = group["TARGET"].to_numpy()
        y_prob = group["DEFAULT_PROBABILITY"].to_numpy()
        at_threshold = threshold_metrics(y_true, y_prob, threshold)

        rows.append({
            "group": str(value),
            "applicants": int(len(group)),
            "population_share": round(len(group) / len(frame), 4),
            "observed_default_rate": round(float(y_true.mean()), 4),
            "mean_predicted": round(float(y_prob.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "approval_rate": at_threshold["approval_rate"],
            "recall": at_threshold["recall"],
            "precision": at_threshold["precision"],
            "false_positive_rate": round(1 - at_threshold["specificity"], 4),
            # Calibration within the group: >1 means the model over-predicts risk here.
            "calibration_ratio": round(
                float(y_prob.mean() / y_true.mean()) if y_true.mean() else float("nan"), 3
            ),
        })
    return sorted(rows, key=lambda row: row["applicants"], reverse=True)


def _fairness_gaps(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Max-minus-min across groups for the three standard parity measures."""
    if len(rows) < 2:
        return {}

    def spread(metric: str) -> float:
        values = [row[metric] for row in rows if not np.isnan(row[metric])]
        return round(max(values) - min(values), 4) if values else float("nan")

    return {
        "demographic_parity_gap": spread("approval_rate"),
        "equal_opportunity_gap": spread("recall"),
        "predictive_parity_gap": spread("precision"),
        "auc_gap": spread("roc_auc"),
        "calibration_ratio_spread": spread("calibration_ratio"),
    }


# --------------------------------------------------------------------------- #
# Distribution shift
# --------------------------------------------------------------------------- #
def population_stability_index(
    reference: pd.Series, comparison: pd.Series, bins: int = 10
) -> float:
    """PSI between two samples of one feature.

    Convention in credit risk: < 0.1 stable, 0.1-0.25 moderate shift,
    > 0.25 material shift. Bin edges come from the reference sample only, which
    is what makes the measure directional.
    """
    reference = reference.dropna()
    comparison = comparison.dropna()
    if reference.empty or comparison.empty:
        return float("nan")

    edges = np.unique(np.nanquantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    comparison_share = np.histogram(comparison, bins=edges)[0] / len(comparison)
    # Floor at a small epsilon: an empty bin would otherwise send PSI to infinity.
    epsilon = 1e-6
    reference_share = np.clip(reference_share, epsilon, None)
    comparison_share = np.clip(comparison_share, epsilon, None)

    return float(
        np.sum((comparison_share - reference_share)
               * np.log(comparison_share / reference_share))
    )


def _shift_report(train: pd.DataFrame, holdout: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for feature in SHIFT_FEATURES:
        if feature not in train or feature not in holdout:
            continue
        psi = population_stability_index(train[feature], holdout[feature])
        rows.append({
            "feature": feature,
            "psi": round(psi, 5),
            "verdict": (
                "stable" if psi < 0.1 else "moderate shift" if psi < 0.25 else "material shift"
            ),
            "train_mean": round(float(train[feature].mean()), 2),
            "holdout_mean": round(float(holdout[feature].mean()), 2),
        })
    return sorted(rows, key=lambda row: row["psi"], reverse=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_group_metrics(rows: list[dict[str, Any]], title: str, filename: str) -> str:
    labels = [row["group"] for row in rows]
    positions = np.arange(len(labels))
    width = 0.26

    figure, axis = plt.subplots(figsize=(6.4, 4))
    axis.bar(positions - width, [r["observed_default_rate"] for r in rows], width,
             label="observed default rate", color="#b7472a")
    axis.bar(positions, [r["mean_predicted"] for r in rows], width,
             label="mean predicted", color="#c9772e")
    axis.bar(positions + width, [1 - r["approval_rate"] for r in rows], width,
             label="decline rate", color="#1f4e79")
    axis.set_xticks(positions, labels, rotation=0)
    axis.set(ylabel="Rate", title=title)
    axis.legend(fontsize=8)

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figures_dir / filename
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path.name


def _plot_shift(rows: list[dict[str, Any]], filename: str = "fairness_shift.png") -> str:
    figure, axis = plt.subplots(figsize=(6.4, 4))
    axis.barh([row["feature"] for row in rows][::-1],
              [row["psi"] for row in rows][::-1], color="#2e7d32")
    axis.axvline(0.1, ls="--", color="#c9772e", linewidth=1, label="0.10 moderate")
    axis.axvline(0.25, ls="--", color="#b7472a", linewidth=1, label="0.25 material")
    axis.set(xlabel="Population stability index", title="Train vs holdout distribution shift")
    axis.legend(fontsize=8)
    axis.tick_params(axis="y", labelsize=8)

    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figures_dir / filename
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path.name


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(save: bool = True) -> dict[str, Any]:
    """Group performance + distribution shift over the scored holdout."""
    scorer = get_scorer()
    holdout = load_holdout().copy()
    holdout["AGE_BAND"] = age_band(holdout["DAYS_BIRTH"])

    gender_rows = _group_metrics(
        holdout[holdout["CODE_GENDER"].isin(["M", "F"])], "CODE_GENDER", scorer.threshold
    )
    age_rows = _group_metrics(holdout, "AGE_BAND", scorer.threshold)

    train = load_dataset()
    train = train[~train["SK_ID_CURR"].isin(holdout["SK_ID_CURR"])]
    shift_rows = _shift_report(train, holdout)

    report: dict[str, Any] = {
        "population": int(len(holdout)),
        "decision_threshold": scorer.threshold,
        "min_group_size": MIN_GROUP_SIZE,
        "groups": {
            "gender": {"rows": gender_rows, "gaps": _fairness_gaps(gender_rows)},
            "age_band": {"rows": age_rows, "gaps": _fairness_gaps(age_rows)},
        },
        "distribution_shift": {
            "label": "train vs holdout population stability index",
            "caveat": (
                "This is NOT temporal drift. application_train.csv carries no time "
                "axis and both splits are random samples of one snapshot, so these "
                "values should be near zero by construction. The measurement is "
                "wired and tested here so that a genuine time-ordered feed can be "
                "monitored without new code."
            ),
            "features": shift_rows,
            "max_psi": round(max((row["psi"] for row in shift_rows), default=0.0), 5),
        },
        "figures": {
            "gender": _plot_group_metrics(
                gender_rows, "Performance by gender (holdout)", "fairness_gender.png"
            ) if gender_rows else None,
            "age_band": _plot_group_metrics(
                age_rows, "Performance by age band (holdout)", "fairness_age.png"
            ) if age_rows else None,
            "shift": _plot_shift(shift_rows) if shift_rows else None,
        },
    }

    if save:
        save_json(report, settings.fairness_path)
    _log_report(report)
    return report


def _log_report(report: dict[str, Any]) -> None:
    for name, block in report["groups"].items():
        gaps = block["gaps"]
        if not gaps:
            continue
        logger.info(
            "%s: approval-rate gap %.3f | recall gap %.3f | AUC gap %.3f",
            name, gaps["demographic_parity_gap"], gaps["equal_opportunity_gap"],
            gaps["auc_gap"],
        )
    logger.info("Max train/holdout PSI: %.5f (expected ~0 on a random split)",
                report["distribution_shift"]["max_psi"])


def load_report() -> dict[str, Any] | None:
    from src.utils.helpers import load_json

    return load_json(settings.fairness_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fairness / model-health report.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    build_report(save=not args.no_save)


if __name__ == "__main__":
    main()
