"""End-to-end training pipeline.

    python -m src.ml.train              # full run
    python -m src.ml.train --sample 50000   # fast smoke run

Pipeline: load -> preprocess -> split -> train -> calibrate -> pick threshold ->
evaluate -> persist artifacts, figures, metrics and scored predictions.

**Split design.** Four disjoint stratified splits, because each downstream step
needs data the previous one did not touch:

    train (60%)   fit the booster
    valid (10%)   early stopping
    calib (10%)   fit the isotonic calibrator AND pick the cost threshold
    holdout (20%) report every headline number; also the applicant pool in the UI

Fitting the calibrator on the early-stopping split, or choosing the threshold on
the split used to report savings, both produce numbers that do not survive
contact with production. Keeping them separate costs 10% of the data and buys an
honest headline.

**Class imbalance.** 8.07% positives. Handled with LightGBM `scale_pos_weight`
(= negatives / positives ~ 11.4), not SMOTE: synthesising 260k minority rows in
143 dimensions is slow, adds no information the tree cannot already find through
reweighting, and interpolates nonsense between categorical codes. The cost is
that raw scores come out badly inflated - which is precisely what the isotonic
calibration step exists to undo.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # figures are written to disk, never displayed
import matplotlib.pyplot as plt  # noqa: E402

from src.data.build_database import write_predictions  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.data.preprocessor import ID_COLUMN, TARGET_COLUMN, CreditRiskPreprocessor  # noqa: E402
from src.ml.evaluate import (  # noqa: E402
    CostModel, assign_risk_band, band_summary, calibration_curve_data,
    calibration_metrics, cost_comparison, decile_lift, discrimination_metrics,
    optimal_threshold, risk_band_edges, threshold_metrics,
)
from src.utils.config import settings  # noqa: E402
from src.utils.helpers import fmt_money, fmt_pct, save_json, set_seed, timed  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 34,
    "max_depth": 8,
    "min_child_samples": 120,      # 307k rows: large leaves keep splits stable
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_split_gain": 0.01,
    "n_estimators": 4000,          # cut short by early stopping, never reached
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 200

# Baselines are deliberately capped: they exist to answer "versus what?", not to
# be tuned competitors. Depth and leaf size keep the forest to ~1 minute on 184k
# rows while still being a fair, converged model rather than a strawman.
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 50,
    "class_weight": "balanced",
    "n_jobs": -1,
}
LOGISTIC_PARAMS: dict = {
    "class_weight": "balanced",
    "max_iter": 2000,     # scaled 143-column input converges well inside this
}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(
    features: pd.DataFrame, target: pd.Series, seed: int
) -> dict[str, np.ndarray]:
    """Stratified train / valid / calib / holdout index arrays."""
    holdout_fraction = settings.holdout_fraction
    valid_fraction = settings.valid_fraction
    calib_fraction = settings.calib_fraction

    index = np.arange(len(features))
    remaining_index, holdout_index = train_test_split(
        index, test_size=holdout_fraction, stratify=target, random_state=seed
    )
    remaining_target = target.iloc[remaining_index]

    # Fractions below are expressed relative to the remaining pool.
    remaining_share = 1 - holdout_fraction
    calib_share = calib_fraction / remaining_share
    remaining_index, calib_index = train_test_split(
        remaining_index, test_size=calib_share, stratify=remaining_target, random_state=seed
    )
    remaining_target = target.iloc[remaining_index]

    valid_share = valid_fraction / (remaining_share - calib_fraction)
    train_index, valid_index = train_test_split(
        remaining_index, test_size=valid_share, stratify=remaining_target, random_state=seed
    )

    splits = {
        "train": train_index, "valid": valid_index,
        "calib": calib_index, "holdout": holdout_index,
    }
    logger.info(
        "Splits: %s",
        ", ".join(
            f"{name}={len(idx):,} ({target.iloc[idx].mean():.2%} default)"
            for name, idx in splits.items()
        ),
    )
    return splits


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_booster(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    categorical_features: list[str],
    seed: int,
) -> lgb.LGBMClassifier:
    """Fit LightGBM with imbalance reweighting and AUC early stopping."""
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    logger.info("scale_pos_weight = %.2f (%s neg / %s pos)",
                scale_pos_weight, f"{negatives:,}", f"{positives:,}")

    model = lgb.LGBMClassifier(
        **LGBM_PARAMS, random_state=seed, scale_pos_weight=scale_pos_weight
    )
    with timed("LightGBM training"):
        model.fit(
            x_train, y_train,
            # eval_X / eval_y rather than the deprecated eval_set tuple list.
            eval_X=x_valid, eval_y=y_valid,
            eval_metric="auc",
            categorical_feature=categorical_features,
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )
    logger.info("Best iteration: %d (valid AUC %.4f)",
                model.best_iteration_, model.best_score_["valid_0"]["auc"])
    return model


def fit_calibrator(y_calib: pd.Series, raw_scores: np.ndarray) -> IsotonicRegression:
    """Isotonic calibration of the reweighted scores back onto true probabilities.

    Isotonic over Platt: `scale_pos_weight` applies a roughly constant odds shift
    that a sigmoid can absorb, but the residual miscalibration in the tails is
    not sigmoid-shaped. With 30k calibration rows there is ample data for the
    non-parametric fit.

    Isotonic is monotone *non-decreasing*, so it never reorders two applicants -
    but it does collapse many distinct scores onto the same step (61k unique raw
    scores become ~150 calibrated levels). Those ties move ROC-AUC by a few
    ten-thousandths rather than leaving it bit-identical. Measured on this model:
    0.7792 raw against 0.7788 calibrated. That is the honest statement; "ranking
    is preserved so AUC is unchanged" is the almost-true version of it.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, y_calib.to_numpy())
    return calibrator


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_figure(figure: plt.Figure, name: str) -> Path:
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figures_dir / name
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_model_diagnostics(
    y_holdout: np.ndarray,
    probabilities: np.ndarray,
    raw_scores: np.ndarray,
    curve: pd.DataFrame,
    threshold: float,
    importance: pd.DataFrame,
) -> list[str]:
    """Write the six figures the UI, the README and the deck all reuse."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    written: list[str] = []

    # 1. ROC
    false_positive_rate, true_positive_rate, _ = roc_curve(y_holdout, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot(false_positive_rate, true_positive_rate, color="#1f4e79", linewidth=2)
    axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="ROC curve (holdout)")
    written.append(_save_figure(figure, "ml_roc_curve.png").name)

    # 2. Precision-recall (the honest one under 8% prevalence)
    precision, recall, _ = precision_recall_curve(y_holdout, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot(recall, precision, color="#b7472a", linewidth=2)
    axis.axhline(float(np.mean(y_holdout)), ls="--", color="#999999", linewidth=1,
                 label=f"base rate {np.mean(y_holdout):.1%}")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (holdout)")
    axis.legend()
    written.append(_save_figure(figure, "ml_pr_curve.png").name)

    # 3. Calibration, before vs after
    before = calibration_curve_data(y_holdout, raw_scores)
    after = calibration_curve_data(y_holdout, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4.2))
    axis.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1, label="perfect")
    axis.plot(before["predicted"], before["observed"], "o-", color="#c9772e",
              label="raw (reweighted)")
    axis.plot(after["predicted"], after["observed"], "o-", color="#1f4e79",
              label="isotonic-calibrated")
    axis.set(xlabel="Predicted probability", ylabel="Observed default rate",
             title="Calibration (holdout)")
    axis.legend()
    written.append(_save_figure(figure, "ml_calibration.png").name)

    # 4. Cost curve
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    axis.plot(curve["threshold"], curve["total_cost"], color="#1f4e79",
              linewidth=2, label="total cost")
    axis.plot(curve["threshold"], curve["default_loss"], "--", color="#b7472a",
              linewidth=1.2, label="credit loss")
    axis.plot(curve["threshold"], curve["opportunity_loss"], "--", color="#2e7d32",
              linewidth=1.2, label="forgone margin")
    axis.axvline(threshold, color="#000000", linewidth=1,
                 label=f"optimum {threshold:.2f}")
    axis.axvline(0.5, color="#999999", ls=":", linewidth=1, label="naive 0.50")
    axis.set(xlabel="Decision threshold", ylabel="Expected portfolio cost",
             title="Expected loss vs threshold")
    axis.legend(fontsize=8)
    written.append(_save_figure(figure, "ml_cost_curve.png").name)

    # 5. Score distribution by outcome
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    axis.hist(probabilities[y_holdout == 0], bins=60, alpha=0.65,
              color="#2e7d32", label="repaid", density=True)
    axis.hist(probabilities[y_holdout == 1], bins=60, alpha=0.65,
              color="#b7472a", label="defaulted", density=True)
    axis.axvline(threshold, color="#000000", linewidth=1, label="decision threshold")
    axis.set(xlabel="Calibrated probability of default", ylabel="Density",
             title="Score separation (holdout)")
    axis.legend()
    written.append(_save_figure(figure, "ml_score_distribution.png").name)

    # 6. Feature importance
    top = importance.head(20).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6.4, 6))
    axis.barh(top["feature"], top["gain"], color="#1f4e79")
    axis.set(xlabel="Total split gain", title="Top 20 features by gain")
    axis.tick_params(axis="y", labelsize=8)
    written.append(_save_figure(figure, "ml_feature_importance.png").name)

    return written


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def benchmark_baselines(
    features: pd.DataFrame,
    target: pd.Series,
    splits: dict[str, np.ndarray],
    lightgbm_holdout: dict[str, float],
    seed: int,
) -> dict[str, dict]:
    """Score two honest baselines on the same holdout, to answer "versus what?".

    Logistic regression and a random forest are fitted on the *same* train split,
    with the *same* features and seed, and scored on the *same* untouched holdout
    as LightGBM. Nothing else would be a fair comparison.

    **Input assumption.** Neither baseline handles NaN, and neither handles
    LightGBM's native categorical splits. The categorical columns are already
    integer-coded by the preprocessor, so they are passed through as ordinary
    numeric columns and the whole matrix is median-imputed. That treats an
    arbitrary code ordering as if it were meaningful, which is exactly the
    handicap a simple baseline carries in real life - and part of why LightGBM
    wins. One-hot encoding 143 columns would be a different, much larger model,
    not a baseline.

    Imputer and scaler are fitted on train only and applied to holdout, so the
    comparison carries no leakage the LightGBM number does not also carry.
    """
    x_train = features.iloc[splits["train"]]
    x_holdout = features.iloc[splits["holdout"]]
    y_train = target.iloc[splits["train"]].to_numpy()
    y_holdout = target.iloc[splits["holdout"]].to_numpy()

    imputer = SimpleImputer(strategy="median")
    x_train_filled = imputer.fit_transform(x_train)
    x_holdout_filled = imputer.transform(x_holdout)

    results: dict[str, dict] = {}

    # --- logistic regression: standard-scaled, because it is not scale-free ---
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_filled)
    x_holdout_scaled = scaler.transform(x_holdout_filled)
    with timed("Logistic regression baseline"):
        logistic = LogisticRegression(**LOGISTIC_PARAMS, random_state=seed)
        logistic.fit(x_train_scaled, y_train)
    results["logistic_regression"] = discrimination_metrics(
        y_holdout, logistic.predict_proba(x_holdout_scaled)[:, 1]
    )

    # --- random forest: no scaling needed, but still needs the imputation ---
    with timed("Random forest baseline"):
        forest = RandomForestClassifier(**RANDOM_FOREST_PARAMS, random_state=seed)
        forest.fit(x_train_filled, y_train)
    results["random_forest"] = discrimination_metrics(
        y_holdout, forest.predict_proba(x_holdout_filled)[:, 1]
    )

    results["lightgbm"] = dict(lightgbm_holdout)

    for name, scores in results.items():
        logger.info("Baseline %-20s holdout ROC-AUC %.4f | PR-AUC %.4f",
                    name, scores["roc_auc"], scores["pr_auc"])
    return results


def plot_baseline_comparison(baselines: dict[str, dict]) -> str:
    """Grouped bar of holdout ROC-AUC and PR-AUC per model."""
    labels = {
        "logistic_regression": "Logistic\nregression",
        "random_forest": "Random\nforest",
        "lightgbm": "LightGBM\n(shipped)",
    }
    order = [name for name in labels if name in baselines]
    positions = np.arange(len(order))
    width = 0.36

    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    roc = [baselines[name]["roc_auc"] for name in order]
    prc = [baselines[name]["pr_auc"] for name in order]
    bars_roc = axis.bar(positions - width / 2, roc, width, label="ROC-AUC",
                        color="#1f4e79")
    bars_prc = axis.bar(positions + width / 2, prc, width, label="PR-AUC",
                        color="#c9772e")
    for bars in (bars_roc, bars_prc):
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    axis.set_xticks(positions, [labels[name] for name in order], fontsize=9)
    axis.set(ylabel="Score", ylim=(0, 1.0),
             title="Baseline comparison (same holdout, same features)")
    axis.legend(fontsize=8)
    return _save_figure(figure, "ml_baseline_comparison.png").name


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_training(sample: int | None = None, seed: int | None = None) -> dict:
    """Execute the full pipeline and return the metrics payload."""
    seed = seed if seed is not None else settings.random_seed
    set_seed(seed)
    settings.ensure_dirs()

    # ---- data ----
    frame = load_dataset(split="train", nrows=sample, use_cache=sample is None)
    if sample and len(frame) > sample:
        frame = frame.sample(n=sample, random_state=seed).reset_index(drop=True)

    identifiers = frame[ID_COLUMN].to_numpy()
    amounts = frame["AMT_CREDIT"].to_numpy(dtype=float)

    preprocessor = CreditRiskPreprocessor()
    features, target = preprocessor.fit_transform(frame)
    if target is None:
        raise ValueError("TARGET column missing - cannot train.")

    splits = make_splits(features, target, seed)
    categorical = preprocessor.state.categorical_features

    # ---- fit ----
    model = train_booster(
        features.iloc[splits["train"]], target.iloc[splits["train"]],
        features.iloc[splits["valid"]], target.iloc[splits["valid"]],
        categorical, seed,
    )

    raw_scores = {
        name: model.predict_proba(features.iloc[index])[:, 1]
        for name, index in splits.items()
    }

    # ---- calibrate on a split used for nothing else ----
    calibrator = fit_calibrator(target.iloc[splits["calib"]], raw_scores["calib"])
    probabilities = {name: calibrator.predict(scores) for name, scores in raw_scores.items()}

    # ---- decide: threshold chosen on calib, reported on holdout ----
    cost_model = CostModel(lgd_rate=settings.lgd_rate, margin_rate=settings.margin_rate)
    threshold, _ = optimal_threshold(
        target.iloc[splits["calib"]].to_numpy(),
        probabilities["calib"],
        amounts[splits["calib"]],
        cost_model,
    )
    edges = risk_band_edges(threshold)

    # ---- evaluate on the untouched holdout ----
    y_holdout = target.iloc[splits["holdout"]].to_numpy()
    p_holdout = probabilities["holdout"]
    holdout_amounts = amounts[splits["holdout"]]

    _, holdout_curve = optimal_threshold(y_holdout, p_holdout, holdout_amounts, cost_model)
    comparison = cost_comparison(y_holdout, p_holdout, holdout_amounts, threshold, cost_model)

    importance = pd.DataFrame({
        "feature": features.columns,
        "gain": model.booster_.feature_importance("gain"),
        "split": model.booster_.feature_importance("split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    discrimination = {
        name: discrimination_metrics(target.iloc[splits[name]].to_numpy(),
                                     probabilities[name])
        for name in ("train", "valid", "holdout")
    }
    baselines = benchmark_baselines(
        features, target, splits, discrimination["holdout"], seed
    )

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(frame)),
        "n_features": int(features.shape[1]),
        "n_categorical": len(categorical),
        "best_iteration": int(model.best_iteration_),
        "scale_pos_weight": round(float(model.get_params()["scale_pos_weight"]), 3),
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
        "discrimination": discrimination,
        "calibration": {
            "before": calibration_metrics(y_holdout, raw_scores["holdout"]),
            "after": calibration_metrics(y_holdout, p_holdout),
            "curve_before": calibration_curve_data(y_holdout, raw_scores["holdout"]),
            "curve_after": calibration_curve_data(y_holdout, p_holdout),
        },
        "decision": {
            "cost_model": cost_model.as_dict(),
            "threshold": threshold,
            "threshold_selected_on": "calib",
            "risk_band_edges": edges,
            "at_threshold": threshold_metrics(y_holdout, p_holdout, threshold),
            "at_naive_half": threshold_metrics(y_holdout, p_holdout, 0.5),
            "cost_comparison": comparison,
            "cost_curve": holdout_curve.to_dict(orient="list"),
        },
        "risk_bands": band_summary(y_holdout, p_holdout, edges).to_dict(orient="records"),
        "decile_lift": decile_lift(y_holdout, p_holdout).to_dict(orient="records"),
        "top_features": importance.head(25).to_dict(orient="records"),
        "baselines": baselines,
        "figures": plot_model_diagnostics(
            y_holdout, p_holdout, raw_scores["holdout"], holdout_curve, threshold, importance
        ) + [plot_baseline_comparison(baselines)],
    }

    # ---- persist ----
    joblib.dump(
        {
            "model": model,
            "feature_names": features.columns.tolist(),
            "categorical_features": categorical,
            "threshold": threshold,
            "risk_band_edges": edges,
            "cost_model": cost_model.as_dict(),
            "params": LGBM_PARAMS,
            "trained_at": metrics["trained_at"],
        },
        settings.model_path,
    )
    joblib.dump(preprocessor, settings.preprocessor_path)
    joblib.dump(calibrator, settings.calibrator_path)
    save_json(metrics, settings.metrics_path)
    save_json(importance.to_dict(orient="records"),
              settings.report_dir / "feature_importance.json")
    logger.info("Artifacts written to %s", settings.model_dir)

    # ---- holdout snapshot: the applicant pool the UI scores and explains ----
    holdout_frame = frame.iloc[splits["holdout"]].copy()
    holdout_frame["DEFAULT_PROBABILITY"] = p_holdout
    holdout_frame["RISK_BAND"] = assign_risk_band(p_holdout, edges)
    holdout_frame.to_parquet(settings.holdout_path, index=False)
    logger.info("Holdout snapshot -> %s", settings.holdout_path.name)

    # ---- scored predictions back into SQLite, so the analyst can query them ----
    split_labels = np.empty(len(frame), dtype=object)
    all_probabilities = np.empty(len(frame), dtype=float)
    for name, index in splits.items():
        split_labels[index] = name
        all_probabilities[index] = probabilities[name]

    predictions = pd.DataFrame({
        "SK_ID_CURR": identifiers,
        "DEFAULT_PROBABILITY": all_probabilities,
        "RISK_BAND": assign_risk_band(all_probabilities, edges),
        "DECISION": np.where(all_probabilities >= threshold, "Decline", "Approve"),
        "DATA_SPLIT": split_labels,
    })
    try:
        write_predictions(predictions)
    except FileNotFoundError:
        logger.warning("SQLite database absent - skipping predictions table. "
                       "Run `python -m src.data.build_database` first.")

    _log_summary(metrics)
    return metrics


def _log_summary(metrics: dict) -> None:
    holdout = metrics["discrimination"]["holdout"]
    decision = metrics["decision"]
    comparison = decision["cost_comparison"]
    logger.info("=" * 68)
    logger.info("Holdout ROC-AUC %.4f | PR-AUC %.4f | KS %.4f",
                holdout["roc_auc"], holdout["pr_auc"], holdout["ks_statistic"])
    logger.info("Brier %.5f -> %.5f after calibration (ECE %.4f -> %.4f)",
                metrics["calibration"]["before"]["brier_score"],
                metrics["calibration"]["after"]["brier_score"],
                metrics["calibration"]["before"]["expected_calibration_error"],
                metrics["calibration"]["after"]["expected_calibration_error"])
    logger.info("Cost-optimal threshold %.3f -> approval rate %s, recall %s",
                decision["threshold"],
                fmt_pct(decision["at_threshold"]["approval_rate"]),
                fmt_pct(decision["at_threshold"]["recall"]))
    logger.info("Expected loss %s vs %s at 0.50 -> saving %s (%s)",
                fmt_money(comparison["cost_optimal"]["total_cost"]),
                fmt_money(comparison["naive_0.5"]["total_cost"]),
                fmt_money(comparison["naive_0.5"]["saving_vs_this"]),
                fmt_pct(comparison["naive_0.5"]["saving_pct"]))
    logger.info("Saving vs approving everyone: %s",
                fmt_money(comparison["approve_everyone"]["saving_vs_this"]))
    for name, scores in metrics.get("baselines", {}).items():
        logger.info("Baseline %-20s ROC-AUC %.4f | PR-AUC %.4f",
                    name, scores["roc_auc"], scores["pr_auc"])
    logger.info("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the credit-risk model.")
    parser.add_argument("--sample", type=int, default=None,
                        help="train on N sampled rows (smoke test)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    metrics = run_training(sample=args.sample, seed=args.seed)
    print(json.dumps(metrics["discrimination"]["holdout"], indent=2))


if __name__ == "__main__":
    main()
