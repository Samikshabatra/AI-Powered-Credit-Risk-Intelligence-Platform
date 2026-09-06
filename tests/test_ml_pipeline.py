"""Evaluation maths, scoring contract and the end-to-end decision path."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.evaluate import (
    CostModel, assign_risk_band, calibration_curve_data, calibration_metrics,
    cost_comparison, decile_lift, discrimination_metrics, expected_cost,
    optimal_threshold, risk_band_edges, threshold_metrics,
)
from tests.conftest import requires_model


# --------------------------------------------------------------------------- #
# Evaluation maths - pure functions, no artifacts needed
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic():
    """A separable-but-noisy population with an 8% event rate, like the real one."""
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.binomial(1, 0.08, n)
    scores = np.clip(rng.beta(2, 20, n) + y * rng.uniform(0.05, 0.35, n), 0, 1)
    amounts = rng.uniform(50_000, 1_500_000, n)
    return y, scores, amounts


def test_discrimination_metrics_are_coherent(synthetic) -> None:
    y, scores, _ = synthetic
    metrics = discrimination_metrics(y, scores)
    assert 0.5 < metrics["roc_auc"] <= 1.0
    assert metrics["gini"] == pytest.approx(2 * metrics["roc_auc"] - 1)
    assert metrics["pr_auc"] > metrics["base_rate"]  # better than random
    assert 0 <= metrics["ks_statistic"] <= 1


def test_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 1, 1])
    assert discrimination_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0


def test_threshold_metrics_confusion_matrix_sums_to_the_population(synthetic) -> None:
    y, scores, _ = synthetic
    metrics = threshold_metrics(y, scores, 0.15)
    total = sum(metrics[k] for k in
                ("true_negatives", "false_positives", "false_negatives", "true_positives"))
    assert total == len(y)
    assert 0 <= metrics["recall"] <= 1


def test_lower_threshold_declines_more(synthetic) -> None:
    y, scores, _ = synthetic
    strict = threshold_metrics(y, scores, 0.05)
    lenient = threshold_metrics(y, scores, 0.40)
    assert strict["approval_rate"] < lenient["approval_rate"]
    assert strict["recall"] >= lenient["recall"]


def test_calibration_curve_uses_equal_count_bins(synthetic) -> None:
    """Equal-width bins are useless at an 8% base rate - everything lands in bin 1."""
    y, scores, _ = synthetic
    curve = calibration_curve_data(y, scores, n_bins=10)
    counts = curve["bin_counts"]
    assert len(counts) == 10
    assert max(counts) - min(counts) <= 1


def test_perfect_calibration_has_near_zero_error() -> None:
    rng = np.random.default_rng(1)
    probabilities = rng.uniform(0.01, 0.5, 20_000)
    outcomes = rng.binomial(1, probabilities)
    metrics = calibration_metrics(outcomes, probabilities)
    assert metrics["expected_calibration_error"] < 0.02


def test_miscalibrated_scores_are_detected() -> None:
    """An inflated score - what scale_pos_weight produces before calibration."""
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.01, 0.3, 10_000)
    outcomes = rng.binomial(1, truth)
    inflated = np.clip(truth * 4, 0, 1)
    assert (calibration_metrics(outcomes, inflated)["expected_calibration_error"]
            > calibration_metrics(outcomes, truth)["expected_calibration_error"])


# --------------------------------------------------------------------------- #
# Cost-sensitive decisioning
# --------------------------------------------------------------------------- #
def test_expected_cost_splits_into_its_two_error_sources() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    amounts = np.array([100.0, 200.0, 300.0, 400.0])
    model = CostModel(lgd_rate=0.5, margin_rate=0.1)

    cost = expected_cost(y, scores, amounts, 0.5, model)
    assert cost["default_loss"] == pytest.approx(300 * 0.5)      # approved and defaulted
    assert cost["opportunity_loss"] == pytest.approx(200 * 0.1)  # declined but good
    assert cost["total_cost"] == pytest.approx(150 + 20)


def test_cost_is_weighted_by_loan_size() -> None:
    """Declining a 2M loan is not the same mistake as declining a 50k one."""
    y = np.array([0, 0])
    scores = np.array([0.9, 0.9])
    model = CostModel()
    small = expected_cost(y, scores, np.array([50_000.0, 50_000.0]), 0.5, model)
    large = expected_cost(y, scores, np.array([2_000_000.0, 2_000_000.0]), 0.5, model)
    assert large["total_cost"] > small["total_cost"] * 10


def test_optimal_threshold_beats_the_naive_half(synthetic) -> None:
    y, scores, amounts = synthetic
    model = CostModel()
    threshold, curve = optimal_threshold(y, scores, amounts, model)
    assert 0.0 < threshold < 1.0

    best = expected_cost(y, scores, amounts, threshold, model)["total_cost"]
    naive = expected_cost(y, scores, amounts, 0.5, model)["total_cost"]
    assert best <= naive
    assert curve["total_cost"].min() == pytest.approx(best, rel=1e-6)


def test_asymmetric_costs_push_the_threshold_down() -> None:
    """When a missed default hurts far more than a lost customer, decline earlier."""
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.1, 3000)
    scores = np.clip(rng.beta(2, 15, 3000) + y * 0.2, 0, 1)
    amounts = np.full(3000, 500_000.0)

    harsh, _ = optimal_threshold(y, scores, amounts, CostModel(lgd_rate=0.9, margin_rate=0.02))
    mild, _ = optimal_threshold(y, scores, amounts, CostModel(lgd_rate=0.2, margin_rate=0.2))
    assert harsh < mild


def test_cost_comparison_reports_a_non_negative_saving(synthetic) -> None:
    y, scores, amounts = synthetic
    model = CostModel()
    threshold, _ = optimal_threshold(y, scores, amounts, model)
    comparison = cost_comparison(y, scores, amounts, threshold, model)
    assert comparison["naive_0.5"]["saving_vs_this"] >= 0
    assert comparison["approve_everyone"]["saving_vs_this"] >= 0
    assert comparison["cost_optimal"]["saving_vs_this"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Risk bands
# --------------------------------------------------------------------------- #
def test_bands_are_anchored_on_the_threshold() -> None:
    edges = risk_band_edges(0.13)
    assert edges["high_min"] == 0.13
    assert edges["low_max"] == pytest.approx(0.065)


def test_band_assignment_scalar_and_vector_agree() -> None:
    edges = risk_band_edges(0.2)
    probabilities = np.array([0.05, 0.15, 0.5])
    vectorised = list(assign_risk_band(probabilities, edges))
    scalars = [assign_risk_band(float(p), edges) for p in probabilities]
    assert vectorised == scalars == ["Low", "Medium", "High"]


def test_decile_lift_ranks_riskiest_first(synthetic) -> None:
    y, scores, _ = synthetic
    table = decile_lift(y, scores)
    assert len(table) == 10
    assert table["default_rate"].iloc[0] > table["default_rate"].iloc[-1]
    assert table["cumulative_defaults_captured"].iloc[-1] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Trained artifacts
# --------------------------------------------------------------------------- #
@requires_model
def test_scorer_produces_a_valid_probability(scorer, sample_applicant) -> None:
    assessment = scorer.assess(sample_applicant)
    assert 0.0 <= assessment.probability <= 1.0
    assert assessment.risk_band in {"Low", "Medium", "High"}
    assert assessment.decision in {"Approve", "Decline"}


@requires_model
def test_decision_agrees_with_the_threshold(scorer, sample_applicant) -> None:
    assessment = scorer.assess(sample_applicant)
    expected = "Decline" if assessment.probability >= scorer.threshold else "Approve"
    assert assessment.decision == expected


@requires_model
def test_scoring_is_deterministic(scorer, sample_applicant) -> None:
    first = scorer.assess(sample_applicant).probability
    second = scorer.assess(sample_applicant).probability
    assert first == second


@requires_model
def test_batch_scoring_matches_single_row_scoring(scorer, holdout) -> None:
    subset = holdout.head(20)
    batch = scorer.score_frame(subset)
    single = scorer.assess(subset.iloc[[3]]).probability
    assert batch["DEFAULT_PROBABILITY"].iloc[3] == pytest.approx(single, rel=1e-9)


@requires_model
def test_calibration_actually_moved_the_scores(scorer, holdout) -> None:
    """Reweighted raw scores are inflated; calibration is what makes them probabilities."""
    features = scorer.build_features(holdout.head(400))
    raw = scorer.raw_scores(features)
    calibrated = scorer.calibrate(raw)
    assert raw.mean() > calibrated.mean() * 2
    assert 0.05 < calibrated.mean() < 0.12


@requires_model
def test_calibration_never_reorders_applicants(scorer, holdout) -> None:
    """Isotonic is monotone non-decreasing: it may create ties, never inversions."""
    features = scorer.build_features(holdout.head(2000))
    raw = scorer.raw_scores(features)
    calibrated = scorer.calibrate(raw)

    ordered = calibrated[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= -1e-12), "Calibration inverted a pair of scores"


@requires_model
def test_calibration_costs_almost_no_discrimination(scorer, holdout) -> None:
    """Ties from the step function move AUC slightly - it must stay negligible."""
    from sklearn.metrics import roc_auc_score

    features = scorer.build_features(holdout)
    raw = scorer.raw_scores(features)
    y = holdout["TARGET"].to_numpy()
    delta = abs(roc_auc_score(y, raw) - roc_auc_score(y, scorer.calibrate(raw)))
    assert delta < 0.001, f"Calibration cost {delta:.4f} of AUC"


@requires_model
def test_what_if_recomputes_engineered_features(scorer, holdout) -> None:
    """Tripling income must move loan-to-income, not just the raw column."""
    row = holdout.iloc[0]
    baseline = scorer.assess(row)
    richer = scorer.what_if(row, {"AMT_INCOME_TOTAL": float(row["AMT_INCOME_TOTAL"]) * 3})
    assert richer.features["CREDIT_INCOME_RATIO"] < baseline.features["CREDIT_INCOME_RATIO"]


@requires_model
def test_holdout_metrics_meet_the_documented_bar() -> None:
    """Guards the README's headline numbers against a silent regression."""
    from src.utils.config import settings
    from src.utils.helpers import load_json

    metrics = load_json(settings.metrics_path)
    if metrics is None:
        pytest.skip("No metrics file.")
    holdout = metrics["discrimination"]["holdout"]
    assert holdout["roc_auc"] > 0.75
    assert holdout["pr_auc"] > 0.20
    assert metrics["calibration"]["after"]["expected_calibration_error"] < 0.02
    assert (metrics["calibration"]["after"]["brier_score"]
            < metrics["calibration"]["before"]["brier_score"])


@requires_model
def test_unknown_applicant_raises_a_helpful_error(scorer) -> None:
    from src.ml.predict import get_applicant

    with pytest.raises(KeyError, match="not in the holdout set"):
        get_applicant(-1)
