"""Business rules derived from the ML model, via a shallow decision-tree surrogate.

    python -m src.rules.rule_extractor

A gradient-boosted ensemble of ~1200 trees is not credit policy: nobody can put
it in a committee pack. So a depth-limited decision tree is fitted to *the
model's own decisions* - not to the raw labels - and its leaves are read back as
IF-THEN rules. That is what makes them rules **of the model**: the surrogate
approximates the deployed policy, and its fidelity score says how well.

Every rule ships with three numbers, because a rule without them is an opinion:

    support    share of applicants the rule covers
    precision  observed default rate among them (the ground truth, not the model)
    lift       precision divided by the portfolio base rate

Two deliberate restrictions keep the output usable as policy:

* **Numeric features only.** A split on "OCCUPATION_TYPE <= 7.5" is an artefact
  of integer encoding, not a rule anyone can apply.
* **No protected attributes.** Gender, age and family status are excluded from
  the rule language even though the model may use them - a rule set is the one
  artefact here that would be applied by hand to a real applicant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

from src.ml.predict import RiskScorer, get_scorer, load_holdout
from src.utils.config import settings
from src.utils.helpers import fmt_pct, humanise_feature, save_json
from src.utils.logger import get_logger
from src.xai.reason_codes import PROTECTED_ATTRIBUTES

logger = get_logger(__name__)

MAX_DEPTH = 4
# Minimum leaf size, as a share of the population the surrogate is fitted on
# rather than a flat count. A rule covering a handful of applicants is not a
# policy, so the floor is real - but a flat 2000 silently produced a *single*
# leaf on a 3,600-row population (fidelity 18.7%, one vacuous rule at 100%
# support), because no split could leave 2000 rows on both sides. Scaling keeps
# the full 60,000-row run identical (60000/30 = 2000) and keeps a smaller
# population, such as the deployment sample, producing real rules.
MIN_SAMPLES_LEAF_FRACTION = 1 / 30
MIN_SAMPLES_LEAF_FLOOR = 50
N_CANDIDATE_FEATURES = 14

# Features whose *units* a policy reader understands without a data dictionary.
# Day-counts are excluded in favour of their engineered year equivalents.
_UNREADABLE = {
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE", "BUREAU_DAYS_CREDIT_MEAN", "BUREAU_DAYS_CREDIT_MIN",
    "BUREAU_DAYS_CREDIT_MAX", "BUREAU_DAYS_ENDDATE_MAX", "PREV_DAYS_DECISION_MAX",
    "PREV_DAYS_DECISION_MIN", "HOUR_APPR_PROCESS_START", "EXT_SOURCE_MEAN_X_AGE",
    "EMPLOYED_TO_AGE_RATIO", "REGION_POPULATION_RELATIVE",
}

# How a value should be rendered inside a rule condition.
_FORMATTERS: dict[str, str] = {
    "AMT_INCOME_TOTAL": "money", "AMT_CREDIT": "money", "AMT_ANNUITY": "money",
    "AMT_GOODS_PRICE": "money", "BUREAU_DEBT_SUM": "money", "BUREAU_CREDIT_SUM": "money",
    "PREV_AMT_CREDIT_MEAN": "money", "INCOME_PER_PERSON": "money",
    "CREDIT_PER_PERSON": "money", "BUREAU_OVERDUE_MAX": "money",
    "EXT_SOURCE_1": "score", "EXT_SOURCE_2": "score", "EXT_SOURCE_3": "score",
    "EXT_SOURCE_MEAN": "score", "EXT_SOURCE_MIN": "score", "EXT_SOURCE_MAX": "score",
    "EXT_SOURCE_STD": "score", "PREV_REFUSAL_RATE": "rate", "PREV_APPROVAL_RATE": "rate",
    "BUREAU_DEBT_CREDIT_RATIO": "rate", "BUREAU_ACTIVE_RATIO": "rate",
    "AGE_YEARS": "years", "EMPLOYMENT_YEARS": "years", "OWN_CAR_AGE": "years",
}


@dataclass
class BusinessRule:
    """One IF-THEN policy rule read off a surrogate leaf."""

    rule_id: int
    conditions: list[str]
    conditions_technical: list[str]
    action: str
    risk_band: str
    applicants: int
    support: float
    precision: float
    lift: float
    model_decline_rate: float

    @property
    def text(self) -> str:
        return f"IF {' AND '.join(self.conditions)} THEN {self.action}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule": self.text,
            "conditions": self.conditions,
            "conditions_technical": self.conditions_technical,
            "action": self.action,
            "risk_band": self.risk_band,
            "applicants": self.applicants,
            "support": round(self.support, 4),
            "precision": round(self.precision, 4),
            "lift": round(self.lift, 3),
            "model_decline_rate": round(self.model_decline_rate, 4),
        }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _format_threshold(feature: str, value: float) -> str:
    kind = _FORMATTERS.get(feature, "auto")
    if kind == "money":
        return f"{value:,.0f}"
    if kind in {"score", "rate"}:
        return f"{value:.2f}"
    if kind == "years":
        return f"{value:.1f} years"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _condition_text(feature: str, threshold: float, go_left: bool) -> str:
    label = humanise_feature(feature)
    operator = "<=" if go_left else ">"
    return f"{label} {operator} {_format_threshold(feature, threshold)}"


# --------------------------------------------------------------------------- #
# Feature selection
# --------------------------------------------------------------------------- #
def select_rule_features(scorer: RiskScorer, n: int = N_CANDIDATE_FEATURES) -> list[str]:
    """Top gain-importance features that are numeric, readable and permitted."""
    importance = pd.DataFrame({
        "feature": scorer.feature_names,
        "gain": scorer.model.booster_.feature_importance("gain"),
    }).sort_values("gain", ascending=False)

    excluded = set(scorer.categorical_features) | PROTECTED_ATTRIBUTES | _UNREADABLE
    eligible = [f for f in importance["feature"] if f not in excluded]
    selected = eligible[:n]
    logger.info("Rule language: %d features (%s...)", len(selected), ", ".join(selected[:4]))
    return selected


# --------------------------------------------------------------------------- #
# Surrogate
# --------------------------------------------------------------------------- #
def min_samples_leaf(population_size: int) -> int:
    """Leaf-size floor for a population of `population_size` rows."""
    return max(MIN_SAMPLES_LEAF_FLOOR,
               int(round(population_size * MIN_SAMPLES_LEAF_FRACTION)))


def fit_surrogate(
    features: pd.DataFrame, model_decisions: np.ndarray, seed: int
) -> tuple[DecisionTreeClassifier, float]:
    """Fit the interpretable stand-in and measure how faithfully it mimics the model."""
    # A surrogate cannot use NaN; median imputation is acceptable *here* because
    # the tree is an explanation of the policy, not the scoring path.
    filled = features.fillna(features.median(numeric_only=True))

    tree = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_leaf=min_samples_leaf(len(filled)),
        class_weight="balanced",
        random_state=seed,
    )
    tree.fit(filled, model_decisions)
    fidelity = float((tree.predict(filled) == model_decisions).mean())
    logger.info("Surrogate fidelity vs the deployed model: %.1f%%", 100 * fidelity)
    return tree, fidelity


def _leaf_paths(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, list[tuple]]:
    """Root-to-leaf condition paths, keyed by leaf node id."""
    structure = tree.tree_
    paths: dict[int, list[tuple]] = {}

    def walk(node: int, conditions: list[tuple]) -> None:
        if structure.feature[node] == _tree.TREE_UNDEFINED:
            paths[node] = conditions
            return
        feature = feature_names[structure.feature[node]]
        threshold = float(structure.threshold[node])
        walk(structure.children_left[node], conditions + [(feature, threshold, True)])
        walk(structure.children_right[node], conditions + [(feature, threshold, False)])

    walk(0, [])
    return paths


def _simplify(conditions: list[tuple]) -> list[tuple]:
    """Collapse repeated splits on one feature into the tightest surviving bound."""
    bounds: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for feature, threshold, go_left in conditions:
        if feature not in bounds:
            bounds[feature] = {}
            order.append(feature)
        key = "upper" if go_left else "lower"
        current = bounds[feature].get(key)
        if current is None:
            bounds[feature][key] = threshold
        else:
            bounds[feature][key] = min(current, threshold) if go_left \
                else max(current, threshold)

    simplified: list[tuple] = []
    for feature in order:
        if "lower" in bounds[feature]:
            simplified.append((feature, bounds[feature]["lower"], False))
        if "upper" in bounds[feature]:
            simplified.append((feature, bounds[feature]["upper"], True))
    return simplified


def _action_for(default_rate: float, base_rate: float, edges: dict[str, float]) -> tuple[str, str]:
    """Map a leaf's observed default rate to a band and a policy action."""
    if default_rate >= edges["high_min"]:
        return "High", "DECLINE - refer to senior underwriting for override"
    if default_rate >= edges["low_max"]:
        return "Medium", "REFER - manual review, verify income, consider risk pricing"
    if default_rate >= base_rate * 0.6:
        return "Low", "APPROVE - standard terms"
    return "Low", "APPROVE - straight-through, eligible for pre-approved limit increase"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_rules(
    sample_size: int = 60_000, seed: int | None = None
) -> dict[str, Any]:
    """Fit the surrogate on the holdout population and read its leaves as rules."""
    seed = seed if seed is not None else settings.random_seed
    scorer = get_scorer()

    population = load_holdout()
    if len(population) > sample_size:
        population = population.sample(n=sample_size, random_state=seed)

    rule_features = select_rule_features(scorer)
    features = scorer.build_features(population)[rule_features]

    probabilities = population["DEFAULT_PROBABILITY"].to_numpy()
    model_decisions = (probabilities >= scorer.threshold).astype(int)
    actual = population["TARGET"].to_numpy()
    base_rate = float(actual.mean())

    tree, fidelity = fit_surrogate(features, model_decisions, seed)

    filled = features.fillna(features.median(numeric_only=True))
    leaf_ids = tree.apply(filled)
    paths = _leaf_paths(tree, rule_features)

    rules: list[BusinessRule] = []
    for rule_id, (leaf, conditions) in enumerate(sorted(paths.items()), start=1):
        mask = leaf_ids == leaf
        count = int(mask.sum())
        if count == 0:
            continue

        leaf_default_rate = float(actual[mask].mean())
        simplified = _simplify(conditions)
        band, action = _action_for(leaf_default_rate, base_rate, scorer.band_edges)

        rules.append(BusinessRule(
            rule_id=rule_id,
            conditions=[_condition_text(f, t, left) for f, t, left in simplified],
            conditions_technical=[
                f"{f} {'<=' if left else '>'} {t:.4f}" for f, t, left in simplified
            ],
            action=action,
            risk_band=band,
            applicants=count,
            support=count / len(population),
            precision=leaf_default_rate,
            lift=leaf_default_rate / base_rate if base_rate else float("nan"),
            model_decline_rate=float(model_decisions[mask].mean()),
        ))

    rules.sort(key=lambda rule: rule.precision, reverse=True)
    for position, rule in enumerate(rules, start=1):
        rule.rule_id = position

    payload = {
        "generated_from": "depth-4 decision-tree surrogate of the LightGBM policy",
        "surrogate_fidelity": round(fidelity, 4),
        "population": int(len(population)),
        "population_source": "holdout",
        "base_default_rate": round(base_rate, 4),
        "decision_threshold": scorer.threshold,
        "risk_band_edges": scorer.band_edges,
        "rule_features": rule_features,
        "excluded_from_rule_language": {
            "protected_attributes": sorted(PROTECTED_ATTRIBUTES),
            "reason": "Rules may be applied manually to a real applicant, so the rule "
                      "language excludes attributes a lender must not decide on.",
        },
        "n_rules": len(rules),
        "rules": [rule.to_dict() for rule in rules],
    }
    save_json(payload, settings.rules_path)
    _log_rules(rules, base_rate)
    return payload


def _log_rules(rules: list[BusinessRule], base_rate: float) -> None:
    logger.info("Extracted %d rules (portfolio base rate %s)", len(rules), fmt_pct(base_rate))
    for rule in rules[:5]:
        logger.info(
            "  [%s] %s -> %s | support %s, default rate %s, lift %.2fx",
            rule.risk_band, " AND ".join(rule.conditions), rule.action.split(" - ")[0],
            fmt_pct(rule.support), fmt_pct(rule.precision), rule.lift,
        )


def load_rules() -> dict[str, Any]:
    """Read the persisted rule set. Backs the agent's `get_policy_rules` tool."""
    from src.utils.helpers import load_json

    payload = load_json(settings.rules_path)
    if payload is None:
        raise FileNotFoundError(
            f"No rules at {settings.rules_path}. Run `python -m src.rules.rule_extractor`."
        )
    return payload


def rules_dataframe() -> pd.DataFrame:
    """Rule set as a table for the UI."""
    payload = load_rules()
    frame = pd.DataFrame([
        {
            "Rule": rule["rule"],
            "Band": rule["risk_band"],
            "Applicants": rule["applicants"],
            "Support": rule["support"],
            "Default rate": rule["precision"],
            "Lift": rule["lift"],
        }
        for rule in payload["rules"]
    ])
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive business rules from the model.")
    parser.add_argument("--sample", type=int, default=60_000)
    args = parser.parse_args()
    extract_rules(sample_size=args.sample)


if __name__ == "__main__":
    main()
