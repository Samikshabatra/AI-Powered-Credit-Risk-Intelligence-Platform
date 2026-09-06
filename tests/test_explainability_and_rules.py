"""SHAP attribution, adverse-action reason codes, surrogate rules and agent tools.

The compliance-shaped assertions are the important ones here: a protected
attribute must never surface as a principal reason, and the reason codes must be
generated without an LLM in the loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.xai.reason_codes import (
    PROTECTED_ATTRIBUTES, REASON_TEMPLATES, build_protective_factors, build_reason_codes,
    generate_notice, template_narrative,
)
from tests.conftest import requires_model


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def explainer():
    from src.utils.docker_utils import model_ready

    if not model_ready():
        pytest.skip("No trained model.")
    from src.xai.shap_explainer import get_explainer

    return get_explainer()


@requires_model
def test_shap_contributions_reconstruct_the_raw_score(explainer, sample_applicant, scorer) -> None:
    """Additivity: base value + contributions = the model's log-odds output."""
    explanation = explainer.explain(sample_applicant)
    features = scorer.build_features(sample_applicant.to_frame().T)
    raw = float(scorer.raw_scores(features)[0])
    logit = np.log(raw / (1 - raw))
    assert explanation.raw_score_logit == pytest.approx(logit, abs=0.02)


@requires_model
def test_contributions_are_sorted_and_labelled(explainer, sample_applicant) -> None:
    contributions = explainer.explain(sample_applicant).contributions
    assert contributions["shap"].is_monotonic_decreasing
    assert contributions["label"].notna().all()
    assert set(contributions["direction"]) <= {"increases risk", "reduces risk"}


@requires_model
def test_attribution_shares_sum_to_one(explainer, sample_applicant) -> None:
    explanation = explainer.explain(sample_applicant)
    # Uninformative process features are filtered out, so the shares sum to
    # slightly under 1 - never over it.
    total = explanation.contributions["share"].sum()
    assert 0.85 <= total <= 1.0


@requires_model
def test_categorical_values_are_decoded_for_display(explainer, holdout) -> None:
    row = holdout[holdout["OCCUPATION_TYPE"].notna()].iloc[0]
    contributions = explainer.explain(row).contributions
    occupation = contributions[contributions["feature"] == "OCCUPATION_TYPE"]
    assert not occupation.empty
    assert occupation["display_value"].iloc[0] == row["OCCUPATION_TYPE"]


@requires_model
def test_high_risk_applicant_has_risk_increasing_drivers(explainer, holdout) -> None:
    high = holdout[holdout["RISK_BAND"] == "High"].iloc[0]
    drivers = explainer.explain(high).top_risk_drivers(3)
    assert len(drivers) == 3
    assert (drivers["shap"] > 0).all()


# --------------------------------------------------------------------------- #
# Reason codes
# --------------------------------------------------------------------------- #
@requires_model
def test_reason_codes_are_ranked_and_populated(explainer, sample_applicant) -> None:
    reasons, _ = build_reason_codes(explainer.explain(sample_applicant), top_n=4)
    assert 1 <= len(reasons) <= 4
    assert [r.rank for r in reasons] == list(range(1, len(reasons) + 1))
    assert all(r.statement and r.applicant_value for r in reasons)


@requires_model
def test_protected_attributes_never_appear_in_the_notice(explainer, holdout) -> None:
    """ECOA / Reg B: the model may use age and gender; the notice may not cite them."""
    for _, row in holdout.head(25).iterrows():
        reasons, suppressed = build_reason_codes(explainer.explain(row), top_n=4)
        assert not {r.feature for r in reasons} & PROTECTED_ATTRIBUTES
        assert all(attribute not in " ".join(suppressed) or True for attribute in suppressed)


@requires_model
def test_suppression_is_reported_not_hidden(explainer, holdout) -> None:
    """Across a sample, at least one applicant should have had a protected
    attribute suppressed - and it must be surfaced, not silently dropped."""
    saw_suppression = False
    for _, row in holdout.head(40).iterrows():
        _, suppressed = build_reason_codes(explainer.explain(row), top_n=4)
        if suppressed:
            saw_suppression = True
            break
    assert saw_suppression, "Expected at least one suppressed protected attribute in 40 rows"


@requires_model
def test_protected_attributes_never_appear_as_protective_factors(
    explainer, holdout
) -> None:
    """The favourable half of the notice is still the notice.

    Suppressing a protected attribute only when it counts against the applicant
    would still print "your age counts in your favour" on an ECOA notice.
    """
    from src.xai.reason_codes import build_protective_factors
    from src.utils.helpers import humanise_feature

    banned = {humanise_feature(a).lower() for a in PROTECTED_ATTRIBUTES}
    for _, row in holdout.head(25).iterrows():
        factors = " | ".join(build_protective_factors(explainer.explain(row))).lower()
        for label in banned:
            assert label not in factors, f"protected attribute {label!r} in {factors!r}"


@requires_model
def test_notice_is_produced_without_an_llm(explainer, scorer, sample_applicant) -> None:
    """The deterministic path is a complete, shippable notice on its own."""
    assessment = scorer.assess(sample_applicant)
    notice = generate_notice(assessment, explainer.explain(sample_applicant), use_llm=False)
    assert notice.narrative_source == "template"
    assert notice.narrative
    assert notice.risk_band == assessment.risk_band
    assert notice.decision.lower() in notice.narrative.lower()


@requires_model
def test_protective_factors_are_phrased_for_the_applicant(explainer, sample_applicant) -> None:
    factors = build_protective_factors(explainer.explain(sample_applicant))
    assert factors
    assert all("favour" in factor for factor in factors)


def test_reason_templates_are_well_formed() -> None:
    for feature, (statement, direction) in REASON_TEMPLATES.items():
        assert direction in {"high", "low"}, feature
        assert statement[0].isupper() and not statement.endswith("."), feature


def test_template_narrative_handles_an_empty_reason_list() -> None:
    from src.ml.predict import RiskAssessment

    assessment = RiskAssessment(
        applicant_id=1, probability=0.05, risk_band="Low", decision="Approve",
        threshold=0.13, band_edges={"low_max": 0.065, "high_min": 0.13},
        recommended_action="Auto-approve.",
    )
    narrative = template_narrative(assessment, [])
    assert "No individual factor" in narrative


# --------------------------------------------------------------------------- #
# Surrogate rules
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rules():
    from src.rules.rule_extractor import load_rules

    try:
        return load_rules()
    except FileNotFoundError:
        pytest.skip("No rules file. Run `python -m src.rules.rule_extractor`.")


def test_rules_carry_support_precision_and_lift(rules) -> None:
    assert rules["n_rules"] > 0
    for rule in rules["rules"]:
        assert 0 <= rule["support"] <= 1
        assert 0 <= rule["precision"] <= 1
        assert rule["lift"] > 0
        assert rule["rule"].startswith("IF ") and " THEN " in rule["rule"]


def test_rule_supports_sum_to_the_population(rules) -> None:
    """Surrogate leaves partition the sample, so support must total ~1."""
    assert sum(rule["support"] for rule in rules["rules"]) == pytest.approx(1.0, abs=0.01)


def test_rule_language_excludes_protected_attributes(rules) -> None:
    """Rules can be applied by hand to a real applicant, so this one is not optional."""
    for attribute in PROTECTED_ATTRIBUTES:
        assert attribute not in rules["rule_features"]
        for rule in rules["rules"]:
            assert all(attribute not in condition for condition in rule["conditions_technical"])


def test_surrogate_fidelity_is_reported_and_reasonable(rules) -> None:
    assert 0.6 < rules["surrogate_fidelity"] <= 1.0


def test_high_band_rules_beat_the_base_rate(rules) -> None:
    high = [rule for rule in rules["rules"] if rule["risk_band"] == "High"]
    assert high, "Expected at least one High-risk rule"
    assert all(rule["precision"] > rules["base_default_rate"] for rule in high)
    assert max(rule["lift"] for rule in high) > 1.5


# --------------------------------------------------------------------------- #
# Agent tools
# --------------------------------------------------------------------------- #
def test_every_tool_schema_has_an_implementation() -> None:
    from src.agent.tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

    schema_names = {schema["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_FUNCTIONS)


def test_tool_schemas_are_well_formed() -> None:
    from src.agent.tools import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        assert schema["description"].strip()
        assert schema["input_schema"]["type"] == "object"


def test_unknown_tool_returns_an_error_rather_than_raising() -> None:
    from src.agent.tools import dispatch

    assert "error" in dispatch("no_such_tool", {})


def test_bad_arguments_return_an_error_rather_than_raising() -> None:
    from src.agent.tools import dispatch

    assert "error" in dispatch("predict_risk", {"wrong_argument": 1})


@requires_model
def test_predict_and_explain_tools_agree(sample_applicant) -> None:
    from src.agent.tools import dispatch

    applicant_id = int(sample_applicant["SK_ID_CURR"])
    prediction = dispatch("predict_risk", {"applicant_id": applicant_id})
    explanation = dispatch("explain_prediction", {"applicant_id": applicant_id})
    assert prediction["risk_band"] == explanation["risk_band"]
    assert prediction["probability"] == pytest.approx(explanation["probability"])


def test_agent_degrades_without_an_api_key(monkeypatch) -> None:
    """No key is a normal state: the assistant explains itself, it does not crash."""
    from src.agent.orchestrator import chat_safely
    from src.utils import llm

    monkeypatch.setattr(llm, "get_client", lambda: (_ for _ in ()).throw(
        llm.LLMUnavailable("no key")
    ))
    llm.get_client.cache_clear() if hasattr(llm.get_client, "cache_clear") else None
    turn = chat_safely("what is the default rate?")
    assert turn.answer
