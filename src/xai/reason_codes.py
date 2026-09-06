"""SHAP contributions -> adverse-action reason codes.

Under ECOA / Regulation B, a US lender that declines an application must give the
applicant the *principal reasons* for the decision - specific and accurate, not
"you failed our scoring model". This module turns the model's own attribution
into exactly that notice.

Two layers, deliberately in this order:

1. **Deterministic templates.** Every reason code is generated from the SHAP
   contribution and the applicant's own value by rule. This layer is auditable,
   reproducible, and works with no API key.
2. **AI phrasing (optional).** The cheap model rewrites the deterministic bullets
   a short, respectful notice. It is given the reason codes as *facts to
   restate*, never as data to reason about, so it cannot invent a reason that
   the model did not produce. If it is unavailable, layer 1 is already a
   complete, shippable notice.

That ordering is the hallucination control: the LLM is a stylist here, not a
decision-maker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.predict import RiskAssessment
from src.utils.config import settings
from src.utils.llm import LLMUnavailable, get_client
from src.utils.logger import get_logger
from src.xai.shap_explainer import ShapExplanation

logger = get_logger(__name__)

# Feature -> (reason when the value is unfavourable, comparison direction).
# `direction` says which way is *worse*: "low" means a small value is riskier.
REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    "EXT_SOURCE_MEAN": ("External credit scores are below the approved-applicant average", "low"),
    "EXT_SOURCE_MIN": ("At least one credit bureau scores this applicant poorly", "low"),
    "EXT_SOURCE_MAX": ("No credit bureau scores this applicant strongly", "low"),
    "EXT_SOURCE_1": ("External credit score 1 is below our approval range", "low"),
    "EXT_SOURCE_2": ("External credit score 2 is below our approval range", "low"),
    "EXT_SOURCE_3": ("External credit score 3 is below our approval range", "low"),
    "EXT_SOURCE_STD": ("Credit bureaus disagree substantially about this applicant", "high"),
    "EXT_SOURCE_COUNT": ("Too few external credit references are available", "low"),
    "CREDIT_INCOME_RATIO": ("The requested loan is large relative to declared income", "high"),
    "ANNUITY_INCOME_RATIO": ("The repayment burden is high relative to income", "high"),
    "CREDIT_TERM": ("The implied repayment term is longer than typical", "high"),
    "GOODS_CREDIT_RATIO": ("The down payment is small relative to the amount financed", "low"),
    "AMT_CREDIT": ("The requested credit amount is high", "high"),
    "AMT_INCOME_TOTAL": ("Declared income is below the approved-applicant average", "low"),
    "AMT_ANNUITY": ("The annual repayment amount is high", "high"),
    "EMPLOYMENT_YEARS": ("Length of current employment is short", "low"),
    "DAYS_EMPLOYED": ("Length of current employment is short", "low"),
    "IS_UNEMPLOYED": ("No current employment is recorded", "high"),
    "EMPLOYED_TO_AGE_RATIO": ("Employment history is short relative to age", "low"),
    "AGE_YEARS": ("Applicant age falls in a higher-risk band", "low"),
    "DAYS_BIRTH": ("Applicant age falls in a higher-risk band", "low"),
    "OCCUPATION_TYPE": ("Recorded occupation is associated with higher default rates", "high"),
    "ORGANIZATION_TYPE": ("Employer industry is associated with higher default rates", "high"),
    "NAME_EDUCATION_TYPE": ("Recorded education level is associated with higher default rates", "high"),
    "NAME_INCOME_TYPE": ("Recorded income type is associated with higher default rates", "high"),
    "NAME_FAMILY_STATUS": ("Recorded family status is associated with higher default rates", "high"),
    "NAME_HOUSING_TYPE": ("Recorded housing situation is associated with higher default rates", "high"),
    "REGION_RATING_CLIENT": ("The applicant's region carries a weaker credit rating", "high"),
    "BUREAU_DEBT_SUM": ("Outstanding debt reported by credit bureaus is high", "high"),
    "BUREAU_DEBT_CREDIT_RATIO": ("A high share of available external credit is already drawn", "high"),
    "BUREAU_DEBT_INCOME_RATIO": ("External debt is high relative to income", "high"),
    "BUREAU_ACTIVE_COUNT": ("Several external credit accounts are currently active", "high"),
    "BUREAU_ACTIVE_RATIO": ("Most external credit accounts remain open", "high"),
    "BUREAU_OVERDUE_SUM": ("Overdue amounts are reported by credit bureaus", "high"),
    "BUREAU_OVERDUE_MAX": ("A past external loan went materially overdue", "high"),
    "BUREAU_DAYS_OVERDUE_MAX": ("A past external loan was overdue for an extended period", "high"),
    "BUREAU_LOAN_COUNT": ("A limited external credit history is on file", "low"),
    "BUREAU_PROLONG_SUM": ("Previous external loans required extension", "high"),
    "PREV_REFUSAL_RATE": ("A high share of previous applications were refused", "high"),
    "PREV_REFUSED_COUNT": ("Previous credit applications were refused", "high"),
    "PREV_APP_COUNT": ("Limited borrowing history with this lender", "low"),
    "PREV_CREDIT_TO_APPLICATION": ("Previously granted amounts fell short of amounts requested", "low"),
    "PREV_CNT_PAYMENT_MEAN": ("Previous loan terms were longer than typical", "high"),
    "PREV_DAYS_DECISION_MAX": ("The most recent credit decision is not recent", "low"),
    "DEF_30_CNT_SOCIAL_CIRCLE": ("Defaults are recorded among the applicant's social circle", "high"),
    "DEF_60_CNT_SOCIAL_CIRCLE": ("Defaults are recorded among the applicant's social circle", "high"),
    "AMT_REQ_CREDIT_BUREAU_YEAR": ("Frequent recent credit enquiries were recorded", "high"),
    "DAYS_LAST_PHONE_CHANGE": ("Contact details were changed recently", "high"),
    "DAYS_ID_PUBLISH": ("Identity documents were issued recently", "high"),
    "DAYS_REGISTRATION": ("Registration details were changed recently", "high"),
    "OWN_CAR_AGE": ("Recorded vehicle age is high", "high"),
    "DOCUMENT_SUBMITTED_COUNT": ("Few supporting documents were submitted", "low"),
    "CONTACT_FLAG_COUNT": ("Few contact details were provided", "low"),
    "REG_CITY_NOT_WORK_CITY": ("The registered address differs from the work address", "high"),
    "FLAG_OWN_REALTY": ("No property ownership is recorded", "high"),
    "FLAG_OWN_CAR": ("No vehicle ownership is recorded", "high"),
    "CNT_CHILDREN": ("Household dependants increase the repayment burden", "high"),
    "CNT_FAM_MEMBERS": ("Household size increases the repayment burden", "high"),
    "INCOME_PER_PERSON": ("Income per household member is low", "low"),
}

# Attributes a lender must not cite as a reason for declining credit.
# The model may still use them; the *notice* may not name them. Surfacing this
# explicitly is the point - see the Model Health & Fairness tab.
PROTECTED_ATTRIBUTES = {"CODE_GENDER", "NAME_FAMILY_STATUS", "AGE_YEARS", "DAYS_BIRTH"}


@dataclass
class ReasonCode:
    """One principal reason, traceable back to the feature that produced it."""

    rank: int
    feature: str
    label: str
    statement: str
    applicant_value: str
    contribution_share: float
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "feature": self.feature,
            "label": self.label,
            "statement": self.statement,
            "applicant_value": self.applicant_value,
            "contribution_share": round(self.contribution_share, 4),
            "protected_attribute": self.protected,
        }


@dataclass
class AdverseActionNotice:
    """The full explanation package rendered on the Explainability page."""

    applicant_id: int | None
    decision: str
    risk_band: str
    probability: float
    reasons: list[ReasonCode]
    protective_factors: list[str]
    narrative: str
    narrative_source: str  # 'llm' or 'template'
    suppressed_protected: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicant_id": self.applicant_id,
            "decision": self.decision,
            "risk_band": self.risk_band,
            "probability": round(self.probability, 4),
            "principal_reasons": [reason.to_dict() for reason in self.reasons],
            "protective_factors": self.protective_factors,
            "narrative": self.narrative,
            "narrative_source": self.narrative_source,
            "suppressed_protected_attributes": self.suppressed_protected,
        }


# --------------------------------------------------------------------------- #
# Layer 1: deterministic reason codes
# --------------------------------------------------------------------------- #
def _fallback_statement(label: str, direction: str) -> str:
    verb = "is higher than typical for approved applicants" if direction == "high" \
        else "is lower than typical for approved applicants"
    return f"{label} {verb}"


def build_reason_codes(
    explanation: ShapExplanation,
    top_n: int = 4,
    exclude_protected: bool = True,
) -> tuple[list[ReasonCode], list[str]]:
    """Turn the strongest risk-increasing contributions into reason codes.

    Protected attributes are dropped from the notice but reported separately, so
    a reviewer can see that the model leaned on one rather than have it quietly
    disappear.
    """
    reasons: list[ReasonCode] = []
    suppressed: list[str] = []
    rank = 1

    for _, row in explanation.contributions[explanation.contributions["shap"] > 0].iterrows():
        feature = row["feature"]
        if exclude_protected and feature in PROTECTED_ATTRIBUTES:
            suppressed.append(str(row["label"]))
            continue

        template = REASON_TEMPLATES.get(feature)
        statement = (
            template[0] if template
            else _fallback_statement(str(row["label"]), "high")
        )
        reasons.append(ReasonCode(
            rank=rank,
            feature=feature,
            label=str(row["label"]),
            statement=statement,
            applicant_value=str(row["display_value"]),
            contribution_share=float(row["share"]),
        ))
        rank += 1
        if len(reasons) >= top_n:
            break

    return reasons, suppressed


def build_protective_factors(
    explanation: ShapExplanation, top_n: int = 3, exclude_protected: bool = True
) -> list[str]:
    """The strongest points in the applicant's favour, phrased for the applicant.

    Protected attributes are excluded here for the same reason they are excluded
    from the principal reasons: this text goes into the notice the applicant
    receives, and a notice that says "your age counts in your favour" cites a
    protected attribute just as plainly as one that says it counts against them.
    Suppressing only the negative direction would be a half-measure that still
    puts age or gender in front of the applicant.
    """
    factors: list[str] = []
    # Ask for extra rows so the list still fills up after protected ones drop out.
    for _, row in explanation.top_protective(top_n + len(PROTECTED_ATTRIBUTES)).iterrows():
        if exclude_protected and row["feature"] in PROTECTED_ATTRIBUTES:
            continue
        factors.append(
            f"{row['label']} ({row['display_value']}) counts in the applicant's favour"
        )
        if len(factors) >= top_n:
            break
    return factors


def template_narrative(
    assessment: RiskAssessment, reasons: list[ReasonCode]
) -> str:
    """A complete, shippable notice with no LLM involved."""
    verdict = (
        f"This applicant is assessed as **{assessment.risk_band} risk** with a "
        f"{assessment.probability * 100:.1f}% probability of default, against a "
        f"decision threshold of {assessment.threshold * 100:.1f}%. "
        f"Recommended action: {assessment.decision.lower()}."
    )
    if not reasons:
        return verdict + " No individual factor stood out as a principal reason."
    bullets = "\n".join(
        f"{reason.rank}. {reason.statement} (applicant value: {reason.applicant_value})."
        for reason in reasons
    )
    return f"{verdict}\n\nPrincipal reasons:\n{bullets}"


# --------------------------------------------------------------------------- #
# Layer 2: AI phrasing
# --------------------------------------------------------------------------- #
_NARRATIVE_SYSTEM = """You write adverse-action explanations for a consumer lender.

You will be given a credit decision and its principal reasons, already computed
by the lender's model. Restate them for the applicant in plain language.

Rules, without exception:
- Use ONLY the reasons supplied. Never add, infer, merge or invent a reason.
- Never mention gender, marital status, age, race, religion or nationality, even
  if the applicant data would suggest them.
- Do not recompute, question or re-rank the risk assessment. Report it.
- Neutral and respectful. No apology, no lecture, no sales pitch.
- 3 to 5 sentences of prose. State the decision, then the reasons in the order
  given, then one concrete, actionable step the applicant could take.
- Plain text only. No headings, no bullet points, no markdown."""


def _narrative_prompt(assessment: RiskAssessment, reasons: list[ReasonCode],
                      protective: list[str]) -> str:
    lines = [
        f"Decision: {assessment.decision}",
        f"Risk band: {assessment.risk_band}",
        f"Probability of default: {assessment.probability * 100:.1f}% "
        f"(decline threshold {assessment.threshold * 100:.1f}%)",
        "",
        "Principal reasons, most important first:",
    ]
    lines += [
        f"{reason.rank}. {reason.statement} (applicant value: {reason.applicant_value})"
        for reason in reasons
    ] or ["(none identified)"]
    if protective:
        lines += ["", "Factors in the applicant's favour:"] + [f"- {p}" for p in protective]
    return "\n".join(lines)


def llm_narrative(
    assessment: RiskAssessment, reasons: list[ReasonCode], protective: list[str]
) -> str:
    """Ask the cheap model to phrase the notice. Raises LLMUnavailable if unconfigured."""
    client = get_client()
    response = client.complete(
        system=_NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": _narrative_prompt(assessment, reasons, protective)}],
        model=settings.summary_model,
        max_tokens=400,
        cache_system=True,     # the system prompt is identical on every applicant
        label="reason_codes",
    )
    return response.text


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_notice(
    assessment: RiskAssessment,
    explanation: ShapExplanation,
    top_n: int = 4,
    use_llm: bool = True,
) -> AdverseActionNotice:
    """Full adverse-action notice: reason codes, protective factors, narrative."""
    reasons, suppressed = build_reason_codes(explanation, top_n=top_n)
    protective = build_protective_factors(explanation)

    narrative = template_narrative(assessment, reasons)
    source = "template"
    if use_llm:
        try:
            narrative = llm_narrative(assessment, reasons, protective)
            source = "llm"
        except LLMUnavailable:
            logger.info("AI model unavailable - using the deterministic narrative.")
        except Exception as exc:  # network hiccup must not break a credit decision
            logger.warning("AI phrasing failed (%s) - falling back to template.", exc)

    return AdverseActionNotice(
        applicant_id=assessment.applicant_id,
        decision=assessment.decision,
        risk_band=assessment.risk_band,
        probability=assessment.probability,
        reasons=reasons,
        protective_factors=protective,
        narrative=narrative,
        narrative_source=source,
        suppressed_protected=suppressed,
    )


def explain_applicant_fully(applicant_id: int, use_llm: bool = True) -> dict[str, Any]:
    """Score + explain + phrase in one call. Backs the agent's explain tool."""
    from src.ml.predict import get_applicant, get_scorer
    from src.xai.shap_explainer import get_explainer

    row = get_applicant(applicant_id)
    assessment = get_scorer().assess(row)
    explanation = get_explainer().explain(row)
    notice = generate_notice(assessment, explanation, use_llm=use_llm)

    return {
        "assessment": assessment.to_dict(),
        "notice": notice.to_dict(),
        "top_contributions": explanation.top_risk_drivers(5)[
            ["feature", "label", "display_value", "shap", "share"]
        ].to_dict(orient="records"),
    }
