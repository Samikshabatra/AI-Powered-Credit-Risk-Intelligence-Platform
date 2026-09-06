"""Tool definitions for the orchestrator agent.

Each tool is a thin adapter over a module that already exists and is already
tested. The agent adds routing, not capability - which is the point: if the
routing misfires, every one of these is still reachable from a button in the UI.

Two conventions hold throughout:

* **Tools never raise.** A failure is returned as `{"error": ...}` so the model
  can explain it to the user or try a different tool, instead of the chat dying.
* **Tools return compact JSON.** Whatever a tool returns is echoed back into the
  model's context, so returning a 200-row frame would be expensive and useless.
  Results are trimmed to what a person would actually be told.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from src.utils.helpers import load_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

MAX_ROWS_RETURNED = 25


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #
def query_data(nl_question: str) -> dict[str, Any]:
    """Answer a question about the portfolio by generating and running SQL."""
    from src.talk_to_data.nl_to_sql import get_agent

    result = get_agent().ask(nl_question)
    payload: dict[str, Any] = {
        "answer": result.answer,
        "sql": result.sql,
        "row_count": result.row_count,
        "attempts": result.attempts,
    }
    if result.success and not result.rows.empty:
        payload["rows"] = json.loads(
            result.rows.head(MAX_ROWS_RETURNED).to_json(orient="records")
        )
    if result.fallback:
        payload["note"] = "This question could not be answered from the warehouse."
    return payload


def predict_risk(applicant_id: int) -> dict[str, Any]:
    """Score one applicant: probability, risk band, decision, recommended action."""
    from src.ml.predict import score_applicant

    try:
        return score_applicant(int(applicant_id)).to_dict()
    except KeyError as exc:
        return {"error": str(exc)}
    except FileNotFoundError as exc:
        return {"error": f"Model artifacts missing: {exc}"}


def explain_prediction(applicant_id: int, top_n: int = 4) -> dict[str, Any]:
    """Explain one applicant's score: SHAP drivers plus adverse-action reasons."""
    from src.xai.reason_codes import explain_applicant_fully

    try:
        # The narrative is generated deterministically here: the orchestrator is
        # already an LLM and will phrase the answer itself, so paying for a
        # second model call to write prose that gets rewritten is wasted spend.
        payload = explain_applicant_fully(int(applicant_id), use_llm=False)
    except KeyError as exc:
        return {"error": str(exc)}
    except FileNotFoundError as exc:
        return {"error": f"Model artifacts missing: {exc}"}

    notice = payload["notice"]
    return {
        "applicant_id": notice["applicant_id"],
        "risk_band": notice["risk_band"],
        "probability": notice["probability"],
        "decision": notice["decision"],
        "principal_reasons": [
            {
                "rank": reason["rank"],
                "reason": reason["statement"],
                "applicant_value": reason["applicant_value"],
                "contribution_share": reason["contribution_share"],
            }
            for reason in notice["principal_reasons"][:top_n]
        ],
        "protective_factors": notice["protective_factors"],
        "suppressed_protected_attributes": notice["suppressed_protected_attributes"],
    }


def get_policy_rules(band: str | None = None, limit: int = 8) -> dict[str, Any]:
    """Return the IF-THEN policy rules derived from the model."""
    from src.rules.rule_extractor import load_rules

    try:
        payload = load_rules()
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    rules = payload["rules"]
    if band:
        rules = [rule for rule in rules if rule["risk_band"].lower() == band.lower()]
    return {
        "surrogate_fidelity": payload["surrogate_fidelity"],
        "base_default_rate": payload["base_default_rate"],
        "n_rules": len(rules),
        "rules": [
            {
                "rule": rule["rule"],
                "band": rule["risk_band"],
                "support": rule["support"],
                "default_rate": rule["precision"],
                "lift": rule["lift"],
            }
            for rule in rules[:limit]
        ],
    }


def list_applicants(band: str | None = None, limit: int = 10) -> dict[str, Any]:
    """List applicant ids from the holdout pool, optionally filtered by risk band."""
    from src.ml.predict import list_applicants as _list

    try:
        frame = _list(limit=min(limit, MAX_ROWS_RETURNED), band=band)
    except FileNotFoundError as exc:
        return {"error": str(exc)}
    return {
        "count": len(frame),
        "applicants": json.loads(
            frame[["SK_ID_CURR", "DEFAULT_PROBABILITY", "RISK_BAND"]].to_json(
                orient="records"
            )
        ),
    }


def get_model_performance() -> dict[str, Any]:
    """Headline model metrics: discrimination, calibration and the cost threshold."""
    from src.utils.config import settings

    metrics = load_json(settings.metrics_path)
    if metrics is None:
        return {"error": "No metrics found. Run `python -m src.ml.train` first."}

    decision = metrics["decision"]
    comparison = decision["cost_comparison"]
    return {
        "trained_at": metrics["trained_at"],
        "holdout": metrics["discrimination"]["holdout"],
        "calibration": {
            "brier_before": metrics["calibration"]["before"]["brier_score"],
            "brier_after": metrics["calibration"]["after"]["brier_score"],
            "expected_calibration_error_after":
                metrics["calibration"]["after"]["expected_calibration_error"],
        },
        "decision_threshold": decision["threshold"],
        "risk_band_edges": decision["risk_band_edges"],
        "approval_rate": decision["at_threshold"]["approval_rate"],
        "recall_at_threshold": decision["at_threshold"]["recall"],
        "expected_loss_saving_vs_naive_threshold":
            comparison["naive_0.5"]["saving_vs_this"],
        "expected_loss_saving_vs_approving_everyone":
            comparison["approve_everyone"]["saving_vs_this"],
        "risk_bands": metrics["risk_bands"],
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "query_data": query_data,
    "predict_risk": predict_risk,
    "explain_prediction": explain_prediction,
    "get_policy_rules": get_policy_rules,
    "list_applicants": list_applicants,
    "get_model_performance": get_model_performance,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "query_data",
        "description": (
            "Answer an aggregate or statistical question about the loan portfolio "
            "(default rates, averages, counts, breakdowns, rankings, comparisons "
            "across groups). Generates SQL, runs it read-only, and returns the "
            "answer with the SQL used. Use this for anything about the population "
            "as a whole rather than one named applicant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nl_question": {
                    "type": "string",
                    "description": "The question, in plain English, self-contained. "
                                   "Resolve pronouns and follow-up references first.",
                },
            },
            "required": ["nl_question"],
        },
    },
    {
        "name": "predict_risk",
        "description": (
            "Score one applicant by id: calibrated probability of default, risk "
            "band (Low/Medium/High), approve-or-decline at the cost-optimal "
            "threshold, and the recommended action. Use when the user names an "
            "applicant id and wants the decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "applicant_id": {
                    "type": "integer",
                    "description": "SK_ID_CURR of an applicant in the holdout set.",
                },
            },
            "required": ["applicant_id"],
        },
    },
    {
        "name": "explain_prediction",
        "description": (
            "Explain why one applicant received their score: the top SHAP risk "
            "drivers translated into adverse-action reason codes, plus the factors "
            "in the applicant's favour. Use for 'why', 'explain', 'what drove this'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "applicant_id": {"type": "integer", "description": "SK_ID_CURR."},
                "top_n": {
                    "type": "integer",
                    "description": "How many principal reasons to return (default 4).",
                },
            },
            "required": ["applicant_id"],
        },
    },
    {
        "name": "get_policy_rules",
        "description": (
            "Return the human-readable IF-THEN credit policy rules derived from the "
            "model by a decision-tree surrogate, each with support, observed default "
            "rate and lift. Use for questions about policy, criteria or 'what rules "
            "does the model imply'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "band": {
                    "type": "string",
                    "enum": ["Low", "Medium", "High"],
                    "description": "Optionally return only rules for one risk band.",
                },
                "limit": {"type": "integer", "description": "Maximum rules to return."},
            },
        },
    },
    {
        "name": "list_applicants",
        "description": (
            "List applicant ids available for scoring, optionally filtered by risk "
            "band. Use when the user wants an example applicant or does not know an id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "band": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "limit": {"type": "integer"},
            },
        },
    },
    {
        "name": "get_model_performance",
        "description": (
            "Return the model's own evaluation results: ROC-AUC, PR-AUC, KS, "
            "calibration before and after, the cost-optimal decision threshold, "
            "approval rate and the expected-loss saving. Use for questions about how "
            "good, accurate or well-calibrated the model is."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name. Never raises - errors come back as data."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        logger.info("tool %s(%s)", name, ", ".join(f"{k}={v!r}" for k, v in arguments.items()))
        return function(**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # a tool failure is a conversation event, not a crash
        logger.exception("Tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
