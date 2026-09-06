"""Streamlit UI for the Credit Risk Intelligence Platform.

    streamlit run app/ui.py

A product dashboard rather than a report: a top bar, a left nav rail, and pages
built from white cards on a light ground. `app/theme.py` holds the design system
and `app/charts.py` the Altair builders, so pages here stay about composition and
data rather than markup.

Eight sections, one idea: a credit decision should be inspectable end to end.
The hero is the **Decision Trace** on the Predict page - one applicant walked
from raw features through the model, the calibrator, the cost-optimal threshold,
the SHAP attribution and the adverse-action reasons to the policy rule covering
them. Every stage is a tested module; this page makes the chain visible.

Two rules run through the whole app:

* **Every LLM feature has a deterministic twin.** Scoring, explanation, rules,
  EDA and fairness all work with no API key. Only the chat assistant and the
  phrasing of the reason-code narrative need the AI agent, and both degrade to a
  message rather than an error.
* **Nothing is computed twice.** Model, explainer and dataframes are cached at
  resource level, so moving a slider re-scores one row instead of reloading a
  640 MB frame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import charts, theme  # noqa: E402
from app.theme import badge, icon  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.docker_utils import artifact_status, missing_data_message  # noqa: E402
from src.utils.helpers import fmt_money, fmt_pct, load_json  # noqa: E402

st.set_page_config(
    page_title="CreditRisk IQ",
    page_icon="::",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.stylesheet(), unsafe_allow_html=True)

HTML = dict(unsafe_allow_html=True)

NAV = [
    "Home", "Predict", "Explainability", "Talk to Data", "EDA & Insights",
    "Model Performance", "Business Rules", "Model Health", "About",
]

NAV_KEY = "nav_page"
PENDING_NAV = "_pending_nav"
PENDING_QUESTION = "_pending_question"

PAGE_SUBTITLES = {
    "Home": "What the platform does, and where to go next",
    "Predict": "Score an applicant and read the decision behind it",
    "Explainability": "Why an applicant was scored the way they were",
    "Talk to Data": "Ask the warehouse a question in plain English",
    "EDA & Insights": "What the data says before any model touches it",
    "Model Performance": "Discrimination, calibration and decision cost",
    "Business Rules": "Policy rules derived from the model",
    "Model Health": "Group performance and population stability",
    "About": "How the platform is built, and what it cannot do",
}

SUGGESTED_QUESTIONS = [
    "What is the default rate by income band?",
    "Which occupations have the highest default rate?",
    "Which features are most correlated with default?",
    "Show high-risk applicants with a low external score",
]


def goto(page: str) -> None:
    """Queue a section change, then rerun.

    Streamlit refuses to mutate a widget's own key after that widget exists, so
    the target is parked under a separate key and applied at the top of the next
    run - before the nav radio is created. This is what makes the buttons on the
    Home cards real navigation rather than decoration.
    """
    st.session_state[PENDING_NAV] = page
    st.rerun()


def ask_assistant(question: str) -> None:
    """Queue a question for the Talk-to-Data assistant and jump to it."""
    st.session_state[PENDING_QUESTION] = question
    goto("Talk to Data")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model artifacts...")
def _scorer():
    from src.ml.predict import get_scorer

    return get_scorer()


@st.cache_resource(show_spinner="Building the SHAP explainer...")
def _explainer():
    from src.xai.shap_explainer import get_explainer

    return get_explainer()


@st.cache_data(show_spinner="Loading the holdout population...")
def _holdout() -> pd.DataFrame:
    from src.ml.predict import load_holdout

    return load_holdout()


@st.cache_data
def _metrics() -> dict | None:
    return load_json(settings.metrics_path)


@st.cache_data
def _eda() -> dict | None:
    return load_json(settings.eda_summary_path)


@st.cache_data
def _rules() -> dict | None:
    return load_json(settings.rules_path)


@st.cache_data
def _fairness() -> dict | None:
    return load_json(settings.fairness_path)


@st.cache_data
def _eval_report() -> dict | None:
    return load_json(settings.eval_results_path)


@st.cache_data
def _importance() -> list | None:
    return load_json(settings.report_dir / "feature_importance.json")


@st.cache_data
def _tables() -> dict:
    from src.talk_to_data.query_runner import table_summary

    try:
        return table_summary()
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _template_applicant() -> pd.Series:
    """A median reference applicant, used as the base for manual scoring.

    A form cannot supply 143 features. The honest construction is to start from
    the portfolio's median applicant and override only the fields the user set -
    and to say so on the page, so nobody reads a manual score as though six
    inputs produced it.
    """
    holdout = _holdout()
    row = holdout.iloc[0].copy()
    for column, value in holdout.select_dtypes("number").median().items():
        row[column] = value
    for column in holdout.select_dtypes(exclude="number").columns:
        mode = holdout[column].mode()
        if not mode.empty:
            row[column] = mode.iloc[0]
    return row


def figure(name: str | None, caption: str | None = None) -> None:
    """Render a PNG written by the pipeline, or say why it is absent."""
    if not name:
        return
    path = settings.figures_dir / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure `{name}` has not been generated yet.")


def require(*keys: str) -> bool:
    """Guard a page on the artifacts it needs, naming the command that builds them."""
    status = artifact_status()
    commands = {
        "database": "python -m src.data.build_database",
        "model": "python -m src.ml.train",
        "holdout": "python -m src.ml.train",
        "rules": "python -m src.rules.rule_extractor",
        "eda": "python -m src.data.eda_insights",
    }
    if "eda" in keys and _eda() is None:
        st.warning("The exploratory analysis has not been run yet.")
        st.code(commands["eda"], language="bash")
        return False
    missing = [k for k in keys if k in status and not status[k]]
    if not missing:
        return True
    if not status.get("raw_data"):
        st.warning(missing_data_message())
        return False
    st.warning("This section needs artifacts that have not been built yet.")
    for key in missing:
        st.code(commands[key], language="bash")
    return False


# --------------------------------------------------------------------------- #
# Chrome
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    from src.utils.llm import llm_available


    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:9px;padding:0 4px 14px">'
            f'<div style="width:30px;height:30px;border-radius:9px;background:{theme.NAVY};'
            f'display:grid;place-items:center">{icon("shield", 17, "#fff")}</div>'
            f'<div><div style="font-weight:800;font-size:14.5px;color:{theme.INK};'
            f'letter-spacing:-.02em">CreditRisk IQ</div>'
            f'<div style="font-size:10.5px;color:{theme.MUTED}">Decision intelligence</div>'
            f"</div></div>",
            **HTML,
        )

        # Apply a queued jump before the widget exists, or Streamlit rejects it.
        if PENDING_NAV in st.session_state:
            st.session_state[NAV_KEY] = st.session_state.pop(PENDING_NAV)
        page = st.radio("Section", NAV, key=NAV_KEY, label_visibility="collapsed")

        st.markdown("<div style='height:12px'></div>", **HTML)
        status = artifact_status()
        rows = [
            ("Raw dataset", status.get("raw_data")),
            ("SQL warehouse", status.get("database")),
            ("Trained model", status.get("model")),
            ("Policy rules", status.get("rules")),
            ("AI API key", llm_available()),
        ]
        items = "".join(
            f'<div style="display:flex;align-items:center;gap:8px;padding:3px 0;'
            f'font-size:12px;color:{theme.MUTED}">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:'
            f'{theme.GREEN if ok else "#cbd5e1"}"></span>{label}</div>'
            for label, ok in rows
        )
        st.markdown(
            f'<div style="border-top:1px solid {theme.LINE};padding:12px 6px 0">'
            f'<div style="font-size:10.5px;font-weight:700;letter-spacing:.13em;'
            f'text-transform:uppercase;color:{theme.FAINT};margin-bottom:8px">'
            f"Build status</div>{items}</div>",
            **HTML,
        )
        if not llm_available():
            st.caption(
                "Without an API key, Talk to Data and the assistant are disabled. "
                "Every other section runs offline."
            )
    return page


# --------------------------------------------------------------------------- #
# Shared fragments
# --------------------------------------------------------------------------- #
def band_distribution_frame(metrics: dict) -> pd.DataFrame:
    frame = pd.DataFrame(metrics["risk_bands"])
    return frame.rename(columns={"population_share": "share"})[
        ["band", "share", "applicants"]
    ]


# Rule conditions carry full human labels, which are too long for a table that
# also has to show support, default rate and lift. These shorten the repeated
# phrases without changing what the condition means.
_RULE_ABBREVIATIONS = {
    "Average external credit score": "Avg ext score",
    "Weakest external credit score": "Min ext score",
    "Strongest external credit score": "Max ext score",
    "Number of external scores available": "Ext scores held",
    "Disagreement between credit bureaus": "Bureau disagreement",
    "Down-payment coverage": "Down-payment cover",
    "Implied repayment term": "Repayment term",
    "Historic refusal rate": "Prior refusal rate",
    "Debt-service-to-income ratio": "Debt service / income",
    "Loan-to-income ratio": "Loan / income",
    "External credit utilisation": "Ext utilisation",
    "Previously granted vs requested": "Granted / requested",
    "Average previous loan term": "Avg prior term",
    "Loan annuity (yearly payment)": "Annuity",
}


def _compact_rule(conditions: list[str]) -> str:
    text = " AND ".join(conditions)
    for long, short in _RULE_ABBREVIATIONS.items():
        text = text.replace(long, short)
    return text


def rules_table(rules: dict) -> pd.DataFrame:
    frame = pd.DataFrame(rules["rules"])
    return pd.DataFrame({
        "#": frame["rule_id"],
        "Rule": frame["conditions"].apply(_compact_rule),
        "Risk level": frame["risk_band"],
        "Support": frame["support"] * 100,
        "Precision": frame["precision"] * 100,
        "Lift": frame["lift"],
    })


def _rule_columns() -> dict:
    # Explicit widths: the rule text is long enough to push the numeric columns
    # off the visible area if the table is left to size itself.
    return {
        "#": st.column_config.NumberColumn(width="small"),
        "Rule": st.column_config.TextColumn(width="medium"),
        "Risk level": st.column_config.TextColumn(width="small"),
        "Support": st.column_config.NumberColumn(format="%.1f%%", width="small"),
        "Precision": st.column_config.NumberColumn("Default rate", format="%.1f%%",
                                                   width="small"),
        "Lift": st.column_config.NumberColumn(format="%.2fx", width="small"),
    }


def fairness_group_frame(fairness: dict, group: str) -> pd.DataFrame:
    rows = []
    for entry in fairness["groups"][group]["rows"]:
        rows.append({"group": entry["group"], "metric": "ROC-AUC",
                     "value": entry["roc_auc"]})
        rows.append({"group": entry["group"], "metric": "Default rate",
                     "value": entry["observed_default_rate"]})
    return pd.DataFrame(rows)


def matching_rule(row: pd.Series, rules: dict | None) -> dict | None:
    """The first surrogate rule whose conditions this applicant satisfies."""
    if not rules:
        return None
    import operator
    import re

    features = _scorer().build_features(row.to_frame().T).iloc[0]
    comparators = {"<=": operator.le, ">": operator.gt,
                   "<": operator.lt, ">=": operator.ge}
    for rule in rules.get("rules", []):
        satisfied = True
        for condition in rule["conditions_technical"]:
            match = re.match(r"^(\S+)\s+(<=|>=|<|>)\s+(-?[\d.]+)$", condition)
            if not match:
                satisfied = False
                break
            column, symbol, threshold = match.groups()
            value = features.get(column)
            if value is None or pd.isna(value) or not comparators[symbol](
                float(value), float(threshold)
            ):
                satisfied = False
                break
        if satisfied:
            return rule
    return None


def decision_trace_steps(assessment, reasons, matched_rule,
                         n_features: int) -> list[dict]:
    """The eight-step audit trail, built from live scoring output."""
    top = reasons[0].statement if reasons else "no dominant driver"
    rule_text = (
        matched_rule["rule"].split(" THEN ")[0][3:]
        if matched_rule else "No surrogate rule covers this applicant"
    )
    return [
        {"title": "Applicant data",
         "detail": (f"Raw application record, applicant {assessment.applicant_id}"
                    if assessment.applicant_id is not None
                    else "Manually entered applicant, no holdout record")},
        {"title": "Feature engineering",
         "detail": f"{n_features} features, 19 engineered, DAYS_EMPLOYED sentinel repaired"},
        {"title": "Model prediction",
         "detail": f"Raw booster score {assessment.raw_score:.3f}, in log-odds space"},
        {"title": "Calibration",
         "detail": f"Calibrated probability {assessment.probability:.3f} "
                   f"({assessment.probability * 100:.1f}%)"},
        {"title": f"Risk decision — {assessment.risk_band} risk",
         "detail": f"Threshold {assessment.threshold:.3f}, cost-optimal. "
                   f"Decision: {assessment.decision}.",
         "alert": assessment.risk_band == "High"},
        {"title": "SHAP explanation", "detail": f"Top driver: {top}"},
        {"title": "Business rules", "detail": rule_text},
        {"title": f"Recommended action — {assessment.decision}",
         "detail": assessment.recommended_action},
    ]


def render_prediction_result(assessment, compact: bool = False) -> None:
    """The gauge, verdict and decision rows shared by Predict and Home."""
    band = assessment.risk_band
    st.markdown(
        theme.svg_gauge(assessment.probability, assessment.threshold, band),
        **HTML,
    )
    st.markdown(
        f'<div style="text-align:center;margin:8px 0 14px">'
        f'<span class="badge" style="background:{theme.BAND_COLOUR[band]};color:#fff;'
        f'font-size:12.5px;padding:6px 18px">{band} Risk</span></div>',
        **HTML,
    )
    edges = assessment.band_edges
    rows = [
        ("Risk band", f"{band} (cuts at {edges['low_max']:.1%} / {edges['high_min']:.1%})"),
        ("Decision threshold", f"{assessment.threshold:.3f}"),
        ("Recommended action",
         badge(assessment.decision,
               band="Low" if assessment.decision == "Approve" else "High")),
    ]
    if not compact:
        expected_loss = (assessment.probability * settings.lgd_rate
                         * float(assessment.features.get("AMT_CREDIT") or 0))
        rows.append(("Expected loss if approved", fmt_money(expected_loss)))
        if assessment.actual_target is not None:
            rows.append(("Ground truth (holdout label)",
                         "Defaulted" if assessment.actual_target else "Repaid"))
    st.markdown(theme.kv_rows(rows), **HTML)


# --------------------------------------------------------------------------- #
# 1. Home
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _hero_preview() -> str:
    """A real applicant, scored live, for the hero card.

    The median applicant rather than an extreme one - deterministic, so the
    landing page does not change its headline example on every rerun, and
    cached so the hero costs one scoring call per session.
    """
    try:
        scorer = _scorer()
        row = _template_applicant().copy()
        row["TARGET"] = np.nan
        row["SK_ID_CURR"] = np.nan
        assessment = scorer.assess(row)
        expected_loss = (assessment.probability * settings.lgd_rate
                         * float(assessment.features.get("AMT_CREDIT") or 0))
        return theme.sample_prediction(
            assessment.probability, assessment.threshold, assessment.risk_band,
            assessment.decision, fmt_money(expected_loss))
    except Exception:
        # The hero must never be the reason the landing page fails to render.
        return ""


def page_home() -> None:
    """What the platform is, how a decision is made, and where to go next."""
    metrics, eda, fairness = _metrics(), _eda(), _fairness()
    status = artifact_status()

    if not any(status.values()):
        st.warning(missing_data_message())
        return

    holdout_metrics = (metrics or {}).get("discrimination", {}).get("holdout", {})
    profile = (eda or {}).get("profile", {})
    target = (eda or {}).get("target", {})
    preview = _hero_preview() if status.get("model") and status.get("holdout") else ""

    st.markdown(
        theme.warm_hero(
            "AI-Powered Credit Risk Intelligence Platform",
            "Credit decisions you can explain, defend and audit",
            "A calibrated LightGBM model on the Home Credit portfolio, with the "
            "things a lender actually needs around it: a threshold set by expected "
            "loss, adverse-action reasons, policy rules, fairness gaps, and an AI "
            "agent that answers questions in plain English.",
            tagline="Data today. A fairer tomorrow.",
            values=["People", "Data", "Fairness", "Opportunity"],
            promise="A fairer financial tomorrow",
            preview=preview),
        **HTML)
    st.markdown("<div style='height:12px'></div>", **HTML)

    st.markdown(theme.kpi_strip([
        ("applicants", f"{profile.get('n_rows', 0):,}", "Loan applications analysed"),
        ("rate", fmt_pct(target.get("default_rate", 0), 2), "Portfolio default rate"),
        ("auc", f"{holdout_metrics.get('roc_auc', 0):.3f}",
         "Holdout ROC-AUC (on 61,503 unseen applicants)"),
    ]), **HTML)
    st.markdown("<div style='height:12px'></div>", **HTML)

    with theme.card(
        "How one decision is produced",
        "The score is step three of seven. Everything after it is what makes the "
        "decision defensible.",
    ):
        st.markdown(theme.pipeline([
            ("applicants", "Application", "122 raw columns"),
            ("features", "Engineering", "sentinels repaired"),
            ("auc", "Risk model", "imbalance-weighted"),
            ("scale", "Calibration", "isotonic, held-back split"),
            ("performance", "Decision", "cost-optimal threshold"),
            ("shield", "Explanation", "SHAP to ECOA reasons"),
            ("rules", "Policy", "surrogate rule + action"),
        ]), **HTML)

    st.markdown("<div style='height:8px'></div>", **HTML)
    st.markdown('<div class="section-head">Explore the platform</div>', **HTML)

    destinations = [
        ("predict", "Risk Prediction", "Score an applicant and read the decision trace.",
         "Predict", theme.BLUE, status.get("model")),
        ("shield", "Explainability", "The adverse-action notice behind a score.",
         "Explainability", "#7c3aed", status.get("model")),
        ("chat", "Talk to Data", "Ask the warehouse a question in plain English.",
         "Talk to Data", "#0891b2", status.get("database")),
        ("eda", "EDA & Insights", "Five findings, each with the consequence it forces.",
         "EDA & Insights", theme.GREEN, eda is not None),
        ("performance", "Model Performance",
         "Discrimination, calibration and the cost of each decision.",
         "Model Performance", "#c2410c", metrics is not None),
        ("rules", "Business Rules", "Policy rules with support, default rate and lift.",
         "Business Rules", "#b45309", status.get("rules")),
        ("health", "Model Health", "Fairness gaps and population stability.",
         "Model Health", "#be123c", fairness is not None),
        ("about", "About", "Stack, architecture and how it fits together.",
         "About", theme.MUTED, True),
    ]

    for row_start in (0, 4):
        columns = st.columns(4, gap="medium")
        for column, (icon_name, title, blurb, target_page, accent, ready) in zip(
            columns, destinations[row_start:row_start + 4]
        ):
            with column:
                with theme.card():
                    st.markdown(
                        theme.section_tile(icon_name, title, blurb, accent), **HTML)
                    if ready:
                        if st.button("Open", use_container_width=True,
                                     key=f"go_{target_page}"):
                            goto(target_page)
                    else:
                        st.button("Not built yet", use_container_width=True,
                                  disabled=True, key=f"go_{target_page}")


def page_predict() -> None:
    st.caption("Enter applicant details or pick a sample applicant to get a risk "
               "score, an explanation and a recommendation.")

    if not require("model", "holdout"):
        return

    from src.xai.reason_codes import build_reason_codes

    scorer = _scorer()
    holdout = _holdout()
    assessment = None
    row = None

    left, right = st.columns([1.25, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            # A radio, not st.tabs: Streamlit renders every tab body on
            # every run, so all three branches executed and whichever
            # assigned last won - the result panel could show the sample
            # applicant while the form displayed manual values.
            mode = st.radio(
                "Input method",
                ["Manual Input", "Batch Upload", "Use Sample Applicant"],
                horizontal=True, label_visibility="collapsed",
                key="predict_mode")
            st.markdown(
                "<style>"
                '[data-testid="stElementContainer"]:has(input[value="Manual Input"])'
                " .stRadio [role=radiogroup]{flex-direction:row;gap:4px;"
                "border-bottom:1px solid #e2e8f0;margin-bottom:10px}"
                '[data-testid="stElementContainer"]:has(input[value="Manual Input"])'
                " [role=radiogroup] label{padding:7px 14px;border-bottom:2px solid "
                "transparent;margin:0}"
                '[data-testid="stElementContainer"]:has(input[value="Manual Input"])'
                " [role=radiogroup] label>div:first-child{display:none!important}"
                '[data-testid="stElementContainer"]:has(input[value="Manual Input"])'
                " [role=radiogroup] label:has(input:checked)"
                "{border-bottom-color:#2563eb}"
                '[data-testid="stElementContainer"]:has(input[value="Manual Input"])'
                " [role=radiogroup] label:has(input:checked) p"
                "{color:#2563eb;font-weight:600}"
                "</style>", **HTML)

            if mode == "Manual Input":
                st.markdown('<div style="font-size:13px;font-weight:700;color:#0f172a;'
                            'margin-bottom:8px">Applicant information</div>', **HTML)
                template = _template_applicant()
                form = st.form("manual_applicant", border=False)
                col_a, col_b = form.columns(2)
                age = col_a.number_input("Age", 18, 90,
                                         int(-template["DAYS_BIRTH"] / 365))
                income = col_b.number_input("Annual income", 25_000, 2_000_000,
                                            int(template["AMT_INCOME_TOTAL"]), step=5_000)
                income_type = col_a.selectbox(
                    "Income type",
                    scorer.preprocessor.category_levels("NAME_INCOME_TYPE") or ["Working"])
                tenure = col_b.number_input("Years at current job", 0, 50, 3)
                ext_score = col_a.number_input(
                    "External credit score 2", 0.0, 1.0,
                    round(float(template["EXT_SOURCE_2"]), 2), step=0.01)
                previous = col_b.number_input("Previous applications", 0, 50, 4)

                with form.expander("More features (optional)"):
                    more_a, more_b = st.columns(2)
                    credit = more_a.number_input("Loan amount", 45_000, 4_000_000,
                                                 int(template["AMT_CREDIT"]), step=15_000)
                    annuity = more_b.number_input("Annual repayment", 2_000, 300_000,
                                                  int(template["AMT_ANNUITY"]), step=1_000)
                    contract = more_a.selectbox(
                        "Contract type",
                        scorer.preprocessor.category_levels("NAME_CONTRACT_TYPE")
                        or ["Cash loans"])
                    education = more_b.selectbox(
                        "Education",
                        scorer.preprocessor.category_levels("NAME_EDUCATION_TYPE")
                        or ["Secondary / secondary special"])

                form.caption(
                    f"The model uses {len(scorer.feature_names)} features. Fields left "
                    "untouched take the portfolio median, so a manual score is a what-if "
                    "against a typical applicant — not a score built from six inputs."
                )
                submitted = form.form_submit_button(
                    "Predict Risk", type="primary", use_container_width=True)
                # Everything above sits in an st.form, so editing a field does
                # not rerun the app: the result panel cannot drift out of step
                # with the inputs, and the decision changes only when Submit is
                # pressed. That is what makes an explicit button safe here.
                row = template.copy()
                # Synthetic applicant: no real outcome, and no real identity.
                row["TARGET"] = np.nan
                row["SK_ID_CURR"] = np.nan
                row["DAYS_BIRTH"] = -age * 365
                row["AMT_INCOME_TOTAL"] = float(income)
                row["NAME_INCOME_TYPE"] = income_type
                row["DAYS_EMPLOYED"] = -tenure * 365
                row["EXT_SOURCE_2"] = float(ext_score)
                row["PREV_APP_COUNT"] = float(previous)
                row["AMT_CREDIT"] = float(credit)
                row["AMT_ANNUITY"] = float(annuity)
                row["NAME_CONTRACT_TYPE"] = contract
                row["NAME_EDUCATION_TYPE"] = education
                if submitted or "manual_row" not in st.session_state:
                    st.session_state["manual_row"] = row
                row = st.session_state["manual_row"]
                assessment = scorer.assess(row)

            if mode == "Use Sample Applicant":
                band = st.selectbox("Risk band", ["All", "High", "Medium", "Low"])
                pool = holdout if band == "All" else holdout[holdout["RISK_BAND"] == band]
                if pool.empty:
                    st.warning("No applicants in that band.")
                else:
                    sample = pool.sample(n=min(40, len(pool)), random_state=7)
                    sample = sample.sort_values("DEFAULT_PROBABILITY", ascending=False)
                    applicant_id = st.selectbox(
                        "Applicant", sample["SK_ID_CURR"].tolist(),
                        format_func=lambda i:
                            f"{i}  —  {sample.loc[i, 'DEFAULT_PROBABILITY']:.1%}")
                    row = holdout.loc[applicant_id]
                    assessment = scorer.assess(row)
                    st.caption(
                        f"Drawn from the {len(holdout):,}-row holdout set, never seen in "
                        "training — so the ground-truth label is a fair check.")

                    from src.utils.helpers import humanise_feature

                    fields = ["CODE_GENDER", "AMT_INCOME_TOTAL", "AMT_CREDIT",
                              "AMT_ANNUITY", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE",
                              "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
                    st.markdown(theme.kv_rows([
                        (humanise_feature(f),
                         "not provided" if pd.isna(row[f]) else (
                             f"{row[f]:,.3f}".rstrip("0").rstrip(".")
                             if isinstance(row[f], (int, float, np.floating))
                             else str(row[f])))
                        for f in fields if f in row.index
                    ]), **HTML)

            if mode == "Batch Upload":
                st.caption(
                    "Upload a CSV with the same columns as `application_train.csv`. "
                    "Columns the model expects but the file omits are filled with NaN, "
                    "which LightGBM handles natively.")
                upload = st.file_uploader("Applicant CSV", type=["csv"],
                                          label_visibility="collapsed")
                if upload is not None:
                    batch = pd.read_csv(upload)
                    scored = scorer.score_frame(batch)
                    st.success(f"Scored {len(scored):,} applicants.")
                    counts = scored["RISK_BAND"].value_counts()
                    st.markdown(theme.stat_row([
                        (f"{int(counts.get(b, 0)):,}", f"{b} risk")
                        for b in ("Low", "Medium", "High")
                    ]), **HTML)
                    st.markdown("<div style='height:8px'></div>", **HTML)
                    st.dataframe(scored.head(200), hide_index=True,
                                 use_container_width=True)
                    st.download_button(
                        "Download scored CSV",
                        scored.to_csv(index=False).encode(),
                        file_name="scored_applicants.csv", mime="text/csv")

    with right:
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title">Prediction result</div>', **HTML)
            if assessment is None:
                st.markdown(theme.notice(
                    "Choose a sample applicant, or fill the form and press "
                    "<b>Predict Risk</b>.", kind="info", icon_name="about"), **HTML)
            else:
                explanation = _explainer().explain(row)
                reasons, suppressed = build_reason_codes(explanation, top_n=3)
                render_prediction_result(assessment)

                st.markdown('<div style="font-size:12px;font-weight:700;color:#0f172a;'
                            'margin:14px 0 6px">Principal reasons</div>', **HTML)
                for reason in reasons:
                    st.markdown(
                        f'<div style="font-size:12.5px;line-height:1.5;margin-bottom:7px">'
                        f"<b>{reason.rank}.</b> {reason.statement}<br>"
                        f'<span style="color:{theme.MUTED};font-size:11.5px">value '
                        f"{reason.applicant_value} &middot; "
                        f"{reason.contribution_share:.0%} of attribution</span></div>",
                        **HTML)
                if suppressed:
                    st.caption("Suppressed as protected attributes: "
                               + ", ".join(suppressed[:3]))

    if assessment is not None:
        st.markdown("<div style='height:8px'></div>", **HTML)
        explanation = _explainer().explain(row)
        reasons, _ = build_reason_codes(explanation, top_n=3)
        trace_col, shap_col = st.columns([1, 1.35], gap="medium")

        with trace_col:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title">Decision Trace</div>'
                            '<div class="card-sub">The complete journey from raw input '
                            "to final decision.</div>", **HTML)
                st.markdown(theme.trace(decision_trace_steps(
                    assessment, reasons, matching_rule(row, _rules()),
                    len(scorer.feature_names))), **HTML)

        with shap_col:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title">What moved this score</div>'
                            '<div class="card-sub">SHAP contributions in log-odds space. '
                            "Red pushes towards default, green towards repayment.</div>",
                            **HTML)
                contributions = explanation.contributions
                top = pd.concat([contributions.head(8), contributions.tail(5)])
                st.altair_chart(charts.contribution_bars(top, height=300),
                                use_container_width=True)
                st.caption(
                    "Calibration is monotone, so the sign and ranking of every "
                    "contribution carry over to the calibrated probability. The "
                    "magnitudes are log-odds, which is why they are never shown as "
                    "percentage points of risk.")

        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown(
                '<div class="card-title">What-if</div>'
                '<div class="card-sub">Change the raw inputs and the decision is '
                "recomputed from scratch - every engineered ratio is rebuilt, not left "
                "stale.</div>", **HTML)
            slider_column, result_column = st.columns([1.5, 1], gap="medium")

            with slider_column:
                first, second = st.columns(2)
                income = first.slider(
                    "Annual income", 25_000, 1_000_000,
                    int(min(float(row["AMT_INCOME_TOTAL"]), 1_000_000)), step=5_000,
                    key="wi_income")
                credit = second.slider(
                    "Loan amount", 45_000, 3_000_000,
                    int(min(float(row["AMT_CREDIT"]), 3_000_000)), step=15_000,
                    key="wi_credit")
                annuity_default = (
                    float(row["AMT_ANNUITY"]) if pd.notna(row["AMT_ANNUITY"]) else 27_000)
                annuity = first.slider(
                    "Annual repayment", 2_000, 260_000,
                    int(min(annuity_default, 260_000)), step=1_000, key="wi_annuity")
                ext_default = (
                    float(row["EXT_SOURCE_2"]) if pd.notna(row["EXT_SOURCE_2"]) else 0.5)
                ext_source = second.slider(
                    "External credit score 2", 0.0, 1.0, ext_default, 0.01,
                    key="wi_ext")

            adjusted = scorer.what_if(row, {
                "AMT_INCOME_TOTAL": float(income),
                "AMT_CREDIT": float(credit),
                "AMT_ANNUITY": float(annuity),
                "EXT_SOURCE_2": float(ext_source),
            })

            with result_column:
                delta = (adjusted.probability - assessment.probability) * 100
                st.metric("Probability of default",
                          f"{adjusted.probability * 100:.1f}%",
                          delta=f"{delta:+.1f} pts", delta_color="inverse")
                st.markdown(
                    f"Original <b>{assessment.probability * 100:.1f}%</b> "
                    f"({assessment.decision}) &rarr; adjusted "
                    f"<b>{adjusted.probability * 100:.1f}%</b> "
                    f"({adjusted.decision})", **HTML)
                if adjusted.decision != assessment.decision:
                    st.success(
                        f"The decision flips from {assessment.decision} to "
                        f"{adjusted.decision}.")
                st.caption(
                    "Loan-to-income moves from "
                    f"{float(row['AMT_CREDIT']) / max(float(row['AMT_INCOME_TOTAL']), 1):.2f}"
                    f" to {credit / max(income, 1):.2f}.")



# --------------------------------------------------------------------------- #
# 3. Talk to Data
# --------------------------------------------------------------------------- #
def _sql_evidence(turn) -> dict:
    """The SQL and rows behind an answer, if the agent reached the warehouse.

    Showing the query is the whole point of a Talk-to-Data feature: an answer a
    reviewer cannot check is worth less than no answer. The agent does not carry
    the rows itself, so they are read back from the NL->SQL sub-agent.
    """
    if "query_data" not in turn.tool_names:
        return {}
    try:
        from src.talk_to_data.nl_to_sql import get_agent

        result = get_agent().last_result
    except Exception:
        return {}
    if result is None:
        return {}
    return {
        "sql": result.executed_sql,
        "rows": result.rows.head(25).to_dict(orient="records"),
        "row_count": result.row_count,
        "elapsed_ms": result.elapsed_ms,
    }


def _render_sql_evidence(payload: dict) -> None:
    """Render the SQL, the returned rows, and a chart when the shape suits one."""
    if not payload or not payload.get("sql"):
        return
    with st.expander(
        f"SQL that produced this answer &middot; {payload.get('row_count', 0)} rows in "
        f"{payload.get('elapsed_ms', 0):.0f} ms"
    ):
        st.code(payload["sql"], language="sql")
        rows = pd.DataFrame(payload.get("rows") or [])
        if rows.empty:
            return
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if len(rows.columns) >= 2:
            chart = charts.sql_result_bars(rows, rows.columns[0], rows.columns[1])
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)


def page_explainability() -> None:
    st.caption("SHAP attribution turned into the principal reasons a lender is "
               "required to give. Pick an applicant to see the notice they would "
               "receive.")

    if not require("model", "holdout"):
        return

    from src.xai.reason_codes import generate_notice

    holdout, scorer, explainer = _holdout(), _scorer(), _explainer()

    controls = st.columns([1, 1.6, 1.4])
    band = controls[0].selectbox("Risk band", ["High", "Medium", "Low", "All"],
                                 key="xai_band")
    pool = holdout if band == "All" else holdout[holdout["RISK_BAND"] == band]
    if pool.empty:
        st.warning("No applicants in that band.")
        return
    sample = pool.sample(n=min(30, len(pool)), random_state=3).sort_values(
        "DEFAULT_PROBABILITY", ascending=False)
    applicant_id = controls[1].selectbox(
        "Applicant", sample["SK_ID_CURR"].tolist(), key="xai_applicant",
        format_func=lambda i: f"{i}  \u2014  {sample.loc[i, 'DEFAULT_PROBABILITY']:.1%}")

    row = holdout.loc[applicant_id]
    assessment = scorer.assess(row)
    explanation = explainer.explain(row)

    st.markdown(
        f"Applicant `{applicant_id}` &mdash; "
        f"{badge(assessment.risk_band, assessment.risk_band)} &mdash; "
        f"<b>{assessment.probability * 100:.1f}%</b> probability of default",
        **HTML)
    st.markdown("<div style='height:8px'></div>", **HTML)

    with theme.card("Applicant profile",
                    "The record the reasons below are describing."):
        from src.utils.helpers import humanise_feature

        profile_fields = [
            "CODE_GENDER", "DAYS_BIRTH", "NAME_INCOME_TYPE", "OCCUPATION_TYPE",
            "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "AMT_INCOME_TOTAL",
            "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED", "EXT_SOURCE_1",
            "EXT_SOURCE_2", "EXT_SOURCE_3", "BUREAU_LOAN_COUNT", "PREV_APP_COUNT",
            "CNT_CHILDREN",
        ]

        def readable(field: str):
            value = row.get(field)
            if value is None or pd.isna(value):
                return "not provided"
            if field == "DAYS_BIRTH":
                return f"{-value / 365:.0f} years"
            if field == "DAYS_EMPLOYED":
                if value == 365243:
                    return "not employed"
                return f"{-value / 365:.1f} years"
            if isinstance(value, (int, float, np.floating)):
                return f"{value:,.3f}".rstrip("0").rstrip(".")
            return str(value)

        columns = st.columns(4)
        for index, field in enumerate(profile_fields):
            if field not in row.index:
                continue
            label = ("Age" if field == "DAYS_BIRTH"
                     else "Employment length" if field == "DAYS_EMPLOYED"
                     else humanise_feature(field))
            columns[index % 4].markdown(
                f'<div style="padding:6px 0;border-bottom:1px solid {theme.LINE}">'
                f'<div style="font-size:10.5px;color:{theme.MUTED}">{label}</div>'
                f'<div style="font-size:13px;font-weight:600;color:{theme.INK}">'
                f"{readable(field)}</div></div>", **HTML)

    st.markdown("<div style='height:8px'></div>", **HTML)

    notice_tab, shap_tab, global_tab = st.tabs(
        ["Adverse-action notice", "SHAP contributions", "Global drivers"])

    with notice_tab:
        left, right = st.columns([1.2, 1], gap="medium")
        with left:
            with theme.card("Notice the applicant would receive"):
                with st.spinner("Generating..."):
                    notice = generate_notice(assessment, explanation, use_llm=False)
                st.markdown(notice.narrative)
                st.caption(
                    "Generated deterministically from the SHAP attribution - no "
                    "language model is involved, so no reason here can be invented.")
        with right:
            with theme.card("Principal reasons"):
                st.dataframe(
                    pd.DataFrame([
                        {"#": r.rank, "Reason": r.statement,
                         "Applicant value": r.applicant_value,
                         "Share": r.contribution_share}
                        for r in notice.reasons]),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "#": st.column_config.NumberColumn(width=40),
                        "Reason": st.column_config.TextColumn(width="large"),
                        "Applicant value": st.column_config.TextColumn(
                            "Value", width=110),
                        "Share": st.column_config.NumberColumn(
                            "Share", format="percent", width=80)})
                if notice.protective_factors:
                    st.markdown("**Counting in the applicant's favour**")
                    for factor in notice.protective_factors:
                        st.markdown(f"- {factor}")
                if notice.suppressed_protected:
                    st.markdown(theme.notice(
                        "Suppressed from the notice as protected attributes under "
                        "ECOA / Regulation B: <b>"
                        + ", ".join(notice.suppressed_protected)
                        + "</b>. The model may use them; the notice may not cite them.",
                        kind="warn", icon_name="scale"), **HTML)

    with shap_tab:
        left, right = st.columns([1.3, 1], gap="medium")
        contributions = explanation.contributions
        with left:
            with theme.card("What moved this score"):
                top = pd.concat([contributions.head(9), contributions.tail(6)])
                st.altair_chart(charts.contribution_bars(top, height=340),
                                use_container_width=True)
                st.caption(
                    "SHAP explains the raw booster output in log-odds space. "
                    "Calibration is monotone, so the sign and ranking of every "
                    "contribution carry over to the calibrated probability; the "
                    "magnitudes do not, which is why they are shown as a share of "
                    "total attribution and never as percentage points of risk.")
        with right:
            with theme.card("Full attribution"):
                st.dataframe(
                    contributions[["label", "display_value", "shap", "share"]].rename(
                        columns={"label": "Feature", "display_value": "Value",
                                 "shap": "SHAP", "share": "Share"}),
                    hide_index=True, use_container_width=True, height=300,
                    column_config={
                        "SHAP": st.column_config.NumberColumn(format="%.3f"),
                        "Share": st.column_config.NumberColumn(format="percent")})
                if st.button("Render waterfall plot", use_container_width=True):
                    with st.spinner("Plotting..."):
                        name = explainer.plot_applicant_waterfall(row)
                    figure(name)

    with global_tab:
        with theme.card("Portfolio-level drivers",
                        "Mean |SHAP| over a holdout sample - what drives the book, "
                        "not one applicant."):
            if st.button("Compute global SHAP (samples 2,000 applicants)"):
                with st.spinner("Computing SHAP over the sample..."):
                    name = explainer.plot_global_summary(holdout)
                    importance = explainer.global_importance(holdout)
                figure(name)
                st.dataframe(
                    importance.head(18)[["label", "mean_abs_shap", "mean_shap"]].rename(
                        columns={"label": "Feature", "mean_abs_shap": "Mean |SHAP|",
                                 "mean_shap": "Mean signed SHAP"}),
                    hide_index=True, use_container_width=True)
            else:
                figure("shap_summary.png")
                figure("ml_feature_importance.png",
                       "LightGBM gain importance, written during training")


def page_talk() -> None:
    st.caption("The AI agent writes the SQL, the guardrails decide whether it runs, "
               "the answer comes back with the query attached.")

    if not require("database"):
        return

    from src.talk_to_data.semantic_layer import render_markdown
    from src.utils.llm import LEDGER, llm_available

    tab_ask, tab_semantic, tab_guard, tab_eval = st.tabs(
        ["Assistant", "Semantic layer", "Guardrails", "Evaluation"])

    with tab_ask:
        if not llm_available():
            st.markdown(theme.notice(
                "The AI assistant needs an API key. Copy "
                "<code>.env.example</code> to <code>.env</code>, set "
                "<code>ANTHROPIC_API_KEY</code>, and restart.<br>Every capability it "
                "fronts is still reachable from the other sections — scoring, "
                "explanation, rules and EDA all run offline.",
                kind="warn", icon_name="about"), **HTML)
        else:
            if "chat" not in st.session_state:
                st.session_state.chat = []

            # Suggested questions are buttons, not decorative chips: clicking one
            # asks it. This is the fastest way to see the agent work.
            st.caption("Try one of these, or type your own below:")
            suggestion_columns = st.columns(2)
            for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
                if suggestion_columns[index % 2].button(
                    suggestion, use_container_width=True, key=f"chip{index}"
                ):
                    st.session_state[PENDING_QUESTION] = suggestion
                    st.rerun()

            for message in st.session_state.chat:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message.get("tools"):
                        st.caption("Routed to: " + ", ".join(message["tools"]))
                    _render_sql_evidence(message)

            typed = st.chat_input(
                "Ask another question... e.g. which occupations default most?")
            # A question queued from a suggestion button or from the Home card.
            question = typed or st.session_state.pop(PENDING_QUESTION, None)
            if question:
                st.session_state.chat.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Routing..."):
                        from src.agent.orchestrator import chat_safely

                        turn = chat_safely(question)
                    st.markdown(turn.answer)
                    if turn.tool_names:
                        st.caption("Routed to: " + ", ".join(turn.tool_names))
                    evidence = _sql_evidence(turn)
                    _render_sql_evidence(evidence)
                st.session_state.chat.append({
                    "role": "assistant", "content": turn.answer,
                    "tools": turn.tool_names, **evidence})

            controls = st.columns([1, 3])
            if st.session_state.chat and controls[0].button("Clear conversation"):
                from src.agent.orchestrator import get_orchestrator

                st.session_state.chat = []
                get_orchestrator(reset=True)
                st.rerun()

            ledger = LEDGER.summary()
            if ledger.get("calls"):
                controls[1].caption(
                    f"{ledger['calls']} model calls this session · "
                    f"{ledger['input_tokens']:,} input tokens · "
                    f"{ledger['cache_read_tokens']:,} served from cache "
                    f"({ledger['input_token_saving_pct']:.0%} input saving)")

    with tab_semantic:
        st.markdown(
            "Ask a model for *the default rate* three times and you can get three "
            "different formulas — one of which returns `0` because SQLite truncates "
            "integer division. Ask for *employment length* and you get a 1000-year "
            "career for 18% of applicants.\n\n"
            "So the formulas are defined once, here, and injected into every prompt.")
        st.markdown(render_markdown())

    with tab_guard:
        st.markdown(
            "Generated SQL is treated as hostile — not because the model is "
            "adversarial, but because a prompt-injected question has to fail on "
            "**mechanism**, not on the model's good manners.")
        st.markdown("""
| # | Guardrail | What it stops |
|---|---|---|
| 1 | Statement allowlist (`sqlparse`) | Anything that is not a single SELECT or WITH |
| 2 | Whole-token keyword denylist | INSERT / UPDATE / DELETE / DROP / ALTER / ATTACH / PRAGMA |
| 3 | Table allowlist | Hallucinated tables and cross-database reads |
| 4 | Read-only connection (`mode=ro`) | Any write that got past 1–3 |
""")
        st.caption(
            "Plus an auto-injected LIMIT on non-aggregate queries, a wall-clock "
            "interrupt through SQLite's progress handler, schema grounding read back "
            "from `sqlite_master`, and a hard refusal when a question cannot be "
            "answered from these tables.")

        with st.expander("Try the validator yourself", expanded=True):
            from src.talk_to_data.query_runner import SQLValidationError, run_sql

            candidate = st.text_area(
                "SQL",
                "SELECT NAME_CONTRACT_TYPE, AVG(TARGET) AS default_rate\n"
                "FROM applications GROUP BY NAME_CONTRACT_TYPE",
                height=110, label_visibility="collapsed")
            if st.button("Validate and run"):
                try:
                    result = run_sql(candidate)
                    st.success(f"Allowed — {result.row_count} rows in "
                               f"{result.elapsed_ms:.0f} ms")
                    st.dataframe(result.rows, hide_index=True, use_container_width=True)
                    if len(result.rows.columns) >= 2:
                        chart = charts.sql_result_bars(
                            result.rows, result.rows.columns[0], result.rows.columns[1])
                        if chart is not None:
                            st.altair_chart(chart, use_container_width=True)
                except SQLValidationError as exc:
                    st.error(f"Blocked: {exc}")
                except Exception as exc:
                    st.error(f"{type(exc).__name__}: {exc}")

    with tab_eval:
        report = _eval_report()
        if report is None:
            st.markdown(theme.notice(
                "The NL-to-SQL evaluation has not been run in this environment "
                "(it needs an API key).", kind="info", icon_name="about"), **HTML)
            st.code("python -m src.talk_to_data.eval_harness", language="bash")
            st.caption(
                "25 labelled cases covering every query pattern, plus the sentinel trap, "
                "a multi-turn follow-up, three unanswerable questions and a "
                "prompt-injection attempt. Expected answers are recomputed from "
                "ground-truth SQL on every run, so a rebuilt database cannot silently "
                "invalidate the labels.")
        else:
            headline = report["headline"]

            def shown(value) -> str:
                return "n/a" if value is None else fmt_pct(value)

            st.markdown(theme.stat_row([
                (shown(headline["answer_accuracy"]),
                 f"Answer accuracy ({report['answerable_cases']} cases)"),
                (shown(headline["execution_accuracy"]), "Execution accuracy"),
                (shown(headline["fallback_accuracy"]),
                 f"Correct refusals ({report['unanswerable_cases']} cases)"),
                (shown(report["reliability"]["retry_rate"]), "Retry rate"),
            ]), **HTML)
            if report["cases"] < 25:
                st.caption(
                    f"This run covered {report['cases']} of the 25 labelled cases. "
                    "Run `python -m src.talk_to_data.eval_harness` without `--ids` "
                    "for the full suite, including the refusal and injection cases."
                )
            st.markdown("<div style='height:10px'></div>", **HTML)
            cases = pd.DataFrame(report["cases_detail"])[
                ["id", "pattern", "question", "correct", "attempts", "latency_ms"]]
            st.dataframe(
                cases, hide_index=True, use_container_width=True,
                # Fit the rows rather than padding blank ones on a partial run.
                height=min(420, 38 * len(cases) + 40),
                column_config={
                    "id": st.column_config.TextColumn("ID", width="small"),
                    "pattern": st.column_config.TextColumn("Pattern", width="small"),
                    "question": st.column_config.TextColumn("Question", width="large"),
                    "correct": st.column_config.CheckboxColumn("Pass", width="small"),
                    "attempts": st.column_config.NumberColumn("Tries", width="small"),
                    "latency_ms": st.column_config.NumberColumn(
                        "Latency (ms)", format="%.0f", width="small")})


# --------------------------------------------------------------------------- #
# 4. EDA & Insights
# --------------------------------------------------------------------------- #
def page_eda() -> None:
    st.caption("The analysis is imported from the same module the notebook uses, "
               "so the two cannot drift apart.")

    if not require("eda"):
        return
    eda = _eda()
    profile, target = eda["profile"], eda["target"]

    st.markdown(theme.stat_row([
        (f"{profile['n_rows']:,}", "Total records"),
        ("122", "Original features"),
        (str(profile["n_columns"] - 122), "Joined aggregates"),
        (fmt_pct(target["default_rate"], 2), "Default rate"),
        (f"{1 - profile['mean_missing_rate']:.1%}", "Data completeness"),
    ]), **HTML)
    st.markdown("<div style='height:10px'></div>", **HTML)

    tab_overview, tab_features, tab_quality, tab_key = st.tabs(
        ["Overview", "Feature analysis", "Data quality", "Key insights"])

    with tab_overview:
        left, right = st.columns([1.35, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Default rate by income band</div>", **HTML)
                frame = pd.DataFrame(eda["insights"]["income_bracket"]["table"])
                st.altair_chart(
                    charts.rate_bars(frame, "income_bracket", height=210,
                                     highlight_above=target["default_rate"]),
                    use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Key insights</div>", **HTML)
                st.markdown(theme.insights([
                    block["headline"] for block in eda["insights"].values()]), **HTML)

    with tab_features:
        for name, block in eda["insights"].items():
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown(
                    f'<div class="card-title" style="font-size:15px">'
                    f'{name.replace("_", " ").title()}</div>'
                    f'<div class="card-sub">{block["headline"]}</div>', **HTML)
                chart_col, table_col = st.columns([1.3, 1], gap="medium")
                frame = pd.DataFrame(block["table"])
                with chart_col:
                    if "default_rate" in frame.columns:
                        st.altair_chart(
                            charts.rate_bars(frame, frame.columns[0], height=200,
                                             highlight_above=target["default_rate"]),
                            use_container_width=True)
                    else:
                        figure(block["figure"])
                with table_col:
                    st.dataframe(frame, hide_index=True, use_container_width=True)

    with tab_quality:
        st.markdown('<div style="font-size:14px;font-weight:700;margin-bottom:6px">'
                    "Anomalies that are not missing values</div>"
                    '<div class="card-sub">A null is easy — every profiler finds it. '
                    "These are values that are present, valid-looking and wrong.</div>",
                    **HTML)
        for anomaly in eda["data_quality"]["anomalies"]:
            st.markdown(theme.notice(anomaly, kind="warn", icon_name="about"), **HTML)

        st.markdown("<div style='height:12px'></div>", **HTML)
        left, right = st.columns([1, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Completeness by feature group</div>", **HTML)
                st.dataframe(
                    pd.DataFrame(eda["data_quality"]["scorecard"]),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "mean_completeness": st.column_config.ProgressColumn(
                            "Mean completeness", format="%.1f%%",
                            min_value=0, max_value=1)})
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Least-populated columns</div>", **HTML)
                figure(eda["data_quality"]["figure"])

    with tab_key:
        st.markdown(theme.insights([
            block["headline"] for block in eda["insights"].values()]), **HTML)
        st.markdown("<div style='height:8px'></div>", **HTML)
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">'
                        "Feature groups</div>", **HTML)
            st.dataframe(
                pd.DataFrame([
                    {"Group": name, "Columns": info["count"],
                     "Examples": ", ".join(info["examples"][:4])}
                    for name, info in eda["feature_groups"].items()]),
                hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# 5. Model Performance
# --------------------------------------------------------------------------- #
def page_performance() -> None:
    st.caption("Evaluated on a 61,503-row holdout the model never saw during "
               "training.")

    if not require("model"):
        return
    metrics = _metrics()
    holdout_metrics = metrics["discrimination"]["holdout"]
    decision = metrics["decision"]

    st.markdown(theme.stat_row([
        (f"{holdout_metrics['roc_auc']:.3f}", "ROC-AUC"),
        (f"{holdout_metrics['pr_auc']:.3f}", "PR-AUC"),
        (f"{metrics['calibration']['after']['brier_score']:.3f}", "Brier score"),
        (f"{metrics['scale_pos_weight']:.2f}", "Class weight"),
        (f"{decision['threshold']:.3f}", "Decision threshold"),
    ]), **HTML)
    st.markdown("<div style='height:10px'></div>", **HTML)

    tab_overview, tab_curves, tab_calib, tab_importance, tab_threshold = st.tabs(
        ["Overview", "ROC & PR", "Calibration", "Feature importance",
         "Threshold analysis"])

    with tab_overview:
        left, right = st.columns([1.15, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Default rate by risk decile</div>", **HTML)
                st.altair_chart(
                    charts.decile_bars(pd.DataFrame(metrics["decile_lift"]), height=215),
                    use_container_width=True)
                st.caption(
                    "Decile 1 is the riskiest tenth of applicants. A model that could "
                    "not rank would show a flat bar at the 8.1% base rate.")
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Discrimination by split</div>", **HTML)
                st.dataframe(
                    pd.DataFrame(metrics["discrimination"]).T.reset_index()
                    .rename(columns={"index": "split"}),
                    hide_index=True, use_container_width=True)
                st.markdown(theme.notice(
                    f"<b>PR-AUC is the honest headline.</b> At a "
                    f"{holdout_metrics['base_rate']:.1%} base rate a random model scores "
                    f"{holdout_metrics['base_rate']:.2f} on PR-AUC but 0.50 on ROC-AUC, "
                    "so ROC-AUC flatters any model that is useless at the top of the "
                    "ranking.", kind="info", icon_name="about"), **HTML)

        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">'
                        "Risk bands on the holdout</div>", **HTML)
            st.dataframe(
                pd.DataFrame(metrics["risk_bands"]), hide_index=True,
                use_container_width=True,
                column_config={
                    "observed_default_rate": st.column_config.NumberColumn(
                        "Observed default rate", format="percent"),
                    "mean_predicted": st.column_config.NumberColumn(
                        "Mean predicted", format="percent"),
                    "population_share": st.column_config.NumberColumn(
                        "Population share", format="percent")})

    with tab_curves:
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">ROC curve'
                            "</div>", **HTML)
                figure("ml_roc_curve.png")
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Precision-recall curve</div>", **HTML)
                figure("ml_pr_curve.png")
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">'
                        "Score separation</div>", **HTML)
            figure("ml_score_distribution.png")

    with tab_calib:
        before = metrics["calibration"]["before"]
        after = metrics["calibration"]["after"]
        left, right = st.columns([1, 1.15], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown(theme.stat_row([
                    (f"{after['brier_score']:.4f}",
                     f"Brier (was {before['brier_score']:.4f})"),
                    (f"{after['expected_calibration_error']:.4f}",
                     f"ECE (was {before['expected_calibration_error']:.4f})"),
                ]), **HTML)
                st.markdown(
                    "<div style='height:10px'></div>"
                    "<p style='font-size:12.5px;line-height:1.6'>"
                    f"<code>scale_pos_weight = {metrics['scale_pos_weight']:.2f}</code> "
                    "fixes the imbalance and wrecks the probabilities as a side effect — "
                    f"the raw model predicts {before['mean_predicted']:.0%} average risk "
                    f"against an actual {before['mean_observed']:.1%}. Isotonic "
                    "calibration, fitted on a split used for nothing else, puts that "
                    "right.</p>"
                    "<p style='font-size:12.5px;line-height:1.6'>Being monotone it never "
                    "reorders two applicants — it only collapses distinct scores onto "
                    "shared steps, which moves ROC-AUC by well under a thousandth. "
                    "Trustworthy probabilities, at effectively no cost to ranking.</p>",
                    **HTML)
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Reliability curve</div>", **HTML)
                st.altair_chart(
                    charts.calibration_curve(
                        pd.DataFrame(metrics["calibration"]["curve_before"]),
                        pd.DataFrame(metrics["calibration"]["curve_after"]),
                        height=250),
                    use_container_width=True)

    with tab_importance:
        left, right = st.columns([1, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "LightGBM gain importance</div>", **HTML)
                importance = _importance()
                if importance:
                    from src.utils.helpers import humanise_feature

                    frame = pd.DataFrame(importance).head(18)
                    frame["label"] = frame["feature"].map(humanise_feature)
                    st.altair_chart(charts.importance_bars(frame, "gain", height=380),
                                    use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">Global SHAP'
                            "</div>"
                            '<div class="card-sub">Mean |SHAP| over a holdout sample — '
                            "what drives the portfolio, not one applicant.</div>", **HTML)
                if st.button("Compute global SHAP (samples 2,000 applicants)"):
                    with st.spinner("Computing SHAP over the sample..."):
                        explainer = _explainer()
                        name = explainer.plot_global_summary(_holdout())
                        global_importance = explainer.global_importance(_holdout())
                    figure(name)
                    st.dataframe(
                        global_importance.head(18)[
                            ["label", "mean_abs_shap", "mean_shap"]],
                        hide_index=True, use_container_width=True)
                else:
                    figure("shap_summary.png")

    with tab_threshold:
        comparison = decision["cost_comparison"]
        cost = decision["cost_model"]
        st.markdown(
            "<p style='font-size:13px;line-height:1.6'>A false negative costs "
            f"<b>{cost['lgd_rate']:.0%} of the loan</b> (loss given default); a false "
            f"positive costs <b>{cost['margin_rate']:.0%}</b> (forgone margin). Both are "
            "weighted by that applicant's actual <code>AMT_CREDIT</code>, because "
            "declining a 2M loan is not the same mistake as declining a 50k one.</p>",
            **HTML)
        left, right = st.columns([1.05, 1], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Expected loss vs threshold</div>", **HTML)
                st.altair_chart(
                    charts.cost_curve(pd.DataFrame(decision["cost_curve"]),
                                      decision["threshold"], height=250),
                    use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Policy comparison</div>", **HTML)
                st.dataframe(
                    pd.DataFrame([
                        {"Policy": name.replace("_", " "),
                         "Threshold": values["threshold"],
                         "Approval rate": values["approval_rate"],
                         "Expected loss": values["total_cost"],
                         "Saving": values["saving_vs_this"]}
                        for name, values in comparison.items()]),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "Approval rate": st.column_config.NumberColumn(format="percent"),
                        "Expected loss": st.column_config.NumberColumn(
                            format="compact"),
                        "Saving": st.column_config.NumberColumn(format="compact")})
                st.markdown(theme.notice(
                    f"<b>{fmt_money(comparison['naive_0.5']['saving_vs_this'])}</b> of "
                    "expected loss avoided versus a naive 0.50 threshold; "
                    f"<b>{fmt_money(comparison['approve_everyone']['saving_vs_this'])}"
                    "</b> versus approving everyone. The threshold is selected on the "
                    "calibration split and reported on the holdout, so the saving is not "
                    "the threshold overfitting its own evaluation set.",
                    kind="good"), **HTML)


# --------------------------------------------------------------------------- #
# 6. Business Rules
# --------------------------------------------------------------------------- #
def page_rules() -> None:
    st.caption("Interpretable rules extracted from the model's own decisions using "
               "a depth-4 surrogate decision tree.")

    if not require("rules"):
        return
    rules = _rules()

    st.markdown(theme.stat_row([
        (str(rules["n_rules"]), "Rules extracted"),
        (fmt_pct(rules["surrogate_fidelity"]), "Surrogate fidelity"),
        (fmt_pct(rules["base_default_rate"], 2), "Portfolio default rate"),
        (f"{rules['decision_threshold']:.3f}", "Decision threshold"),
    ]), **HTML)
    st.markdown("<div style='height:10px'></div>", **HTML)

    left, right = st.columns([1.5, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown(
                f'<div class="card-title">Top rules ({rules["n_rules"]})</div>'
                '<div class="card-sub">Support is the share of applicants covered. '
                "Default rate is measured against ground-truth outcomes, not model "
                "predictions — so a rule is defensible on its own evidence.</div>",
                **HTML)
            st.dataframe(rules_table(rules), hide_index=True, use_container_width=True,
                         height=440, column_config=_rule_columns())

    with right:
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">Lift by rule'
                        "</div>", **HTML)
            frame = pd.DataFrame(rules["rules"])
            frame["label"] = frame.apply(
                lambda r: f"#{r['rule_id']} · {r['risk_band']}", axis=1)
            st.altair_chart(
                charts.importance_bars(frame.head(10), "lift", height=270,
                                       colour=theme.RED),
                use_container_width=True)
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">'
                        "Why these features</div>", **HTML)
            protected = ", ".join(
                rules["excluded_from_rule_language"]["protected_attributes"])
            st.markdown(
                "<p style='font-size:12.5px;line-height:1.6'>The rule language is "
                f"restricted to <b>{len(rules['rule_features'])} numeric, readable, "
                "permitted features</b>. Excluded on purpose:</p>"
                "<ul style='font-size:12.5px;line-height:1.6;padding-left:18px'>"
                "<li><b>Categorical features</b> — a split on "
                "<code>OCCUPATION_TYPE &lt;= 7.5</code> is an artefact of integer "
                "encoding, not a rule anyone can apply by hand.</li>"
                f"<li><b>Protected attributes</b> ({protected}) — a rule set is the one "
                "artefact here a human might apply directly.</li>"
                "<li><b>Raw day-counts</b> — superseded by their engineered year "
                "equivalents.</li></ul>",
                **HTML)


# --------------------------------------------------------------------------- #
# 7. Model Health
# --------------------------------------------------------------------------- #
def page_health() -> None:
    st.caption("A model can hold 0.78 AUC overall while approving one group at a "
               "materially different rate. Measured here, and labelled honestly.")

    if not require("model"):
        return
    fairness = _fairness()
    if fairness is None:
        st.warning("The fairness report has not been generated yet.")
        st.code("python -m src.monitoring.fairness", language="bash")
        return

    tab_fairness, tab_shift, tab_quality = st.tabs(
        ["Fairness", "Distribution comparison", "Data quality"])

    with tab_fairness:
        for name, block in fairness["groups"].items():
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown(
                    f'<div class="card-title" style="font-size:15px">Performance by '
                    f'{name.replace("_", " ")}</div>', **HTML)
                chart_col, table_col = st.columns([1, 1.25], gap="medium")
                with chart_col:
                    st.altair_chart(
                        charts.grouped_bars(fairness_group_frame(fairness, name),
                                            "group", "metric", "value", height=200,
                                            label=""),
                        use_container_width=True)
                with table_col:
                    st.dataframe(
                        pd.DataFrame(block["rows"])[[
                            "group", "applicants", "observed_default_rate", "roc_auc",
                            "approval_rate", "recall"]],
                        hide_index=True, use_container_width=True,
                        column_config={
                            "group": st.column_config.TextColumn("Group"),
                            "applicants": st.column_config.NumberColumn(
                                "Applicants", format="localized"),
                            "observed_default_rate": st.column_config.NumberColumn(
                                "Default rate", format="percent"),
                            "roc_auc": st.column_config.NumberColumn(
                                "ROC-AUC", format="%.3f"),
                            "approval_rate": st.column_config.NumberColumn(
                                "Approval rate", format="percent"),
                            "recall": st.column_config.NumberColumn(
                                "Recall", format="percent")})
                gaps = block["gaps"]
                if gaps:
                    st.markdown(theme.stat_row([
                        (fmt_pct(gaps["demographic_parity_gap"]),
                         "Demographic parity gap"),
                        (fmt_pct(gaps["equal_opportunity_gap"]),
                         "Equal opportunity gap"),
                        (fmt_pct(gaps["predictive_parity_gap"]),
                         "Predictive parity gap"),
                        (f"{gaps['auc_gap']:.3f}", "AUC gap"),
                    ]), **HTML)

        st.markdown(theme.notice(
            "None of these gaps are <b>fixed</b> here. Which parity a lender should "
            "enforce is a policy and legal question, not a modelling one — the "
            "platform's job is to measure them and put them on screen.",
            kind="info", icon_name="about"), **HTML)

    with tab_shift:
        shift = fairness["distribution_shift"]
        with st.container(border=True):
            st.markdown(theme.CARD_MARK, **HTML)
            st.markdown('<div class="card-title" style="font-size:14px">'
                        "Train vs holdout population stability</div>", **HTML)
            st.markdown(theme.notice(
                f"<b>This is not drift, and it is labelled honestly.</b> "
                f"{shift['caveat']}", kind="warn", icon_name="about"), **HTML)
            st.markdown("<div style='height:10px'></div>", **HTML)
            chart_col, table_col = st.columns([1, 1], gap="medium")
            frame = pd.DataFrame(shift["features"])
            with chart_col:
                st.altair_chart(charts.psi_bars(frame, height=250),
                                use_container_width=True)
            with table_col:
                st.dataframe(frame, hide_index=True, use_container_width=True)

    with tab_quality:
        eda = _eda()
        if eda:
            with st.container(border=True):
                st.markdown(theme.CARD_MARK, **HTML)
                st.markdown('<div class="card-title" style="font-size:14px">'
                            "Completeness by feature group</div>", **HTML)
                st.dataframe(pd.DataFrame(eda["data_quality"]["scorecard"]),
                             hide_index=True, use_container_width=True)
            for anomaly in eda["data_quality"]["anomalies"]:
                st.markdown(theme.notice(anomaly, kind="warn", icon_name="about"), **HTML)
        else:
            st.info("Run the EDA to populate this tab.")


# --------------------------------------------------------------------------- #
# 8. About
# --------------------------------------------------------------------------- #
@st.cache_data
def _stack_groups() -> dict:
    """The technology stack, with versions read from requirements.txt.

    Reading the pin file rather than hardcoding a list means the About page
    cannot claim a version the project does not actually install.
    """
    from src.utils.config import PROJECT_ROOT

    versions: dict[str, str] = {}
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, _, version = line.partition("==")
            versions[name.strip().lower()] = version.strip()

    def pin(package: str, role: str) -> tuple[str, str]:
        version = versions.get(package)
        return (package, f"{version}" if version else role)

    return {
        "Language & data": [
            ("python", "3.11"),
            pin("pandas", "dataframes"),
            pin("numpy", "arrays"),
            pin("pyarrow", "parquet cache"),
        ],
        "Machine learning": [
            pin("lightgbm", "gradient boosting"),
            pin("scikit-learn", "splits, calibration, surrogate tree"),
            pin("shap", "TreeExplainer"),
            pin("joblib", "artifact serialisation"),
        ],
        "LLM": [
            pin("anthropic", "SDK"),
            ("claude-sonnet-5", "NL to SQL, agent routing"),
            ("claude-haiku-4-5", "row summarisation, reason phrasing"),
            pin("sqlparse", "SQL guardrails"),
        ],
        "Application": [
            pin("streamlit", "UI"),
            ("altair", "interactive charts (ships with Streamlit)"),
            pin("matplotlib", "static report figures"),
            ("SQLite", "read-only analytical warehouse"),
        ],
        "Config & infrastructure": [
            pin("pydantic-settings", "typed configuration"),
            ("Docker", "multi-stage build + compose"),
            pin("pytest", "test suite"),
            pin("nbconvert", "notebook export"),
        ],
    }


def page_about() -> None:
    st.markdown(
        theme.hero(
            "Building a fairer financial future with data and AI",
            "This platform demonstrates how AI and data can be used responsibly to "
            "assess credit risk, enable more inclusive lending, and create greater "
            "opportunities for people around the world.",
            eyebrow="About this platform"),
        **HTML)
    st.markdown("<div style='height:12px'></div>", **HTML)

    with st.container(border=True):
        st.markdown(theme.CARD_MARK, **HTML)
        st.markdown(theme.footer_columns([
            ("scale", "Our Mission",
             "Use AI to enable fairer, faster and more transparent credit decisions."),
            ("eda", "The Dataset",
             "Home Credit Default Risk from Kaggle — 307,511 applications, 122 columns, "
             "an 8.07% default rate."),
            ("shield", "Responsible AI",
             "Adverse-action reason codes, protected-attribute suppression, and "
             "published fairness gaps."),
            ("spark", "Open Innovation",
             "End-to-end AI engineering: data, model, explanation, policy, agent and "
             "deployment."),
        ]), **HTML)

    st.markdown("<div style='height:10px'></div>", **HTML)
    with theme.card("How a decision is produced"):
            st.code(
                "application row\n"
                "   |- clean          sentinels repaired, outliers clipped\n"
                "   |- engineer       19 domain features\n"
                "   |- LightGBM       scale_pos_weight = 11.39\n"
                "   |- calibrate      isotonic, on a held-back split\n"
                "   |- decide         cost-optimal threshold, weighted by loan amount\n"
                "   |- SHAP           per-applicant attribution\n"
                "   |- reason codes   ECOA-style principal reasons\n"
                "   +- policy rule    the surrogate rule covering this applicant",
                language="text")

    st.markdown("<div style='height:10px'></div>", **HTML)
    with theme.card("Architecture",
                    "Every box is a module that exists in the repository."):
        st.markdown(theme.architecture([
            ("Data", [
                ("application_train.csv", "307,511 rows"),
                ("bureau.csv", "external credit history"),
                ("previous_application.csv", "prior applications"),
            ]),
            ("Processing", [
                ("loader.py", "join + aggregate"),
                ("preprocessor.py", "clean, engineer, encode"),
                ("build_database.py", "SQLite warehouse"),
            ]),
            ("Model", [
                ("train.py", "LightGBM + isotonic + cost threshold"),
                ("evaluate.py", "ROC/PR, calibration, expected loss"),
            ]),
            ("Intelligence", [
                ("predict.py", "score, band, decide"),
                ("shap_explainer.py", "attribution"),
                ("reason_codes.py", "ECOA reasons"),
                ("rule_extractor.py", "surrogate policy"),
                ("fairness.py", "group gaps, PSI"),
            ]),
            ("Language", [
                ("semantic_layer.py", "canonical metrics"),
                ("query_runner.py", "4 SQL guardrails"),
                ("nl_to_sql.py", "self-correcting NL to SQL"),
                ("eval_harness.py", "25 labelled cases"),
            ]),
            ("Agent", [
                ("orchestrator.py", "AI tool-use loop"),
                ("tools.py", "6 tools over the modules above"),
            ]),
            ("Interface", [
                ("app/ui.py", "9 sections"),
                ("theme.py", "design system"),
                ("charts.py", "Altair builders"),
            ]),
        ]), **HTML)


    st.markdown("<div style='height:10px'></div>", **HTML)
    with theme.card("Technology stack",
                    "Versions are read from requirements.txt at render time, so this "
                    "list cannot drift from what is actually installed."):
        st.markdown(theme.stack_table(_stack_groups()), **HTML)

# --------------------------------------------------------------------------- #
PAGES = {
    "Home": page_home,
    "Predict": page_predict,
    "Explainability": page_explainability,
    "Talk to Data": page_talk,
    "EDA & Insights": page_eda,
    "Model Performance": page_performance,
    "Business Rules": page_rules,
    "Model Health": page_health,
    "About": page_about,
}


def main() -> None:
    page = sidebar()
    st.markdown(theme.topbar(page, PAGE_SUBTITLES.get(page, "")), **HTML)
    PAGES[page]()


if __name__ == "__main__":
    main()
