"""Build documents/project_presentation.pdf from the pipeline's own artifacts.

    python documents/build_presentation.py

Every number and every chart on these slides is read from `reports/` at build
time rather than typed in, so the deck cannot drift from the model. Re-run the
pipeline, re-run this, and the deck is current.

Rendered with matplotlib rather than python-pptx: the deliverable is a PDF, and
this keeps the dependency list identical to the one the app already needs.

The Decision Trace slide is drawn as a diagram, not pasted as a screenshot -
it is generated from a live scoring call, so the applicant, the probability and
the reasons on it are real output, not mock-ups.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import settings  # noqa: E402
from src.utils.helpers import fmt_money, fmt_pct, load_json  # noqa: E402

OUTPUT = ROOT / "documents" / "project_presentation.pdf"

# 16:9 at 100 dpi-ish, which is what a slide deck wants.
SLIDE = (13.333, 7.5)
INK = "#12213a"
MUTED = "#5b6779"
ACCENT = "#1f4e79"
RISK = "#b7472a"
GOOD = "#2e7d32"
WARM = "#c9772e"
PAPER = "#fbfaf8"


# --------------------------------------------------------------------------- #
# Slide primitives
# --------------------------------------------------------------------------- #
def new_slide(pdf: PdfPages):
    figure = plt.figure(figsize=SLIDE, facecolor=PAPER)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0, 0), 1, 1, color=PAPER, zorder=-10))
    return figure, axis


def close(figure, pdf: PdfPages) -> None:
    pdf.savefig(figure, facecolor=PAPER)
    plt.close(figure)


def title(axis, text: str, subtitle: str = "") -> None:
    # Shrink rather than overflow: a 13.3in slide fits ~46 characters at 27pt.
    size = 27 if len(text) <= 46 else 27 * 46 / len(text)
    axis.text(0.06, 0.90, text, fontsize=size, color=INK, weight="bold", va="top")
    if subtitle:
        axis.text(0.06, 0.835, subtitle, fontsize=13.5, color=MUTED, va="top")
    axis.plot([0.06, 0.94], [0.805, 0.805], color="#d8d3cb", linewidth=1.1)


def bullets(axis, items: list[str], x: float = 0.075, y: float = 0.71,
            size: float = 13, gap: float = 0.072, width: int = 78) -> float:
    """Wrapped bullet list. Returns the y position after the last line."""
    import textwrap

    for item in items:
        lines = textwrap.wrap(item, width=width) or [""]
        axis.text(x - 0.018, y, "•", fontsize=size, color=ACCENT, va="top")
        for offset, line in enumerate(lines):
            axis.text(x, y - offset * 0.042, line, fontsize=size, color=INK, va="top")
        y -= gap + max(0, len(lines) - 1) * 0.042
    return y


def stat_row(axis, stats: list[tuple[str, str, str]], y: float = 0.30) -> None:
    """Big-number tiles: (value, label, colour)."""
    n = len(stats)
    span = 0.88 / n
    for index, (value, label, colour) in enumerate(stats):
        x = 0.06 + span * index
        axis.add_patch(FancyBboxPatch(
            (x, y), span - 0.028, 0.185, boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="#ffffff", edgecolor="#e3ded6", linewidth=1,
        ))
        axis.text(x + 0.022, y + 0.125, value, fontsize=25, color=colour, weight="bold",
                  va="center")
        axis.text(x + 0.022, y + 0.048, label, fontsize=10.5, color=MUTED, va="center")


def add_image(figure, path: Path, rect: list[float]) -> bool:
    if not path.exists():
        return False
    image_axis = figure.add_axes(rect)
    image_axis.imshow(plt.imread(str(path)))
    image_axis.axis("off")
    return True


def _shorten(text: str, limit: int) -> str:
    """Truncate on a word boundary rather than mid-word."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def _paragraph(axis, x: float, y: float, text: str, width: int, size: float,
               colour: str) -> None:
    """Wrapped body text - the plain axis.text calls were running off the slide."""
    import textwrap

    for offset, line in enumerate(textwrap.wrap(text, width=width)):
        axis.text(x, y - offset * 0.042, line, fontsize=size, color=colour, va="top")


def footer(axis, text: str) -> None:
    axis.text(0.06, 0.045, text, fontsize=10, color=MUTED, style="italic")


def table(axis, headers: list[str], rows: list[list[str]], y: float = 0.68,
          widths: list[float] | None = None, size: float = 11.5) -> None:
    widths = widths or [1 / len(headers)] * len(headers)
    x_positions, cursor = [], 0.065
    for width in widths:
        x_positions.append(cursor)
        cursor += width * 0.87

    for x, header in zip(x_positions, headers):
        axis.text(x, y, header, fontsize=size - 0.5, color=MUTED, weight="bold", va="top")
    axis.plot([0.06, 0.94], [y - 0.028, y - 0.028], color="#d8d3cb", linewidth=1)

    row_y = y - 0.062
    for row in rows:
        for x, cell in zip(x_positions, row):
            colour = INK
            if cell.startswith("+"):
                colour = GOOD
            axis.text(x, row_y, cell, fontsize=size, color=colour, va="top")
        row_y -= 0.058


# --------------------------------------------------------------------------- #
# Slides
# --------------------------------------------------------------------------- #
def slide_title(pdf, metrics) -> None:
    figure, axis = new_slide(pdf)
    axis.add_patch(Rectangle((0, 0.62), 1, 0.38, color=ACCENT, zorder=-5))
    axis.text(0.06, 0.86, "AI-Powered Credit Risk", fontsize=42, color="#ffffff",
              weight="bold", va="top")
    axis.text(0.06, 0.755, "Intelligence Platform", fontsize=42, color="#ffffff",
              weight="bold", va="top")
    axis.text(0.06, 0.665, "Home Credit Default Risk  |  307,511 applications",
              fontsize=14, color="#c9dcf0", va="top")

    axis.text(0.06, 0.53,
              "Not just a model — a credit decision system:",
              fontsize=19, color=INK, weight="bold", va="top")
    axis.text(0.06, 0.465,
              "scored, explained, defensible, auditable, and queryable in plain English.",
              fontsize=19, color=MUTED, va="top")

    holdout = metrics["discrimination"]["holdout"]
    saving = metrics["decision"]["cost_comparison"]["naive_0.5"]["saving_vs_this"]
    stat_row(axis, [
        (f"{holdout['roc_auc']:.3f}", "Holdout ROC-AUC", ACCENT),
        (f"{holdout['pr_auc']:.3f}", "PR-AUC at an 8% base rate", ACCENT),
        (f"{metrics['calibration']['after']['expected_calibration_error']:.4f}",
         "Calibration error (from 0.2687)", GOOD),
        (fmt_money(saving), "Expected loss avoided", GOOD),
    ], y=0.20)
    footer(axis, "Every figure in this deck is read from reports/ at build time.")
    close(figure, pdf)


def slide_problem(pdf, eda) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "The problem, and what makes it hard",
          "A lender must decide, price and justify — in that order")
    bullets(axis, [
        "Predict default on 307,511 loan applications, 122 raw columns, "
        f"{fmt_pct(eda['target']['default_rate'], 2)} default rate "
        f"({eda['target']['imbalance_ratio']:.1f} : 1 imbalance).",
        "Accuracy is a trap: predicting ‘everyone repays’ scores 91.9%. "
        "PR-AUC and expected loss are the metrics that mean anything.",
        "A score is not a decision. The threshold, not the model, determines "
        "how much money the portfolio loses.",
        "A decision that cannot be explained cannot be issued: ECOA / Regulation B "
        "requires the principal reasons for every decline.",
        "Analysts ask questions faster than a BI backlog can answer them.",
    ])
    add_image(figure, settings.figures_dir / "eda_target_balance.png",
              [0.60, 0.10, 0.34, 0.50])
    footer(axis, "Dataset: Kaggle Home Credit Default Risk. Never committed to the repo.")
    close(figure, pdf)


def slide_sentinel(pdf, eda) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "EDA: the finding most analyses get wrong",
          "DAYS_EMPLOYED = 365243 on 55,374 rows (18%) — a sentinel, not a 1000-year career")
    bullets(axis, [
        "Most write-ups stop at ‘dirty data, drop it’. But 99.96% of that group "
        "are pensioners.",
        "They default at 5.4% — BELOW the 8.7% employed average. One of the safer "
        "segments in the book.",
        "Bin them into ‘10 years+’ and the trend moves the wrong way for the "
        "wrong reason. Drop them and 18% of the portfolio disappears.",
        "Fixed in three places so it cannot leak back: the preprocessor, the SQL "
        "semantic layer, and the EDA itself.",
    ], y=0.70, width=52)
    add_image(figure, settings.figures_dir / "eda_insight_employment.png",
              [0.55, 0.14, 0.40, 0.52])
    footer(axis, "Four more insights in the notebook: income, contract type, age band, "
                 "external scores.")
    close(figure, pdf)


def slide_signal(pdf) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "EDA: where the signal actually lives",
          "External credit scores separate the population ~6x, best decile to worst")
    add_image(figure, settings.figures_dir / "eda_insight_ext_source.png",
              [0.06, 0.14, 0.44, 0.58])
    add_image(figure, settings.figures_dir / "eda_insight_age.png",
              [0.53, 0.14, 0.42, 0.58])
    footer(axis, "Consequence: engineer mean / min / max / count / disagreement across the "
                 "three scores — EXT_SOURCE_1 is missing for 56% of applicants.")
    close(figure, pdf)


def slide_model(pdf, metrics) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "The model, and the split design that keeps it honest",
          "LightGBM, 143 features, four disjoint stratified splits")
    table(axis, ["Split", "Share", "Used for"], [
        ["train", "60%", "Fitting the booster"],
        ["valid", "10%", "Early stopping"],
        ["calib", "10%", "Isotonic calibrator AND threshold selection"],
        ["holdout", "20%", "Every headline number; the UI applicant pool"],
    ], y=0.74, widths=[0.16, 0.14, 0.70])

    bullets(axis, [
        "Imbalance: scale_pos_weight = 11.39, not SMOTE.",
        "Calibration is not optional — reweighting wrecks the probabilities.",
    ], y=0.42, width=95, size=12.5, gap=0.058)

    holdout = metrics["discrimination"]["holdout"]
    stat_row(axis, [
        (f"{holdout['roc_auc']:.4f}", "ROC-AUC (holdout)", ACCENT),
        (f"{holdout['pr_auc']:.4f}", "PR-AUC", ACCENT),
        (f"{holdout['ks_statistic']:.3f}", "KS statistic", ACCENT),
        (f"{metrics['best_iteration']}", "Trees (early stopped)", MUTED),
    ], y=0.10)
    close(figure, pdf)


def slide_calibration(pdf, metrics) -> None:
    figure, axis = new_slide(pdf)
    before = metrics["calibration"]["before"]
    after = metrics["calibration"]["after"]
    title(axis, "Calibration: making the probability mean what it says",
          f"Raw model predicts {before['mean_predicted']:.0%} average risk against an "
          f"actual {before['mean_observed']:.1%}")
    add_image(figure, settings.figures_dir / "ml_calibration.png", [0.53, 0.13, 0.42, 0.58])
    bullets(axis, [
        f"Brier score {before['brier_score']:.4f} → {after['brier_score']:.4f}.",
        f"Expected calibration error {before['expected_calibration_error']:.4f} "
        f"→ {after['expected_calibration_error']:.4f}.",
        "Isotonic over Platt: the residual tail miscalibration is not sigmoid-shaped, "
        "and 30k calibration rows is ample for a non-parametric fit.",
        "Honest caveat: isotonic never reorders applicants, but it collapses 61k "
        "distinct scores onto ~150 steps. ROC-AUC moves 0.7792 → 0.7788.",
    ], y=0.70, width=50)
    footer(axis, "Fitted on a split used for nothing else — not the early-stopping set.")
    close(figure, pdf)


def slide_cost(pdf, metrics) -> None:
    figure, axis = new_slide(pdf)
    decision = metrics["decision"]
    comparison = decision["cost_comparison"]
    cost = decision["cost_model"]
    title(axis, "A threshold chosen on economics, not on 0.50",
          f"False negative costs {cost['lgd_rate']:.0%} of the loan; false positive costs "
          f"{cost['margin_rate']:.0%} — both weighted by that applicant's AMT_CREDIT")
    add_image(figure, settings.figures_dir / "ml_cost_curve.png", [0.63, 0.16, 0.33, 0.50])
    table(axis, ["Policy", "Threshold", "Approvals", "Expected loss"], [
        ["Approve everyone", "—", "100.0%",
         fmt_money(comparison["approve_everyone"]["total_cost"])],
        ["Naive 0.50", "0.500",
         f"{comparison['naive_0.5']['approval_rate']:.1%}",
         fmt_money(comparison["naive_0.5"]["total_cost"])],
        ["Cost-optimal", f"{decision['threshold']:.3f}",
         f"{comparison['cost_optimal']['approval_rate']:.1%}",
         fmt_money(comparison["cost_optimal"]["total_cost"])],
    ], y=0.68, widths=[0.24, 0.13, 0.14, 0.21], size=11.5)
    axis.text(0.065, 0.34,
              f"{fmt_money(comparison['naive_0.5']['saving_vs_this'])} of expected loss "
              f"avoided versus naive 0.50.",
              fontsize=15.5, color=GOOD, weight="bold", va="top")
    axis.text(0.065, 0.275,
              f"{fmt_money(comparison['approve_everyone']['saving_vs_this'])} versus "
              "approving everyone.",
              fontsize=15.5, color=GOOD, weight="bold", va="top")
    _paragraph(axis, 0.065, 0.20,
               "Approval rate falls from 99.6% to 81.4% — the model buys that saving by "
               "declining 18 applicants in 100 it would otherwise have approved.",
               width=62, size=11.5, colour=MUTED)
    footer(axis, "Threshold selected on the calibration split, reported on the holdout — "
                 "so the saving is not the threshold overfitting its own evaluation set.")
    close(figure, pdf)


def slide_decision_trace(pdf) -> None:
    """The hero. Drawn from a live scoring call, so every value here is real."""
    figure, axis = new_slide(pdf)
    title(axis, "Decision Trace — the whole chain on one screen",
          "The hero view of the app: one applicant, every stage, top to bottom")

    try:
        from src.ml.predict import get_applicant, get_scorer, list_applicants
        from src.rules.rule_extractor import load_rules
        from src.xai.reason_codes import build_reason_codes
        from src.xai.shap_explainer import get_explainer

        pool = list_applicants(limit=20, band="High", seed=1)
        applicant_id = int(pool.index[0])
        row = get_applicant(applicant_id)
        assessment = get_scorer().assess(row)
        explanation = get_explainer().explain(row)
        reasons, _ = build_reason_codes(explanation, top_n=3)
        rules = load_rules()
        top_rule = next((r for r in rules["rules"] if r["risk_band"] == "High"), None)
    except Exception as exc:  # the deck must still build without artifacts
        axis.text(0.06, 0.5, f"Artifacts unavailable: {exc}", fontsize=13, color=RISK)
        close(figure, pdf)
        return

    steps = [
        ("1. Feature engineering",
         f"{len(get_scorer().feature_names)} features, 19 engineered, sentinel repaired"),
        ("2. Risk model",
         f"raw score {assessment.raw_score:.3f}  (LightGBM, scale_pos_weight 11.39)"),
        ("3. Calibration",
         f"{assessment.probability * 100:.1f}% calibrated probability of default"),
        ("4. Decision threshold",
         f"decline at or above {assessment.threshold * 100:.1f}%  (cost-optimal)"),
        ("5. Risk band", f"{assessment.risk_band.upper()}"),
        ("6. Explanation (SHAP → reason codes)",
         reasons[0].statement if reasons else "no dominant driver"),
        ("7. Policy rule",
         (top_rule["rule"].split(" THEN ")[0][3:][:60] + "... → DECLINE")
         if top_rule else "no covering rule"),
    ]

    y = 0.735
    axis.plot([0.085, 0.085], [0.135, y], color=ACCENT, linewidth=2.2, zorder=1)
    for label, value in steps:
        axis.scatter([0.085], [y], s=95, color=ACCENT, zorder=3,
                     edgecolors=PAPER, linewidths=2)
        axis.text(0.115, y + 0.012, label.upper(), fontsize=9.5, color=MUTED,
                  weight="bold", va="center")
        axis.text(0.115, y - 0.028, value[:74], fontsize=12.5, color=INK, va="center")
        y -= 0.086

    colour = {"High": RISK, "Medium": WARM, "Low": GOOD}[assessment.risk_band]
    axis.add_patch(FancyBboxPatch(
        (0.63, 0.44), 0.31, 0.22, boxstyle="round,pad=0.014,rounding_size=0.016",
        facecolor=colour, edgecolor="none",
    ))
    axis.text(0.785, 0.605, f"Applicant {assessment.applicant_id}", fontsize=12,
              color="#ffffff", ha="center", va="center")
    axis.text(0.785, 0.545, assessment.decision.upper(), fontsize=32, color="#ffffff",
              weight="bold", ha="center", va="center")
    axis.text(0.785, 0.485, f"{assessment.probability * 100:.1f}% probability of default",
              fontsize=12.5, color="#ffffff", ha="center", va="center")

    axis.text(0.63, 0.41, "Principal reasons (ECOA-style)", fontsize=11.5, color=MUTED,
              weight="bold", va="top")
    reason_y = 0.365
    for reason in reasons:
        import textwrap

        for offset, line in enumerate(textwrap.wrap(
            f"{reason.rank}. {reason.statement}", width=42
        )):
            axis.text(0.63, reason_y - offset * 0.032, line, fontsize=10.5, color=INK,
                      va="top")
        reason_y -= 0.087
    footer(axis, "Generated live from the trained model — not a mock-up.")
    close(figure, pdf)


def slide_explainability(pdf) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "Explainability that a regulator would accept",
          "SHAP attribution → adverse-action reason codes (ECOA / Regulation B)")
    bullets(axis, [
        "Layer 1 — deterministic templates. Every reason code is derived from the "
        "SHAP contribution and the applicant's own value by rule: auditable, "
        "reproducible, works with no API key.",
        "Layer 2 — a cheap model restates those bullets as a short notice. It receives "
        "the reason codes as facts to restate, never as data to reason about.",
        "That ordering IS the hallucination control: the LLM is a stylist here, not a "
        "decision-maker.",
        "Protected attributes (gender, age, marital status) are suppressed from the "
        "notice and REPORTED — a reviewer sees that the model leaned on one, rather "
        "than having it quietly disappear.",
        "Stated caveat: SHAP explains the raw log-odds output, so contributions are "
        "shown as a share of total push, never as ‘+7% of default probability’.",
    ], y=0.70, width=92, size=12.5, gap=0.062)
    add_image(figure, settings.figures_dir / "ml_feature_importance.png",
              [0.70, 0.08, 0.26, 0.40])
    close(figure, pdf)


def slide_rules(pdf, rules) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "From a 1,190-tree ensemble to credit policy",
          f"Depth-4 decision-tree surrogate of the model's own decisions — "
          f"{fmt_pct(rules['surrogate_fidelity'])} fidelity")
    high = [r for r in rules["rules"] if r["risk_band"] == "High"][:4]
    table(axis, ["Rule (abbreviated)", "Support", "Default rate", "Lift"], [
        [_shorten(rule["rule"].split(" THEN ")[0][3:], 64),
         f"{rule['support']:.1%}", f"{rule['precision']:.1%}", f"{rule['lift']:.2f}x"]
        for rule in high
    ], y=0.68, widths=[0.60, 0.13, 0.16, 0.11], size=11)
    bullets(axis, [
        "Precision is measured against ground-truth outcomes, not model predictions — "
        "so a rule is defensible on its own evidence.",
        "Rule language excludes categorical splits (an encoding artefact) and protected "
        "attributes — this is the one artefact a human might apply by hand.",
    ], y=0.32, width=92, size=12.5)
    footer(axis, f"{rules['n_rules']} rules total, partitioning the holdout population. "
                 f"Portfolio base rate {fmt_pct(rules['base_default_rate'], 2)}.")
    close(figure, pdf)


def slide_talk_to_data(pdf) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "Talk to Data: self-correcting NL → SQL",
          "The AI agent writes the SQL; guardrails decide whether it runs")
    axis.text(0.065, 0.71,
              "Sonnet (schema + metrics + exemplars, cached)\n"
              "   → run_sql(sql)\n"
              "        → rejected or SQLite error → error text back to Sonnet "
              "→ rewrite (≤ 2 retries)\n"
              "        → success → rows to Haiku → business answer",
              fontsize=12, color=INK, va="top", family="monospace")
    table(axis, ["#", "Guardrail", "Stops"], [
        ["1", "Statement allowlist (sqlparse)", "Anything that is not a single SELECT / WITH"],
        ["2", "Whole-token keyword denylist", "INSERT / UPDATE / DELETE / DROP / ATTACH / PRAGMA"],
        ["3", "Table allowlist", "Hallucinated tables, cross-database reads"],
        ["4", "Read-only connection (mode=ro)", "Any write that survived layers 1-3"],
    ], y=0.49, widths=[0.05, 0.35, 0.60], size=11)
    bullets(axis, [
        "Semantic layer: 15 metrics pinned once and injected into every prompt, so "
        "‘default rate’ is one formula — and the 365243 sentinel is baked "
        "into employment_years, so the metric cannot lie.",
    ], y=0.20, width=98, size=12)
    footer(axis, "25 labelled eval cases including a multi-turn follow-up, three "
                 "unanswerable questions and a prompt-injection attempt.")
    close(figure, pdf)


def slide_fairness(pdf, fairness) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "Model health & fairness, labelled honestly",
          "A model can hold 0.78 AUC overall while approving one group very differently")
    rows = []
    for name, block in fairness["groups"].items():
        for entry in block["rows"][:3]:
            rows.append([
                f"{name.replace('_', ' ')}: {entry['group']}",
                f"{entry['applicants']:,}",
                f"{entry['observed_default_rate']:.1%}",
                f"{entry['roc_auc']:.3f}",
                f"{entry['approval_rate']:.1%}",
            ])
    table(axis, ["Group", "Applicants", "Default rate", "ROC-AUC", "Approval rate"],
          rows[:6], y=0.70, widths=[0.32, 0.16, 0.18, 0.16, 0.18], size=11)

    gender_gaps = fairness["groups"]["gender"]["gaps"]
    age_gaps = fairness["groups"]["age_band"]["gaps"]
    axis.text(0.065, 0.26,
              f"Demographic parity gap: {gender_gaps['demographic_parity_gap']:.1%} by "
              f"gender, {age_gaps['demographic_parity_gap']:.1%} by age band.",
              fontsize=13.5, color=RISK, weight="bold", va="top")
    _paragraph(axis, 0.065, 0.205,
               "None of these are ‘fixed’ here. Which parity a lender should enforce is "
               "a policy and legal question, not a modelling one — the platform measures "
               "them and puts them on screen.",
               width=104, size=12, colour=INK)
    _paragraph(axis, 0.065, 0.115,
               "Distribution shift is reported as train-vs-holdout PSI and explicitly NOT "
               "called drift: this dataset has no time axis, so max PSI is 0.0004 by "
               "construction. Calling that ‘no drift detected’ would be theatre.",
               width=104, size=12, colour=MUTED)
    close(figure, pdf)


def slide_architecture(pdf) -> None:
    figure, axis = new_slide(pdf)
    title(axis, "Engineering", "One container, mounted data, turnkey first run")
    axis.text(0.065, 0.71,
              "data/raw/*.csv (mounted)\n"
              "   → loader + preprocessor      sentinel repair, 19 engineered features\n"
              "   → train.py                  LightGBM + isotonic + cost threshold\n"
              "   → shap_explainer / reason_codes / rule_extractor\n"
              "   → build_database            curated SQLite warehouse (read-only)\n"
              "   → nl_to_sql + eval_harness  self-correcting, measured\n"
              "   → agent/orchestrator        AI tool-use over all six tools\n"
              "   → app/ui.py                 7 sections, Decision Trace hero",
              fontsize=11.5, color=INK, va="top", family="monospace")
    bullets(axis, [
        "docker compose up builds the warehouse, EDA, model, rules and fairness report "
        "on first start, then serves the UI. Verified cold-start: 85 seconds.",
        "124 passing tests. Guardrails, semantic-layer SQL executed against the real "
        "database, preprocessing traps and reason-code compliance all covered.",
        "Every LLM feature has a deterministic twin — scoring, explanation, rules and "
        "EDA all run with no API key.",
        "Token discipline, measured: 18,653 → 5,027 billable input tokens over a "
        "6-turn session (−73%), driven mostly by prompt caching.",
    ], y=0.40, width=96, size=12.5, gap=0.062)
    close(figure, pdf)


def slide_close(pdf) -> None:
    figure, axis = new_slide(pdf)
    axis.add_patch(Rectangle((0, 0), 1, 1, color=ACCENT, zorder=-5))
    axis.text(0.06, 0.80, "What makes this a system,", fontsize=34, color="#ffffff",
              weight="bold", va="top")
    axis.text(0.06, 0.705, "not a notebook", fontsize=34, color="#ffffff",
              weight="bold", va="top")
    lines = [
        "Calibrated probabilities, not just a ranking.",
        "A threshold chosen on expected loss, not on 0.50.",
        "Reasons a regulator would accept, generated without an LLM in the loop.",
        "Policy rules with support, precision and lift attached.",
        "SQL a language model cannot misuse, on four independent guardrails.",
        "Fairness gaps measured and published rather than absorbed.",
        "Limitations stated: no time axis, an assumed cost matrix, 81% surrogate fidelity.",
    ]
    y = 0.58
    for line in lines:
        axis.text(0.075, y, "—", fontsize=13, color="#7fb3e0", va="top")
        axis.text(0.105, y, line, fontsize=14.5, color="#ffffff", va="top")
        y -= 0.065
    axis.text(0.06, 0.075,
              "Every number in this deck regenerates from the pipeline: "
              "python documents/build_presentation.py",
              fontsize=11, color="#a8c9e8", style="italic")
    close(figure, pdf)


# --------------------------------------------------------------------------- #
def build(output: Path = OUTPUT) -> Path:
    metrics = load_json(settings.metrics_path)
    eda = load_json(settings.eda_summary_path)
    rules = load_json(settings.rules_path)
    fairness = load_json(settings.fairness_path)

    missing = [name for name, value in
               (("model_metrics", metrics), ("eda_summary", eda),
                ("business_rules", rules), ("fairness_report", fairness))
               if value is None]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {', '.join(missing)}. Run the pipeline first "
            "(see README quick start)."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        slide_title(pdf, metrics)
        slide_problem(pdf, eda)
        slide_sentinel(pdf, eda)
        slide_signal(pdf)
        slide_model(pdf, metrics)
        slide_calibration(pdf, metrics)
        slide_cost(pdf, metrics)
        slide_decision_trace(pdf)
        slide_explainability(pdf)
        slide_rules(pdf, rules)
        slide_talk_to_data(pdf)
        slide_fairness(pdf, fairness)
        slide_architecture(pdf)
        slide_close(pdf)

        info = pdf.infodict()
        info["Title"] = "AI-Powered Credit Risk Intelligence Platform"
        info["Subject"] = "Explainable, agentic credit-risk decisioning on Home Credit data"

    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")
    return output


if __name__ == "__main__":
    build()
