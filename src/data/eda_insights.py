"""Exploratory analysis: the five business insights, plus the data-quality scorecard.

The analysis lives here rather than inside the notebook so that exactly one
implementation feeds all three consumers - `notebooks/eda.ipynb`, the Streamlit
EDA tab, and the figures in the deck. A notebook that recomputes its own charts
is a notebook that eventually disagrees with the app.

Every insight returns `(table, figure_filename)`: the table is the evidence, the
figure is how it gets communicated, and `run_full_eda()` persists both.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.feature_dictionary import group_columns  # noqa: E402
from src.data.loader import dataset_profile, load_dataset  # noqa: E402
from src.data.preprocessor import missing_report  # noqa: E402
from src.utils.config import settings  # noqa: E402
from src.utils.helpers import DAYS_EMPLOYED_SENTINEL, save_json  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

RISK_COLOUR = "#b7472a"
SAFE_COLOUR = "#1f4e79"
ACCENT_COLOUR = "#c9772e"
MIN_GROUP = 500


def _save(figure: plt.Figure, filename: str) -> str:
    settings.figures_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figures_dir / filename
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return path.name


def _rate_bar(
    table: pd.DataFrame, label_column: str, title: str, filename: str,
    base_rate: float | None = None, rotate: int = 0,
) -> str:
    """Standard default-rate-by-group bar chart with the portfolio baseline drawn in."""
    figure, axis = plt.subplots(figsize=(6.6, 4))
    colours = [
        RISK_COLOUR if rate > (base_rate or 0) else SAFE_COLOUR
        for rate in table["default_rate"]
    ]
    axis.bar(table[label_column].astype(str), table["default_rate"], color=colours)
    if base_rate is not None:
        axis.axhline(base_rate, ls="--", color="#444444", linewidth=1,
                     label=f"portfolio average {base_rate:.1%}")
        axis.legend(fontsize=8)
    axis.set(ylabel="Default rate", title=title)
    axis.tick_params(axis="x", labelrotation=rotate, labelsize=8)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    return _save(figure, filename)


# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
def target_summary(frame: pd.DataFrame) -> tuple[dict[str, Any], str]:
    """Class balance - the single fact that shapes every modelling choice."""
    counts = frame["TARGET"].value_counts().sort_index()
    base_rate = float(frame["TARGET"].mean())

    figure, axis = plt.subplots(figsize=(4.6, 3.8))
    axis.bar(["Repaid (0)", "Defaulted (1)"], counts.values,
             color=[SAFE_COLOUR, RISK_COLOUR])
    for position, value in enumerate(counts.values):
        axis.text(position, value, f"{value:,}\n{value / len(frame):.1%}",
                  ha="center", va="bottom", fontsize=9)
    axis.set(ylabel="Applicants", title="Class imbalance in the training data")
    axis.margins(y=0.18)
    filename = _save(figure, "eda_target_balance.png")

    return {
        "repaid": int(counts.get(0, 0)),
        "defaulted": int(counts.get(1, 0)),
        "default_rate": round(base_rate, 4),
        "imbalance_ratio": round(float(counts.get(0, 0) / max(counts.get(1, 1), 1)), 2),
    }, filename


def feature_categorisation(frame: pd.DataFrame) -> dict[str, Any]:
    """Columns bucketed into the business feature groups."""
    grouped = group_columns([c for c in frame.columns if c not in {"TARGET", "SK_ID_CURR"}])
    return {
        group: {"count": len(columns), "examples": columns[:5]}
        for group, columns in sorted(grouped.items(), key=lambda kv: -len(kv[1]))
    }


def data_quality_scorecard(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """One table: completeness and anomaly flags per feature group.

    The anomaly column is the point. A generic profiler reports "DAYS_EMPLOYED:
    0% missing, max 365243" and moves on; this flags it as the sentinel it is.
    """
    report = missing_report(frame)
    groups = group_columns(list(frame.columns))

    rows: list[dict[str, Any]] = []
    for group, columns in groups.items():
        subset = report.loc[[c for c in columns if c in report.index]]
        if subset.empty:
            continue
        rows.append({
            "feature_group": group,
            "columns": len(subset),
            "mean_completeness": round(float(1 - subset["missing_rate"].mean()), 4),
            "worst_completeness": round(float(1 - subset["missing_rate"].max()), 4),
            "columns_over_40pct_missing": int((subset["missing_rate"] > 0.4).sum()),
            "fully_populated_columns": int((subset["missing_rate"] == 0).sum()),
        })

    scorecard = pd.DataFrame(rows).sort_values("mean_completeness")

    anomalies = []
    if "DAYS_EMPLOYED" in frame:
        count = int((frame["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum())
        if count:
            anomalies.append(
                f"DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} on {count:,} rows "
                f"({count / len(frame):.1%}) - a 'not employed' sentinel, not 1000 years"
            )
    if "CODE_GENDER" in frame:
        count = int((frame["CODE_GENDER"] == "XNA").sum())
        if count:
            anomalies.append(f"CODE_GENDER = 'XNA' on {count} rows - unknown, not a third class")
    if "AMT_INCOME_TOTAL" in frame:
        maximum = float(frame["AMT_INCOME_TOTAL"].max())
        median = float(frame["AMT_INCOME_TOTAL"].median())
        anomalies.append(
            f"AMT_INCOME_TOTAL max is {maximum:,.0f} against a median of {median:,.0f} "
            f"({maximum / median:.0f}x) - a single extreme outlier that distorts every ratio"
        )
    if "OBS_30_CNT_SOCIAL_CIRCLE" in frame:
        count = int((frame["OBS_30_CNT_SOCIAL_CIRCLE"] > 100).sum())
        if count:
            anomalies.append(
                f"OBS_30_CNT_SOCIAL_CIRCLE exceeds 100 on {count} rows - implausible"
            )
    scorecard.attrs["anomalies"] = anomalies

    # Chart: the worst-populated columns, since 49 columns are >40% missing.
    worst = report.head(20)
    figure, axis = plt.subplots(figsize=(6.6, 6))
    axis.barh(worst.index[::-1], (worst["missing_rate"] * 100)[::-1], color=ACCENT_COLOUR)
    axis.set(xlabel="% missing", title="20 least-populated columns")
    axis.tick_params(axis="y", labelsize=7)
    filename = _save(figure, "eda_missingness.png")
    scorecard.attrs["figure"] = filename

    return scorecard, filename


# --------------------------------------------------------------------------- #
# The five business insights
# --------------------------------------------------------------------------- #
def insight_income(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """1. Default rate by income bracket."""
    working = frame[["AMT_INCOME_TOTAL", "TARGET"]].copy()
    working["income_bracket"] = pd.cut(
        working["AMT_INCOME_TOTAL"],
        bins=[0, 100_000, 150_000, 200_000, 300_000, np.inf],
        labels=["<100k", "100-150k", "150-200k", "200-300k", "300k+"],
    )
    table = (
        working.groupby("income_bracket", observed=True)
        .agg(applicants=("TARGET", "size"), default_rate=("TARGET", "mean"))
        .reset_index()
    )
    filename = _rate_bar(
        table, "income_bracket", "Default rate by income bracket",
        "eda_insight_income.png", base_rate=float(frame["TARGET"].mean()),
    )
    return table.round(4), filename


def insight_contract_type(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """2. Default rate by contract type (cash vs revolving)."""
    table = (
        frame.groupby("NAME_CONTRACT_TYPE", observed=True)
        .agg(applicants=("TARGET", "size"),
             default_rate=("TARGET", "mean"),
             avg_credit=("AMT_CREDIT", "mean"))
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    filename = _rate_bar(
        table, "NAME_CONTRACT_TYPE", "Default rate by contract type",
        "eda_insight_contract.png", base_rate=float(frame["TARGET"].mean()),
    )
    return table.round(4), filename


def insight_age(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """3. Default rate by age band, engineered from DAYS_BIRTH."""
    working = frame[["DAYS_BIRTH", "TARGET"]].copy()
    working["age_years"] = -working["DAYS_BIRTH"] / 365.0
    working["age_band"] = pd.cut(
        working["age_years"], bins=[0, 25, 30, 35, 40, 50, 60, 200],
        labels=["<25", "25-29", "30-34", "35-39", "40-49", "50-59", "60+"], right=False,
    )
    table = (
        working.groupby("age_band", observed=True)
        .agg(applicants=("TARGET", "size"), default_rate=("TARGET", "mean"))
        .reset_index()
    )
    filename = _rate_bar(
        table, "age_band", "Default rate by age band", "eda_insight_age.png",
        base_rate=float(frame["TARGET"].mean()),
    )
    return table.round(4), filename


def insight_external_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """4. Default rate across external credit-score deciles."""
    rows: list[dict[str, Any]] = []
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    colours = {"EXT_SOURCE_1": "#1f4e79", "EXT_SOURCE_2": "#c9772e", "EXT_SOURCE_3": "#2e7d32"}

    for column in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        if column not in frame:
            continue
        subset = frame[[column, "TARGET"]].dropna()
        subset = subset.assign(
            decile=pd.qcut(subset[column].rank(method="first"), 10, labels=False) + 1
        )
        grouped = subset.groupby("decile").agg(
            default_rate=("TARGET", "mean"), applicants=("TARGET", "size")
        )
        axis.plot(grouped.index, grouped["default_rate"], "o-",
                  color=colours[column], label=column, linewidth=1.8)
        rows.append({
            "score": column,
            "coverage": round(float(frame[column].notna().mean()), 4),
            "correlation_with_target": round(float(frame[column].corr(frame["TARGET"])), 4),
            "default_rate_worst_decile": round(float(grouped["default_rate"].iloc[0]), 4),
            "default_rate_best_decile": round(float(grouped["default_rate"].iloc[-1]), 4),
            "risk_ratio": round(
                float(grouped["default_rate"].iloc[0] / grouped["default_rate"].iloc[-1]), 2
            ),
        })

    axis.axhline(float(frame["TARGET"].mean()), ls="--", color="#444444", linewidth=1,
                 label="portfolio average")
    axis.set(xlabel="Score decile (1 = lowest score)", ylabel="Default rate",
             title="Default rate by external credit-score decile")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(fontsize=8)
    filename = _save(figure, "eda_insight_ext_source.png")
    return pd.DataFrame(rows), filename


def insight_employment(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """5. Default rate by employment length, with the 365243 sentinel isolated.

    Split out rather than binned in: those 55k rows are not long-tenured
    employees, they are applicants with no employment record, and they carry a
    materially different default rate. Averaging them into the top bucket is the
    single most common analysis error on this dataset.
    """
    working = frame[["DAYS_EMPLOYED", "TARGET"]].copy()
    sentinel = working["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    working["employment_years"] = np.where(
        sentinel, np.nan, -working["DAYS_EMPLOYED"] / 365.0
    )
    working["band"] = pd.cut(
        working["employment_years"], bins=[0, 1, 3, 5, 10, 100],
        labels=["<1y", "1-3y", "3-5y", "5-10y", "10y+"], right=False,
    ).astype(object)
    working.loc[sentinel, "band"] = "Not employed (sentinel)"

    order = ["<1y", "1-3y", "3-5y", "5-10y", "10y+", "Not employed (sentinel)"]
    table = (
        working.dropna(subset=["band"])
        .groupby("band", observed=True)
        .agg(applicants=("TARGET", "size"), default_rate=("TARGET", "mean"))
        .reindex([b for b in order if b in working["band"].unique()])
        .reset_index()
    )
    filename = _rate_bar(
        table, "band", "Default rate by employment length",
        "eda_insight_employment.png", base_rate=float(frame["TARGET"].mean()), rotate=20,
    )
    return table.round(4), filename


INSIGHTS = {
    "income_bracket": (insight_income, "Default risk falls as income rises, but the effect is "
                                       "far weaker than credit-bureau signal."),
    "contract_type": (insight_contract_type, "Cash loans default materially more often than "
                                             "revolving credit."),
    "age_band": (insight_age, "Default risk falls monotonically with age; under-25s are the "
                              "riskiest band in the portfolio."),
    "external_scores": (insight_external_scores, "External credit scores are the strongest "
                                                 "single predictor in the dataset."),
    "employment_length": (insight_employment, "Default risk falls with tenure - and the "
                                              "365243 sentinel group defaults LESS than "
                                              "employed applicants, because it is mostly "
                                              "pensioners. Binning it in would invert the "
                                              "trend it sits on."),
}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_full_eda(frame: pd.DataFrame | None = None, save: bool = True) -> dict[str, Any]:
    """Run every analysis, write the figures, and persist the summary JSON."""
    frame = load_dataset() if frame is None else frame
    settings.ensure_dirs()

    balance, balance_figure = target_summary(frame)
    scorecard, missing_figure = data_quality_scorecard(frame)

    insights: dict[str, Any] = {}
    for name, (function, headline) in INSIGHTS.items():
        table, figure = function(frame)
        insights[name] = {
            "headline": headline,
            "figure": figure,
            "table": table.to_dict(orient="records"),
        }
        logger.info("Insight '%s' -> %s", name, figure)

    summary = {
        "profile": dataset_profile(frame),
        "target": balance,
        "target_figure": balance_figure,
        "feature_groups": feature_categorisation(frame),
        "data_quality": {
            "scorecard": scorecard.to_dict(orient="records"),
            "anomalies": scorecard.attrs.get("anomalies", []),
            "figure": missing_figure,
        },
        "insights": insights,
    }

    if save:
        save_json(summary, settings.eda_summary_path)
    return summary


def load_summary() -> dict[str, Any] | None:
    from src.utils.helpers import load_json

    return load_json(settings.eda_summary_path)


if __name__ == "__main__":
    run_full_eda()
