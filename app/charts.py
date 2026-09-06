"""Altair chart builders styled to the dashboard design.

Altair rather than plotly: it ships inside Streamlit, so the charts add nothing
to `requirements.txt` or to the Docker image, and they stay interactive (hover
values) where the matplotlib figures written by the pipeline are static PNGs.

Both exist on purpose. The PNGs in `reports/figures/` are the reproducible
artifacts the notebook, the README and the PDF deck all cite; these are the
live, hoverable versions the app renders. The numbers behind them are the same.

Every chart shares `_base()` so axis weight, gridlines, fonts and padding are
defined once - the fastest way to make twelve charts read as one system.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.theme import AMBER, BLUE, BLUE_SOFT, FAINT, GREEN, INK, LINE, MUTED, RED

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
BAND_SCALE = alt.Scale(domain=["Low", "Medium", "High"], range=[GREEN, AMBER, RED])


def _configure(chart):
    """Shared look: light horizontal grid only, no chart border.

    Split out from `_base` because a FacetChart takes no `height` - that has to
    be set on the inner spec before faceting, and passing it at the top level
    raises SchemaValidationError only when the chart is serialised.
    """
    return (
        chart
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont=FONT, titleFont=FONT, labelColor=FAINT, titleColor=MUTED,
            labelFontSize=10.5, titleFontSize=10.5, titlePadding=8,
            domainColor=LINE, tickColor=LINE, gridColor="#f1f5f9", grid=False,
        )
        .configure_legend(
            labelFont=FONT, titleFont=FONT, labelColor=MUTED, titleColor=MUTED,
            labelFontSize=11, titleFontSize=11, symbolType="circle", symbolSize=90,
        )
        .configure_title(font=FONT, fontSize=12.5, color=INK, anchor="start")
    )


def _base(chart: alt.Chart, height: int) -> alt.Chart:
    return _configure(chart.properties(height=height))


def _pct_axis(title: str = "") -> alt.Axis:
    return alt.Axis(format=".0%", grid=True, title=title, tickCount=5)


# --------------------------------------------------------------------------- #
# Categorical rate bars - the workhorse of the EDA sections
# --------------------------------------------------------------------------- #
def rate_bars(
    frame: pd.DataFrame,
    category: str,
    value: str = "default_rate",
    height: int = 190,
    label: str = "Default rate",
    colour: str = BLUE,
    highlight_above: float | None = None,
    show_values: bool = True,
) -> alt.LayerChart:
    """Bar chart of a rate by category, optionally reddening bars above a baseline."""
    data = frame.copy()
    data[category] = data[category].astype(str)

    encode_colour = alt.value(colour)
    if highlight_above is not None:
        data["_above"] = data[value] > highlight_above
        encode_colour = alt.condition(
            alt.datum._above, alt.value(RED), alt.value(colour)
        )

    bars = alt.Chart(data).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=26,
    ).encode(
        x=alt.X(f"{category}:N", sort=None, axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y(f"{value}:Q", axis=_pct_axis(label)),
        color=encode_colour,
        tooltip=[
            alt.Tooltip(f"{category}:N", title="Group"),
            alt.Tooltip(f"{value}:Q", title=label, format=".2%"),
        ] + ([alt.Tooltip("applicants:Q", title="Applicants", format=",")]
             if "applicants" in data.columns else []),
    )

    layers = [bars]
    if show_values:
        layers.append(
            alt.Chart(data).mark_text(
                dy=-7, fontSize=10.5, font=FONT, color=INK, fontWeight=600,
            ).encode(
                x=alt.X(f"{category}:N", sort=None),
                y=alt.Y(f"{value}:Q"),
                text=alt.Text(f"{value}:Q", format=".1%"),
            )
        )
    if highlight_above is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [highlight_above]})).mark_rule(
                strokeDash=[4, 4], color=MUTED, strokeWidth=1,
            ).encode(y="y:Q")
        )
    return _base(alt.layer(*layers), height)


def grouped_bars(
    frame: pd.DataFrame, category: str, series: str, value: str,
    height: int = 210, label: str = "Rate", palette: list[str] | None = None,
) -> alt.FacetChart:
    """Per-group metrics, one small panel per metric.

    Faceted with an independent y-scale rather than grouped on a shared axis:
    ROC-AUC sits around 0.78 and the default rate around 0.08, so plotting both
    against one scale flattens the default-rate bars to invisible slivers. Two
    panels keep the comparison across groups - which is the fairness question -
    while letting each metric use its own range.
    """
    bars = alt.Chart().mark_bar(
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3, size=24,
    ).encode(
        x=alt.X(f"{category}:N", sort=None, axis=alt.Axis(title=None, labelAngle=0)),
        y=alt.Y(f"{value}:Q", axis=alt.Axis(grid=True, title=None, tickCount=4)),
        color=alt.Color(
            f"{series}:N",
            scale=alt.Scale(range=palette or [BLUE, "#93c5fd"]),
            legend=None,
        ),
        tooltip=[category, series, alt.Tooltip(f"{value}:Q", format=".3f")],
    )
    labels = alt.Chart().mark_text(
        dy=-7, fontSize=9.5, font=FONT, color=INK, fontWeight=600,
    ).encode(
        x=alt.X(f"{category}:N", sort=None),
        y=alt.Y(f"{value}:Q"),
        text=alt.Text(f"{value}:Q", format=".3f"),
    )
    chart = (
        alt.layer(bars, labels, data=frame)
        # Height belongs to the inner spec, and the panel has to be wide enough
        # for every category label or Vega silently drops alternate ones.
        .properties(height=height, width=max(130, 38 * frame[category].nunique()))
        .facet(column=alt.Column(
            f"{series}:N", title=None,
            header=alt.Header(labelFont=FONT, labelFontSize=11, labelColor=MUTED,
                              labelFontWeight=600)))
        .resolve_scale(y="independent")
    )
    return _configure(chart)


# --------------------------------------------------------------------------- #
# Donut - risk distribution
# --------------------------------------------------------------------------- #
def risk_donut(frame: pd.DataFrame, height: int = 210,
               centre_value: str = "", centre_label: str = "") -> alt.LayerChart:
    """Band distribution as a donut, with a value in the hole."""
    arc = alt.Chart(frame).mark_arc(
        innerRadius=58, outerRadius=84, cornerRadius=3, padAngle=0.012,
    ).encode(
        theta=alt.Theta("share:Q", stack=True),
        color=alt.Color(
            "band:N", scale=BAND_SCALE,
            legend=alt.Legend(title=None, orient="right", direction="vertical"),
        ),
        tooltip=[
            alt.Tooltip("band:N", title="Risk band"),
            alt.Tooltip("share:Q", title="Share", format=".1%"),
            alt.Tooltip("applicants:Q", title="Applicants", format=","),
        ],
    )
    text = alt.Chart(pd.DataFrame({"v": [centre_value], "l": [centre_label]}))
    centre = alt.layer(
        text.mark_text(fontSize=19, fontWeight=800, color=INK, font=FONT, dy=-8)
            .encode(text="v:N"),
        text.mark_text(fontSize=10.5, color=MUTED, font=FONT, dy=11).encode(text="l:N"),
    )
    return _base(alt.layer(arc, centre), height)


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def line_curve(
    frame: pd.DataFrame, x: str, y: str, height: int = 175,
    x_title: str = "", y_title: str = "", colour: str = BLUE,
    reference: pd.DataFrame | None = None, area: bool = True,
) -> alt.LayerChart:
    """A single curve with an optional dashed reference line (ROC / PR / cost)."""
    base = alt.Chart(frame)
    line = base.mark_line(color=colour, strokeWidth=2.2).encode(
        x=alt.X(f"{x}:Q", axis=alt.Axis(title=x_title, grid=False)),
        y=alt.Y(f"{y}:Q", axis=alt.Axis(title=y_title, grid=True, tickCount=5)),
        tooltip=[alt.Tooltip(f"{x}:Q", format=".3f"),
                 alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    layers = [line]
    if area:
        layers.insert(0, base.mark_area(color=colour, opacity=0.10).encode(
            x=f"{x}:Q", y=f"{y}:Q"))
    if reference is not None:
        layers.append(
            alt.Chart(reference).mark_line(
                color=FAINT, strokeDash=[4, 4], strokeWidth=1.4,
            ).encode(x=f"{x}:Q", y=f"{y}:Q")
        )
    return _base(alt.layer(*layers), height)


def calibration_curve(before: pd.DataFrame, after: pd.DataFrame,
                      height: int = 200) -> alt.LayerChart:
    """Reliability curve, raw against calibrated, with the perfect diagonal."""
    diagonal = pd.DataFrame({"predicted": [0, 1], "observed": [0, 1]})
    layers = [
        alt.Chart(diagonal).mark_line(
            color=FAINT, strokeDash=[4, 4], strokeWidth=1.3,
        ).encode(x="predicted:Q", y="observed:Q"),
    ]
    for frame, colour, name in ((before, AMBER, "raw (reweighted)"),
                                (after, BLUE, "isotonic-calibrated")):
        data = frame.assign(series=name)
        layers.append(
            alt.Chart(data).mark_line(point=alt.OverlayMarkDef(size=42), strokeWidth=2.2)
            .encode(
                x=alt.X("predicted:Q", axis=alt.Axis(title="Predicted probability",
                                                     grid=False)),
                y=alt.Y("observed:Q", axis=alt.Axis(title="Observed default rate",
                                                    grid=True)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(domain=["raw (reweighted)", "isotonic-calibrated"],
                                    range=[AMBER, BLUE]),
                    legend=alt.Legend(title=None, orient="top-left"),
                ),
                tooltip=[alt.Tooltip("predicted:Q", format=".3f"),
                         alt.Tooltip("observed:Q", format=".3f")],
            )
        )
    return _base(alt.layer(*layers), height)


def cost_curve(frame: pd.DataFrame, optimal: float, naive: float = 0.5,
               height: int = 205) -> alt.LayerChart:
    """Expected portfolio loss across thresholds, with both policies marked."""
    melted = frame.melt(
        id_vars="threshold",
        value_vars=["total_cost", "default_loss", "opportunity_loss"],
        var_name="component", value_name="cost",
    )
    labels = {"total_cost": "Total cost", "default_loss": "Credit loss",
              "opportunity_loss": "Forgone margin"}
    melted["component"] = melted["component"].map(labels)

    lines = alt.Chart(melted).mark_line(strokeWidth=2).encode(
        x=alt.X("threshold:Q", axis=alt.Axis(title="Decision threshold", grid=False)),
        y=alt.Y("cost:Q", axis=alt.Axis(title="Expected loss", format="~s", grid=True)),
        color=alt.Color(
            "component:N",
            scale=alt.Scale(domain=list(labels.values()), range=[BLUE, RED, GREEN]),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
        strokeDash=alt.condition(
            alt.datum.component == "Total cost", alt.value([1, 0]), alt.value([4, 3])
        ),
        tooltip=[alt.Tooltip("threshold:Q", format=".3f"),
                 alt.Tooltip("component:N"), alt.Tooltip("cost:Q", format=",.0f")],
    )
    marks = alt.Chart(pd.DataFrame({
        "threshold": [optimal, naive],
        "label": [f"optimum {optimal:.2f}", f"naive {naive:.2f}"],
    })).mark_rule(strokeWidth=1.2).encode(
        x="threshold:Q",
        color=alt.value(INK),
        strokeDash=alt.condition(
            alt.datum.threshold == optimal, alt.value([1, 0]), alt.value([2, 3])
        ),
        tooltip="label:N",
    )
    return _base(alt.layer(lines, marks), height)


def score_histogram(frame: pd.DataFrame, threshold: float,
                    height: int = 195) -> alt.LayerChart:
    """Calibrated score distribution, split by realised outcome."""
    hist = alt.Chart(frame).mark_bar(opacity=0.68).encode(
        x=alt.X("probability:Q", bin=alt.Bin(maxbins=52),
                axis=alt.Axis(title="Calibrated probability of default", format=".0%",
                              grid=False)),
        y=alt.Y("count()", stack=None, axis=alt.Axis(title="Applicants", grid=True)),
        color=alt.Color(
            "outcome:N",
            scale=alt.Scale(domain=["Repaid", "Defaulted"], range=[GREEN, RED]),
            legend=alt.Legend(title=None, orient="top-right"),
        ),
        tooltip=[alt.Tooltip("count()", title="Applicants", format=",")],
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
        color=INK, strokeWidth=1.3,
    ).encode(x="t:Q", tooltip=alt.value(f"threshold {threshold:.3f}"))
    return _base(alt.layer(hist, rule), height)


def contribution_bars(frame: pd.DataFrame, height: int = 260) -> alt.Chart:
    """Signed SHAP contributions for one applicant, strongest first."""
    chart = alt.Chart(frame).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=16,
    ).encode(
        y=alt.Y("label:N", sort=None, axis=alt.Axis(title=None, labelLimit=190)),
        x=alt.X("shap:Q", axis=alt.Axis(title="SHAP contribution (log-odds)", grid=True)),
        color=alt.condition(alt.datum.shap > 0, alt.value(RED), alt.value(GREEN)),
        tooltip=[
            alt.Tooltip("label:N", title="Feature"),
            alt.Tooltip("display_value:N", title="Value"),
            alt.Tooltip("shap:Q", title="SHAP", format="+.3f"),
            alt.Tooltip("share:Q", title="Share of attribution", format=".1%"),
        ],
    )
    return _base(chart, height)


def importance_bars(frame: pd.DataFrame, value: str = "gain",
                    height: int = 300, colour: str = BLUE) -> alt.Chart:
    """Horizontal ranking - feature importance, rule lift, and similar."""
    chart = alt.Chart(frame).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=15,
    ).encode(
        y=alt.Y("label:N", sort="-x", axis=alt.Axis(title=None, labelLimit=210)),
        x=alt.X(f"{value}:Q", axis=alt.Axis(title=None, grid=True, format="~s")),
        color=alt.value(colour),
        tooltip=["label:N", alt.Tooltip(f"{value}:Q", format=",.4f")],
    )
    return _base(chart, height)


def missingness_bars(frame: pd.DataFrame, height: int = 300) -> alt.Chart:
    chart = alt.Chart(frame).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=13,
    ).encode(
        y=alt.Y("column:N", sort="-x", axis=alt.Axis(title=None, labelLimit=230)),
        x=alt.X("missing_rate:Q", axis=alt.Axis(title="Missing", format=".0%", grid=True)),
        color=alt.value(AMBER),
        tooltip=["column:N", alt.Tooltip("missing_rate:Q", format=".1%")],
    )
    return _base(chart, height)


def psi_bars(frame: pd.DataFrame, height: int = 220) -> alt.LayerChart:
    bars = alt.Chart(frame).mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=15,
    ).encode(
        y=alt.Y("feature:N", sort="-x", axis=alt.Axis(title=None, labelLimit=210)),
        x=alt.X("psi:Q", axis=alt.Axis(title="Population stability index", grid=True)),
        color=alt.value(GREEN),
        tooltip=["feature:N", alt.Tooltip("psi:Q", format=".5f"), "verdict:N"],
    )
    thresholds = alt.Chart(pd.DataFrame({
        "x": [0.1, 0.25], "label": ["0.10 moderate", "0.25 material"],
    })).mark_rule(strokeDash=[4, 4], strokeWidth=1).encode(
        x="x:Q",
        color=alt.condition(alt.datum.x == 0.1, alt.value(AMBER), alt.value(RED)),
        tooltip="label:N",
    )
    return _base(alt.layer(bars, thresholds), height)


def decile_bars(frame: pd.DataFrame, height: int = 200) -> alt.LayerChart:
    bars = alt.Chart(frame).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=22,
    ).encode(
        x=alt.X("decile:O", axis=alt.Axis(title="Risk decile (1 = riskiest)",
                                          labelAngle=0)),
        y=alt.Y("default_rate:Q", axis=_pct_axis("Default rate")),
        color=alt.value(BLUE),
        tooltip=[
            alt.Tooltip("decile:O"),
            alt.Tooltip("default_rate:Q", title="Default rate", format=".2%"),
            alt.Tooltip("lift:Q", title="Lift", format=".2f"),
            alt.Tooltip("applicants:Q", title="Applicants", format=","),
        ],
    )
    return _base(alt.layer(bars), height)


def sql_result_bars(frame: pd.DataFrame, category: str, value: str,
                    height: int = 175) -> alt.LayerChart | None:
    """Best-effort chart of a Talk-to-Data result set.

    Returns None when the result is not chartable (one row, or no numeric
    column), so the caller can fall back to the table alone rather than
    rendering something meaningless.
    """
    if frame.empty or len(frame) < 2 or category not in frame or value not in frame:
        return None
    data = frame.head(12).copy()
    data[category] = data[category].astype(str)
    is_rate = bool(data[value].between(0, 1).all())

    bars = alt.Chart(data).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24,
    ).encode(
        x=alt.X(f"{category}:N", sort=None, axis=alt.Axis(title=None, labelAngle=0,
                                                          labelLimit=90)),
        y=alt.Y(f"{value}:Q", axis=alt.Axis(
            title=None, grid=True, format=".0%" if is_rate else "~s")),
        color=alt.value(BLUE),
        tooltip=[category, alt.Tooltip(f"{value}:Q",
                                       format=".2%" if is_rate else ",.0f")],
    )
    labels = alt.Chart(data).mark_text(
        dy=-7, fontSize=10, font=FONT, color=INK, fontWeight=600,
    ).encode(
        x=alt.X(f"{category}:N", sort=None), y=alt.Y(f"{value}:Q"),
        text=alt.Text(f"{value}:Q", format=".1%" if is_rate else ",.0f"),
    )
    return _base(alt.layer(bars, labels), height)
