"""Chart theme: one palette, one Plotly template, applied everywhere.

Colour assignment follows the job each scale does:

* **Categorical** (language, intent, queue) — fixed hue order, never cycled. A
  language keeps its hue when a filter removes another language, so `en` is
  always blue and `hi` is always orange across every view.
* **Sequential** (compliance adherence heatmap) — a single blue hue, light→dark.
* **Diverging** (sentiment, -1..+1) — blue↔red with a neutral gray midpoint, so
  "no sentiment" reads as nothing rather than as a colour.
* **Status** (risk bands, compliance outcomes) — a reserved four-colour scale
  that is never reused for a data series, and is always paired with a text label
  so state is not carried by colour alone.

The four categorical slots were validated with the palette checker in both
modes. Light mode returns a contrast WARN for the aqua and yellow slots against
the light surface; the required relief is shipped — every chart using them
carries direct labels, and every view has a "Show data table" toggle.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SEQUENTIAL_BLUE = [
    [0.00, "#cde2fb"], [0.17, "#b7d3f6"], [0.33, "#9ec5f4"], [0.50, "#6da7ec"],
    [0.67, "#3987e5"], [0.83, "#256abf"], [1.00, "#0d366b"],
]

# Diverging: blue (negative) -> neutral gray -> red (positive) is inverted for
# sentiment, where negative must read as the alarming pole.
DIVERGING_SENTIMENT = [
    [0.00, "#d03b3b"], [0.25, "#e88a8a"], [0.50, "#f0efec"],
    [0.75, "#86b6ef"], [1.00, "#1c5cab"],
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

RISK_BAND_COLOR = {
    "LOW": STATUS["good"],
    "MEDIUM": STATUS["warning"],
    "HIGH": STATUS["serious"],
    "CRITICAL": STATUS["critical"],
}
RISK_BAND_ICON = {"LOW": "●", "MEDIUM": "▲", "HIGH": "◆", "CRITICAL": "■"}

COMPLIANCE_OUTCOME_COLOR = {
    "PRESENT": STATUS["good"],
    "PARTIAL": STATUS["warning"],
    "ABSENT": STATUS["serious"],
    "VIOLATED": STATUS["critical"],
    "NOT_APPLICABLE": INK_MUTED,
}

# Language keeps a fixed slot across every chart in the app.
LANGUAGE_COLOR = {
    "en": CATEGORICAL[0],
    "hi": CATEGORICAL[1],
    "hi_latn": CATEGORICAL[2],
    "mixed": CATEGORICAL[3],
    "und": INK_MUTED,
}
LANGUAGE_LABEL = {
    "en": "English",
    "hi": "Hindi (Devanagari)",
    "hi_latn": "Hindi (romanised)",
    "mixed": "Code-switched",
    "und": "Unknown",
}

SPEAKER_COLOR = {"AGENT": CATEGORICAL[0], "CUSTOMER": CATEGORICAL[1], "IVR": INK_MUTED, "UNKNOWN": INK_MUTED}


def register_template() -> None:
    """Install the shared template so no individual chart restyles axes."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "size": 13, "color": INK_SECONDARY},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=CATEGORICAL,
        title={"font": {"size": 15, "color": INK_PRIMARY}, "x": 0, "xanchor": "left"},
        margin={"l": 56, "r": 24, "t": 48, "b": 44},
        xaxis={
            "gridcolor": GRID, "gridwidth": 1, "zeroline": False,
            "linecolor": AXIS, "ticks": "outside", "tickcolor": AXIS,
            "tickfont": {"color": INK_MUTED, "size": 12},
            "title": {"font": {"color": INK_SECONDARY, "size": 12}},
        },
        yaxis={
            "gridcolor": GRID, "gridwidth": 1, "zeroline": False,
            "linecolor": AXIS, "ticks": "outside", "tickcolor": AXIS,
            "tickfont": {"color": INK_MUTED, "size": 12},
            "title": {"font": {"color": INK_SECONDARY, "size": 12}},
        },
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "left", "x": 0,
            "font": {"color": INK_SECONDARY, "size": 12},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": SURFACE, "bordercolor": AXIS,
            "font": {"color": INK_PRIMARY, "size": 12},
        },
        hovermode="closest",
    )
    pio.templates["ccma"] = template
    pio.templates.default = "ccma"


CSS = f"""
<style>
  .stApp {{ background: {PAGE}; }}
  .ccma-tile {{
      background: {SURFACE};
      border: 1px solid rgba(11,11,11,0.10);
      border-radius: 10px;
      padding: 14px 16px;
      height: 100%;
  }}
  .ccma-tile-label {{
      color: {INK_SECONDARY}; font-size: 12px; font-weight: 600;
      letter-spacing: .02em; text-transform: uppercase; margin-bottom: 6px;
  }}
  .ccma-tile-value {{
      color: {INK_PRIMARY}; font-size: 30px; font-weight: 650; line-height: 1.1;
  }}
  .ccma-tile-sub {{ color: {INK_MUTED}; font-size: 12px; margin-top: 6px; }}
  .ccma-caveat {{
      color: {INK_MUTED}; font-size: 11.5px; font-style: italic;
      border-left: 2px solid {AXIS}; padding-left: 8px; margin-top: 4px;
  }}
  .ccma-turn {{
      background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08);
      border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
  }}
  .ccma-turn-meta {{ font-size: 11.5px; color: {INK_MUTED}; margin-bottom: 4px; }}
  .ccma-turn-text {{ font-size: 14px; color: {INK_PRIMARY}; line-height: 1.5; }}
  .ccma-turn-deva {{ font-size: 13px; color: {INK_SECONDARY}; margin-top: 4px; }}
  .ccma-chip {{
      display: inline-block; padding: 1px 7px; border-radius: 999px;
      font-size: 11px; font-weight: 600; margin-right: 6px;
  }}
</style>
"""


def tile(label: str, value: str, sub: str = "", caveat: str = "") -> str:
    caveat_html = f'<div class="ccma-caveat">{caveat}</div>' if caveat else ""
    sub_html = f'<div class="ccma-tile-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="ccma-tile"><div class="ccma-tile-label">{label}</div>'
        f'<div class="ccma-tile-value">{value}</div>{sub_html}{caveat_html}</div>'
    )


def chip(text: str, color: str) -> str:
    return f'<span class="ccma-chip" style="background:{color}22;color:{color};">{text}</span>'
