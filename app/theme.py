"""Design system for the dashboard: tokens, CSS, and HTML component helpers.

Streamlit's default look is a document; the design this implements is a product
dashboard - a light blue-grey ground, white rounded cards, a left nav rail, and
dense KPI tiles. Getting there means overriding a fair amount of Streamlit
chrome, so all of it lives here rather than being sprinkled through the pages.

Two rules kept throughout:

* **Components return HTML strings**, so a page composes them with one
  `st.markdown(..., unsafe_allow_html=True)` instead of many. Fewer wrapper divs
  from Streamlit means the CSS grid actually behaves.
* **Every number rendered here comes from the caller**, never from a literal.
  The mock-up this design follows carried placeholder figures (ROC-AUC 0.843,
  a 14.7% score); the real model's numbers are different and are what ship.
"""

from __future__ import annotations

import contextlib
from html import escape
from urllib.parse import quote

import streamlit as st

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
INK = "#0f172a"          # headings
BODY = "#334155"         # body copy
MUTED = "#64748b"        # captions, labels
FAINT = "#94a3b8"        # axis text
NAVY = "#0f2b46"         # dark buttons, hero, promo panel
BLUE = "#2563eb"         # primary accent, bars, links
BLUE_SOFT = "#dbeafe"    # tinted fills
BLUE_RAIL = "#eff6ff"    # active nav pill
GREEN = "#16a34a"
AMBER = "#f59e0b"
RED = "#dc2626"
LINE = "#e2e8f0"         # borders
PAGE = "#eef2f7"         # page ground
CARD = "#ffffff"

BAND_COLOUR = {"Low": GREEN, "Medium": AMBER, "High": RED}
BAND_TINT = {"Low": "#dcfce7", "Medium": "#fef3c7", "High": "#fee2e2"}

# Icons are inline SVG rather than emoji: emoji render at wildly different
# sizes across platforms, which wrecks a tile grid.
ICONS = {
    "home": "M3 10.5 12 3l9 7.5M5 9.5V20h14V9.5",
    "predict": "M4 19h16M7 16V9M12 16V5M17 16v-4",
    "chat": "M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.5A8 8 0 1 1 21 12Z",
    "eda": "M4 20V4M4 20h16M8 16V9M13 16v-5M18 16V6",
    "performance": "M4 17l5-5 3 3 7-7M14 8h6v6",
    "rules": "M6 4h9l4 4v12H6zM14 4v5h5M9 13h7M9 17h5",
    "health": "M12 21s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.6-7 10-7 10Z",
    "about": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 11v5M12 8h.01",
    "applicants": "M16 20v-1a4 4 0 0 0-8 0v1M12 11a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z",
    "rate": "M3 17l5.5-5.5 3 3L21 5M21 5h-5M21 5v5",
    "auc": "M5 20V9M12 20V4M19 20v-7",
    "features": "M12 3v12M12 3 8 7M12 3l4 4M4 17v3h16v-3",
    "modules": "M12 3 3 8l9 5 9-5-9-5ZM3 13l9 5 9-5M3 18l9 5 9-5",
    "shield": "M12 3 5 6v6c0 4.5 3 7.6 7 9 4-1.4 7-4.5 7-9V6l-7-3Z",
    "spark": "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18",
    "scale": "M12 3v18M7 7l-4 7h8zM17 7l-4 7h8zM6 21h12",
    "globe": "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM3 12h18M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18Z",
    "download": "M12 3v10m0 0 4-4m-4 4-4-4M4 17v3h16v-3",
    "book": "M4 5a2 2 0 0 1 2-2h12v18H6a2 2 0 0 1-2-2zM8 7h7M8 11h7",
    "search": "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14ZM20 20l-4-4",
    "bell": "M6 9a6 6 0 1 1 12 0c0 5 2 6 2 6H4s2-1 2-6M10 20a2 2 0 0 0 4 0",
    "check": "M20 6 9 17l-5-5",
    "arrow": "M5 12h14M13 6l6 6-6 6",
}


def icon(name: str, size: int = 18, colour: str = "currentColor",
         width: float = 1.7) -> str:
    """Inline SVG icon. Returns markup, not a Streamlit element."""
    path = ICONS.get(name, ICONS["about"])
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{colour}" stroke-width="{width}" stroke-linecap="round" '
        f'stroke-linejoin="round" class="ic"><path d="{path}"/></svg>'
    )


def icon_data_uri(name: str, colour: str) -> str:
    """The same icon as a CSS-usable data URI, for ::before pseudo-elements."""
    path = ICONS.get(name, ICONS["about"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        f'viewBox="0 0 24 24" fill="none" stroke="{colour}" stroke-width="1.7" '
        f'stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>'
    )
    return f'url("data:image/svg+xml,{quote(svg)}")'


# Nav rail icons, in the order the radio renders them.
NAV_ICONS = ["home", "predict", "shield", "chat", "eda", "performance",
             "rules", "health", "about"]


def _nav_icon_rules() -> str:
    """One ::before rule per nav item.

    Streamlit's radio labels take plain text, so the icons are attached by
    position with nth-of-type rather than injected into the label markup.
    """
    rules = []
    for index, name in enumerate(NAV_ICONS, start=1):
        base = (f'section[data-testid="stSidebar"] [role="radiogroup"] '
                f"label:nth-of-type({index})")
        rules.append(
            f'{base} [data-testid="stMarkdownContainer"]::before {{'
            f"content:''; width:18px; height:18px; margin-right:10px; flex:none;"
            f"background-image:{icon_data_uri(name, MUTED)};"
            f"background-repeat:no-repeat; background-size:18px; }}"
        )
        rules.append(
            f'{base}:has(input:checked) [data-testid="stMarkdownContainer"]::before {{'
            f"background-image:{icon_data_uri(name, BLUE)}; }}"
        )
    return "\n".join(rules)


# The exact ancestor chain from a card container to its marker span. Pinning
# `:first-child` keeps an outer column from matching a card nested inside it -
# Streamlit gives plain columns the same testid as bordered containers.
_CARD_HAS = (
    ':has(> div > [data-testid="stVerticalBlock"] > '
    '[data-testid="stElementContainer"]:first-child .cardmark)'
)
CARD_SELECTOR = f'[data-testid="stVerticalBlockBorderWrapper"]{_CARD_HAS}'

# Emitted as the first element of every card container; the CSS above keys
# on it to tell a real card from a plain Streamlit column.
CARD_MARK = '<span class="cardmark"></span>'


@contextlib.contextmanager
def card(title: str = "", subtitle: str = "", title_size: int = 17):
    """A bordered container that reliably picks up the card styling.

    The marker span must be the container's first element - that is what the
    CSS keys on, and what stops an enclosing column from being painted as a card
    too.
    """
    box = st.container(border=True)
    with box:
        head = '<span class="cardmark"></span>'
        if title:
            head += (f'<div class="card-title" style="font-size:{title_size}px">'
                     f"{escape(title)}</div>")
        if subtitle:
            head += f'<div class="card-sub">{escape(subtitle)}</div>'
        st.markdown(head, unsafe_allow_html=True)
        yield box


def svg_gauge(probability: float, threshold: float, band: str,
              size: int = 190) -> str:
    """Half-ring gauge for one applicant's probability of default.

    Drawn as SVG rather than as a chart because the geometry has to be exact:
    the decision threshold sits at the centre of the arc, so the visual question
    is 'which side of the line is this applicant on', not 'what fraction of 100%
    is this'. On a 0-1 scale an 8%-base-rate portfolio renders as slivers.
    """
    ceiling = max(threshold * 2, 1e-6)
    fraction = min(max(probability / ceiling, 0.0), 1.0)
    colour = BAND_COLOUR.get(band, BLUE)

    radius = 74
    circumference = 3.14159 * radius            # half circle
    filled = circumference * fraction
    width, height = 190, 116
    centre_x, centre_y = width / 2, 100
    arc = (f"M {centre_x - radius} {centre_y} "
           f"A {radius} {radius} 0 0 1 {centre_x + radius} {centre_y}")

    # Tick at the threshold, which sits exactly halfway along the arc.
    return f"""
<div class="gauge" style="width:{size}px">
 <svg viewBox="0 0 {width} {height}" width="100%">
   <path d="{arc}" fill="none" stroke="#eef2f7" stroke-width="15"
         stroke-linecap="round"/>
   <path d="{arc}" fill="none" stroke="{colour}" stroke-width="15"
         stroke-linecap="round"
         stroke-dasharray="{filled:.2f} {circumference:.2f}"/>
   <line x1="{centre_x}" y1="{centre_y - radius - 9}" x2="{centre_x}"
         y2="{centre_y - radius + 9}" stroke="{INK}" stroke-width="2"/>
   <text x="{centre_x}" y="{centre_y - 22}" text-anchor="middle"
         font-size="27" font-weight="800" fill="{INK}"
         font-family="Inter, sans-serif">{probability * 100:.1f}%</text>
   <text x="{centre_x}" y="{centre_y - 4}" text-anchor="middle" font-size="10"
         fill="{MUTED}" font-family="Inter, sans-serif">probability of default</text>
 </svg>
 <div class="gauge-scale">
   <span>0%</span><span>decline at {threshold * 100:.1f}%</span>
   <span>{ceiling * 100:.0f}%</span>
 </div>
</div>
"""


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
def stylesheet() -> str:
    nav_icons = _nav_icon_rules()
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
  --ink:{INK}; --body:{BODY}; --muted:{MUTED}; --faint:{FAINT};
  --navy:{NAVY}; --blue:{BLUE}; --blue-soft:{BLUE_SOFT}; --rail:{BLUE_RAIL};
  --green:{GREEN}; --amber:{AMBER}; --red:{RED};
  --line:{LINE}; --page:{PAGE}; --card:{CARD};
  --radius:16px;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10);
}}

html, body, [class*="css"], .stApp {{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}}
.stApp {{ background:var(--page); }}

/* Strip Streamlit chrome that fights the layout. */
#MainMenu, footer, header[data-testid="stHeader"] {{ display:none !important; }}
[data-testid="stDecoration"] {{ display:none !important; }}
.block-container {{ padding:1.15rem 1.6rem 3rem; max-width:1520px; }}
[data-testid="stVerticalBlock"] {{ gap:0.85rem; }}
[data-testid="stElementContainer"]:has(> .stHtml:empty) {{ display:none; }}

h1,h2,h3,h4 {{ color:var(--ink); letter-spacing:-.02em; font-weight:700; }}
p, li, span, label {{ color:var(--body); }}
a {{ color:var(--blue); text-decoration:none; }}
.ic {{ vertical-align:-3px; }}

/* ---------------- top bar ---------------- */
.topbar {{
  display:flex; align-items:center; gap:18px; padding:10px 18px; margin-bottom:14px;
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  box-shadow:var(--shadow);
}}
.topbar .brand {{ display:flex; align-items:center; gap:9px; font-weight:800;
  color:var(--ink); font-size:14.5px; letter-spacing:-.02em; white-space:nowrap; }}
.topbar .brand .dot {{ width:26px; height:26px; border-radius:8px; background:var(--navy);
  display:grid; place-items:center; color:#fff; }}
.topbar-title {{ flex:1; border-left:1px solid var(--line); padding-left:18px; }}
.topbar-title .t {{ font-size:14px; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; }}
.topbar-sub {{ font-size:11.5px; color:var(--muted); margin-top:1px; }}
.topbar .avatar {{ display:flex; align-items:center; gap:9px; white-space:nowrap; }}
.topbar .avatar .circle {{ width:30px; height:30px; border-radius:50%;
  background:var(--navy); color:#fff; display:grid; place-items:center;
  font-size:12.5px; font-weight:700; }}
.topbar .avatar span {{ font-size:13px; color:var(--ink); font-weight:600; }}

/* ---------------- cards ---------------- */
.card {{
  background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
  padding:20px 22px; box-shadow:var(--shadow); height:100%;
}}
.card.tight {{ padding:16px 18px; }}
.card-title {{ font-size:17px; font-weight:700; color:var(--ink); margin:0 0 3px;
  letter-spacing:-.02em; }}
.card-sub {{ font-size:12.5px; color:var(--muted); margin:0 0 14px; line-height:1.5; }}
.card-head {{ display:flex; align-items:flex-start; justify-content:space-between;
  gap:14px; }}

/* ---------------- hero ---------------- */
.hero {{
  position:relative; overflow:hidden; border-radius:var(--radius);
  border:1px solid var(--line); box-shadow:var(--shadow); padding:34px 36px 26px;
  background:
    radial-gradient(120% 130% at 88% 12%, rgba(37,99,235,.20) 0%, rgba(37,99,235,0) 55%),
    radial-gradient(90% 120% at 96% 96%, rgba(245,158,11,.20) 0%, rgba(245,158,11,0) 52%),
    linear-gradient(140deg,#ffffff 0%,#f4f8ff 48%,#eaf1fe 100%);
}}
.hero:after {{
  content:""; position:absolute; right:-70px; top:-60px; width:340px; height:340px;
  border-radius:50%; border:26px solid rgba(37,99,235,.07);
}}
.hero .eyebrow {{ font-size:11px; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--blue); margin-bottom:14px; }}
.hero h1 {{ font-size:clamp(26px,2.6vw,38px); line-height:1.15; margin:0 0 14px;
  font-weight:800; max-width:16ch; }}
.hero p {{ font-size:14.5px; color:var(--body); max-width:46ch; line-height:1.62;
  margin:0 0 22px; }}
.hero .btns {{ display:flex; gap:12px; flex-wrap:wrap; }}
.hero .btn-primary {{ background:var(--navy); color:#fff; font-weight:600; font-size:13.5px;
  padding:11px 22px; border-radius:10px; display:inline-flex; align-items:center; gap:9px; }}
.hero .btn-ghost {{ background:#fff; color:var(--ink); font-weight:600; font-size:13.5px;
  padding:11px 22px; border-radius:10px; border:1px solid var(--line);
  display:inline-flex; align-items:center; gap:9px; }}


/* ---------------- warm landing hero ---------------- */
.whero {{
  position:relative; display:grid;
  grid-template-columns:minmax(380px,1fr) minmax(360px,420px) minmax(0,180px);
  align-items:center; gap:0; overflow:hidden;
  border-radius:18px; border:1px solid #e7ded0; box-shadow:var(--shadow);
  background:linear-gradient(115deg,#faf6ef 0%,#f4ece0 46%,#eee3d3 100%);
}}
@media (max-width:1250px) {{
  .whero {{ grid-template-columns:minmax(320px,1fr) minmax(320px,380px); }}
  .whero-values {{ display:none; }}
}}
.whero-left {{ min-width:0; padding:38px 26px 38px 42px; }}
.whero-eyebrow {{ font-size:10.5px; font-weight:700; letter-spacing:.2em;
  text-transform:uppercase; color:#8a7a63; margin-bottom:18px; }}
.whero h1 {{ font-size:clamp(30px,3.1vw,44px); line-height:1.08; margin:0 0 18px;
  font-weight:800; color:#1c1a17 !important; letter-spacing:-.035em; max-width:15ch; }}
.whero p {{ font-size:13.5px; line-height:1.7; color:#57503f !important; margin:0;
  max-width:52ch; }}
.whero-tagline {{ display:flex; align-items:center; gap:14px; margin-top:26px;
  font-size:13.5px; color:#3c352a; font-weight:500; }}
.whero-tagline i {{ display:block; width:46px; height:1.5px; background:#3c352a;
  position:relative; }}
.whero-tagline i:after {{ content:""; position:absolute; right:0; top:-3px;
  width:7px; height:7px; border-top:1.5px solid #3c352a;
  border-right:1.5px solid #3c352a; transform:rotate(45deg); }}

.whero-right {{ position:relative; min-width:0; padding:26px 0; }}
.whero-values {{ min-width:0; padding:26px 30px 26px 30px; margin-left:4px;
  border-left:1px solid rgba(60,53,42,.16); }}
.whero-words {{ display:flex; flex-direction:column; gap:7px; }}
.whero-words span {{ font-size:12px; font-weight:600; letter-spacing:.18em;
  text-transform:uppercase; color:#6b6152; }}
.whero-rule {{ width:52px; height:1px; background:rgba(60,53,42,.3); margin:20px 0 16px; }}
.whero-promise {{ font-size:14px; font-weight:700; letter-spacing:.05em;
  line-height:1.5; text-transform:uppercase; color:#2c261d; max-width:11ch; }}

/* ---------------- sample prediction card ---------------- */
.sample-card {{ background:#fff; border:1px solid #ece4d7;
  border-radius:16px; padding:18px 20px 16px;
  box-shadow:0 22px 46px -24px rgba(60,53,42,.45); }}
.sc-head {{ display:flex; align-items:center; justify-content:space-between;
  margin-bottom:2px; }}
.sc-title {{ font-size:14px; font-weight:700; color:var(--ink); }}
.sc-pill {{ font-size:11px; font-weight:700; padding:4px 11px; border-radius:999px;
  white-space:nowrap; }}
.sc-delta {{ text-align:center; font-size:11.5px; font-weight:600; margin:-2px 0 12px; }}
.sc-rows {{ border-top:1px solid var(--line); padding-top:4px; }}
.sc-row {{ display:flex; align-items:center; gap:10px; padding:9px 0;
  border-bottom:1px solid #f1f5f9; }}
.sc-row:last-child {{ border-bottom:none; }}
.sc-ic {{ width:26px; height:26px; border-radius:8px; background:var(--rail);
  display:grid; place-items:center; flex:none; }}
.sc-label {{ flex:1; font-size:12px; color:var(--muted); }}
.sc-value {{ font-size:12.5px; font-weight:700; color:var(--ink); }}
.sc-link {{ font-size:12px; font-weight:600; color:var(--blue); white-space:nowrap; }}

/* ---------------- KPI strip ---------------- */
.kstrip {{ display:grid; background:var(--card); border:1px solid var(--line);
  border-radius:14px; box-shadow:var(--shadow); overflow:hidden; }}
.kstrip-item {{ display:flex; align-items:center; gap:13px; padding:16px 20px;
  border-right:1px solid var(--line); min-width:0; }}
.kstrip-item:last-child {{ border-right:none; }}
.kstrip-ic {{ width:38px; height:38px; border-radius:11px; background:var(--rail);
  color:var(--blue); display:grid; place-items:center; flex:none; }}
.kstrip-item .v {{ font-size:21px; font-weight:800; color:var(--ink);
  letter-spacing:-.03em; line-height:1.15; }}
.kstrip-item .l {{ font-size:11.5px; color:var(--muted); margin-top:2px;
  line-height:1.35; }}

/* ---------------- pipeline stages ---------------- */
.stages {{ position:relative; display:grid;
  grid-template-columns:repeat(7,minmax(0,1fr)); gap:10px; padding-top:6px; }}
.stage-rail {{ position:absolute; top:23px; left:7%; right:7%; height:2px;
  background:linear-gradient(90deg,#dbeafe,#2563eb 45%,#2563eb 55%,#dbeafe); }}
.stage {{ position:relative; text-align:center; padding-top:0; }}
.stage-num {{ width:32px; height:32px; margin:0 auto 12px; border-radius:50%;
  background:#fff; border:2px solid var(--blue); color:var(--blue); font-size:12.5px;
  font-weight:800; display:grid; place-items:center; position:relative; z-index:1; }}
.stage-icon {{ display:grid; place-items:center; color:var(--faint); margin-bottom:6px; }}
.stage-title {{ font-size:12.5px; font-weight:700; color:var(--ink); line-height:1.25; }}
.stage-detail {{ font-size:10.5px; color:var(--muted); line-height:1.4; margin-top:4px; }}

/* ---------------- section tiles ---------------- */
.tile {{ border-left:3px solid var(--accent); padding-left:13px; margin:-2px 0 2px; }}
.tile-head {{ display:flex; align-items:center; gap:9px; margin-bottom:7px; }}
.tile-icon {{ width:30px; height:30px; border-radius:9px; display:grid;
  place-items:center; flex:none; background:color-mix(in srgb, var(--accent) 12%, #fff); }}
.tile-title {{ font-size:14px; font-weight:700; color:var(--ink);
  letter-spacing:-.01em; }}
.tile-blurb {{ font-size:12.5px; color:var(--muted); line-height:1.5;
  min-height:32px; }}

/* ---------------- markdown tables ---------------- */
/* Untouched, these inherited nothing and rendered as pale grey on pale grey -
   which is what made the assistant's answer tables look empty. */
.stMarkdown table {{ width:100%; border-collapse:collapse; font-size:12.5px;
  margin:10px 0; border:1px solid var(--line); border-radius:10px;
  overflow:hidden; }}
.stMarkdown thead th {{ background:#f1f5f9; color:var(--ink) !important;
  font-weight:700; text-align:left; padding:9px 12px; font-size:11.5px;
  border-bottom:1px solid var(--line); }}
.stMarkdown tbody td {{ padding:8px 12px; color:var(--body) !important;
  border-bottom:1px solid #eef2f7; vertical-align:top; }}
.stMarkdown tbody tr:last-child td {{ border-bottom:none; }}
.stMarkdown tbody tr:nth-child(even) {{ background:#fcfdff; }}
.stMarkdown tbody td:first-child {{ font-weight:600; color:var(--ink) !important; }}

/* ---------------- chat ---------------- */
[data-testid="stChatMessage"] {{ padding:12px 14px; gap:11px;
  background:transparent; border-radius:12px; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
  background:#f1f5f9; }}
[data-testid="stChatMessage"] p {{ margin-bottom:.35rem; line-height:1.55; }}
[data-testid="stChatMessage"] ul {{ margin:.3rem 0 .5rem; padding-left:1.15rem; }}
[data-testid="stChatMessage"] li {{ margin-bottom:.15rem; }}
[data-testid="stChatMessage"] [data-testid="stVerticalBlock"] {{ gap:.35rem; }}
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3, [data-testid="stChatMessage"] h4 {{
  font-size:14px; margin:.4rem 0 .25rem; }}

/* ---------------- architecture diagram ---------------- */
.arch {{ display:flex; flex-direction:column; gap:0; }}
.arch-layer {{ display:flex; align-items:center; gap:14px; padding:11px 14px;
  border:1px solid var(--line); border-radius:12px; background:#fbfcfe; }}
.arch-label {{ flex:none; width:118px; font-size:10.5px; font-weight:700;
  letter-spacing:.11em; text-transform:uppercase; color:var(--blue); }}
.arch-items {{ display:flex; flex-wrap:wrap; gap:7px; }}
.arch-item {{ background:#fff; border:1px solid var(--line); border-radius:8px;
  padding:5px 10px; font-size:11.5px; color:var(--ink); font-weight:600; }}
.arch-item span {{ color:var(--muted); font-weight:400; }}
.arch-join {{ height:16px; margin-left:60px; border-left:2px solid #cbd5e1; }}

/* ---------------- misc components ---------------- */
.stat-row {{ display:grid; gap:0; border:1px solid var(--line); border-radius:12px;
  overflow:hidden; }}
.stat {{ padding:13px 16px; border-right:1px solid var(--line); }}
.stat:last-child {{ border-right:none; }}
.stat .v {{ font-size:18px; font-weight:800; color:var(--ink); letter-spacing:-.03em;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.stat .l {{ font-size:10.5px; color:var(--muted); margin-top:2px; line-height:1.3;
  word-break:break-word; }}

.badge {{ display:inline-block; padding:3px 11px; border-radius:999px; font-size:11.5px;
  font-weight:600; }}

.kv {{ display:flex; justify-content:space-between; align-items:center;
  padding:9px 0; border-bottom:1px solid var(--line); font-size:12.5px; }}
.kv:last-child {{ border-bottom:none; }}
.kv .k {{ color:var(--muted); }}
.kv .v {{ font-weight:600; color:var(--ink); }}

.insight {{ display:flex; gap:11px; margin-bottom:13px; }}
.insight .n {{ width:22px; height:22px; border-radius:50%; background:var(--blue);
  color:#fff; font-size:11px; font-weight:700; display:grid; place-items:center;
  flex:none; margin-top:1px; }}
.insight p {{ margin:0; font-size:12.5px; line-height:1.55; color:var(--body); }}

.trace-item {{ display:flex; gap:12px; padding-bottom:15px; position:relative; }}
.trace-item:not(:last-child):before {{ content:""; position:absolute; left:11px; top:26px;
  bottom:0; width:2px; background:var(--line); }}
.trace-item .n {{ width:24px; height:24px; border-radius:50%; background:var(--blue);
  color:#fff; font-size:11.5px; font-weight:700; display:grid; place-items:center;
  flex:none; z-index:1; }}
.trace-item.alert .n {{ background:var(--red); }}
.trace-item .t {{ font-size:13px; font-weight:600; color:var(--ink); line-height:1.35; }}
.trace-item .d {{ font-size:11.5px; color:var(--muted); margin-top:2px; line-height:1.45; }}
.trace-item.alert .t {{ color:var(--red); }}

.notice {{ display:flex; gap:11px; align-items:flex-start; padding:12px 15px;
  border-radius:11px; font-size:12.5px; line-height:1.5; }}
.notice.info {{ background:#f1f5f9; color:var(--body); }}
.notice.good {{ background:#f0fdf4; color:#166534; }}
.notice.warn {{ background:#fffbeb; color:#92400e; }}
.notice .ic {{ flex:none; margin-top:1px; }}

.gauge {{ margin:2px auto 6px; }}
.gauge-scale {{ display:flex; justify-content:space-between; font-size:10px;
  color:var(--faint); margin-top:-6px; padding:0 4px; }}
.gauge-scale span:nth-child(2) {{ color:var(--muted); font-weight:600; }}
.gauge-wrap {{ text-align:center; }}
.gauge-val {{ font-size:34px; font-weight:800; color:var(--ink); letter-spacing:-.035em; }}
.gauge-lab {{ font-size:11.5px; color:var(--muted); }}

.section-head {{ font-size:11px; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--blue); margin-bottom:12px; }}

.foot-col b {{ display:block; font-size:13.5px; color:var(--ink); margin:12px 0 6px;
  font-weight:700; }}
.foot-col p {{ font-size:12.5px; color:var(--muted); line-height:1.55; margin:0; }}
.foot-col .box {{ width:38px; height:38px; border-radius:11px; background:var(--rail);
  color:var(--blue); display:grid; place-items:center; }}

/* ---------------- sidebar as nav rail ---------------- */
section[data-testid="stSidebar"] {{ background:var(--card); border-right:1px solid var(--line); }}
section[data-testid="stSidebar"] > div {{ padding-top:1.1rem; }}
section[data-testid="stSidebar"] [role="radiogroup"] {{ gap:3px; }}
section[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding:9px 13px; border-radius:10px; margin:0; transition:background .12s ease;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background:#f8fafc; }}
section[data-testid="stSidebar"] [role="radiogroup"] label p {{
  font-size:13.5px; font-weight:500; color:var(--body);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background:var(--rail);
}}
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
  color:var(--blue); font-weight:600;
}}
section[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] {{
  display:flex; align-items:center;
}}
section[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
  display:none !important;
}}
{nav_icons}

/* ---------------- Streamlit widget restyle ---------------- */
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid var(--line);
  padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ height:34px; padding:0 14px; font-size:13px;
  font-weight:500; color:var(--muted); }}
.stTabs [aria-selected="true"] {{ color:var(--blue) !important; font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:var(--blue); }}
.stTabs [data-baseweb="tab-border"] {{ display:none; }}

.stButton > button {{ border-radius:10px; font-size:13px; font-weight:600;
  border:1px solid var(--line); padding:8px 18px; background:#fff; color:var(--ink); }}
.stButton > button:hover {{ border-color:var(--blue); color:var(--blue); }}
/* A form submit button is .stFormSubmitButton, not .stButton - without it the
   Predict button fell back to Streamlit's default red. */
.stButton > button[kind="primary"], .stButton > button[kind="primaryFormSubmit"],
.stButton > button[data-testid="baseButton-primary"],
.stFormSubmitButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primaryFormSubmit"],
.stFormSubmitButton > button {{
  background:var(--navy) !important; border-color:var(--navy) !important;
}}
.stFormSubmitButton > button p, .stFormSubmitButton > button div {{
  color:#fff !important;
}}
.stFormSubmitButton > button:hover {{ background:#16395c !important; }}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primaryFormSubmit"] p,
.stButton > button[data-testid="baseButton-primary"] p,
.stButton > button[kind="primary"] div,
.stButton > button[kind="primaryFormSubmit"] div {{ color:#fff !important; }}
.stButton > button[kind="primary"]:hover {{ background:#16395c !important; }}

div[data-baseweb="input"] input, div[data-baseweb="select"] > div,
.stNumberInput input, .stTextInput input {{ font-size:13px; border-radius:9px; }}
.stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {{
  font-size:11.5px !important; color:var(--muted) !important; font-weight:500 !important;
}}
[data-testid="stMetricValue"] {{ font-size:22px; font-weight:800; color:var(--ink); }}
[data-testid="stMetricLabel"] {{ font-size:11.5px; color:var(--muted); }}

/* `theme.card()` is the card primitive wherever a card must hold Streamlit
   widgets - an HTML card cannot wrap them. Streamlit gives plain columns the
   same testid as bordered containers, so the marker span (pinned as the
   container's first child) is what separates a real card from a column. */
{CARD_SELECTOR} {{
  background:var(--card); border:1px solid var(--line) !important;
  border-radius:var(--radius); padding:18px 20px; box-shadow:var(--shadow);
}}
{CARD_SELECTOR} {CARD_SELECTOR} {{
  box-shadow:none; padding:13px 15px; border-radius:12px; background:#fbfcfe;
}}
.cardmark {{ display:none; }}
/* Everything that is not a marked card keeps Streamlit's plain layout. */
[data-testid="stVerticalBlockBorderWrapper"]:not({_CARD_HAS}) {{
  border:none !important; background:transparent; box-shadow:none; padding:0;
}}

[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:11px; }}
.stExpander {{ border:1px solid var(--line); border-radius:11px; background:#fff; }}
[data-testid="stChatInput"] textarea {{ font-size:13px; }}
.stAlert {{ border-radius:11px; font-size:12.5px; }}
hr {{ margin:.6rem 0; border-color:var(--line); }}
</style>
"""


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def topbar(title: str, subtitle: str = "", user: str = "Samiksha") -> str:
    """Identity strip: brand, the current section, and who is signed in.

    Deliberately carries nothing clickable. The original comp had marketing nav,
    a search field and a notification bell; none of them had anywhere to go, and
    a control that does nothing is worse than no control. Navigation lives in
    the rail, where it actually works.
    """
    initial = escape(user[:1].upper())
    caption = f'<div class="topbar-sub">{escape(subtitle)}</div>' if subtitle else ""
    return f"""
<div class="topbar">
  <div class="brand"><div class="dot">{icon('shield', 15, '#fff')}</div>CreditRisk IQ</div>
  <div class="topbar-title"><div class="t">{escape(title)}</div>{caption}</div>
  <div class="avatar"><div class="circle">{initial}</div><span>{escape(user)}</span></div>
</div>
"""


def hero(headline: str, body: str,
         eyebrow: str = "Credit Risk Intelligence") -> str:
    """A single banner. Used only by About now - the landing page dropped it."""
    return f"""
<div class="hero">
  <div class="eyebrow">{escape(eyebrow)}</div>
  <h1>{escape(headline)}</h1>
  <p>{escape(body)}</p>
</div>
"""


def card_open(title: str = "", subtitle: str = "", tight: bool = False,
              right: str = "") -> str:
    head = ""
    if title:
        inner = (f'<div><div class="card-title">{escape(title)}</div>'
                 f'<div class="card-sub">{escape(subtitle)}</div></div>')
        head = f'<div class="card-head">{inner}{right}</div>' if right else inner
    return f'<div class="card{" tight" if tight else ""}">{head}'


def card_close() -> str:
    return "</div>"


def warm_hero(eyebrow: str, headline: str, body: str, tagline: str,
              values: list[str], promise: str, preview: str = "") -> str:
    """Light hero: statement left, live sample prediction floating over it."""
    word_list = "".join(f"<span>{escape(word)}</span>" for word in values)
    return f"""
<div class="whero">
  <div class="whero-left">
    <div class="whero-eyebrow">{escape(eyebrow)}</div>
    <h1>{escape(headline)}</h1>
    <p>{escape(body)}</p>
    <div class="whero-tagline"><i></i>{escape(tagline)}</div>
  </div>
  <div class="whero-right">{preview}</div>
  <div class="whero-values">
    <div class="whero-words">{word_list}</div>
    <div class="whero-rule"></div>
    <div class="whero-promise">{escape(promise)}</div>
  </div>
</div>
"""


def sample_prediction(probability: float, threshold: float, band: str,
                      decision: str, expected_loss: str) -> str:
    """The white decision card the hero floats: real output, not an illustration."""
    colour = BAND_COLOUR.get(band, BLUE)
    tint = BAND_TINT.get(band, "#f1f5f9")
    ceiling = max(threshold * 2, 1e-6)
    direction = "&darr;" if probability < threshold else "&uarr;"
    return f"""
<div class="sample-card">
  <div class="sc-head">
    <span class="sc-title">Sample prediction</span>
    <span class="sc-pill" style="background:{tint};color:{colour}">{escape(band)} Risk</span>
  </div>
  {svg_gauge(probability, threshold, band, size=210)}
  <div class="sc-delta" style="color:{colour}">
    {direction} {abs(probability - threshold) * 100:.1f} pts vs. threshold
  </div>
  <div class="sc-rows">
    <div class="sc-row"><span class="sc-ic">{icon('check', 14, BLUE)}</span>
      <span class="sc-label">Recommended action</span>
      <span class="sc-pill" style="background:{tint};color:{colour}">{escape(decision)}</span>
    </div>
    <div class="sc-row"><span class="sc-ic">{icon('rate', 14, BLUE)}</span>
      <span class="sc-label">Expected loss (if approved)</span>
      <span class="sc-value">{escape(expected_loss)}</span>
    </div>
    <div class="sc-row"><span class="sc-ic">{icon('shield', 14, BLUE)}</span>
      <span class="sc-label">Key drivers</span>
      <span class="sc-link">See Explainability &rarr;</span>
    </div>
  </div>
</div>
"""


def kpi_strip(items: list[tuple[str, str, str]]) -> str:
    """Horizontal fact band: (icon, value, label)."""
    cells = "".join(
        f'<div class="kstrip-item"><div class="kstrip-ic">{icon(name, 18)}</div>'
        f'<div><div class="v">{escape(value)}</div>'
        f'<div class="l">{escape(label)}</div></div></div>'
        for name, value, label in items
    )
    return (f'<div class="kstrip" style="grid-template-columns:'
            f'repeat({len(items)},minmax(0,1fr))">{cells}</div>')


def pipeline(stages: list[tuple[str, str, str]]) -> str:
    """Numbered flow of the decision pipeline: (icon, title, detail).

    A diagram rather than prose because the single most useful thing a first-time
    viewer can learn is that this is a chain, not a model - the score is only the
    third of seven steps.
    """
    nodes = []
    for index, (icon_name, title, detail) in enumerate(stages, start=1):
        nodes.append(
            f'<div class="stage">'
            f'<div class="stage-num">{index}</div>'
            f'<div class="stage-icon">{icon(icon_name, 16)}</div>'
            f'<div class="stage-title">{escape(title)}</div>'
            f'<div class="stage-detail">{escape(detail)}</div>'
            f"</div>"
        )
    return f'<div class="stages"><div class="stage-rail"></div>{"".join(nodes)}</div>'


def section_tile(icon_name: str, title: str, blurb: str, accent: str) -> str:
    """Header for a navigation tile - the button itself is a Streamlit widget."""
    return (
        f'<div class="tile" style="--accent:{accent}">'
        f'<div class="tile-head">'
        f'<div class="tile-icon">{icon(icon_name, 17, accent)}</div>'
        f'<div class="tile-title">{escape(title)}</div></div>'
        f'<div class="tile-blurb">{escape(blurb)}</div></div>'
    )


def architecture(layers: list[tuple[str, list[tuple[str, str]]]]) -> str:
    """Layered architecture diagram: (layer name, [(module, role), ...]).

    Drawn from the real module names so the diagram and the repository cannot
    disagree - a picture of an architecture nobody shipped is worse than none.
    """
    blocks = []
    for index, (name, items) in enumerate(layers):
        if index:
            blocks.append('<div class="arch-join"></div>')
        chips = "".join(
            f'<div class="arch-item">{escape(module)}'
            f"<span> &middot; {escape(role)}</span></div>"
            for module, role in items
        )
        blocks.append(
            f'<div class="arch-layer"><div class="arch-label">{escape(name)}</div>'
            f'<div class="arch-items">{chips}</div></div>'
        )
    return f'<div class="arch">{"".join(blocks)}</div>'


def stack_table(groups: dict[str, list[tuple[str, str]]]) -> str:
    """Technology stack, grouped, as (name, version-or-role) pairs."""
    blocks = []
    for group, items in groups.items():
        rows = "".join(
            f'<div class="stack-row"><span class="n">{escape(name)}</span>'
            f'<span class="v">{escape(detail)}</span></div>'
            for name, detail in items
        )
        blocks.append(
            f'<div class="stack-group"><div class="stack-head">{escape(group)}</div>'
            f"{rows}</div>"
        )
    return f'<div class="stack">{"".join(blocks)}</div>'


def stat_row(items: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="stat"><div class="v">{escape(v)}</div>'
        f'<div class="l">{escape(l)}</div></div>' for v, l in items
    )
    return (f'<div class="stat-row" '
            f'style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">'
            f'{cells}</div>')


def badge(text: str, band: str | None = None, colour: str | None = None,
          tint: str | None = None) -> str:
    colour = colour or BAND_COLOUR.get(band or "", MUTED)
    tint = tint or BAND_TINT.get(band or "", "#f1f5f9")
    return (f'<span class="badge" style="background:{tint};color:{colour}">'
            f'{escape(text)}</span>')


def kv_rows(rows: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="kv"><span class="k">{escape(k)}</span>'
        f'<span class="v">{v}</span></div>' for k, v in rows
    )


def insights(items: list[str]) -> str:
    return "".join(
        f'<div class="insight"><div class="n">{i}</div><p>{escape(text)}</p></div>'
        for i, text in enumerate(items, start=1)
    )


def trace(steps: list[dict]) -> str:
    """steps: {title, detail, alert?}"""
    out = []
    for index, step in enumerate(steps, start=1):
        klass = "trace-item alert" if step.get("alert") else "trace-item"
        out.append(f"""
<div class="{klass}">
  <div class="n">{index}</div>
  <div><div class="t">{escape(step['title'])}</div>
       <div class="d">{escape(step['detail'])}</div></div>
</div>""")
    return "".join(out)


def notice(text: str, kind: str = "info", icon_name: str = "check") -> str:
    colour = {"info": MUTED, "good": GREEN, "warn": AMBER}[kind]
    return (f'<div class="notice {kind}">{icon(icon_name, 16, colour)}'
            f'<div>{text}</div></div>')


def gauge_caption(value: str, label: str) -> str:
    return (f'<div class="gauge-wrap"><div class="gauge-val">{escape(value)}</div>'
            f'<div class="gauge-lab">{escape(label)}</div></div>')


def footer_columns(items: list[tuple[str, str, str]]) -> str:
    """items: (icon, title, body)"""
    cells = "".join(
        f'<div class="foot-col"><div class="box">{icon(ic, 19)}</div>'
        f"<b>{escape(t)}</b><p>{escape(b)}</p></div>" for ic, t, b in items
    )
    return (f'<div style="display:grid;grid-template-columns:'
            f'repeat({len(items)},minmax(0,1fr));gap:24px">{cells}</div>')
