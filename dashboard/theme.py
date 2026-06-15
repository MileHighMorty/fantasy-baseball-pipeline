"""Single source of truth for the dashboard's visual design system.

Sleek-modern dark (Linear/Vercel direction): a calm neutral charcoal frame so
the saturated green/red of buy-sell signals is where the eye goes — color lives
in the data, not the chrome. Every hex value, the owner palette, the Plotly
chart theme, the diverging colour scales, and the injected CSS are defined here
once and imported by dashboard/app.py. `.streamlit/config.toml` mirrors the core
palette for the Streamlit chrome; keep the two in sync.

Presentation only — nothing here knows what the numbers mean, only how they look.
"""

# ── core palette ─────────────────────────────────────────────────────
# Deep neutral base, lifted surfaces, one refined accent, semantic data colors.

BG = "#0d0f12"          # near-black charcoal base, not pure black
SURFACE = "#16191d"     # lifted cards / tables / sidebar
SURFACE_2 = "#1e2228"   # hover / elevated
BORDER = "#2a2f37"      # subtle dividers and gridlines
TEXT = "#e8eaed"        # high-contrast primary text
TEXT_MUTED = "#9aa0a6"  # secondary text, captions, axis labels

ACCENT = "#4d9fff"      # refined blue (NOT acid) — active nav, key numbers, my team
ACCENT_BG = "#14233a"   # muted dark blue for accent-tinted table cells

# Semantic data colors, tuned for a dark base (calm, not neon). These carry the
# meaning: green = buy/breakout/good, amber = neutral/caution, red = sell/regress.
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"

# Darker muted backgrounds for table cells; the bright color above is the text,
# so a cell reads as glowing semantic text on a quiet tinted ground.
GOOD_BG = "#132d1c"
WARN_BG = "#2e2410"
BAD_BG = "#2d1518"

# Neutral cell ground for "not mine / not available" rows (no semantic charge).
NEUTRAL_BG = SURFACE_2

FONT_FAMILY = "Inter, ui-sans-serif, system-ui, -apple-system, sans-serif"


# ── owner / team palette ─────────────────────────────────────────────
# Muted, desaturated ring harmonized with the dark frame — NOT the old bright
# flat-UI set. Hues are spaced around the wheel but deliberately SKIP the blue
# band (~210-260°), which is reserved for the accent so my own team reads as the
# one bright point. FA is a neutral gray: it is not a team, so it must not look
# like one (this also fixes the old collision where FA reused a team's color).

MY_TEAM = "Rutsch Hour"
FA_COLOR = "#6b7280"  # neutral gray — "not a team"

OWNER_COLORS = {
    "Ben": "#c2906b",                 # warm tan
    "Chad": "#c2a86b",                # amber
    "George": "#a8b06b",              # olive
    "J-Rod Show": "#84b06b",          # green
    "Jorp": "#6bb083",                # emerald
    "Mullets": "#6bb0a3",             # teal
    "Negs": "#6ba8b0",                # cyan
    "One Pathetic Luzar": "#936bb0",  # violet
    "Porter": "#b06baa",              # magenta
    "Professor McGonigle": "#b06b85", # pink
    "Young Gunz": "#b06b6f",          # dusty red
    MY_TEAM: ACCENT,                  # my team is the bright accent star
    "FA": FA_COLOR,
}


# ── diverging / sequential colour scales ─────────────────────────────
# Buy/sell semantics preserved: green = buy, red = sell, neutral midpoint.

# Diverging scale for signed buy-magnitude charts (negative = sell/red,
# 0 = neutral, positive = buy/green). Used with color_continuous_midpoint=0.
DIVERGING_SCALE = [
    [0.0, BAD],
    [0.5, "#5a6069"],  # neutral mid-gray
    [1.0, GOOD],
]

# Sequential scale for single-lens breakout charts where every point is already
# a buy and color encodes only the STRENGTH of the buy — so it runs neutral-dark
# green to bright green (never red, which would mislabel a buy as a sell).
SEQUENTIAL_GOOD = [
    [0.0, "#1f3d2b"],
    [1.0, GOOD],
]

# Discrete colorway for grouped bar charts: accent first, muted gray second, then
# the semantic colors. Keeps "actual vs expected" bars calm and legible.
CHART_COLORWAY = [ACCENT, TEXT_MUTED, GOOD, WARN, BAD]


def apply_chart_theme(fig, *, height=None):
    """Apply the shared dark chart theme to a Plotly figure, in place.

    Sets the charcoal paper/plot grounds, the Inter font in TEXT, subtle BORDER
    gridlines (no heavy chart junk), a consistent margin, and the shared
    colorway. Every chart calls this so heights/fonts/grounds stop being ad hoc.

    Args:
        fig: A Plotly figure (px.* or go.Figure).
        height: Optional pixel height; left untouched when None.

    Returns:
        The same figure, themed (returned for chaining convenience).
    """
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family=FONT_FAMILY, size=13),
        title_font=dict(color=TEXT, family=FONT_FAMILY, size=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        coloraxis_colorbar=dict(
            tickfont=dict(color=TEXT_MUTED), title_font=dict(color=TEXT_MUTED)
        ),
        margin=dict(l=48, r=24, t=56, b=48),
        colorway=CHART_COLORWAY,
    )
    if height is not None:
        fig.update_layout(height=height)
    axis = dict(
        gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
        title_font=dict(color=TEXT_MUTED), tickfont=dict(color=TEXT_MUTED),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


# ── injected CSS ─────────────────────────────────────────────────────
# One disciplined block for what config.toml can't reach: a real type face
# (Inter), tightened heading weights/tracking, sidebar polish, and metric tiles.
# Driven entirely by the constants above so nothing is hardcoded twice.

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {FONT_FAMILY};
}}
.stApp {{ background-color: {BG}; }}

section[data-testid="stSidebar"] {{
    background-color: {SURFACE};
    border-right: 1px solid {BORDER};
}}

h1, h2, h3 {{
    color: {TEXT};
    font-weight: 600;
    letter-spacing: -0.011em;
}}
h1 {{ font-weight: 700; letter-spacing: -0.021em; }}

a, a:visited {{ color: {ACCENT}; }}

hr {{ border-color: {BORDER}; }}

[data-testid="stMetric"] {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 16px;
}}
[data-testid="stMetricValue"] {{ color: {ACCENT}; font-weight: 600; }}
[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED}; }}
</style>
"""


# ── table cell stylers (semantic background + bright text) ───────────
# Centralized so every page's pandas Styler reads from one palette.


def gap_stability_style(val):
    """Roster-health gap magnitude: small gap = good, large = divergence (bad).

    Semantics preserved from the original: a small expected-vs-actual gap is
    stable (good/green), a large one is a significant divergence (bad/red).
    """
    try:
        gap = abs(float(val))
    except (ValueError, TypeError):
        return ""
    if gap > 0.050:
        return f"background-color: {BAD_BG}; color: {BAD}"
    if gap > 0.020:
        return f"background-color: {WARN_BG}; color: {WARN}"
    return f"background-color: {GOOD_BG}; color: {GOOD}"


def score_style(val):
    """0-100 stream/quality score: >70 good, 50-70 caution, <50 bad."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 70:
        return f"background-color: {GOOD_BG}; color: {GOOD}"
    if v >= 50:
        return f"background-color: {WARN_BG}; color: {WARN}"
    return f"background-color: {BAD_BG}; color: {BAD}"


def edge_style(edge_text):
    """Matchup Edge cell: My Edge = good, Opp Edge = bad, Close = caution."""
    if "My Edge" in edge_text:
        return f"background-color: {GOOD_BG}; color: {GOOD}"
    if "Opp Edge" in edge_text:
        return f"background-color: {BAD_BG}; color: {BAD}"
    if "Close" in edge_text:
        return f"background-color: {WARN_BG}; color: {WARN}"
    return ""


def ownership_style(val):
    """Prospect ownership cell: FA = good, my team = accent, others = neutral."""
    if val == "FA":
        return f"background-color: {GOOD_BG}; color: {GOOD}"
    if val == MY_TEAM:
        return f"background-color: {ACCENT_BG}; color: {ACCENT}"
    return f"background-color: {NEUTRAL_BG}; color: {TEXT_MUTED}"


def upgrade_style(val):
    """Net-upgrade score: >15 strong (good), 10-15 marginal (caution), else none.

    Always sets an explicit text color so it stays readable on the dark base.
    """
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 15:
        return f"background-color: {GOOD_BG}; color: {GOOD}"
    if v >= 10:
        return f"background-color: {WARN_BG}; color: {WARN}"
    return ""


def hot_row_style(width):
    """Full-row tint for a hot prospect (subtle caution-amber ground)."""
    return [f"background-color: {WARN_BG}"] * width


def callup_row_style(width):
    """Full-row tint for a call-up candidate (subtle good-green ground)."""
    return [f"background-color: {GOOD_BG}"] * width
