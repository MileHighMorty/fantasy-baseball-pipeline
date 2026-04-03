"""Fantasy Baseball Pipeline — Streamlit dashboard."""

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on the path so we can import scripts.weekly_refresh
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLD = PROJECT_ROOT / "gold" / "data"
SILVER = PROJECT_ROOT / "silver" / "data"
FANTRAX = Path.home() / "projects" / "dynasty-cap-manager" / "bronze" / "data" / "fantrax"
MY_TEAM = "Rutsch Hour"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_fantrax(prefix: str) -> pd.DataFrame | None:
    """Load the most recent date-stamped Fantrax CSV matching *prefix*."""
    files = sorted(FANTRAX.glob(f"{prefix}_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])


def _load_my_roster() -> set[str]:
    """Return set of player names on my Fantrax roster."""
    df = _latest_fantrax("my_roster")
    if df is None:
        return set()
    return set(df.loc[df["player_name"].notna(), "player_name"].str.strip())


def _load_all_rosters() -> pd.DataFrame | None:
    """Load all rosters and return DataFrame with team_name and player_name."""
    return _latest_fantrax("all_rosters")


def _ownership_status(player_name: str, all_rosters: pd.DataFrame, my_names: set[str]) -> str:
    """Return 'My Team', 'FA', or the owning team name."""
    if player_name in my_names:
        return MY_TEAM
    match = all_rosters.loc[
        all_rosters["player_name"].str.strip() == player_name
    ]
    if match.empty:
        return "FA"
    return match.iloc[0]["team_name"]


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    st.warning(f"File not found: {path.name}")
    return None


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    st.warning(f"File not found: {path.name}")
    return None


def _last_refreshed() -> str | None:
    """Return the most recent modified timestamp among gold/data/ files."""
    files = list(GOLD.glob("*"))
    if not files:
        return None
    latest = max(f.stat().st_mtime for f in files)
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S")


def _color_xwoba_gap(val):
    """Styler function for xwoba-minus-woba gap magnitude."""
    try:
        gap = abs(float(val))
    except (ValueError, TypeError):
        return ""
    if gap > 0.050:
        return "background-color: #ff6b6b; color: white"
    if gap > 0.020:
        return "background-color: #ffd43b"
    return "background-color: #69db7c"


def _score_color(val):
    """Styler function for stream_score / generic 0-100 scores."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 70:
        return "background-color: #69db7c"
    if v >= 50:
        return "background-color: #ffd43b"
    return "background-color: #ff6b6b; color: white"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_session_prep():
    st.header("Session Prep")

    # Last refreshed
    ts = _last_refreshed()
    if ts:
        st.caption(f"Last refreshed: **{ts}**")
    else:
        st.warning("No data files found in gold/data/")

    # --- My Roster Health ---
    st.subheader("My Roster Health")
    my_names = _load_my_roster()
    if not my_names:
        st.warning("Could not load my_roster CSV from Fantrax data.")

    hitters = _load_parquet(SILVER / "statcast_hitters.parquet")
    pitchers = _load_parquet(SILVER / "statcast_pitchers.parquet")

    roster_frames = []
    if hitters is not None:
        h = hitters.copy()
        h["player_type"] = "Hitter"
        roster_frames.append(h)
    if pitchers is not None:
        p = pitchers.copy()
        p["player_type"] = "Pitcher"
        roster_frames.append(p)

    if roster_frames:
        roster = pd.concat(roster_frames, ignore_index=True)
        # Filter to only my Fantrax roster players
        if my_names:
            roster = roster[roster["player_name"].isin(my_names)]
            if roster.empty:
                st.info("No statcast data found for your roster players yet.")
        gap_col = "xwoba_minus_woba" if "xwoba_minus_woba" in roster.columns else "est_woba_minus_woba_diff"
        display_cols = ["player_name", "team", "position", "player_type"]
        if "woba" in roster.columns:
            display_cols.append("woba")
        if "est_woba" in roster.columns:
            display_cols.append("est_woba")
        if gap_col in roster.columns:
            display_cols.append(gap_col)
            styled = (
                roster[display_cols]
                .sort_values(gap_col, key=lambda s: s.abs(), ascending=False)
                .style.map(_color_xwoba_gap, subset=[gap_col])
                .format({c: "{:.3f}" for c in display_cols if c not in ("player_name", "team", "position", "player_type")})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.dataframe(roster[display_cols], use_container_width=True, hide_index=True)

    # --- Breakout adds ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Breakout Hitter Adds")
        bh = _load_csv(GOLD / "breakout_hitters.csv")
        if bh is not None:
            show_cols = [c for c in ["player_name", "team", "position", "est_woba", "woba", "xwoba_minus_woba", "hard_hit_percentile", "barrel_percentile"] if c in bh.columns]
            st.dataframe(bh[show_cols].head(10), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top 10 Breakout Pitcher Adds")
        bp = _load_csv(GOLD / "breakout_pitchers.csv")
        if bp is not None:
            show_cols = [c for c in ["player_name", "team", "position", "xera", "era", "xera_minus_era", "k_percent", "barrel_percentile"] if c in bp.columns]
            st.dataframe(bp[show_cols].head(10), use_container_width=True, hide_index=True)


def page_breakout_board():
    st.header("Breakout Board")

    bh = _load_csv(GOLD / "breakout_hitters.csv")
    if bh is None:
        return

    required = {"est_woba", "woba", "player_name", "team"}
    if not required.issubset(bh.columns):
        st.error(f"Missing columns. Need {required}, have {set(bh.columns)}")
        return

    # --- Add ownership status column ---
    my_names = _load_my_roster()
    all_rosters = _load_all_rosters()
    if all_rosters is not None and my_names:
        bh["ownership"] = bh["player_name"].apply(
            lambda n: _ownership_status(n, all_rosters, my_names)
        )
    else:
        bh["ownership"] = "Unknown"

    size_col = "hard_hit_percent" if "hard_hit_percent" in bh.columns else None
    # Fallback: some datasets name it differently
    if size_col is None and "avg_hit_speed" in bh.columns:
        size_col = "avg_hit_speed"

    hover_fields = {
        "player_name": True,
        "team": True,
        "ownership": True,
    }
    if "hard_hit_percent" in bh.columns:
        hover_fields["hard_hit_percent"] = ":.1f"
    if "brl_percent" in bh.columns:
        hover_fields["brl_percent"] = ":.1f"

    # Color by ownership: My Team = green, FA = blue, other teams = gray
    color_map = {MY_TEAM: "#2ecc71", "FA": "#3498db"}
    fig = px.scatter(
        bh,
        x="est_woba",
        y="woba",
        color="ownership",
        color_discrete_map=color_map,
        symbol="ownership",
        symbol_map={MY_TEAM: "star", "FA": "circle"},
        size=size_col,
        size_max=18,
        hover_data=hover_fields,
        labels={"est_woba": "xwOBA (Expected)", "woba": "wOBA (Actual)", "ownership": "Ownership"},
        title="xwOBA vs wOBA — Players Above the Line Are Underperforming (Buy Candidates)",
    )

    # Diagonal x=y line
    lo = min(bh["est_woba"].min(), bh["woba"].min()) - 0.010
    hi = max(bh["est_woba"].max(), bh["woba"].max()) + 0.010
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
            name="x = y",
        )
    )

    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def page_sp_streaming():
    st.header("SP Streaming Picks")

    sp = _load_csv(GOLD / "sp_streaming_picks.csv")
    if sp is None:
        return

    sort_col = "stream_score" if "stream_score" in sp.columns else sp.columns[-1]
    sp = sp.sort_values(sort_col, ascending=False)

    styled = (
        sp.style
        .map(_score_color, subset=[sort_col] if sort_col in sp.columns else [])
        .format({c: "{:.1f}" for c in sp.select_dtypes("number").columns})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def page_regression_watch():
    st.header("Regression Watch")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hitters")
        rh = _load_csv(GOLD / "regression_hitters.csv")
        if rh is not None:
            gap_col = "xwoba_minus_woba" if "xwoba_minus_woba" in rh.columns else "est_woba_minus_woba_diff"
            if gap_col in rh.columns:
                rh = rh.sort_values(gap_col, ascending=True)  # most negative = biggest overperformer
            show_cols = [c for c in ["player_name", "team", "position", "woba", "est_woba", gap_col] if c in rh.columns]
            st.dataframe(rh[show_cols], use_container_width=True, hide_index=True)

            # Bar chart — top 10 overperformers
            if "woba" in rh.columns and "est_woba" in rh.columns:
                top10 = rh.head(10).copy()
                fig = go.Figure()
                fig.add_trace(go.Bar(name="wOBA (Actual)", x=top10["player_name"], y=top10["woba"]))
                fig.add_trace(go.Bar(name="xwOBA (Expected)", x=top10["player_name"], y=top10["est_woba"]))
                fig.update_layout(barmode="group", title="Top 10 Overperforming Hitters", height=400)
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pitchers")
        rp = _load_csv(GOLD / "regression_pitchers.csv")
        if rp is not None:
            gap_col = "xera_minus_era" if "xera_minus_era" in rp.columns else "era_minus_xera_diff"
            if gap_col in rp.columns:
                rp = rp.sort_values(gap_col, ascending=True)  # most negative = biggest overperformer (ERA below xERA)
            show_cols = [c for c in ["player_name", "team", "position", "era", "xera", gap_col] if c in rp.columns]
            st.dataframe(rp[show_cols], use_container_width=True, hide_index=True)

            if "era" in rp.columns and "xera" in rp.columns:
                top10 = rp.head(10).copy()
                fig = go.Figure()
                fig.add_trace(go.Bar(name="ERA (Actual)", x=top10["player_name"], y=top10["era"]))
                fig.add_trace(go.Bar(name="xERA (Expected)", x=top10["player_name"], y=top10["xera"]))
                fig.update_layout(barmode="group", title="Top 10 Overperforming Pitchers", height=400)
                st.plotly_chart(fig, use_container_width=True)


def page_prospect_pipeline():
    st.header("Prospect Pipeline")

    prospects = _load_csv(GOLD / "prospect_alerts.csv")
    if prospects is None:
        return

    # Highlight call-up candidates
    if "callup_candidate" in prospects.columns:
        def _highlight_callup(row):
            if row.get("callup_candidate") in (True, "True", "true", 1, "1", "Yes", "yes"):
                return ["background-color: #a9e34b"] * len(row)
            return [""] * len(row)

        styled = prospects.style.apply(_highlight_callup, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(prospects, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAGES = {
    "Session Prep": page_session_prep,
    "Breakout Board": page_breakout_board,
    "SP Streaming": page_sp_streaming,
    "Regression Watch": page_regression_watch,
    "Prospect Pipeline": page_prospect_pipeline,
}


def main():
    st.set_page_config(page_title="Fantasy Baseball Pipeline", layout="wide")

    st.sidebar.title("Fantasy Baseball Pipeline")
    selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Running weekly refresh..."):
            try:
                from scripts.weekly_refresh import main as refresh_main
                refresh_main()
                st.sidebar.success("Refresh complete!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    PAGES[selection]()


if __name__ == "__main__":
    main()
