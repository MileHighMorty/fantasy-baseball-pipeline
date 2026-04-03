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


def _load_my_roster_df() -> pd.DataFrame | None:
    """Return DataFrame of my Fantrax roster with player_name and position."""
    df = _latest_fantrax("my_roster")
    if df is None:
        return None
    df["player_name"] = df["player_name"].str.strip()
    return df[["player_name", "position"]].rename(columns={"position": "fantrax_position"})


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
    st.caption(
        "xwOBA minus wOBA measures the gap between a player's expected production "
        "(based on exit velocity, launch angle, and sprint speed) and their actual results. "
        "Green (within 0.020) means performing as expected. "
        "Yellow (gap > 0.020) means slight over/underperformance. "
        "Red (gap > 0.050) means significant divergence — positive red means the player "
        "is unlucky and due for better results, negative red means they're getting lucky "
        "and regression is likely."
    )
    my_names = _load_my_roster()
    my_roster_df = _load_my_roster_df()
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
        # Merge Fantrax position data to replace statcast position
        if my_roster_df is not None:
            roster = roster.merge(my_roster_df, on="player_name", how="left")
            roster["position"] = roster["fantrax_position"].fillna(roster.get("position", ""))
            roster.drop(columns=["fantrax_position"], inplace=True)

        # Position sort order for hitters
        _POS_ORDER = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "UT": 6}

        hit_col, pit_col = st.columns(2)

        # --- Hitters table (left) ---
        with hit_col:
            st.markdown("**Hitters**")
            hit_df = roster[roster["player_type"] == "Hitter"].copy()
            if not hit_df.empty:
                gap_col = "xwoba_minus_woba" if "xwoba_minus_woba" in hit_df.columns else "est_woba_minus_woba_diff"
                hit_df["_pos_rank"] = hit_df["position"].map(_POS_ORDER).fillna(99)
                sort_cols = ["_pos_rank"]
                sort_asc = [True]
                if gap_col in hit_df.columns:
                    sort_cols.append(gap_col)
                    sort_asc.append(False)
                hit_df = hit_df.sort_values(sort_cols, ascending=sort_asc).drop(columns=["_pos_rank"])

                h_display = ["player_name", "team", "position"]
                if "woba" in hit_df.columns:
                    h_display.append("woba")
                if "est_woba" in hit_df.columns:
                    h_display.append("est_woba")
                if gap_col in hit_df.columns:
                    h_display.append(gap_col)

                fmt = {c: "{:.3f}" for c in h_display if c not in ("player_name", "team", "position")}
                styled_h = hit_df[h_display].style.format(fmt)
                if gap_col in hit_df.columns:
                    styled_h = styled_h.map(_color_xwoba_gap, subset=[gap_col])
                st.dataframe(styled_h, use_container_width=True, hide_index=True,
                             height=max(400, 35 * len(hit_df) + 40))
            else:
                st.info("No hitter statcast data available.")

        # --- Pitchers table (right) ---
        with pit_col:
            st.markdown("**Pitchers**")
            pit_df = roster[roster["player_type"] == "Pitcher"].copy()
            if not pit_df.empty:
                era_gap = "xera_minus_era" if "xera_minus_era" in pit_df.columns else None
                if era_gap and era_gap in pit_df.columns:
                    pit_df = pit_df.sort_values(era_gap, ascending=False)

                p_display = ["player_name", "team", "position"]
                if "xera" in pit_df.columns:
                    p_display.append("xera")
                if "era" in pit_df.columns:
                    p_display.append("era")
                if era_gap and era_gap in pit_df.columns:
                    p_display.append(era_gap)

                fmt = {c: "{:.2f}" for c in p_display if c not in ("player_name", "team", "position")}
                styled_p = pit_df[p_display].style.format(fmt)
                if era_gap and era_gap in pit_df.columns:
                    styled_p = styled_p.map(_color_xwoba_gap, subset=[era_gap])
                st.dataframe(styled_p, use_container_width=True, hide_index=True,
                             height=max(400, 35 * len(pit_df) + 40))
            else:
                st.info("No pitcher statcast data available.")

    # --- Breakout adds (free agents only) ---
    col1, col2 = st.columns(2)
    breakout_caption = (
        "These are free agents whose underlying Statcast quality significantly exceeds "
        "their surface stats. They are hitting the ball hard but not getting results yet "
        "— the process is right, the outcomes haven't caught up. These are buy-low "
        "candidates before the market notices."
    )
    with col1:
        st.subheader("Top 10 Breakout Hitter Adds")
        st.caption(breakout_caption)
        bh = _load_csv(GOLD / "breakout_hitters_fa.csv")
        if bh is not None:
            show_cols = [c for c in ["player_name", "team", "position", "est_woba", "woba", "xwoba_minus_woba", "hard_hit_percentile", "barrel_percentile"] if c in bh.columns]
            st.dataframe(bh[show_cols].head(10), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top 10 Breakout Pitcher Adds")
        st.caption(breakout_caption)
        bp = _load_csv(GOLD / "breakout_pitchers_fa.csv")
        if bp is not None:
            show_cols = [c for c in ["player_name", "team", "position", "xera", "era", "xera_minus_era", "k_percent", "barrel_percentile"] if c in bp.columns]
            st.dataframe(bp[show_cols].head(10), use_container_width=True, hide_index=True)


def page_breakout_board():
    st.header("Breakout Board")

    st.caption(
        "Players above the diagonal line have actual wOBA lower than their expected wOBA "
        "— meaning their underlying contact quality is better than their results show. "
        "These are buy candidates. Players below the line are overperforming their Statcast "
        "profile and may regress. Dot size reflects hard-hit percentage. Hover over any dot "
        "for the full profile."
    )

    bh = _load_csv(GOLD / "breakout_hitters_all.csv")
    if bh is None:
        return

    required = {"est_woba", "woba", "player_name", "team"}
    if not required.issubset(bh.columns):
        st.error(f"Missing columns. Need {required}, have {set(bh.columns)}")
        return

    # Ownership comes pre-tagged from breakout_detector
    if "ownership" not in bh.columns:
        bh["ownership"] = "Unknown"

    # Use actual team names for the legend (FA stays as-is)
    bh["status"] = bh["ownership"]

    size_col = "hard_hit_percentile" if "hard_hit_percentile" in bh.columns else None
    if size_col is None and "avg_hit_speed" in bh.columns:
        size_col = "avg_hit_speed"

    # Extract last name + team abbreviation for FA text labels
    bh["label"] = bh["player_name"].str.split().str[-1] + " (" + bh["team"] + ")"

    # Build hover tooltip fields using actual column names
    hover_fields = {
        "player_name": True,
        "team": True,
        "position": True,
        "est_woba": ":.3f",
        "woba": ":.3f",
    }
    for hf in ["xwoba_minus_woba", "est_woba_minus_woba_diff"]:
        if hf in bh.columns:
            hover_fields[hf] = ":.3f"
            break
    if "hard_hit_percentile" in bh.columns:
        hover_fields["hard_hit_percentile"] = ":.1f"
    if "brl_percent" in bh.columns:
        hover_fields["brl_percent"] = ":.1f"

    # Build color/symbol/opacity maps for all 12 teams + FA
    _LEAGUE_TEAMS = [
        "Ben", "Chad", "George", "J-Rod Show", "Jorp", "Luke",
        "Mullets", "Negs", "One Pathetic Luzar", "Porter",
        "Professor McGonigle", "Rutsch Hour",
    ]
    # Owned team palette (distinct colors for each team)
    _TEAM_COLORS = [
        "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c", "#3498db",
        "#9b59b6", "#e84393", "#fd79a8", "#00cec9", "#6c5ce7", "#ffeaa7",
    ]
    color_map = {"FA": "#3498db"}
    symbol_map = {"FA": "circle"}
    opacity_map = {"FA": 1.0}
    for i, team in enumerate(_LEAGUE_TEAMS):
        color_map[team] = _TEAM_COLORS[i]
        symbol_map[team] = "star" if team == MY_TEAM else "diamond"
        opacity_map[team] = 1.0 if team == MY_TEAM else 0.4

    # Ensure FA dots are larger by adding a size boost column
    if size_col:
        bh["_plot_size"] = bh[size_col]
        bh.loc[bh["status"] == "FA", "_plot_size"] = bh.loc[bh["status"] == "FA", size_col] * 1.4
        plot_size_col = "_plot_size"
    else:
        plot_size_col = None

    category_order = ["FA"] + sorted(set(bh["status"].unique()) - {"FA"})

    fig = px.scatter(
        bh,
        x="est_woba",
        y="woba",
        color="status",
        color_discrete_map=color_map,
        symbol="status",
        symbol_map=symbol_map,
        size=plot_size_col,
        size_max=20,
        hover_data=hover_fields,
        labels={"est_woba": "xwOBA (Expected)", "woba": "wOBA (Actual)", "status": "Owner"},
        title="xwOBA vs wOBA — Players Above the Line Are Underperforming (Buy Candidates)",
        category_orders={"status": category_order},
    )

    # Set opacity per trace so owned players fade into the background
    for trace in fig.data:
        trace.opacity = opacity_map.get(trace.name, 0.6)

    # Add visible text labels (last name + team) on FA dots only
    fa_data = bh[bh["status"] == "FA"]
    if not fa_data.empty:
        fig.add_trace(
            go.Scatter(
                x=fa_data["est_woba"],
                y=fa_data["woba"] + 0.004,
                mode="text",
                text=fa_data["label"],
                textposition="top center",
                textfont=dict(size=12, color="white"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Diagonal x=y line — white dashed for visibility on dark background
    lo = min(bh["est_woba"].min(), bh["woba"].min()) - 0.010
    hi = max(bh["est_woba"].max(), bh["woba"].max()) + 0.010
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color="white", width=2),
            showlegend=False,
            name="x = y",
        )
    )

    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def page_sp_streaming():
    st.header("SP Streaming Picks")
    st.caption(
        "Starting pitchers ranked by streaming value for this week. Score is weighted: "
        "40% pitcher xERA (lower is better), 30% strikeout rate (higher is better), "
        "30% opponent weakness (low wRC+ and high K% is better). Green scores are "
        "high-confidence streams, yellow are matchup-dependent, red are risky."
    )

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
        st.caption(
            "These players have actual results significantly better than their underlying "
            "Statcast metrics suggest. A large negative xwOBA gap means their batting average, "
            "home runs, or OBP are inflated by luck (high BABIP, unsustainable HR/FB rate). "
            "Consider selling high in trades before regression hits."
        )
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
        st.caption(
            "These pitchers have ERAs significantly lower than their expected ERA (xERA). "
            "A large negative gap means they've been getting lucky with strand rate, BABIP "
            "against, or sequencing. Their stuff quality doesn't support the current ERA. "
            "Expect the ERA to rise."
        )
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


def _ownership_color(val):
    """Styler function for prospect ownership column."""
    if val == "FA":
        return "background-color: #69db7c; color: black"
    if val == MY_TEAM:
        return "background-color: #ffd43b; color: black"
    return "background-color: #dee2e6; color: black"


def _upgrade_color(val):
    """Styler function for net upgrade score — green > 15, yellow 10-15."""
    try:
        v = float(val)
    except (ValueError, TypeError):
        return ""
    if v > 15:
        return "background-color: #69db7c"
    if v >= 10:
        return "background-color: #ffd43b"
    return ""


def page_add_drop():
    st.header("Suggested Roster Moves")
    st.caption(
        "These suggestions compare your weakest rostered player at each position "
        "against the best available free agent. A swap is suggested when the FA's "
        "underlying Statcast metrics significantly exceed your current player. "
        "Dynasty warnings flag young players who should be traded rather than dropped."
    )

    suggestions = _load_csv(GOLD / "add_drop_suggestions.csv")
    if suggestions is None:
        return
    if suggestions.empty:
        st.success("No suggested moves — your roster is solid at every position.")
        return

    # Filter out sub-threshold rows (keep net_upgrade >= 10)
    suggestions = suggestions[suggestions["net_upgrade"] >= 10].copy()
    if suggestions.empty:
        st.success("No significant upgrades available right now.")
        return

    display_cols = [
        "position",
        "drop_candidate",
        "drop_score",
        "add_candidate",
        "add_score",
        "net_upgrade",
        "dynasty_warning",
    ]
    available = [c for c in display_cols if c in suggestions.columns]
    display_df = suggestions[available].rename(columns={
        "position": "Position",
        "drop_candidate": "Drop",
        "drop_score": "Drop Score",
        "add_candidate": "Add",
        "add_score": "Add Score",
        "net_upgrade": "Net Upgrade",
        "dynasty_warning": "Dynasty Warning",
    })

    fmt = {"Drop Score": "{:.1f}", "Add Score": "{:.1f}", "Net Upgrade": "{:.1f}"}
    styled = display_df.style.format(fmt, na_rep="—")
    styled = styled.map(_upgrade_color, subset=["Net Upgrade"])

    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=max(400, 35 * len(display_df) + 40))

    st.caption(
        "These are data-driven suggestions, not automatic decisions. "
        "Always consider lineup context, team quality, and upcoming schedule "
        "before making moves."
    )


def page_prospect_pipeline():
    st.header("Prospect Pipeline")
    st.caption(
        "Tracked minor league prospects with current stats and call-up indicators. "
        "A prospect is flagged as a call-up candidate if performing significantly above "
        "their level (wRC+ > 130 for hitters, ERA < 3.50 with WHIP < 1.25 for pitchers "
        "at Double-A or higher) and on the 40-man roster."
    )

    prospects = _load_csv(GOLD / "prospect_alerts.csv")
    if prospects is None:
        return

    # FA filter toggle
    show_fa_only = st.checkbox("Show FAs only", value=False)
    if show_fa_only and "ownership" in prospects.columns:
        prospects = prospects[prospects["ownership"] == "FA"]

    if prospects.empty:
        st.info("No prospects match the current filter.")
        return

    # Add hot indicator column
    if "is_hot" in prospects.columns:
        prospects["hot"] = prospects["is_hot"].apply(
            lambda x: "\U0001f525" if x in (True, "True", "true", 1, "1") else ""
        )

    # Mark prospects without stats
    if "has_stats" in prospects.columns:
        no_stats = prospects["has_stats"].apply(
            lambda x: x in (False, "False", "false", 0, "0")
        )
        for col in ["avg", "obp", "slg", "era", "whip", "k_per_9"]:
            if col in prospects.columns:
                prospects.loc[no_stats, col] = None

    # Build display columns
    display_cols = ["hot"] if "hot" in prospects.columns else []
    display_cols += ["name", "team", "position", "level"]
    if "age" in prospects.columns:
        display_cols.append("age")
    display_cols.append("ownership")

    # Add stat columns
    for c in ["avg", "obp", "slg", "k_pct", "bb_pct", "hr",
              "era", "whip", "k_per_9", "bb_per_9", "ip"]:
        if c in prospects.columns and prospects[c].notna().any():
            display_cols.append(c)

    for c in ["on_40_man", "callup_candidate", "heat_score"]:
        if c in prospects.columns:
            display_cols.append(c)

    available = [c for c in display_cols if c in prospects.columns]
    display_df = prospects[available].copy()

    # Format numeric columns
    fmt = {}
    for c in ["avg", "obp", "slg"]:
        if c in display_df.columns:
            fmt[c] = "{:.3f}"
    for c in ["era", "whip", "k_per_9", "bb_per_9", "ip", "heat_score"]:
        if c in display_df.columns:
            fmt[c] = "{:.1f}"

    # Apply styling
    styled = display_df.style.format(fmt, na_rep="—")

    if "ownership" in display_df.columns:
        styled = styled.map(_ownership_color, subset=["ownership"])

    # Highlight hot prospects
    if "is_hot" in prospects.columns:
        def _highlight_hot(row):
            is_hot_val = prospects.iloc[row.name].get("is_hot", False) if row.name < len(prospects) else False
            if is_hot_val in (True, "True", "true", 1, "1"):
                return ["background-color: #fff3e0"] * len(row)
            return [""] * len(row)
        # Only apply row highlight if hot column exists
        hot_mask = prospects["is_hot"].apply(lambda x: x in (True, "True", "true", 1, "1"))
        if hot_mask.any():
            def _hot_row_style(row):
                idx = row.name
                if idx in hot_mask.index and hot_mask.loc[idx]:
                    return ["background-color: #fff3e0"] * len(row)
                return [""] * len(row)
            styled = styled.apply(_hot_row_style, axis=1)

    # Highlight call-up candidates
    if "callup_candidate" in display_df.columns:
        def _highlight_callup(row):
            if row.get("callup_candidate") in (True, "True", "true", 1, "1", "Yes", "yes"):
                return ["background-color: #a9e34b"] * len(row)
            return [""] * len(row)
        styled = styled.apply(_highlight_callup, axis=1)

    st.dataframe(styled, use_container_width=True, hide_index=True,
                 height=max(500, 35 * len(display_df) + 40))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PAGES = {
    "Session Prep": page_session_prep,
    "Breakout Board": page_breakout_board,
    "SP Streaming": page_sp_streaming,
    "Regression Watch": page_regression_watch,
    "Add/Drop Suggestions": page_add_drop,
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
                refresh_main(argv=[])
                st.sidebar.success("Refresh complete!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    PAGES[selection]()


if __name__ == "__main__":
    main()
