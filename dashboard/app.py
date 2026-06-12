"""Fantasy Baseball Pipeline — Streamlit dashboard."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import logging

import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from rapidfuzz import process, fuzz

# Ensure project root is on the path so we can import scripts.weekly_refresh
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLD = PROJECT_ROOT / "gold" / "data"
SILVER = PROJECT_ROOT / "silver" / "data"
FANGRAPHS = PROJECT_ROOT / "bronze" / "data" / "fangraphs"
FANTRAX = PROJECT_ROOT / "bronze" / "data" / "fantrax"
MY_TEAM = "Rutsch Hour"
FUZZY_THRESHOLD = 85


@st.cache_data(ttl=3600)
def _load_id_map() -> pd.DataFrame | None:
    """Load the pre-built player ID map."""
    path = SILVER / "player_id_map.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

logger = logging.getLogger(__name__)

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
    names = df.loc[df["player_name"].notna(), "player_name"].str.strip()
    return set(names[names != "None"])


def _load_my_roster_df() -> pd.DataFrame | None:
    """Return DataFrame of my Fantrax roster with player_name and position."""
    df = _latest_fantrax("my_roster")
    if df is None:
        return None
    df["player_name"] = df["player_name"].str.strip()
    return df[["player_name", "position"]].rename(columns={"position": "fantrax_position"})


def _fuzzy_merge(
    base: pd.DataFrame,
    right: pd.DataFrame,
    right_name_col: str = "player_name",
    cols_to_add: list[str] | None = None,
    threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """LEFT JOIN *right* onto *base* by fuzzy-matching player names.

    Only columns listed in *cols_to_add* are brought over (or all non-name
    columns if None).  Existing columns in *base* are NOT overwritten.
    """
    if right is None or right.empty:
        return base

    right_names = right[right_name_col].dropna().unique().tolist()
    if not right_names:
        return base

    if cols_to_add is None:
        cols_to_add = [c for c in right.columns if c != right_name_col]
    # Only bring columns that don't already exist in base
    cols_to_add = [c for c in cols_to_add if c not in base.columns]
    if not cols_to_add:
        return base

    matched_rows = []
    for _, row in base.iterrows():
        result = process.extractOne(
            row["player_name"], right_names,
            scorer=fuzz.token_sort_ratio, score_cutoff=threshold,
        )
        if result is not None:
            match_name = result[0]
            match_row = right.loc[right[right_name_col] == match_name].iloc[0]
            matched_rows.append({c: match_row[c] for c in cols_to_add if c in match_row.index})
        else:
            matched_rows.append({c: np.nan for c in cols_to_add})

    extra = pd.DataFrame(matched_rows, index=base.index)
    return pd.concat([base, extra], axis=1)


def _load_fangraphs_team_pos() -> pd.DataFrame:
    """Load team and position info from FanGraphs batting + pitching CSVs."""
    frames = []
    for pattern, pos_source in [("*batting*.csv", "batting"), ("*pitching*.csv", "pitching")]:
        files = sorted(FANGRAPHS.glob(pattern))
        if not files:
            continue
        df = pd.read_csv(files[-1], usecols=lambda c: c in ("Name", "Team"))
        df = df.rename(columns={"Name": "player_name", "Team": "fg_team"})
        if pos_source == "pitching":
            df["fg_position"] = "P"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["player_name", "fg_team", "fg_position"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="player_name", keep="first")


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


# MLB API abbreviation → Savant/FanGraphs abbreviation mapping (only mismatches)
_MLB_TO_SAVANT_ABBR = {
    "SF": "SFG", "TB": "TBR", "SD": "SDP", "KC": "KCR",
    "CWS": "CWS", "WSH": "WSN", "AZ": "ARI",
}


@st.cache_data(ttl=3600)
def _fetch_weekly_gp() -> dict[str, int] | None:
    """Fetch MLB schedule for the next 7 days and return {team_abbr: game_count}."""
    today = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=6)).strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?startDate={today}&endDate={end}&sportId=1&hydrate=team"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Failed to fetch MLB schedule for GP_week column")
        return None

    counts: dict[str, int] = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            for side in ("away", "home"):
                abbr = game.get("teams", {}).get(side, {}).get("team", {}).get("abbreviation")
                if abbr:
                    abbr = _MLB_TO_SAVANT_ABBR.get(abbr, abbr)
                    counts[abbr] = counts.get(abbr, 0) + 1
    return counts


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

def _render_matchup_overview():
    """Weekly Matchup Overview: head-to-head category comparison."""
    _PITCHER_POS = {"SP", "RP", "P"}

    opponent_name = st.sidebar.text_input("Opponent Team Name", value="Ben")

    all_rosters = _load_all_rosters()
    my_roster_df = _load_my_roster_df()
    hitters_sc = _load_parquet(SILVER / "statcast_hitters.parquet")
    pitchers_sc = _load_parquet(SILVER / "statcast_pitchers.parquet")

    if any(x is None for x in [all_rosters, my_roster_df, hitters_sc, pitchers_sc]):
        st.warning("Missing data for matchup overview.")
        return

    # Split opponent roster
    opp = all_rosters[all_rosters["team_name"].str.lower() == opponent_name.strip().lower()].copy()
    if opp.empty:
        st.info(f"No roster found for '{opponent_name}'.")
        return
    opp["player_name"] = opp["player_name"].str.strip()
    opp["player_type"] = opp["position"].apply(
        lambda p: "Pitcher" if p in _PITCHER_POS else "Hitter"
    )

    # Split my roster
    my = my_roster_df.copy()
    my = my[my["player_name"].notna() & (my["player_name"] != "None")]
    my["player_type"] = my["fantrax_position"].apply(
        lambda p: "Pitcher" if p in _PITCHER_POS else "Hitter"
    )

    id_map = _load_id_map()

    def _match_names_to_statcast(names: list[str], statcast_df: pd.DataFrame) -> pd.DataFrame:
        """Match player names to statcast data via ID map (savant_player_id join)."""
        if id_map is not None and "savant_player_id" in statcast_df.columns:
            idm = id_map[["player_name", "savant_player_id"]].drop_duplicates(
                subset=["player_name"], keep="first"
            )
            name_df = pd.DataFrame({"player_name": names})
            joined = name_df.merge(idm, on="player_name", how="inner")
            result = joined.merge(statcast_df, on="savant_player_id", how="inner", suffixes=("", "_sc"))
            if "player_name_sc" in result.columns:
                result = result.drop(columns=["player_name_sc"])
            return result
        # Fallback: fuzzy match
        sc_names = statcast_df["player_name"].dropna().unique().tolist()
        rows = []
        for name in names:
            res = process.extractOne(name, sc_names, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
            if res:
                rows.append(statcast_df[statcast_df["player_name"] == res[0]].iloc[0])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    my_h = _match_names_to_statcast(my[my["player_type"] == "Hitter"]["player_name"].tolist(), hitters_sc)
    opp_h = _match_names_to_statcast(opp[opp["player_type"] == "Hitter"]["player_name"].tolist(), hitters_sc)
    my_p = _match_names_to_statcast(my[my["player_type"] == "Pitcher"]["player_name"].tolist(), pitchers_sc)
    opp_p = _match_names_to_statcast(opp[opp["player_type"] == "Pitcher"]["player_name"].tolist(), pitchers_sc)

    # Build category comparisons
    def _safe(val, fmt=".3f"):
        return f"{val:{fmt}}" if pd.notna(val) else "N/A"

    my_hr = int((my_h["barrel_percentile"] > 50).sum()) if not my_h.empty else 0
    opp_hr = int((opp_h["barrel_percentile"] > 50).sum()) if not opp_h.empty else 0

    my_sb = int((my_h["sprint_speed"] > 28).sum()) if not my_h.empty and "sprint_speed" in my_h.columns else 0
    opp_sb = int((opp_h["sprint_speed"] > 28).sum()) if not opp_h.empty and "sprint_speed" in opp_h.columns else 0

    my_obp = my_h["est_woba"].mean() if not my_h.empty else np.nan
    opp_obp = opp_h["est_woba"].mean() if not opp_h.empty else np.nan

    my_k = len(my_p) if not my_p.empty else 0
    opp_k = len(opp_p) if not opp_p.empty else 0

    my_era = my_p["xera"].mean() if not my_p.empty else np.nan
    opp_era = opp_p["xera"].mean() if not opp_p.empty else np.nan

    # Detect RPs
    opp_rp_names = opp[opp["position"] == "RP"]["player_name"].tolist()
    opp_rp_count = len(opp_rp_names)

    categories = [
        ("HR (barrel%ile>50)", str(my_hr), str(opp_hr), my_hr - opp_hr),
        ("SB (speed>28)", str(my_sb), str(opp_sb), my_sb - opp_sb),
        ("OBP (avg xwOBA)", _safe(my_obp), _safe(opp_obp),
         (my_obp - opp_obp) if pd.notna(my_obp) and pd.notna(opp_obp) else 0),
        ("K (SP count)", str(my_k), str(opp_k), my_k - opp_k),
        ("ERA (avg xERA)", _safe(my_era), _safe(opp_era),
         (opp_era - my_era) if pd.notna(my_era) and pd.notna(opp_era) else 0),  # lower is better
        ("SVH", "PUNT", str(opp_rp_count) + " RPs", -1),  # always red
    ]

    rows = []
    my_wins = 0
    opp_wins = 0
    for cat, my_val, opp_val, diff in categories:
        if diff > 0.001:
            edge = "✅ My Edge"
            my_wins += 1
        elif diff < -0.001:
            edge = "❌ Opp Edge"
            opp_wins += 1
        else:
            edge = "⚠️ Close"
        rows.append({"Category": cat, "My Team": my_val, opponent_name: opp_val, "Edge": edge})

    with st.expander("Weekly Matchup Overview", expanded=True):
        comp_df = pd.DataFrame(rows)
        st.dataframe(
            comp_df.style.apply(
                lambda col: [
                    "background-color: #1a472a" if "My Edge" in v
                    else "background-color: #5c1a1a" if "Opp Edge" in v
                    else "background-color: #4a4a00" if "Close" in v
                    else ""
                    for v in col
                ] if col.name == "Edge" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(f"**Projected: {my_wins}-{opp_wins}**")


def page_session_prep():
    st.header("Session Prep")

    # Last refreshed
    ts = _last_refreshed()
    if ts:
        st.caption(f"Last refreshed: **{ts}**")
    else:
        st.warning("No data files found in gold/data/")

    # --- Weekly Matchup Overview ---
    _render_matchup_overview()

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

    # Start from FULL Fantrax roster, split by position
    my_roster_df = _load_my_roster_df()
    if my_roster_df is None or my_roster_df.empty:
        st.warning("Could not load my_roster CSV from Fantrax data.")
        return

    # Drop empty roster slots (player_name is "None" or NaN)
    my_roster_df = my_roster_df[
        my_roster_df["player_name"].notna()
        & (my_roster_df["player_name"] != "None")
    ].copy()

    _PITCHER_POSITIONS = {"SP", "RP", "P"}
    my_roster_df["player_type"] = my_roster_df["fantrax_position"].apply(
        lambda p: "Pitcher" if p in _PITCHER_POSITIONS else "Hitter"
    )

    hitters_statcast = _load_parquet(SILVER / "statcast_hitters.parquet")
    pitchers_statcast = _load_parquet(SILVER / "statcast_pitchers.parquet")

    roster_hitters = my_roster_df[my_roster_df["player_type"] == "Hitter"].copy()
    roster_pitchers = my_roster_df[my_roster_df["player_type"] == "Pitcher"].copy()

    # --- Use ID map for team/position and savant_player_id join ---
    id_map = _load_id_map()

    _hitter_metric_cols = ["woba", "est_woba", "xwoba_minus_woba",
                           "hard_hit_percentile", "barrel_percentile"]
    _pitcher_metric_cols = ["xera", "era", "xera_minus_era",
                            "k_percent", "barrel_percentile"]

    if id_map is not None:
        # Join ID map onto roster to get savant_player_id and team
        for df, statcast, metric_cols in [
            (roster_hitters, hitters_statcast, _hitter_metric_cols),
            (roster_pitchers, pitchers_statcast, _pitcher_metric_cols),
        ]:
            # Merge ID map for savant_player_id and team
            idm = id_map[["player_name", "savant_player_id", "team"]].drop_duplicates(
                subset=["player_name"], keep="first"
            )
            df_merged = df.merge(idm, on="player_name", how="left")

            # Join statcast by savant_player_id (exact join, no fuzzy)
            if statcast is not None and "savant_player_id" in statcast.columns:
                sc_cols = ["savant_player_id"] + [c for c in metric_cols if c in statcast.columns]
                sc_subset = statcast[sc_cols].drop_duplicates(subset=["savant_player_id"])
                df_merged = df_merged.merge(sc_subset, on="savant_player_id", how="left", suffixes=("", "_sc"))
                # If team came from ID map as NaN, try statcast
                if "team_sc" in df_merged.columns:
                    df_merged["team"] = df_merged["team"].fillna(df_merged["team_sc"])
                    df_merged = df_merged.drop(columns=["team_sc"])

            # Copy results back
            for col in ["team", "savant_player_id"] + metric_cols:
                if col in df_merged.columns:
                    df[col] = df_merged[col].values
    else:
        # Fallback: fuzzy merge if no ID map
        if hitters_statcast is not None:
            roster_hitters = _fuzzy_merge(
                roster_hitters, hitters_statcast,
                cols_to_add=["team"] + _hitter_metric_cols,
            )
        if pitchers_statcast is not None:
            roster_pitchers = _fuzzy_merge(
                roster_pitchers, pitchers_statcast,
                cols_to_add=["team"] + _pitcher_metric_cols,
            )

    # Fantrax position is ALWAYS the authoritative position
    for df in (roster_hitters, roster_pitchers):
        df["position"] = df["fantrax_position"]

    # Position sort order for hitters
    _POS_ORDER = {"C": 0, "1B": 1, "2B": 2, "3B": 3, "SS": 4, "OF": 5, "UT": 6}

    hit_col, pit_col = st.columns(2)

    # --- Hitters table (left) ---
    with hit_col:
        st.markdown("**Hitters**")
        hit_df = roster_hitters.copy()
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
            styled_h = hit_df[h_display].style.format(fmt, na_rep="-")
            if gap_col in hit_df.columns:
                styled_h = styled_h.map(_color_xwoba_gap, subset=[gap_col])
            st.dataframe(styled_h, use_container_width=True, hide_index=True,
                         height=max(400, 35 * len(hit_df) + 40))
        else:
            st.info("No hitters on roster.")

    # --- Pitchers table (right) ---
    with pit_col:
        st.markdown("**Pitchers**")
        pit_df = roster_pitchers.copy()
        if not pit_df.empty:
            era_gap = "xera_minus_era" if "xera_minus_era" in pit_df.columns else None
            if era_gap and era_gap in pit_df.columns:
                # Sort pitchers with data first (by gap desc), then those without data
                pit_df["_has_data"] = pit_df[era_gap].notna().astype(int)
                pit_df = pit_df.sort_values(["_has_data", era_gap], ascending=[False, False]).drop(columns=["_has_data"])

            p_display = ["player_name", "team", "position"]
            if "xera" in pit_df.columns:
                p_display.append("xera")
            if "era" in pit_df.columns:
                p_display.append("era")
            if era_gap and era_gap in pit_df.columns:
                p_display.append(era_gap)

            fmt = {c: "{:.2f}" for c in p_display if c not in ("player_name", "team", "position")}
            styled_p = pit_df[p_display].style.format(fmt, na_rep="-")
            if era_gap and era_gap in pit_df.columns:
                styled_p = styled_p.map(_color_xwoba_gap, subset=[era_gap])
            st.dataframe(styled_p, use_container_width=True, hide_index=True,
                         height=max(400, 35 * len(pit_df) + 40))
        else:
            st.info("No pitchers on roster.")

    # --- Breakout adds (free agents only) ---
    col1, col2 = st.columns(2)
    breakout_caption = (
        "These are free agents whose underlying Statcast quality significantly exceeds "
        "their surface stats. They are hitting the ball hard but not getting results yet "
        "— the process is right, the outcomes haven't caught up. These are buy-low "
        "candidates before the market notices."
    )

    # Load FanGraphs data once for position fill on FA breakout tables
    fg_lookup = _load_fangraphs_team_pos()

    breakout_id_map = _load_id_map()

    def _fill_fa_position(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing position/team on FA breakout players from ID map or FanGraphs."""
        if df is None or df.empty:
            return df
        df = df.copy()

        # Try ID map first
        if breakout_id_map is not None:
            idm = breakout_id_map[["player_name", "team", "position"]].drop_duplicates(
                subset=["player_name"], keep="first"
            )
            for col in ("team", "position"):
                if col not in df.columns:
                    df[col] = None
                missing = df[col].isna() | (df[col].astype(str).isin(["None", ""]))
                if missing.any():
                    fill = df.loc[missing].merge(
                        idm[["player_name", col]], on="player_name", how="left", suffixes=("_old", "")
                    )
                    if f"{col}_old" in fill.columns:
                        df.loc[missing, col] = fill[col].values
                    elif col in fill.columns:
                        df.loc[missing, col] = fill[col].values

        # Fallback: FanGraphs fuzzy match for remaining gaps
        for col_to_fill, fg_col in [("position", "fg_position"), ("team", "fg_team")]:
            if col_to_fill not in df.columns:
                df[col_to_fill] = None
            missing = df[col_to_fill].isna() | (df[col_to_fill].astype(str).isin(["None", ""]))
            if not missing.any() or fg_lookup.empty:
                continue
            for idx in df.index[missing]:
                name = df.at[idx, "player_name"]
                result = process.extractOne(
                    name, fg_lookup["player_name"].tolist(),
                    scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD,
                )
                if result is not None:
                    fg_row = fg_lookup.loc[fg_lookup["player_name"] == result[0]].iloc[0]
                    if fg_col in fg_row.index and pd.notna(fg_row[fg_col]):
                        df.at[idx, col_to_fill] = fg_row[fg_col]

        return df

    weekly_gp = _fetch_weekly_gp()

    def _add_gp_week(df: pd.DataFrame) -> pd.DataFrame:
        """Add GP_week column from MLB schedule if available."""
        if weekly_gp and "team" in df.columns:
            df["GP_week"] = df["team"].map(weekly_gp)
        return df

    with col1:
        st.subheader("Top 10 Breakout Hitter Adds")
        st.caption(breakout_caption)
        bh = _load_csv(GOLD / "breakout_hitters_fa.csv")
        if bh is not None:
            bh = _fill_fa_position(bh)
            bh = _add_gp_week(bh)
            show_cols = [c for c in ["player_name", "GP_week", "team", "position", "est_woba", "woba", "xwoba_minus_woba", "hard_hit_percentile", "barrel_percentile"] if c in bh.columns]
            fmt = {c: "{:.3f}" for c in show_cols if c not in ("player_name", "GP_week", "team", "position", "hard_hit_percentile", "barrel_percentile")}
            fmt.update({c: "{:.0f}" for c in show_cols if c in ("hard_hit_percentile", "barrel_percentile", "GP_week")})
            st.dataframe(bh[show_cols].head(10).style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Top 10 Breakout Pitcher Adds")
        st.caption(breakout_caption)
        bp = _load_csv(GOLD / "breakout_pitchers_fa.csv")
        if bp is not None:
            bp = _fill_fa_position(bp)
            bp = _add_gp_week(bp)
            show_cols = [c for c in ["player_name", "GP_week", "team", "position", "xera", "era", "xera_minus_era", "k_percent", "barrel_percentile"] if c in bp.columns]
            fmt = {c: "{:.2f}" for c in show_cols if c not in ("player_name", "GP_week", "team", "position", "k_percent", "barrel_percentile")}
            fmt.update({c: "{:.1f}" for c in show_cols if c in ("k_percent", "barrel_percentile")})
            fmt.update({c: "{:.0f}" for c in show_cols if c in ("GP_week",)})
            st.dataframe(bp[show_cols].head(10).style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)


def _breakout_style_maps():
    """Return shared color/symbol/opacity maps for breakout charts."""
    _LEAGUE_TEAMS = [
        "Ben", "Chad", "George", "J-Rod Show", "Jorp", "Mullets",
        "Negs", "One Pathetic Luzar", "Porter",
        "Professor McGonigle", "Rutsch Hour", "Young Gunz",
    ]
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
    return color_map, symbol_map, opacity_map


def _add_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Add ownership column if not already present, using ID map or all_rosters."""
    if "ownership" in df.columns:
        df["status"] = df["ownership"]
        return df

    # Try ID map first for ownership lookup
    id_map = _load_id_map()
    if id_map is not None:
        lookup_cols = ["player_name", "fantrax_team_name"]
        has_status = "status" in id_map.columns
        if has_status:
            lookup_cols.append("status")
        ownership_lookup = id_map[lookup_cols].rename(
            columns={"status": "idmap_status"}
        )
        # Statcast frames use plain source names; strip Fantrax's two-way
        # role suffix ("Shohei Ohtani-P") so those rows can join.
        ownership_lookup = ownership_lookup.assign(
            player_name=ownership_lookup["player_name"].str.replace(
                r"-[HP]$", "", regex=True
            )
        ).drop_duplicates(subset=["player_name"], keep="first")

        merged = df.merge(ownership_lookup, on="player_name", how="left")
        if has_status:
            # The id_map's own status column is the ownership signal:
            # 'owned' rows show the owning fantasy team, everything else
            # (status 'fa' or no id_map row at all) is a free agent.
            df["status"] = np.where(
                merged["idmap_status"] == "owned",
                merged["fantrax_team_name"],
                "FA",
            )
        else:
            # Older id_map without a status column: FA rows carry
            # empty-string team names, which fillna alone would miss.
            df["status"] = (
                merged["fantrax_team_name"].replace("", np.nan).fillna("FA").values
            )
        return df

    # Fallback: all_rosters
    all_rosters = _load_all_rosters()
    my_names = _load_my_roster()
    if all_rosters is not None:
        df["status"] = df["player_name"].apply(
            lambda n: _ownership_status(n, all_rosters, my_names)
        )
    else:
        df["status"] = df["player_name"].apply(
            lambda n: MY_TEAM if n in my_names else "Unknown"
        )
    return df


def page_breakout_board():
    st.header("Breakout Board")

    color_map, symbol_map, opacity_map = _breakout_style_maps()

    # ---- Hitter Breakout Board ----
    st.subheader("Hitter Breakout Board")
    st.caption(
        "Players above the line have better underlying contact quality than their "
        "results show. Buy candidates."
    )

    bh = _load_csv(GOLD / "breakout_hitters_all.csv")
    if bh is not None:
        required = {"est_woba", "woba", "player_name", "team"}
        if not required.issubset(bh.columns):
            st.error(f"Missing columns. Need {required}, have {set(bh.columns)}")
        else:
            bh = _add_ownership(bh)

            size_col = "hard_hit_percentile" if "hard_hit_percentile" in bh.columns else None
            if size_col is None and "avg_hit_speed" in bh.columns:
                size_col = "avg_hit_speed"

            bh["label"] = bh["player_name"].str.split().str[-1] + " (" + bh["team"] + ")"

            hover_fields = {
                "player_name": True, "team": True, "position": True,
                "est_woba": ":.3f", "woba": ":.3f",
            }
            for hf in ["xwoba_minus_woba", "est_woba_minus_woba_diff"]:
                if hf in bh.columns:
                    hover_fields[hf] = ":.3f"
                    break
            if "hard_hit_percentile" in bh.columns:
                hover_fields["hard_hit_percentile"] = ":.1f"
            if "brl_percent" in bh.columns:
                hover_fields["brl_percent"] = ":.1f"

            if size_col:
                bh["_plot_size"] = bh[size_col]
                bh.loc[bh["status"] == "FA", "_plot_size"] = bh.loc[bh["status"] == "FA", size_col] * 1.4
                plot_size_col = "_plot_size"
            else:
                plot_size_col = None

            category_order = ["FA"] + sorted(set(bh["status"].unique()) - {"FA"})

            fig_h = px.scatter(
                bh, x="est_woba", y="woba",
                color="status", color_discrete_map=color_map,
                symbol="status", symbol_map=symbol_map,
                size=plot_size_col, size_max=20,
                hover_data=hover_fields,
                labels={"est_woba": "xwOBA (Expected)", "woba": "wOBA (Actual)", "status": "Owner"},
                title="xwOBA vs wOBA",
                category_orders={"status": category_order},
            )
            for trace in fig_h.data:
                trace.opacity = opacity_map.get(trace.name, 0.6)

            fa_data = bh[bh["status"] == "FA"]
            if not fa_data.empty:
                fig_h.add_trace(go.Scatter(
                    x=fa_data["est_woba"], y=fa_data["woba"] + 0.004,
                    mode="text", text=fa_data["label"], textposition="top center",
                    textfont=dict(size=12, color="white"),
                    showlegend=False, hoverinfo="skip",
                ))

            lo = min(bh["est_woba"].min(), bh["woba"].min()) - 0.010
            hi = max(bh["est_woba"].max(), bh["woba"].max()) + 0.010
            fig_h.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(dash="dash", color="white", width=2),
                showlegend=False, name="x = y",
            ))
            fig_h.update_layout(height=650)
            st.plotly_chart(fig_h, use_container_width=True)

    # ---- Pitcher Breakout Board ----
    st.subheader("Pitcher Breakout Board")
    st.caption(
        "Players below the line have ERAs inflated beyond what their stuff quality "
        "suggests. Their xERA says they should be better. Buy candidates."
    )

    bp = _load_parquet(SILVER / "statcast_pitchers.parquet")
    if bp is not None:
        required_p = {"xera", "era", "player_name", "team"}
        if not required_p.issubset(bp.columns):
            st.error(f"Missing pitcher columns. Need {required_p}, have {set(bp.columns)}")
        else:
            bp = _add_ownership(bp)

            bp["label"] = bp["player_name"].str.split().str[-1] + " (" + bp["team"] + ")"

            hover_fields_p = {
                "player_name": True, "team": True,
                "xera": ":.2f", "era": ":.2f",
            }
            if "xera_minus_era" in bp.columns:
                hover_fields_p["xera_minus_era"] = ":.2f"
            if "k_percent" in bp.columns:
                hover_fields_p["k_percent"] = ":.1f"
            if "barrel_percentile" in bp.columns:
                hover_fields_p["barrel_percentile"] = ":.1f"

            p_size_col = "hard_hit_percentile" if "hard_hit_percentile" in bp.columns else None
            if p_size_col:
                bp["_plot_size"] = bp[p_size_col]
                bp.loc[bp["status"] == "FA", "_plot_size"] = bp.loc[bp["status"] == "FA", p_size_col] * 1.4
                plot_size_p = "_plot_size"
            else:
                plot_size_p = None

            category_order_p = ["FA"] + sorted(set(bp["status"].unique()) - {"FA"})

            fig_p = px.scatter(
                bp, x="xera", y="era",
                color="status", color_discrete_map=color_map,
                symbol="status", symbol_map=symbol_map,
                size=plot_size_p, size_max=20,
                hover_data=hover_fields_p,
                labels={"xera": "xERA (Expected)", "era": "ERA (Actual)", "status": "Owner"},
                title="xERA vs ERA",
                category_orders={"status": category_order_p},
            )
            for trace in fig_p.data:
                trace.opacity = opacity_map.get(trace.name, 0.6)

            fa_pitchers = bp[bp["status"] == "FA"]
            if not fa_pitchers.empty:
                fig_p.add_trace(go.Scatter(
                    x=fa_pitchers["xera"], y=fa_pitchers["era"] + 0.08,
                    mode="text", text=fa_pitchers["label"], textposition="top center",
                    textfont=dict(size=12, color="white"),
                    showlegend=False, hoverinfo="skip",
                ))

            lo_p = min(bp["xera"].min(), bp["era"].min()) - 0.20
            hi_p = max(bp["xera"].max(), bp["era"].max()) + 0.20
            fig_p.add_trace(go.Scatter(
                x=[lo_p, hi_p], y=[lo_p, hi_p], mode="lines",
                line=dict(dash="dash", color="white", width=2),
                showlegend=False, name="x = y",
            ))
            fig_p.update_layout(height=650)
            st.plotly_chart(fig_p, use_container_width=True)


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
