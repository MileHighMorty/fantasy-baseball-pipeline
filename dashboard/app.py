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
import yaml
from rapidfuzz import process, fuzz

# Ensure project root is on the path so we can import scripts.weekly_refresh
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gold.ownership import attach_status  # noqa: E402  (needs PROJECT_ROOT on path)
from dashboard import theme  # noqa: E402  (design system: palette, chart theme, stylers)

GOLD = PROJECT_ROOT / "gold" / "data"
SILVER = PROJECT_ROOT / "silver" / "data"
FANGRAPHS = PROJECT_ROOT / "bronze" / "data" / "fangraphs"
FANTRAX = PROJECT_ROOT / "bronze" / "data" / "fantrax"
CONFIG = PROJECT_ROOT / "config" / "settings.yaml"
MY_TEAM = "Rutsch Hour"
FUZZY_THRESHOLD = 85

# The 12 fantasy teams in the league. Single source for the breakout-board
# team picker and the per-owner style map.
LEAGUE_TEAMS = [
    "Ben", "Chad", "George", "J-Rod Show", "Jorp", "Mullets",
    "Negs", "One Pathetic Luzar", "Porter",
    "Professor McGonigle", "Rutsch Hour", "Young Gunz",
]
OTHER_TEAMS = [t for t in LEAGUE_TEAMS if t != MY_TEAM]


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

_PITCHER_POSITIONS = {"SP", "RP", "P"}


def _is_pitcher(position: str | None) -> bool:
    """Classify a Fantrax position string as a pitcher.

    Fantrax encodes multi-eligibility as a comma list ("SP,RP"), so a plain
    set-membership test misses those and drops the player into the hitters
    table. Split on comma and treat the player as a pitcher if ANY eligibility
    token is a pitching slot.
    """
    return any(tok.strip() in _PITCHER_POSITIONS for tok in str(position).split(","))


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


@st.cache_data(ttl=3600)
def _load_fangraphs_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-player FanGraphs batting and pitching stat columns for the matchup board.

    Pulls only the columns the 9-category aggregation needs, each keyed by
    ``IDfg`` for an id-map join. Counting stats are season totals; ``OBP`` /
    ``ERA`` / ``WHIP`` are per-player rates weighted at aggregation time.

    Returns:
        ``(batting, pitching)``. Either is an empty frame (correct columns)
        when its CSV is absent, so callers never KeyError.
    """
    bat_cols = ["IDfg", "HR", "R", "RBI", "SB", "OBP", "PA"]
    pit_cols = ["IDfg", "SO", "W", "ERA", "WHIP", "IP", "SV"]

    def _latest(pattern: str, cols: list[str]) -> pd.DataFrame:
        files = sorted(FANGRAPHS.glob(pattern))
        if not files:
            return pd.DataFrame(columns=cols)
        return pd.read_csv(files[-1], usecols=lambda c: c in cols)

    return _latest("*_batting.csv", bat_cols), _latest("*_pitching.csv", pit_cols)


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
    """Styler for xwoba-minus-woba gap magnitude (delegates to the theme)."""
    return theme.gap_stability_style(val)


def _score_color(val):
    """Styler for stream_score / generic 0-100 scores (delegates to the theme)."""
    return theme.score_style(val)


def _edge_color(edge_text):
    """Styler for the matchup Edge column (delegates to the theme)."""
    return theme.edge_style(edge_text)


def _configured_opponent() -> str | None:
    """Return ``fantrax.current_opponent`` from settings.yaml, or None.

    Used only to pick the opponent dropdown's default. Missing file or key,
    or a non-string/blank value, yields None so the caller falls back to the
    first team rather than crashing.
    """
    try:
        with open(CONFIG, encoding="utf-8") as f:
            settings = yaml.safe_load(f) or {}
        value = settings.get("fantrax", {}).get("current_opponent")
        return value if isinstance(value, str) and value.strip() else None
    except (OSError, yaml.YAMLError):
        return None


def _dash_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the literal string ``"None"`` to NaN so na_rep renders it as "-".

    A roster player with no Statcast row (prospect/IL) can carry the string
    ``"None"`` in a text column; the Styler's ``na_rep`` catches real NaN but
    not that string. Only object columns are touched, so numeric formatting is
    unaffected.
    """
    out = df.copy()
    obj_cols = out.select_dtypes(include="object").columns
    if len(obj_cols):
        out[obj_cols] = out[obj_cols].replace("None", np.nan)
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

# "Close" margins for the matchup edge call, as named constants rather than
# magic numbers. Counting categories use a relative margin (a 1-HR gap on 80 is
# noise); rate categories use absolute margins on their natural scale.
_CLOSE_COUNTING_PCT = 0.03   # within 3% of the larger total → Close
_CLOSE_OBP = 0.005           # OBP points
_CLOSE_ERA = 0.20            # earned runs / 9
_CLOSE_WHIP = 0.03           # baserunners / IP


def _aggregate_hitting(
    roster: pd.DataFrame, fg_batting: pd.DataFrame, id_map: pd.DataFrame | None
) -> tuple[dict, int, int]:
    """Aggregate a roster's hitters into the five hitting categories.

    Splits to hitters first (so a pitcher's near-empty qual=0 batting row never
    pollutes the totals), joins to FanGraphs by ``fangraphs_id``, SUMs the
    counting stats, and PA-weights OBP — ``Σ(OBP*PA)/Σ PA``, the pooled team
    OBP, not a naive mean. Unresolved hitters (no FanGraphs row) drop out via
    skipna sums rather than breaking the aggregate.

    Args:
        roster: A team's rows with a ``player_type`` column.
        fg_batting: FanGraphs batting stats keyed by ``IDfg``.
        id_map: The player ID map (``player_name`` → ``fangraphs_id``).

    Returns:
        ``(totals, n_resolved, n_hitters)`` where totals has HR/R/RBI/SB/OBP.
    """
    hitters = roster[roster["player_type"] == "Hitter"]
    n_hitters = len(hitters)
    empty = {"HR": 0, "R": 0, "RBI": 0, "SB": 0, "OBP": np.nan}
    if id_map is None or fg_batting.empty or n_hitters == 0:
        return empty, 0, n_hitters

    idm = id_map[["player_name", "fangraphs_id"]].drop_duplicates("player_name")
    joined = hitters.merge(idm, on="player_name", how="left").merge(
        fg_batting, left_on="fangraphs_id", right_on="IDfg", how="left"
    )
    resolved = joined[joined["IDfg"].notna()]
    n_resolved = len(resolved)

    pa = resolved["PA"].sum()
    obp = (resolved["OBP"] * resolved["PA"]).sum() / pa if pa > 0 else np.nan
    totals = {
        "HR": resolved["HR"].sum(),
        "R": resolved["R"].sum(),
        "RBI": resolved["RBI"].sum(),
        "SB": resolved["SB"].sum(),
        "OBP": obp,
    }
    return totals, n_resolved, n_hitters


def _aggregate_pitching(
    roster: pd.DataFrame, fg_pitching: pd.DataFrame, id_map: pd.DataFrame | None
) -> tuple[dict, int, int]:
    """Aggregate a roster's pitchers into the four computed pitching categories.

    Splits to pitchers first, joins to FanGraphs by ``fangraphs_id``, SUMs K
    (``SO``) and W, and IP-weights ERA and WHIP — ``Σ(rate*IP)/Σ IP``. That
    weighting is exact, not an approximation: since ``ER = ERA*IP/9`` and
    ``BB+H = WHIP*IP``, the IP-weighted mean equals the pooled ``Σ ER /Σ IP *9``
    and ``Σ(BB+H)/Σ IP`` despite the raw ER/BB/H columns being absent. Saves are
    summed for the (punted) SVH row's context.

    Args:
        roster: A team's rows with a ``player_type`` column.
        fg_pitching: FanGraphs pitching stats keyed by ``IDfg``.
        id_map: The player ID map (``player_name`` → ``fangraphs_id``).

    Returns:
        ``(totals, n_resolved, n_pitchers)`` where totals has K/W/ERA/WHIP/SV.
    """
    pitchers = roster[roster["player_type"] == "Pitcher"]
    n_pitchers = len(pitchers)
    empty = {"K": 0, "W": 0, "ERA": np.nan, "WHIP": np.nan, "SV": 0}
    if id_map is None or fg_pitching.empty or n_pitchers == 0:
        return empty, 0, n_pitchers

    idm = id_map[["player_name", "fangraphs_id"]].drop_duplicates("player_name")
    joined = pitchers.merge(idm, on="player_name", how="left").merge(
        fg_pitching, left_on="fangraphs_id", right_on="IDfg", how="left"
    )
    resolved = joined[joined["IDfg"].notna()]
    n_resolved = len(resolved)

    ip = resolved["IP"].sum()
    era = (resolved["ERA"] * resolved["IP"]).sum() / ip if ip > 0 else np.nan
    whip = (resolved["WHIP"] * resolved["IP"]).sum() / ip if ip > 0 else np.nan
    totals = {
        "K": resolved["SO"].sum(),
        "W": resolved["W"].sum(),
        "ERA": era,
        "WHIP": whip,
        "SV": resolved["SV"].sum(),
    }
    return totals, n_resolved, n_pitchers


def _category_edge(my_val: float, opp_val: float, higher_better: bool, margin: float) -> str:
    """Return the styled edge label for one category.

    ``margin`` is the absolute "too close to call" band. A missing value (no
    resolved players on a side) yields Close rather than a false edge.
    """
    if pd.isna(my_val) or pd.isna(opp_val):
        return "⚠️ Close"
    if abs(my_val - opp_val) <= margin:
        return "⚠️ Close"
    my_better = my_val > opp_val if higher_better else my_val < opp_val
    return "✅ My Edge" if my_better else "❌ Opp Edge"


def _render_matchup_overview():
    """Roster Strength Comparison: season-to-date OBP 5x5 board, my team vs opponent.

    Both rosters come from all_rosters and are aggregated against the live
    FanGraphs (qual=0) stats via the id map. Counting categories are summed,
    OBP/ERA/WHIP are sample-weighted, and SVH is conceded (punt strategy). The
    figures are season-to-date production — a roster-strength comparison, not a
    projected weekly result.
    """
    all_rosters = _load_all_rosters()
    fg_batting, fg_pitching = _load_fangraphs_stats()
    id_map = _load_id_map()

    if all_rosters is None or fg_batting.empty or fg_pitching.empty:
        st.warning("Missing data for roster strength comparison.")
        return

    # Opponent picker: the real league teams from the roster data, minus my own.
    # Default to config fantrax.current_opponent when set, else the first team.
    opponent_choices = sorted(
        t for t in all_rosters["team_name"].dropna().unique().tolist()
        if t != MY_TEAM
    )
    if not opponent_choices:
        st.info("No opponent teams found in roster data.")
        return
    configured = _configured_opponent()
    default_idx = (
        opponent_choices.index(configured) if configured in opponent_choices else 0
    )
    opponent_name = st.sidebar.selectbox(
        "Opponent Team", opponent_choices, index=default_idx
    )

    def _team_roster(name: str) -> pd.DataFrame:
        """One team's roster rows from all_rosters, tagged hitter/pitcher."""
        r = all_rosters[
            all_rosters["team_name"].str.lower() == name.strip().lower()
        ].copy()
        r["player_name"] = r["player_name"].str.strip()
        r["player_type"] = r["position"].apply(
            lambda p: "Pitcher" if _is_pitcher(p) else "Hitter"
        )
        return r

    my_roster = _team_roster(MY_TEAM)
    opp_roster = _team_roster(opponent_name)
    if my_roster.empty or opp_roster.empty:
        st.info("Could not load both rosters for the matchup.")
        return

    my_h, my_h_n, my_h_tot = _aggregate_hitting(my_roster, fg_batting, id_map)
    opp_h, opp_h_n, opp_h_tot = _aggregate_hitting(opp_roster, fg_batting, id_map)
    my_p, my_p_n, my_p_tot = _aggregate_pitching(my_roster, fg_pitching, id_map)
    opp_p, opp_p_n, opp_p_tot = _aggregate_pitching(opp_roster, fg_pitching, id_map)

    def _count_margin(a: float, b: float) -> float:
        """Relative close-band for a counting category (3% of the larger side)."""
        return _CLOSE_COUNTING_PCT * max(a, b, 1)

    # (label, my, opp, higher_better, margin, value-format)
    specs = [
        ("HR", my_h["HR"], opp_h["HR"], True, _count_margin(my_h["HR"], opp_h["HR"]), "int"),
        ("R", my_h["R"], opp_h["R"], True, _count_margin(my_h["R"], opp_h["R"]), "int"),
        ("RBI", my_h["RBI"], opp_h["RBI"], True, _count_margin(my_h["RBI"], opp_h["RBI"]), "int"),
        ("SB", my_h["SB"], opp_h["SB"], True, _count_margin(my_h["SB"], opp_h["SB"]), "int"),
        ("OBP", my_h["OBP"], opp_h["OBP"], True, _CLOSE_OBP, "obp"),
        ("K", my_p["K"], opp_p["K"], True, _count_margin(my_p["K"], opp_p["K"]), "int"),
        ("W", my_p["W"], opp_p["W"], True, _count_margin(my_p["W"], opp_p["W"]), "int"),
        ("ERA", my_p["ERA"], opp_p["ERA"], False, _CLOSE_ERA, "rate2"),
        ("WHIP", my_p["WHIP"], opp_p["WHIP"], False, _CLOSE_WHIP, "rate2"),
    ]

    def _fmt(val, kind: str) -> str:
        if pd.isna(val):
            return "—"
        if kind == "int":
            return f"{val:.0f}"
        if kind == "obp":
            return f"{val:.3f}"
        return f"{val:.2f}"

    rows = []
    my_wins = opp_wins = 0
    for label, mv, ov, higher_better, margin, kind in specs:
        edge = _category_edge(mv, ov, higher_better, margin)
        if "My Edge" in edge:
            my_wins += 1
        elif "Opp Edge" in edge:
            opp_wins += 1
        rows.append({
            "Category": label, "My Team": _fmt(mv, kind),
            opponent_name: _fmt(ov, kind), "Edge": edge,
        })

    # SVH is punted: we concede it. Show opponent's save total as context.
    rows.append({
        "Category": "SVH", "My Team": "PUNT",
        opponent_name: f"{opp_p['SV']:.0f} SV", "Edge": "❌ Opp Edge",
    })
    opp_wins += 1

    with st.expander("Roster Strength Comparison", expanded=True):
        st.caption(
            "Season-to-date production of each roster's players, by category. "
            "This compares overall roster strength — whose team has produced more "
            "this season — not a projected weekly result. Counting rows are "
            "season-to-date totals; OBP/ERA/WHIP are season-to-date rates "
            "(OBP PA-weighted, ERA/WHIP IP-weighted)."
        )
        comp_df = pd.DataFrame(rows)
        st.dataframe(
            comp_df.style.apply(
                lambda col: [_edge_color(v) for v in col]
                if col.name == "Edge" else [""] * len(col),
                axis=0,
            ),
            use_container_width=True,
            hide_index=True,
        )
        # Season-long edge per category, NOT a projected weekly score: how many
        # of the nine computed categories my roster currently leads.
        st.metric("Season-to-date roster edge", f"Stronger in {my_wins} of 9 categories")
        # Coverage honesty: counting categories only include players that resolve
        # to a FanGraphs row, so make the denominator visible per side.
        st.caption(
            f"Hitting totals from {my_h_n}/{my_h_tot} (me) and "
            f"{opp_h_n}/{opp_h_tot} ({opponent_name}) rostered hitters; "
            f"pitching from {my_p_n}/{my_p_tot} and {opp_p_n}/{opp_p_tot} pitchers. "
            "Unresolved bench/injured/minor-league players are excluded from "
            "counting totals."
        )
        st.caption(
            "Season-to-date roster comparison. A true rest-of-week matchup "
            "projection (games remaining × per-game rates, probable starters) "
            "is a planned enhancement."
        )


def page_session_prep():
    st.header("Session Prep")

    # Last refreshed
    ts = _last_refreshed()
    if ts:
        st.caption(f"Last refreshed: **{ts}**")
    else:
        st.warning("No data files found in gold/data/")

    # --- Roster Strength Comparison ---
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

    my_roster_df["player_type"] = my_roster_df["fantrax_position"].apply(
        lambda p: "Pitcher" if _is_pitcher(p) else "Hitter"
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
            styled_h = _dash_missing(hit_df[h_display]).style.format(fmt, na_rep="-")
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
            styled_p = _dash_missing(pit_df[p_display]).style.format(fmt, na_rep="-")
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
                elif df[col].dtype != object:
                    # A column read entirely empty from CSV infers as float64;
                    # pandas 3.0 refuses to assign a string into it. Coerce to
                    # object so the fill below works regardless of source dtype.
                    df[col] = df[col].astype(object)
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
            elif df[col_to_fill].dtype != object:
                df[col_to_fill] = df[col_to_fill].astype(object)
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
            st.dataframe(_dash_missing(bh[show_cols].head(10)).style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)

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
            st.dataframe(_dash_missing(bp[show_cols].head(10)).style.format(fmt, na_rep="-"), use_container_width=True, hide_index=True)


def _breakout_style_maps():
    """Return shared color/symbol/opacity maps for owner-coloured charts.

    Colours come from the single-source theme palette (muted, harmonized with
    the dark frame; my team is the bright accent, FA a neutral gray) rather than
    the old bright flat-UI set.
    """
    color_map = {"FA": theme.OWNER_COLORS["FA"]}
    symbol_map = {"FA": "circle"}
    opacity_map = {"FA": 1.0}
    for team in LEAGUE_TEAMS:
        color_map[team] = theme.OWNER_COLORS.get(team, theme.TEXT_MUTED)
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


def _breakout_lens_scatter(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    gap: str,
    x_label: str,
    y_label: str,
    title: str,
    diag_pad: float,
    label_dy: float,
    gap_fmt: str,
):
    """Build a single-lens breakout scatter coloured and sized by gap magnitude.

    ``gap`` is the distance off the x=y diagonal (xwOBA-wOBA for hitters,
    xERA-ERA for pitchers) and is always positive within the breakout set, so
    a continuous colour+size scale makes the strongest buy signals pop without
    leaning on owner colour (moot in a single-lens view).  Each point is
    labelled with the player's last name and the gap magnitude.

    Args:
        df: Already ownership-filtered breakout frame (non-empty).
        x, y: Expected vs actual metric columns.
        gap: Signed off-diagonal gap column used for colour, size, and label.
        x_label, y_label, title: Display strings.
        diag_pad: Padding to extend the x=y reference line past the data.
        label_dy: Vertical offset for the text labels above each point.
        gap_fmt: Numeric format (e.g. ``".3f"``) for the gap in the label.

    Returns:
        A Plotly figure.
    """
    df = df.copy()
    # Format the gap with its OWN sign (the "+" format flag), so a positive
    # hitter buy reads "+0.055" and a negative pitcher buy reads "-1.78" — a
    # hardcoded "+" used to render negative gaps as "+-1.78".
    df["_gap_label"] = (
        df["player_name"].str.split().str[-1]
        + " (" + df[gap].map(lambda v: format(v, f"+{gap_fmt}")) + ")"
    )

    hover = {x: f":{gap_fmt}", y: f":{gap_fmt}", gap: f":{gap_fmt}"}
    for opt in ("team", "position"):
        if opt in df.columns:
            hover[opt] = True

    # Marker size encodes the MAGNITUDE of the gap; direction (buy vs sell) is
    # already carried by colour. The pitcher gap (xera-era) is negative for a
    # breakout by the correct sign convention, and plotly rejects negative
    # marker sizes — so size must use the absolute value for both player types.
    df["_size"] = df[gap].abs()

    # Every point here is already a buy (single breakout lens); colour encodes
    # only the STRENGTH of the buy, so it runs neutral->green by magnitude. A
    # diverging red/green scale would wrongly paint the strongest pitcher buys
    # (negative signed gap) red, so magnitude + a sequential-green scale is the
    # semantically correct recolour of the old Plasma.
    fig = px.scatter(
        df, x=x, y=y,
        color="_size", color_continuous_scale=theme.SEQUENTIAL_GOOD,
        size="_size", size_max=24,
        hover_name="player_name", hover_data={**hover, "_size": False},
        labels={x: x_label, y: y_label, "_size": "Buy strength"},
        title=title,
    )

    lo = min(df[x].min(), df[y].min()) - diag_pad
    hi = max(df[x].max(), df[y].max()) + diag_pad
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color=theme.TEXT_MUTED, width=2),
        showlegend=False, name="x = y", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y] + label_dy, mode="text", text=df["_gap_label"],
        textposition="top center", textfont=dict(size=11, color=theme.TEXT),
        showlegend=False, hoverinfo="skip",
    ))
    theme.apply_chart_theme(fig, height=620)
    return fig


# SP-focus toggle for the pitcher views. SVH is punted and the league rewards
# starter volume, so relievers are noise on the breakout/regression lists — the
# full-Savant population floods them with RPs. The pitcher views default to
# startable arms; "All pitchers" restores relievers. View-only: the gold CSVs
# and parquets keep every reliever, this just filters what is plotted.
PITCHER_SCOPE_STARTABLE = "Startable (SP/SP,RP)"
PITCHER_SCOPE_ALL = "All pitchers"
PITCHER_SCOPE_OPTIONS = [PITCHER_SCOPE_STARTABLE, PITCHER_SCOPE_ALL]


def _pitcher_scope_toggle(key: str) -> str:
    """Render the SP-focus radio and return the selected scope.

    Keyed so the Breakout Board and Regression Watch toggles stay independent.

    Args:
        key: Unique Streamlit widget key for this instance of the toggle.

    Returns:
        The selected scope string (one of :data:`PITCHER_SCOPE_OPTIONS`).
    """
    return st.radio(
        "Pitcher scope", PITCHER_SCOPE_OPTIONS, horizontal=True, key=key
    )


def _filter_startable(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Keep only startable pitchers when *scope* asks for it.

    "Startable" keeps rows whose Fantrax ``position`` eligibility contains
    "SP" (so "SP" and "SP,RP" stay, pure "RP" drops). One contains-"SP" test
    serves both pitcher frames despite their different sourcing: the breakout
    CSV's ``position`` is the post-ownership Fantrax string, the regression /
    statcast frame's is the silver Statcast string, but both carry the same
    SP/RP/SP,RP values. A NaN position is unresolved and treated as NOT
    startable (dropped). Defensive: an absent ``position`` column or a None
    frame is returned untouched rather than raising.

    Args:
        df: A pitcher frame (or None when the CSV/parquet is missing).
        scope: The selected pitcher scope; only filters under "Startable".

    Returns:
        The filtered frame (or *df* unchanged when not filtering).
    """
    if df is None or scope != PITCHER_SCOPE_STARTABLE or "position" not in df.columns:
        return df
    return df[df["position"].str.contains("SP", na=False)]


def _render_breakout_lens(df, own_value, lens, *, noun, required, **scatter_kwargs):
    """Filter a breakout frame to one ownership lens and render its scatter.

    Renders nothing but a friendly message when the lens is empty — FA and
    per-team slices are legitimately thin or zero, and an empty chart reads
    as broken.

    Args:
        df: A breakout frame (or None when the CSV is missing).
        own_value: The ``ownership`` value this lens selects.
        lens: The lens label, used for the empty-state message.
        noun: "hitter" or "pitcher", for messages.
        required: Columns the scatter needs.
        **scatter_kwargs: Forwarded to :func:`_breakout_lens_scatter`.
    """
    if df is None:
        return
    if not required.issubset(df.columns):
        st.error(f"Missing {noun} columns. Need {required}, have {set(df.columns)}")
        return
    sub = df[df["ownership"] == own_value].sort_values(
        scatter_kwargs["gap"], ascending=False
    )
    if sub.empty:
        where = {"My Roster": "your roster", "Available (FA)": "free agents"}.get(
            lens, own_value
        )
        st.info(f"No {noun} breakout candidates for {where}.")
        return
    st.plotly_chart(
        _breakout_lens_scatter(sub, **scatter_kwargs), use_container_width=True
    )


def _stacked_panel(
    attached: pd.DataFrame,
    *,
    x: str,
    y: str,
    gap_col: str,
    invert_gap: bool,
    x_label: str,
    y_label: str,
    title: str,
    diag_pad: float,
    gap_fmt: str,
    noun: str,
) -> tuple[int, int]:
    """Render one roster-vs-available gap panel; return (n_roster, n_fa) plotted.

    ``buy_magnitude`` is normalized so POSITIVE = buy/breakout for BOTH hitters
    and pitchers — the pitcher gap (xera-era, where buy is negative) is negated
    so a single diverging colour scale reads the same direction in both panels.
    Roster players are stars, free agents circles.

    Args:
        attached: Statcast frame with ``status``/``fantrax_team_name`` attached.
        x, y: Expected vs actual metric columns.
        gap_col: Signed gap column (``xwoba_minus_woba`` / ``xera_minus_era``).
        invert_gap: Negate the gap so positive = buy (True for pitchers).
        x_label, y_label, title: Display strings.
        diag_pad: Padding for the x=y reference line.
        gap_fmt: Numeric format for hover values.
        noun: "hitter"/"pitcher" for the empty-state message.

    Returns:
        ``(n_roster, n_fa)`` actually plotted.
    """
    is_mine = (attached["status"] == "owned") & (
        attached["fantrax_team_name"] == MY_TEAM
    )
    sub = pd.concat(
        [attached[is_mine], attached[attached["status"] == "fa"]], ignore_index=True
    ).dropna(subset=[x, y, gap_col])

    if sub.empty:
        st.info(f"No qualified {noun}s to show for this view.")
        return 0, 0

    mine = (sub["status"] == "owned") & (sub["fantrax_team_name"] == MY_TEAM)
    sub["group"] = np.where(mine, "My Roster", "Available (FA)")
    # Normalize: positive = buy/breakout for both panels.
    sub["buy_magnitude"] = -sub[gap_col] if invert_gap else sub[gap_col]
    sub["_marker"] = np.where(mine, 15, 9)  # roster stars read larger

    hover = {"group": True, "_marker": False,
             x: f":{gap_fmt}", y: f":{gap_fmt}", "buy_magnitude": f":{gap_fmt}"}
    if "fantrax_position" in sub.columns:
        hover["fantrax_position"] = True

    fig = px.scatter(
        sub, x=x, y=y,
        color="buy_magnitude", color_continuous_scale=theme.DIVERGING_SCALE,
        color_continuous_midpoint=0,
        symbol="group", symbol_map={"My Roster": "star", "Available (FA)": "circle"},
        size="_marker", size_max=16,
        hover_name="player_name", hover_data=hover,
        labels={x: x_label, y: y_label, "buy_magnitude": "Buy signal"},
        title=title,
    )
    lo = min(sub[x].min(), sub[y].min()) - diag_pad
    hi = max(sub[x].max(), sub[y].max()) + diag_pad
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color=theme.TEXT_MUTED, width=2),
        showlegend=False, name="x = y", hoverinfo="skip",
    ))
    theme.apply_chart_theme(fig, height=600)
    st.plotly_chart(fig, use_container_width=True)
    return int(mine.sum()), int((~mine).sum())


def _render_roster_vs_available():
    """Stacked roster-vs-available gap view (the fourth Breakout Board lens).

    Reads the silver Statcast parquets directly (the full qualified population,
    including the "performing as expected" middle a roster baseline needs) and
    attaches ownership/position by vendor id — deliberately NOT the
    breakout_/regression_ CSVs, which drop the middle and carry the inverted
    pitcher buy/sell labels.
    """
    st.caption(
        "Your roster (★) vs available free agents (●) on one gap heat map. "
        "GREEN = BUY (underlying quality beats surface results, expect positive "
        "regression); RED = SELL (results beat quality, expect negative "
        "regression); the dashed line is break-even. Qualified players shown — "
        "roster players below the Statcast qualification floor (minors / IL / "
        "low PA-IP) do not appear."
    )

    sh = _load_parquet(SILVER / "statcast_hitters.parquet")
    sp = _load_parquet(SILVER / "statcast_pitchers.parquet")
    if sh is None or sp is None:
        return

    st.subheader("Hitters — xwOBA vs wOBA")
    nr, nfa = _stacked_panel(
        attach_status(sh),
        x="est_woba", y="woba", gap_col="xwoba_minus_woba", invert_gap=False,
        x_label="xwOBA (Expected)", y_label="wOBA (Actual)",
        title="Hitters: roster vs available", diag_pad=0.010, gap_fmt=".3f",
        noun="hitter",
    )
    st.caption(f"My Roster: {nr} · Available: {nfa}")

    st.subheader("Pitchers — xERA vs ERA")
    scope = _pitcher_scope_toggle("roster_vs_avail_pitcher_scope")
    st.caption(
        "Default shows startable pitchers (SP/SP,RP) — SVH is punted, so relievers "
        "are noise. Switch to **All pitchers** to include relievers."
    )
    sp = _filter_startable(sp, scope)
    nr, nfa = _stacked_panel(
        attach_status(sp),
        x="xera", y="era", gap_col="xera_minus_era", invert_gap=True,
        x_label="xERA (Expected)", y_label="ERA (Actual)",
        title="Pitchers: roster vs available", diag_pad=0.20, gap_fmt=".2f",
        noun="pitcher",
    )
    st.caption(f"My Roster: {nr} · Available: {nfa}")


def page_breakout_board():
    st.header("Breakout Board")
    st.caption(
        "One breakout signal — underlying Statcast quality vs surface results — "
        "through ownership lenses."
    )

    lens = st.radio(
        "Lens",
        ["My Roster", "Available (FA)", "Trade Targets", "Roster vs Available"],
        horizontal=True,
        index=1,  # default to Available (FA): buy-low adds are the most actionable view
    )

    # New stacked view: separate loader + render path, so an incomplete version
    # can never break the three existing lenses below.
    if lens == "Roster vs Available":
        _render_roster_vs_available()
        return

    if lens == "My Roster":
        own_value = MY_TEAM
        st.caption(
            "Your players' breakout/regression signal — hold, promote, or watch "
            "for warning signs."
        )
    elif lens == "Available (FA)":
        own_value = "FA"
        st.caption("Buy-low adds available now — biggest gap leads.")
    else:  # Trade Targets
        own_value = st.selectbox("Team", OTHER_TEAMS)
        st.caption(f"Undervalued players on {own_value} — buy-low trade targets.")

    bh = _load_csv(GOLD / "breakout_hitters_all.csv")
    bp = _load_csv(GOLD / "breakout_pitchers_all.csv")

    st.subheader("Hitter Breakout — xwOBA vs wOBA")
    _render_breakout_lens(
        bh, own_value, lens, noun="hitter",
        required={"est_woba", "woba", "xwoba_minus_woba", "ownership", "player_name"},
        x="est_woba", y="woba", gap="xwoba_minus_woba",
        x_label="xwOBA (Expected)", y_label="wOBA (Actual)",
        title="xwOBA vs wOBA", diag_pad=0.010, label_dy=0.004, gap_fmt=".3f",
    )

    st.subheader("Pitcher Breakout — xERA vs ERA")
    scope = _pitcher_scope_toggle("breakout_pitcher_scope")
    st.caption(
        "Default shows startable pitchers (SP/SP,RP) — SVH is punted, so relievers "
        "are noise. Switch to **All pitchers** to include relievers."
    )
    bp = _filter_startable(bp, scope)
    _render_breakout_lens(
        bp, own_value, lens, noun="pitcher",
        required={"xera", "era", "xera_minus_era", "ownership", "player_name"},
        x="xera", y="era", gap="xera_minus_era",
        x_label="xERA (Expected)", y_label="ERA (Actual)",
        title="xERA vs ERA", diag_pad=0.20, label_dy=0.08, gap_fmt=".2f",
    )


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


def _my_roster_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter a stats frame to MY rostered players via resolved vendor-id ownership.

    Uses :func:`attach_status` (savant/fangraphs id join through the id map) —
    the same ownership resolution the breakout board uses — so the filter is
    identity-based, never name matching. Returns only rows the id map resolves
    to ``status == "owned"`` on my fantasy team.
    """
    attached = attach_status(df)
    return attached[
        (attached["status"] == "owned")
        & (attached["fantrax_team_name"] == MY_TEAM)
    ]


def page_regression_watch():
    st.header("Regression Watch")
    st.caption(
        "Your rostered players whose results are outrunning their underlying "
        "Statcast metrics — sell-high or hold-with-caution candidates. Only your "
        "roster is shown; other teams' regression is not your concern."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hitters")
        st.caption(
            "Actual results significantly better than the underlying Statcast metrics "
            "suggest — a large negative xwOBA gap means the line is inflated by luck "
            "(high BABIP, unsustainable HR/FB). Sell-high or hold-with-caution."
        )
        rh = _load_csv(GOLD / "regression_hitters.csv")
        if rh is not None:
            rh = _my_roster_only(rh)
            if rh.empty:
                st.info("No significant regression flags on your roster right now.")
            else:
                gap_col = "xwoba_minus_woba" if "xwoba_minus_woba" in rh.columns else "est_woba_minus_woba_diff"
                if gap_col in rh.columns:
                    rh = rh.sort_values(gap_col, ascending=True)  # most negative = biggest overperformer
                show_cols = [c for c in ["player_name", "team", "position", "woba", "est_woba", gap_col] if c in rh.columns]
                st.dataframe(rh[show_cols], use_container_width=True, hide_index=True)

                # Bar chart — my regressing hitters (the curated set, not a top-10 cut)
                if "woba" in rh.columns and "est_woba" in rh.columns:
                    chart = rh.head(10).copy()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="wOBA (Actual)", x=chart["player_name"], y=chart["woba"], marker_color=theme.ACCENT))
                    fig.add_trace(go.Bar(name="xwOBA (Expected)", x=chart["player_name"], y=chart["est_woba"], marker_color=theme.TEXT_MUTED))
                    fig.update_layout(barmode="group", title="My Overperforming Hitters")
                    theme.apply_chart_theme(fig, height=400)
                    st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Pitchers")
        st.caption(
            "ERAs significantly lower than expected ERA (xERA) — a large positive "
            "xERA-minus-ERA gap means luck with strand rate, BABIP, or sequencing the "
            "stuff doesn't support. Expect the ERA to rise."
        )
        scope = _pitcher_scope_toggle("regression_pitcher_scope")
        st.caption(
            "Default shows startable pitchers (SP/SP,RP) — SVH is punted, so relievers "
            "are noise. Switch to **All pitchers** to include relievers."
        )
        rp = _load_csv(GOLD / "regression_pitchers.csv")
        rp = _filter_startable(rp, scope)
        if rp is not None:
            rp = _my_roster_only(rp)
            if rp.empty:
                st.info("No significant regression flags on your roster right now.")
            else:
                gap_col = "xera_minus_era" if "xera_minus_era" in rp.columns else "era_minus_xera_diff"
                if gap_col in rp.columns:
                    rp = rp.sort_values(gap_col, ascending=False)  # most positive = luckiest overperformer (ERA below xERA)
                show_cols = [c for c in ["player_name", "team", "position", "era", "xera", gap_col] if c in rp.columns]
                st.dataframe(rp[show_cols], use_container_width=True, hide_index=True)

                if "era" in rp.columns and "xera" in rp.columns:
                    chart = rp.head(10).copy()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="ERA (Actual)", x=chart["player_name"], y=chart["era"], marker_color=theme.ACCENT))
                    fig.add_trace(go.Bar(name="xERA (Expected)", x=chart["player_name"], y=chart["xera"], marker_color=theme.TEXT_MUTED))
                    fig.update_layout(barmode="group", title="My Overperforming Pitchers")
                    theme.apply_chart_theme(fig, height=400)
                    st.plotly_chart(fig, use_container_width=True)


def _ownership_color(val):
    """Styler for prospect ownership column (delegates to the theme)."""
    return theme.ownership_style(val)


def _upgrade_color(val):
    """Styler for net upgrade score (delegates to the theme).

    The theme version always sets an explicit text color, fixing the old
    no-text-color branches that would be unreadable on the dark base.
    """
    return theme.upgrade_style(val)


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

    # Highlight hot prospects (subtle caution-amber row tint from the theme)
    if "is_hot" in prospects.columns:
        hot_mask = prospects["is_hot"].apply(lambda x: x in (True, "True", "true", 1, "1"))
        if hot_mask.any():
            def _hot_row_style(row):
                idx = row.name
                if idx in hot_mask.index and hot_mask.loc[idx]:
                    return theme.hot_row_style(len(row))
                return [""] * len(row)
            styled = styled.apply(_hot_row_style, axis=1)

    # Highlight call-up candidates (subtle good-green row tint from the theme)
    if "callup_candidate" in display_df.columns:
        def _highlight_callup(row):
            if row.get("callup_candidate") in (True, "True", "true", 1, "1", "Yes", "yes"):
                return theme.callup_row_style(len(row))
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
    # Single, centralized CSS block for what config.toml can't reach (the Inter
    # face, heading weight/tracking, sidebar + metric polish). Driven by theme
    # constants so colors are never hardcoded twice.
    st.markdown(theme.CUSTOM_CSS, unsafe_allow_html=True)

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
