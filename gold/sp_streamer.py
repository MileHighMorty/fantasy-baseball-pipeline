"""SP streamer recommending starting pitcher pickups by matchup quality.

Identifies the best streaming pitcher options for today by combining
pitcher quality metrics (xERA, K%) with opponent offensive weakness
(wRC+, K%).  Pitchers are ranked by a weighted stream_score so the
best single-day spot starts float to the top.

Inputs:
    silver/data/statcast_pitchers.parquet
    bronze/data/mlb/{today}_games.csv
    bronze/data/fangraphs/{today}_batting.csv

Outputs:
    gold/data/sp_streaming_picks.csv
"""

import datetime
import pathlib

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
BRONZE_MLB = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "mlb"
BRONZE_FG = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fangraphs"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ── scoring weights ──────────────────────────────────────────────────

WEIGHTS = {
    "xera_score": 0.40,
    "k_score": 0.30,
    "opp_score": 0.30,
}

TOP_N = 25

# ── team-name mapping (MLB full name → 3-letter code) ───────────────

TEAM_ABBREV = {
    "Arizona Diamondbacks": "ARI",
    "Athletics": "ATH",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


# ── loaders ──────────────────────────────────────────────────────────


def load_pitchers() -> pd.DataFrame:
    """Load enriched Statcast pitcher data from the silver layer.

    Returns:
        DataFrame with xERA, K%, and batted-ball metrics for each pitcher.
    """
    return pd.read_parquet(SILVER_DIR / "statcast_pitchers.parquet")


def load_games(today: str) -> pd.DataFrame:
    """Load today's schedule and probable pitchers from the bronze layer.

    Args:
        today: Date string in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame with one row per game containing probable pitchers.
    """
    return pd.read_csv(BRONZE_MLB / f"{today}_games.csv")


def load_team_batting(today: str) -> pd.DataFrame:
    """Load FanGraphs batting stats and aggregate to team level.

    Computes team-level offensive summary (mean wRC+, mean K%) so we can
    identify weak-hitting lineups that are favourable streaming targets.

    Args:
        today: Date string in ``YYYY-MM-DD`` format.

    Returns:
        DataFrame indexed by team abbreviation with ``team_wrc_plus``
        and ``team_k_pct`` columns.
    """
    path = BRONZE_FG / f"{today}_batting.csv"
    if not path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(path)

    # Convert K% from string-like fraction to float if needed
    if raw["K%"].dtype == object:
        raw["K%"] = raw["K%"].str.rstrip("%").astype(float) / 100

    team_stats = (
        raw.groupby("Team")
        .agg(team_wrc_plus=("wRC+", "mean"), team_k_pct=("K%", "mean"))
        .reset_index()
        .rename(columns={"Team": "opp_abbrev"})
    )
    return team_stats


# ── matchup builder ──────────────────────────────────────────────────


def build_matchups(games: pd.DataFrame) -> pd.DataFrame:
    """Unpivot the games schedule into one row per pitcher start.

    Each game has an away and a home probable pitcher.  This function
    produces a flat table with columns ``pitcher_name``, ``team`` (the
    pitcher's team abbreviation), and ``opponent`` (the opposing team
    abbreviation).

    Args:
        games: Raw games DataFrame from :func:`load_games`.

    Returns:
        DataFrame with ``pitcher_name``, ``team``, ``opponent``, and
        ``venue`` columns.
    """
    rows = []
    for _, g in games.iterrows():
        away_abbrev = TEAM_ABBREV.get(g["away_team"], "")
        home_abbrev = TEAM_ABBREV.get(g["home_team"], "")

        if pd.notna(g.get("away_pitcher")) and g["away_pitcher"]:
            rows.append({
                "pitcher_name": g["away_pitcher"],
                "team": away_abbrev,
                "opponent": home_abbrev,
                "venue": g.get("venue", ""),
            })
        if pd.notna(g.get("home_pitcher")) and g["home_pitcher"]:
            rows.append({
                "pitcher_name": g["home_pitcher"],
                "team": home_abbrev,
                "opponent": away_abbrev,
                "venue": g.get("venue", ""),
            })

    return pd.DataFrame(rows)


# ── scoring ──────────────────────────────────────────────────────────


def score_streamers(
    matchups: pd.DataFrame,
    pitchers: pd.DataFrame,
    team_batting: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate a stream_score for each scheduled starter.

    Component scores (all scaled 0-100 via percentile rank):
        - ``xera_score``: inverse xERA rank (lower xERA → higher score).
        - ``k_score``: K% rank (higher K% → higher score).
        - ``opp_score``: opponent weakness composite — high opponent K%
          and low opponent wRC+ both boost this score.

    The final ``stream_score`` is a weighted average of the three
    components.

    Args:
        matchups: Output of :func:`build_matchups`.
        pitchers: Silver-layer pitcher DataFrame.
        team_batting: Team-level offensive stats from :func:`load_team_batting`.

    Returns:
        Scored DataFrame sorted by ``stream_score`` descending.
    """
    # Join pitcher quality metrics
    merged = matchups.merge(
        pitchers[["player_name", "team", "xera", "k_percent"]],
        left_on=["pitcher_name", "team"],
        right_on=["player_name", "team"],
        how="left",
    )

    # Drop pitchers without statcast data
    merged = merged.dropna(subset=["xera"]).copy()

    if merged.empty:
        return merged

    # xERA score — lower is better
    merged["xera_score"] = (
        merged["xera"]
        .rank(pct=True, ascending=False, na_option="bottom")
        .mul(100)
        .round(1)
    )

    # K% score — higher is better
    merged["k_score"] = (
        merged["k_percent"]
        .rank(pct=True, na_option="bottom")
        .mul(100)
        .round(1)
    )

    # Opponent weakness score
    if not team_batting.empty:
        merged = merged.merge(
            team_batting,
            left_on="opponent",
            right_on="opp_abbrev",
            how="left",
        )
        # Weak opponents: low wRC+ and high K%
        merged["opp_wrc_score"] = (
            merged["team_wrc_plus"]
            .rank(pct=True, ascending=False, na_option="bottom")
            .mul(100)
            .round(1)
        )
        merged["opp_k_score"] = (
            merged["team_k_pct"]
            .rank(pct=True, na_option="bottom")
            .mul(100)
            .round(1)
        )
        merged["opp_score"] = (
            (merged["opp_wrc_score"] + merged["opp_k_score"]) / 2
        ).round(1)
    else:
        print("  WARNING: no team batting data; opp_score set to 50")
        merged["opp_score"] = 50.0

    # Weighted composite
    merged["stream_score"] = sum(
        merged[col] * weight for col, weight in WEIGHTS.items()
    ).round(1)

    return (
        merged.sort_values("stream_score", ascending=False)
        .reset_index(drop=True)
    )


# ── display ──────────────────────────────────────────────────────────


def print_streamer_table(df: pd.DataFrame, n: int = TOP_N) -> None:
    """Print the top-N streaming picks as a formatted console table.

    Args:
        df: Scored streamer DataFrame (already sorted).
        n: Number of rows to display.
    """
    cols = [
        "pitcher_name",
        "team",
        "opponent",
        "xera",
        "k_percent",
        "stream_score",
    ]
    names = {
        "pitcher_name": "Pitcher",
        "team": "Team",
        "opponent": "Opp",
        "xera": "xERA",
        "k_percent": "K%",
        "stream_score": "Score",
    }
    display = df.head(n)[[c for c in cols if c in df.columns]].rename(columns=names)
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Load data, score streaming pitchers, display, and save results."""
    today = datetime.date.today().isoformat()

    print(f"Loading today's games ({today})...")
    games = load_games(today)
    print(f"  {len(games)} games on the schedule")

    print("Building pitcher matchups...")
    matchups = build_matchups(games)
    print(f"  {len(matchups)} probable starters found")

    print("Loading silver-layer pitcher data...")
    pitchers = load_pitchers()
    print(f"  {len(pitchers)} pitchers loaded")

    print("Loading team batting data...")
    team_batting = load_team_batting(today)
    if team_batting.empty:
        print("  No team batting file found; opponent scores will be neutral")
    else:
        print(f"  {len(team_batting)} teams loaded")

    print("Scoring streamers...\n")
    scored = score_streamers(matchups, pitchers, team_batting)

    if scored.empty:
        print("  No pitchers matched — check data freshness")
        return

    print(f"  Top {min(TOP_N, len(scored))} SP streaming picks:\n")
    print_streamer_table(scored)
    print()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "sp_streaming_picks.csv"
    scored.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
