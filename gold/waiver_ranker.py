"""Waiver wire ranker scoring available free agents by projected value.

Scores hitters on HR, RBI, R, SB, OBP category contribution and pitchers
on K, W, ERA, WHIP contribution (SVH is punted).  Players are ranked by
a weighted composite score so the best waiver pickups float to the top.

Inputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet

Outputs:
    gold/data/waiver_hitters_ranked.csv
    gold/data/waiver_pitchers_ranked.csv
"""

import pathlib

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ── weights ──────────────────────────────────────────────────────────

HITTER_WEIGHTS = {
    "hr_score": 0.35,
    "speed_score": 0.20,
    "obp_score": 0.45,
}

PITCHER_WEIGHTS = {
    "k_score": 0.40,
    "era_score": 0.35,
    "whip_score": 0.25,
}

TOP_N = 25


# ── loaders ──────────────────────────────────────────────────────────


def load_hitters() -> pd.DataFrame:
    """Load enriched Statcast hitter data from the silver layer.

    Returns:
        DataFrame with batted-ball metrics and percentile ranks.
    """
    return pd.read_parquet(SILVER_DIR / "statcast_hitters.parquet")


def load_pitchers() -> pd.DataFrame:
    """Load enriched Statcast pitcher data from the silver layer.

    Returns:
        DataFrame with xERA, barrel, and strikeout metrics.
    """
    return pd.read_parquet(SILVER_DIR / "statcast_pitchers.parquet")


# ── scoring – hitters ────────────────────────────────────────────────


def score_hitters(df: pd.DataFrame) -> pd.DataFrame:
    """Score each hitter on HR, SB, and OBP category contribution.

    Component scores (all scaled 0-100 via percentile rank):
        - ``hr_score``:  barrel_percentile (higher barrels = more HR upside)
        - ``speed_score``: sprint_speed percentile rank if available, else 0
        - ``obp_score``:  xwOBA percentile rank as an OBP proxy

    Args:
        df: Enriched hitter DataFrame from the silver layer.

    Returns:
        DataFrame with component scores and a composite_hitter_score,
        sorted descending by composite score.
    """
    scored = df.copy()

    # HR upside – barrel percentile is already 0-100
    scored["hr_score"] = scored["barrel_percentile"].fillna(0)

    # Speed / SB upside
    if "sprint_speed" in scored.columns:
        scored["speed_score"] = (
            scored["sprint_speed"]
            .rank(pct=True, na_option="bottom")
            .mul(100)
            .round(1)
        )
    else:
        print("  WARNING: sprint_speed not available; speed_score set to 0")
        scored["speed_score"] = 0.0

    # OBP proxy – percentile-rank xwOBA (est_woba column)
    scored["obp_score"] = (
        scored["est_woba"]
        .rank(pct=True, na_option="bottom")
        .mul(100)
        .round(1)
    )

    # Weighted composite
    scored["composite_hitter_score"] = sum(
        scored[col] * weight for col, weight in HITTER_WEIGHTS.items()
    ).round(1)

    return (
        scored.sort_values("composite_hitter_score", ascending=False)
        .reset_index(drop=True)
    )


# ── scoring – pitchers ──────────────────────────────────────────────


def score_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Score each pitcher on K, ERA, and WHIP category contribution.

    Component scores (all scaled 0-100 via percentile rank):
        - ``k_score``:    k_percent (or k_per_9) percentile rank
        - ``era_score``:  inverse xERA percentile rank (lower xERA = higher)
        - ``whip_score``: inverse (est_ba + walk proxy) percentile rank

    SVH is intentionally ignored (punt saves strategy).

    Args:
        df: Enriched pitcher DataFrame from the silver layer.

    Returns:
        DataFrame with component scores and a composite_pitcher_score,
        sorted descending by composite score.
    """
    scored = df.copy()

    # K upside
    if "k_percent" in scored.columns:
        k_col = "k_percent"
    elif "k_per_9" in scored.columns:
        k_col = "k_per_9"
    else:
        k_col = None
        print("  WARNING: no strikeout column found; k_score set to 0")

    if k_col:
        scored["k_score"] = (
            scored[k_col]
            .rank(pct=True, na_option="bottom")
            .mul(100)
            .round(1)
        )
    else:
        scored["k_score"] = 0.0

    # ERA – lower xERA is better, so rank ascending=False
    scored["era_score"] = (
        scored["xera"]
        .rank(pct=True, ascending=False, na_option="bottom")
        .mul(100)
        .round(1)
    )

    # WHIP proxy – lower xBA means fewer hits allowed
    scored["whip_score"] = (
        scored["est_ba"]
        .rank(pct=True, ascending=False, na_option="bottom")
        .mul(100)
        .round(1)
    )

    # Weighted composite
    scored["composite_pitcher_score"] = sum(
        scored[col] * weight for col, weight in PITCHER_WEIGHTS.items()
    ).round(1)

    return (
        scored.sort_values("composite_pitcher_score", ascending=False)
        .reset_index(drop=True)
    )


# ── display ──────────────────────────────────────────────────────────


def print_hitter_table(df: pd.DataFrame, n: int = TOP_N) -> None:
    """Print the top-N ranked hitters to the console.

    Args:
        df: Scored hitter DataFrame (already sorted).
        n: Number of rows to display.
    """
    cols = [
        "player_name",
        "team",
        "position",
        "hr_score",
        "speed_score",
        "obp_score",
        "composite_hitter_score",
    ]
    names = {
        "player_name": "Name",
        "team": "Team",
        "position": "Pos",
        "hr_score": "HR",
        "speed_score": "Speed",
        "obp_score": "OBP",
        "composite_hitter_score": "Composite",
    }
    display = df.head(n)[cols].rename(columns=names)
    print(display.to_string(index=False))


def print_pitcher_table(df: pd.DataFrame, n: int = TOP_N) -> None:
    """Print the top-N ranked pitchers to the console.

    Args:
        df: Scored pitcher DataFrame (already sorted).
        n: Number of rows to display.
    """
    cols = [
        "player_name",
        "team",
        "k_score",
        "era_score",
        "whip_score",
        "composite_pitcher_score",
    ]
    names = {
        "player_name": "Name",
        "team": "Team",
        "k_score": "K",
        "era_score": "ERA",
        "whip_score": "WHIP",
        "composite_pitcher_score": "Composite",
    }
    display = df.head(n)[cols].rename(columns=names)
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Score and rank waiver wire targets, then save results to gold layer."""
    print("Loading silver-layer hitter data...")
    hitters = load_hitters()
    print(f"  {len(hitters)} hitters loaded")

    print("Scoring hitters...")
    ranked_h = score_hitters(hitters)
    print(f"  Top {TOP_N} waiver hitters:\n")
    if not ranked_h.empty:
        print_hitter_table(ranked_h)
    print()

    print("Loading silver-layer pitcher data...")
    pitchers = load_pitchers()
    print(f"  {len(pitchers)} pitchers loaded")

    print("Scoring pitchers...")
    ranked_p = score_pitchers(pitchers)
    print(f"  Top {TOP_N} waiver pitchers:\n")
    if not ranked_p.empty:
        print_pitcher_table(ranked_p)
    print()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    h_path = GOLD_DIR / "waiver_hitters_ranked.csv"
    ranked_h.to_csv(h_path, index=False)
    print(f"Saved {h_path}")

    p_path = GOLD_DIR / "waiver_pitchers_ranked.csv"
    ranked_p.to_csv(p_path, index=False)
    print(f"Saved {p_path}")


if __name__ == "__main__":
    main()
