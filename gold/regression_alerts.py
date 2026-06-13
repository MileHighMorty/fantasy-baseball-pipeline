"""Regression alerts flagging players over-performing expected stats.

Identifies sell-high candidates whose surface results exceed their
underlying quality metrics, suggesting negative regression ahead.

Inputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet

Outputs:
    gold/data/regression_hitters.csv
    gold/data/regression_pitchers.csv
"""

import pathlib

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ── thresholds ───────────────────────────────────────────────────────

HITTER_XWOBA_GAP = -0.030

# Sign convention (mirror image of breakout_detector; must match the Stage B
# "Roster vs Available" lens):
#   hitter gap = est_woba - woba  -> NEGATIVE = overperforming = SELL (regress)
#   pitcher gap = xera - era      -> POSITIVE = xERA above ERA = lucky = ERA
#                                    should rise = SELL (regress)
# So the pitcher regression threshold is POSITIVE: a candidate must score AT OR
# ABOVE it. Do not flip this to a negative value — that re-inverts the bug and
# surfaces unlucky under-performers (buys) as sells.
PITCHER_XERA_GAP = 0.50

# Outcome-quality guard. A sell must be driven by the GAP (overperformance);
# contact-quality percentiles no longer create flags, because as an OR-clause
# they flagged directional BUYS as sells (e.g. low hard-hit Mookie Betts with a
# positive xwOBA gap). The guard then drops any candidate whose EXPECTED stat is
# still good: regressing toward a still-above-average level is not a sell. These
# are absolute league-average anchors (not population percentiles) so the cutoff
# is stable and interpretable and does not drift with who happens to qualify
# this week: ~.320 is league-average wOBA, ~3.50 is a clearly-good ERA/xERA.
# This is what keeps elite arms like Skubal/Yamamoto/Ohtani (xERA < 3.50) off
# the sell list even when their ERA ran below their xERA.
HITTER_QUALITY_FLOOR = 0.320
PITCHER_QUALITY_FLOOR = 3.50

# Two-way players (e.g. Ohtani) are never acquirable as a single-role asset, so
# they don't belong on a sell list regardless of their numbers. Excluded
# independently of the quality guard, because a future *mediocre* two-way player
# would pass the guard yet still isn't a real sell. Mirrors breakout_detector.
TWO_WAY_EXCLUDE = {"Shohei Ohtani"}


# ── loaders ──────────────────────────────────────────────────────────


def load_hitters() -> pd.DataFrame:
    """Load enriched Statcast hitter data from the silver layer.

    Returns:
        DataFrame with expected-stat differentials and percentile ranks.
    """
    return pd.read_parquet(SILVER_DIR / "statcast_hitters.parquet")


def load_pitchers() -> pd.DataFrame:
    """Load enriched Statcast pitcher data from the silver layer.

    Returns:
        DataFrame with xERA differentials and percentile ranks.
    """
    return pd.read_parquet(SILVER_DIR / "statcast_pitchers.parquet")


# ── detection ────────────────────────────────────────────────────────


def detect_regression_hitters(df: pd.DataFrame) -> pd.DataFrame:
    """Flag sell-high hitter candidates based on expected-stat gaps.

    A hitter qualifies when BOTH hold:
        - ``xwoba_minus_woba <= -0.030`` (results exceed underlying quality —
          overperforming; the gap is the required driver of a sell)
        - ``est_woba <= 0.320`` (expected output is not still above league
          average — otherwise this is regression to a still-good level, not a
          sell)

    Contact quality (hard_hit_percentile) is no longer part of the condition:
    as an OR-clause it created sells that contradicted the gap.

    Args:
        df: Enriched hitter DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xwoba_minus_woba`` ascending
        (biggest overperformers first).
    """
    mask = (
        (df["xwoba_minus_woba"] <= HITTER_XWOBA_GAP)
        & (df["est_woba"] <= HITTER_QUALITY_FLOOR)
    )
    return (
        df.loc[mask]
        .sort_values("xwoba_minus_woba", ascending=True)
        .reset_index(drop=True)
    )


def detect_regression_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag sell-high pitcher candidates based on expected-stat gaps.

    A pitcher qualifies when BOTH hold:
        - ``xera_minus_era >= 0.50`` (xERA above ERA: lucky, ERA should rise —
          the gap is the required driver of a sell)
        - ``xera >= 3.50`` (xERA is not still elite — otherwise this is
          regression to a still-good level, not a sell)

    Contact quality (hard_hit_percentile) is no longer part of the condition:
    as an OR-clause it flagged unlucky, negative-gap pitchers (e.g. Fried,
    Luzardo — those are buys) as sells.

    Args:
        df: Enriched pitcher DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xera_minus_era`` descending
        (biggest overperformers — luckiest — first).
    """
    mask = (
        (df["xera_minus_era"] >= PITCHER_XERA_GAP)
        & (df["xera"] >= PITCHER_QUALITY_FLOOR)
    )
    return (
        df.loc[mask]
        .sort_values("xera_minus_era", ascending=False)
        .reset_index(drop=True)
    )


def drop_two_way(df: pd.DataFrame) -> pd.DataFrame:
    """Remove two-way players (see ``TWO_WAY_EXCLUDE``) from a regression frame.

    Statcast-sourced names are already clean (no Fantrax "-H"/"-P" suffix), so a
    direct name membership test is enough here.

    Args:
        df: A regression frame with a ``player_name`` column.

    Returns:
        The frame with any ``TWO_WAY_EXCLUDE`` players removed.
    """
    return df.loc[~df["player_name"].isin(TWO_WAY_EXCLUDE)].reset_index(drop=True)


# ── display ──────────────────────────────────────────────────────────


def print_hitter_table(df: pd.DataFrame) -> None:
    """Print a formatted regression hitters table to the console.

    Args:
        df: Regression hitter DataFrame.
    """
    display = df[
        [
            "player_name",
            "team",
            "position",
            "est_woba",
            "woba",
            "xwoba_minus_woba",
            "hard_hit_percentile",
            "brl_percent",
        ]
    ].rename(
        columns={
            "player_name": "Name",
            "team": "Team",
            "position": "Pos",
            "est_woba": "xwOBA",
            "woba": "wOBA",
            "xwoba_minus_woba": "Gap",
            "hard_hit_percentile": "HardHit%",
            "brl_percent": "Barrel%",
        }
    )
    print(display.to_string(index=False))


def print_pitcher_table(df: pd.DataFrame) -> None:
    """Print a formatted regression pitchers table to the console.

    Args:
        df: Regression pitcher DataFrame.
    """
    display = df[
        [
            "player_name",
            "team",
            "xera",
            "era",
            "xera_minus_era",
            "hard_hit_percentile",
            "brl_percent",
        ]
    ].rename(
        columns={
            "player_name": "Name",
            "team": "Team",
            "xera": "xERA",
            "era": "ERA",
            "xera_minus_era": "Gap",
            "hard_hit_percentile": "HardHit%",
            "brl_percent": "Barrel%",
        }
    )
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Detect regression candidates and save results to gold layer."""
    print("Loading silver-layer hitter data...")
    hitters = load_hitters()
    print(f"  {len(hitters)} hitters loaded")

    print("Loading silver-layer pitcher data...")
    pitchers = load_pitchers()
    print(f"  {len(pitchers)} pitchers loaded")

    print("Detecting sell-high hitters...")
    regression_h = drop_two_way(detect_regression_hitters(hitters))
    print(f"  {len(regression_h)} regression hitter candidates\n")
    if not regression_h.empty:
        print_hitter_table(regression_h)
    print()

    print("Detecting sell-high pitchers...")
    regression_p = drop_two_way(detect_regression_pitchers(pitchers))
    print(f"  {len(regression_p)} regression pitcher candidates\n")
    if not regression_p.empty:
        print_pitcher_table(regression_p)
    print()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    h_path = GOLD_DIR / "regression_hitters.csv"
    regression_h.to_csv(h_path, index=False)
    print(f"Saved {h_path}")

    p_path = GOLD_DIR / "regression_pitchers.csv"
    regression_p.to_csv(p_path, index=False)
    print(f"Saved {p_path}")


if __name__ == "__main__":
    main()
