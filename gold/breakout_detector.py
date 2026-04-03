"""Breakout detector identifying players with underlying skill improvements.

Flags hitters and pitchers whose expected stats significantly outpace
their surface-level results, suggesting positive regression ahead.

Inputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet

Outputs:
    gold/data/breakout_hitters.csv
    gold/data/breakout_pitchers.csv
"""

import pathlib

import pandas as pd

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ── thresholds ───────────────────────────────────────────────────────

HITTER_XWOBA_GAP = 0.030
HITTER_HARD_HIT_PCTL = 40

PITCHER_XERA_GAP = 0.50
PITCHER_K_BB_PCT = 10


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


def detect_breakout_hitters(df: pd.DataFrame) -> pd.DataFrame:
    """Flag hitter breakout candidates based on expected-stat gaps.

    A hitter qualifies when ALL conditions are true:
        - ``xwoba_minus_woba >= 0.030`` (underlying quality exceeds results)
        - ``hard_hit_percentile >= 40`` (real contact quality, not BABIP luck)

    Args:
        df: Enriched hitter DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xwoba_minus_woba`` descending.
    """
    mask = (
        (df["xwoba_minus_woba"] >= HITTER_XWOBA_GAP)
        & (df["hard_hit_percentile"] >= HITTER_HARD_HIT_PCTL)
    )
    return (
        df.loc[mask]
        .sort_values("xwoba_minus_woba", ascending=False)
        .reset_index(drop=True)
    )


def detect_breakout_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag pitcher breakout candidates based on expected-stat gaps.

    A pitcher qualifies when ALL conditions are true:
        - ``xera_minus_era >= 0.50`` (ERA should be lower than it is)
        - ``k_minus_bb_pct >= 10`` (solid strikeout-walk differential)

    If ``k_minus_bb_pct`` is not available in the source data, that
    filter is skipped and a warning is printed.

    Args:
        df: Enriched pitcher DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xera_minus_era`` descending.
    """
    mask = df["xera_minus_era"] >= PITCHER_XERA_GAP

    if "k_minus_bb_pct" in df.columns:
        mask = mask & (df["k_minus_bb_pct"] >= PITCHER_K_BB_PCT)
    else:
        print(
            "  WARNING: k_minus_bb_pct not available in pitcher data; "
            "skipping K-BB% filter"
        )

    return (
        df.loc[mask]
        .sort_values("xera_minus_era", ascending=False)
        .reset_index(drop=True)
    )


# ── display ──────────────────────────────────────────────────────────


def print_hitter_table(df: pd.DataFrame) -> None:
    """Print a formatted breakout hitters table to the console.

    Args:
        df: Breakout hitter DataFrame.
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
    """Print a formatted breakout pitchers table to the console.

    Args:
        df: Breakout pitcher DataFrame.
    """
    cols = ["player_name", "team", "xera", "era", "xera_minus_era"]
    names = {
        "player_name": "Name",
        "team": "Team",
        "xera": "xERA",
        "era": "ERA",
        "xera_minus_era": "Gap",
    }

    if "k_minus_bb_pct" in df.columns:
        cols.append("k_minus_bb_pct")
        names["k_minus_bb_pct"] = "K-BB%"

    cols.append("brl_percent")
    names["brl_percent"] = "Barrel%"

    display = df[cols].rename(columns=names)
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Detect breakout candidates and save results to gold layer."""
    print("Loading silver-layer hitter data...")
    hitters = load_hitters()
    print(f"  {len(hitters)} hitters loaded")

    print("Loading silver-layer pitcher data...")
    pitchers = load_pitchers()
    print(f"  {len(pitchers)} pitchers loaded")

    print("Detecting breakout hitters...")
    breakout_h = detect_breakout_hitters(hitters)
    print(f"  {len(breakout_h)} breakout hitter candidates\n")
    if not breakout_h.empty:
        print_hitter_table(breakout_h)
    print()

    print("Detecting breakout pitchers...")
    breakout_p = detect_breakout_pitchers(pitchers)
    print(f"  {len(breakout_p)} breakout pitcher candidates\n")
    if not breakout_p.empty:
        print_pitcher_table(breakout_p)
    print()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    h_path = GOLD_DIR / "breakout_hitters.csv"
    breakout_h.to_csv(h_path, index=False)
    print(f"Saved {h_path}")

    p_path = GOLD_DIR / "breakout_pitchers.csv"
    breakout_p.to_csv(p_path, index=False)
    print(f"Saved {p_path}")


if __name__ == "__main__":
    main()
