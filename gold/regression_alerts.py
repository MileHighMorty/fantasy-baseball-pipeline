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
HITTER_HARD_HIT_PCTL = 25

PITCHER_XERA_GAP = -0.50
PITCHER_HARD_HIT_PCTL = 75


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

    A hitter qualifies when ANY condition is true:
        - ``xwoba_minus_woba <= -0.030`` (results exceeding underlying quality)
        - ``hard_hit_percentile <= 25`` (weak contact getting lucky results)

    Args:
        df: Enriched hitter DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xwoba_minus_woba`` ascending
        (biggest overperformers first).
    """
    mask = (
        (df["xwoba_minus_woba"] <= HITTER_XWOBA_GAP)
        | (df["hard_hit_percentile"] <= HITTER_HARD_HIT_PCTL)
    )
    return (
        df.loc[mask]
        .sort_values("xwoba_minus_woba", ascending=True)
        .reset_index(drop=True)
    )


def detect_regression_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag sell-high pitcher candidates based on expected-stat gaps.

    A pitcher qualifies when ANY condition is true:
        - ``xera_minus_era <= -0.50`` (ERA better than it should be)
        - ``hard_hit_percentile >= 75`` (getting hit hard but ERA looks fine)

    Args:
        df: Enriched pitcher DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xera_minus_era`` ascending
        (biggest overperformers first).
    """
    mask = (
        (df["xera_minus_era"] <= PITCHER_XERA_GAP)
        | (df["hard_hit_percentile"] >= PITCHER_HARD_HIT_PCTL)
    )
    return (
        df.loc[mask]
        .sort_values("xera_minus_era", ascending=True)
        .reset_index(drop=True)
    )


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
    regression_h = detect_regression_hitters(hitters)
    print(f"  {len(regression_h)} regression hitter candidates\n")
    if not regression_h.empty:
        print_hitter_table(regression_h)
    print()

    print("Detecting sell-high pitchers...")
    regression_p = detect_regression_pitchers(pitchers)
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
