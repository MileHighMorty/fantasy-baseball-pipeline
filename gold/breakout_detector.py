"""Breakout detector identifying players with underlying skill improvements.

Flags hitters and pitchers whose expected stats significantly outpace
their surface-level results, suggesting positive regression ahead.

Inputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet
    bronze/data/fantrax/all_rosters_*.csv

Outputs:
    gold/data/breakout_hitters_all.csv
    gold/data/breakout_pitchers_all.csv
    gold/data/breakout_hitters_fa.csv
    gold/data/breakout_pitchers_fa.csv
"""

import pathlib

import pandas as pd
from rapidfuzz import process, fuzz

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fantrax"

FUZZY_THRESHOLD = 90

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


# ── roster / ownership ──────────────────────────────────────────────


def load_all_rosters() -> pd.DataFrame | None:
    """Load the latest date-stamped all_rosters CSV from Fantrax."""
    files = sorted(FANTRAX_DIR.glob("all_rosters_*.csv"))
    if not files:
        print("  WARNING: No all_rosters CSV found in", FANTRAX_DIR)
        return None
    latest = files[-1]
    print(f"  Loaded roster file: {latest.name}")
    df = pd.read_csv(latest)
    df["player_name"] = df["player_name"].str.strip()
    return df


def _build_owned_set(rosters: pd.DataFrame) -> set[str]:
    """Return the set of owned player names (exclude 'None' placeholders)."""
    names = rosters.loc[rosters["player_name"].notna(), "player_name"]
    return {n for n in names if n != "None"}


def tag_ownership(
    breakout: pd.DataFrame,
    rosters: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add an 'ownership' column: team name if owned, 'FA' if not."""
    df = breakout.copy()
    if rosters is None:
        df["ownership"] = "Unknown"
        return df

    owned_names = _build_owned_set(rosters)
    # Build a lookup: owned_name -> team_name
    roster_lookup: dict[str, str] = {}
    for _, row in rosters.iterrows():
        pn = row["player_name"]
        if pd.notna(pn) and pn != "None":
            roster_lookup[pn] = row["team_name"]

    owned_list = list(owned_names)

    def _match(player: str) -> str:
        if not owned_list:
            return "FA"
        result = process.extractOne(
            player, owned_list, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD
        )
        if result is None:
            return "FA"
        matched_name = result[0]
        return roster_lookup.get(matched_name, "FA")

    df["ownership"] = df["player_name"].apply(_match)
    return df


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

    Primary filter (required):
        - ``xera_minus_era >= 0.50`` (ERA should be lower than it is)

    Optional secondary filter (applied only when data is available and valid):
        - ``k_minus_bb_pct`` — used for sorting, not hard filtering

    Args:
        df: Enriched pitcher DataFrame from the silver layer.

    Returns:
        Filtered DataFrame ranked by ``xera_minus_era`` descending.
    """
    print(f"  Pitcher columns available: {list(df.columns)}")

    mask = df["xera_minus_era"] >= PITCHER_XERA_GAP
    print(f"  Pitchers with xera_minus_era >= {PITCHER_XERA_GAP}: {mask.sum()}")

    if "k_minus_bb_pct" in df.columns and df["k_minus_bb_pct"].notna().any():
        print(f"  k_minus_bb_pct range: {df['k_minus_bb_pct'].min():.3f} – {df['k_minus_bb_pct'].max():.3f}")
        print("  k_minus_bb_pct used for sorting (not as hard filter)")
    else:
        print("  WARNING: k_minus_bb_pct not available or empty; skipping")

    result = df.loc[mask].copy()

    # Sort by xera gap, then k-bb% as tiebreaker if available
    sort_cols = ["xera_minus_era"]
    sort_asc = [False]
    if "k_minus_bb_pct" in result.columns and result["k_minus_bb_pct"].notna().any():
        sort_cols.append("k_minus_bb_pct")
        sort_asc.append(False)

    return result.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)


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

    print("Loading Fantrax rosters for ownership tagging...")
    rosters = load_all_rosters()

    print("Detecting breakout hitters...")
    breakout_h = detect_breakout_hitters(hitters)
    breakout_h = tag_ownership(breakout_h, rosters)
    fa_h = breakout_h[breakout_h["ownership"] == "FA"].reset_index(drop=True)
    print(f"  {len(breakout_h)} breakout hitter candidates ({len(fa_h)} free agents)\n")
    if not breakout_h.empty:
        print_hitter_table(breakout_h)
    print()

    print("Detecting breakout pitchers...")
    breakout_p = detect_breakout_pitchers(pitchers)
    print(f"  {len(breakout_p)} total pitcher breakout candidates")
    breakout_p = tag_ownership(breakout_p, rosters)
    fa_p = breakout_p[breakout_p["ownership"] == "FA"].reset_index(drop=True)
    print(f"  {len(fa_p)} FA pitcher breakout candidates\n")
    if not breakout_p.empty:
        print_pitcher_table(breakout_p)
    print()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    for df, name in [
        (breakout_h, "breakout_hitters_all.csv"),
        (fa_h, "breakout_hitters_fa.csv"),
        (breakout_p, "breakout_pitchers_all.csv"),
        (fa_p, "breakout_pitchers_fa.csv"),
    ]:
        path = GOLD_DIR / name
        df.to_csv(path, index=False)
        print(f"Saved {path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
