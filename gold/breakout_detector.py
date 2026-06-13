"""Breakout detector identifying players with underlying skill improvements.

Flags hitters and pitchers whose expected stats significantly outpace
their surface-level results, suggesting positive regression ahead.

Inputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet
    silver/data/player_id_map.parquet   (ownership: status owned/fa)

Outputs:
    gold/data/breakout_hitters_all.csv
    gold/data/breakout_pitchers_all.csv
    gold/data/breakout_hitters_fa.csv
    gold/data/breakout_pitchers_fa.csv
"""

import pathlib

import pandas as pd

from gold.ownership import attach_status

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

# ── thresholds ───────────────────────────────────────────────────────

HITTER_XWOBA_GAP = 0.030
HITTER_HARD_HIT_PCTL = 40

PITCHER_XERA_GAP = 0.50
PITCHER_K_BB_PCT = 10

# Two-way players are noise on every breakout lens: they are never a free
# agent, never a clean trade/add target, never acquirable.  We drop them from
# the breakout OUTPUTS ONLY, after ownership is derived — their identities and
# their (possibly two) correct owners stay fully intact in the id_map and
# player master, which this module never writes.  Seeded with the league's
# only current two-way player; add canonical names here as needed.
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


# ── ownership ────────────────────────────────────────────────────────


def tag_ownership(breakout: pd.DataFrame) -> pd.DataFrame:
    """Add 'ownership' and authoritative 'position' from resolved identity.

    Joins to the silver ID map on savant_player_id (fallback fangraphs_id)
    via :func:`gold.ownership.attach_status`, so a Fantrax two-way suffix or
    an accent/spelling difference (e.g. Statcast's "Lance McCullers Jr." vs
    the id map's "Lance Mccullers Jr") can never mislabel a player the way
    the old token-ratio name match could.

    Two columns are set from the join:

    * ``ownership`` — the owning fantasy team when rostered, else ``"FA"``
      (covering id-map status 'fa' and any player with no id-map row).
    * ``position`` — overwritten with the Fantrax multi-position eligibility
      string ("C,1B", "SP,RP").  The breakout frame's incoming position comes
      from the FanGraphs-derived player universe, which is null for hitters
      and a blanket "P" for pitchers; the Fantrax value is the authority.
      Kept as the canonical comma string — explosion is a display concern.
      Falls back to the incoming value only when a player has no id-map row,
      so position is never silently dropped.

    Args:
        breakout: A breakout frame carrying ``savant_player_id`` and
            ``fangraphs_id`` (both present on the Statcast-sourced frames).

    Returns:
        The frame with ``ownership`` set and ``position`` replaced.
    """
    attached = attach_status(breakout)
    df = breakout.copy()
    df["ownership"] = (
        attached["fantrax_team_name"]
        .where(attached["status"] == "owned", "FA")
        .values
    )
    df["position"] = (
        attached["fantrax_position"]
        .where(attached["fantrax_position"].notna(), df.get("position"))
        .values
    )
    return df


def drop_two_way(df: pd.DataFrame) -> pd.DataFrame:
    """Remove two-way players (see ``TWO_WAY_EXCLUDE``) from a breakout frame.

    Matches on the canonical name so a Fantrax role suffix ("-H"/"-P") can
    never sneak a two-way player past the filter, though Statcast-sourced
    breakout names are already clean.

    Args:
        df: A breakout frame with a ``player_name`` column, already tagged
            with ownership.

    Returns:
        The frame with any ``TWO_WAY_EXCLUDE`` players removed.
    """
    canonical = df["player_name"].str.replace(r"-[HP]$", "", regex=True).str.strip()
    return df.loc[~canonical.isin(TWO_WAY_EXCLUDE)].reset_index(drop=True)


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

    print("Detecting breakout hitters...")
    breakout_h = detect_breakout_hitters(hitters)
    breakout_h = tag_ownership(breakout_h)
    before_h = len(breakout_h)
    breakout_h = drop_two_way(breakout_h)
    if before_h != len(breakout_h):
        print(f"  Excluded {before_h - len(breakout_h)} two-way player(s) from breakout output")
    fa_h = breakout_h[breakout_h["ownership"] == "FA"].reset_index(drop=True)
    print(f"  {len(breakout_h)} breakout hitter candidates ({len(fa_h)} free agents)\n")
    if not breakout_h.empty:
        print_hitter_table(breakout_h)
    print()

    print("Detecting breakout pitchers...")
    breakout_p = detect_breakout_pitchers(pitchers)
    print(f"  {len(breakout_p)} total pitcher breakout candidates")
    breakout_p = tag_ownership(breakout_p)
    before_p = len(breakout_p)
    breakout_p = drop_two_way(breakout_p)
    if before_p != len(breakout_p):
        print(f"  Excluded {before_p - len(breakout_p)} two-way player(s) from breakout output")
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
