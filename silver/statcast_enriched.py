"""Statcast data enriched with percentiles and expected-stat differentials.

Joins Baseball Savant expected-stats and batted-ball leaderboard data
onto the player universe, then merges selected FanGraphs columns
(speed score for hitters, K% for pitchers), and computes derived metrics
(xwOBA-minus-wOBA, percentile ranks for barrel rate, etc.) that surface
buy-low / sell-high candidates in fantasy leagues.

Inputs:
    silver/data/player_universe.parquet
    bronze/data/savant/<latest>_batting.csv
    bronze/data/savant/<latest>_pitching.csv
    bronze/data/savant/<latest>_batting_statcast.csv
    bronze/data/savant/<latest>_pitching_statcast.csv
    bronze/data/fangraphs/<latest>_batting.csv
    bronze/data/fangraphs/<latest>_pitching.csv

Outputs:
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet
"""

import pathlib

import pandas as pd

from silver.freshness import warn_if_stale_fangraphs

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
BRONZE_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data"


# ── helpers ──────────────────────────────────────────────────────────


def _latest_csv(directory: pathlib.Path, suffix: str) -> pathlib.Path:
    """Return the most recent CSV in *directory* whose name ends with *suffix*.

    Args:
        directory: Folder to search.
        suffix: Trailing part of the filename before ``.csv``
                (e.g. ``"batting"``).

    Returns:
        Path to the newest matching file.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    matches = sorted(directory.glob(f"*_{suffix}.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No *_{suffix}.csv files found in {directory}"
        )
    return matches[-1]


# ── loaders ──────────────────────────────────────────────────────────


def _load_id_bridge() -> pd.DataFrame:
    """Load the savant_player_id -> fangraphs_id/team/position bridge.

    Sourced from the resolved player_id_map (NOT player_universe), which
    already links the two vendor id systems. Used to left-join FanGraphs
    columns onto the full Savant base; a player with no id_map row simply
    gets a null fangraphs_id and therefore null FanGraphs columns.

    Returns:
        DataFrame keyed by ``savant_player_id`` with ``fangraphs_id``,
        ``team``, and ``position``.  Empty (correct columns) if the map
        has not been built yet.
    """
    cols = ["savant_player_id", "fangraphs_id", "team", "position"]
    path = DATA_DIR / "player_id_map.parquet"
    if not path.exists():
        return pd.DataFrame(columns=cols)
    idm = pd.read_parquet(path, columns=cols)
    idm = idm.dropna(subset=["savant_player_id"]).copy()
    idm["savant_player_id"] = idm["savant_player_id"].astype("int64")
    # One bridge row per savant id (a two-way player has two id_map rows with
    # the same savant id; either carries the same team, so keep the first).
    return idm.drop_duplicates("savant_player_id", keep="first")


def build_player_base(
    savant_batting: pd.DataFrame, savant_pitching: pd.DataFrame
) -> pd.DataFrame:
    """Build the enrichment base from the FULL Savant population.

    Replaces the old player_universe base (Savant INNER JOIN a stale
    FanGraphs snapshot), which gated coverage to whatever survived the
    intersection (54 of 366 pitchers).  The base is the union of every
    Savant batting and pitching player id, bridged to FanGraphs ids and
    team/position via :func:`_load_id_bridge`.  FanGraphs is then a pure
    left-join enrichment in :func:`enrich_hitters` / :func:`enrich_pitchers`,
    null where no FanGraphs row exists.

    Args:
        savant_batting: Merged Savant batting DataFrame.
        savant_pitching: Merged Savant pitching DataFrame.

    Returns:
        DataFrame with ``player_name``, ``savant_player_id``,
        ``fangraphs_id``, ``team``, ``position``, ``age`` (age is null;
        it was a FanGraphs field the gold modules do not score on).
    """
    ids = pd.concat(
        [
            savant_batting[["player_id", "last_name, first_name"]],
            savant_pitching[["player_id", "last_name, first_name"]],
        ],
        ignore_index=True,
    ).drop_duplicates("player_id")
    ids = ids.rename(columns={"player_id": "savant_player_id"})
    ids["savant_player_id"] = ids["savant_player_id"].astype("int64")
    # Savant names are "Last, First"; the rest of the pipeline uses "First Last".
    ids["player_name"] = ids["last_name, first_name"].apply(
        lambda x: " ".join(reversed(str(x).split(", ")))
    )

    base = ids.merge(_load_id_bridge(), on="savant_player_id", how="left")
    base["age"] = pd.NA
    return base[
        ["player_name", "savant_player_id", "fangraphs_id", "team", "position", "age"]
    ]


def load_savant_batting() -> pd.DataFrame:
    """Load and merge the latest Savant expected-stats and batted-ball batting CSVs.

    The expected-stats CSV provides xBA, xSLG, xwOBA and their
    differentials.  The batted-ball CSV adds barrel rate, hard-hit rate,
    and exit-velocity metrics.  The two are joined on ``player_id``.

    Returns:
        Merged DataFrame of Savant batting stats keyed by ``player_id``.
    """
    expected = pd.read_csv(_latest_csv(BRONZE_DIR / "savant", "batting"))
    statcast = pd.read_csv(
        _latest_csv(BRONZE_DIR / "savant", "batting_statcast")
    )
    return expected.merge(statcast, on="player_id", suffixes=("", "_sc"))


def load_savant_pitching() -> pd.DataFrame:
    """Load and merge the latest Savant expected-stats and batted-ball pitching CSVs.

    The expected-stats CSV provides xBA-against, xwOBA-against, xERA,
    and their differentials.  The batted-ball CSV adds barrel rate and
    hard-hit rate allowed.  The two are joined on ``player_id``.

    Returns:
        Merged DataFrame of Savant pitching stats keyed by ``player_id``.
    """
    expected = pd.read_csv(_latest_csv(BRONZE_DIR / "savant", "pitching"))
    statcast = pd.read_csv(
        _latest_csv(BRONZE_DIR / "savant", "pitching_statcast")
    )
    return expected.merge(statcast, on="player_id", suffixes=("", "_sc"))


def load_fangraphs_batting() -> pd.DataFrame:
    """Load the latest FanGraphs batting leaderboard CSV.

    Returns:
        DataFrame keyed by ``IDfg`` with the ``Spd`` (speed) column.
    """
    path = _latest_csv(BRONZE_DIR / "fangraphs", "batting")
    warn_if_stale_fangraphs(path)
    return pd.read_csv(path, usecols=["IDfg", "Spd"])


def load_fangraphs_pitching() -> pd.DataFrame:
    """Load the latest FanGraphs pitching leaderboard CSV.

    Returns:
        DataFrame keyed by ``IDfg`` with ``K%``, ``K/9``, and ``BB%``.
    """
    path = _latest_csv(BRONZE_DIR / "fangraphs", "pitching")
    warn_if_stale_fangraphs(path)
    return pd.read_csv(path, usecols=["IDfg", "K%", "K/9", "BB%"])


# ── enrichment ───────────────────────────────────────────────────────


def enrich_hitters(
    base: pd.DataFrame, savant_batting: pd.DataFrame,
    fg_batting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join Savant batting stats onto the player universe and add derived metrics.

    Derived columns:
        - ``xwoba_minus_woba``: positive means the hitter is underperforming
          relative to contact quality (buy-low signal).
        - ``xba_minus_ba``: positive means hits should come.
        - ``hard_hit_percentile``: rank of ``ev95percent`` (0-100).
        - ``barrel_percentile``: rank of ``brl_percent`` (0-100).
        - ``chase_rate_percentile``: rank of ``anglesweetspotpercent``
          (0-100) as a proxy for plate discipline.
        - ``sprint_speed``: FanGraphs ``Spd`` score (SB proxy), if available.

    Args:
        base: Full-Savant player base from :func:`build_player_base`.
        savant_batting: Merged Savant batting DataFrame from
            :func:`load_savant_batting`.
        fg_batting: FanGraphs batting DataFrame from
            :func:`load_fangraphs_batting`.  Optional; when ``None`` the
            ``sprint_speed`` column is omitted.  Left-joined, so it is null
            for players with no FanGraphs row.

    Returns:
        Enriched hitter DataFrame sorted by ``xwoba_minus_woba`` descending.
    """
    merged = base.merge(
        savant_batting,
        left_on="savant_player_id",
        right_on="player_id",
        how="inner",
    )

    # FanGraphs speed score (left join; null for players with no FanGraphs row).
    # Cast the FanGraphs id to nullable Int64 so it matches the bridge's id
    # dtype (the base carries nulls for unbridged players).
    if fg_batting is not None:
        fg = fg_batting.rename(columns={"IDfg": "fangraphs_id", "Spd": "sprint_speed"})
        fg["fangraphs_id"] = fg["fangraphs_id"].astype("Int64")
        merged = merged.merge(fg, on="fangraphs_id", how="left")

    # Expected-stat differentials
    merged["xwoba_minus_woba"] = merged["est_woba"] - merged["woba"]
    merged["xba_minus_ba"] = merged["est_ba"] - merged["ba"]

    # Percentile ranks (0-100)
    merged["hard_hit_percentile"] = merged["ev95percent"].rank(pct=True) * 100
    merged["barrel_percentile"] = merged["brl_percent"].rank(pct=True) * 100
    # Sweet-spot % is a proxy for contact discipline; higher is better
    merged["chase_rate_percentile"] = (
        merged["anglesweetspotpercent"].rank(pct=True) * 100
    )

    return merged.sort_values("xwoba_minus_woba", ascending=False).reset_index(
        drop=True
    )


def enrich_pitchers(
    base: pd.DataFrame, savant_pitching: pd.DataFrame,
    fg_pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join Savant pitching stats onto the player universe and add derived metrics.

    Derived columns:
        - ``xera_minus_era``: positive means ERA should regress downward
          (buy-low signal for pitchers).
        - ``k_percent``: FanGraphs ``K%`` (strikeout rate), if available.
        - ``k_per_9``: FanGraphs ``K/9``, if available.
        - ``k_minus_bb_pct``: ``k_percent - bb_percent`` when both are
          present.
        - ``barrel_percentile``: rank of ``brl_percent`` (0-100,
          inverted so high percentile = fewer barrels allowed).
        - ``hard_hit_percentile``: rank of ``ev95percent`` (0-100,
          inverted so high percentile = less hard contact allowed).

    Args:
        base: Full-Savant player base from :func:`build_player_base`.
        savant_pitching: Merged Savant pitching DataFrame from
            :func:`load_savant_pitching`.
        fg_pitching: FanGraphs pitching DataFrame from
            :func:`load_fangraphs_pitching`.  Optional; when ``None`` the
            ``k_percent`` / ``k_per_9`` columns are omitted.  Left-joined, so
            they are null for players with no FanGraphs row.

    Returns:
        Enriched pitcher DataFrame sorted by ``xera_minus_era`` descending.
    """
    merged = base.merge(
        savant_pitching,
        left_on="savant_player_id",
        right_on="player_id",
        how="inner",
    )

    # FanGraphs strikeout and walk rates (left join; null for players with no
    # FanGraphs row). Cast the id to nullable Int64 to match the bridge dtype.
    if fg_pitching is not None:
        fg_renamed = fg_pitching.rename(columns={
            "IDfg": "fangraphs_id",
            "K%": "k_percent",
            "K/9": "k_per_9",
            "BB%": "bb_percent",
        })
        fg_renamed["fangraphs_id"] = fg_renamed["fangraphs_id"].astype("Int64")
        merged = merged.merge(fg_renamed, on="fangraphs_id", how="left")

    # Expected-stat differentials
    merged["xera_minus_era"] = merged["xera"] - merged["era"]

    # K-BB% when the columns are present
    if {"k_percent", "bb_percent"}.issubset(merged.columns):
        merged["k_minus_bb_pct"] = merged["k_percent"] - merged["bb_percent"]

    # Percentile ranks (0-100), inverted: lower barrel/hard-hit is better
    merged["barrel_percentile"] = (
        merged["brl_percent"].rank(pct=True, ascending=False) * 100
    )
    merged["hard_hit_percentile"] = (
        merged["ev95percent"].rank(pct=True, ascending=False) * 100
    )

    return merged.sort_values("xera_minus_era", ascending=False).reset_index(
        drop=True
    )


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Build enriched Statcast hitter and pitcher tables."""
    print("Loading Savant batting leaderboard...")
    savant_batting = load_savant_batting()
    print(f"  {len(savant_batting)} batting rows")

    print("Loading Savant pitching leaderboard...")
    savant_pitching = load_savant_pitching()
    print(f"  {len(savant_pitching)} pitching rows")

    print("Building full-Savant player base (id-bridged via player_id_map)...")
    base = build_player_base(savant_batting, savant_pitching)
    print(f"  {len(base)} players in base ({base['fangraphs_id'].notna().sum()} with a FanGraphs id)")

    print("Loading FanGraphs batting leaderboard...")
    fg_batting = load_fangraphs_batting()
    print(f"  {len(fg_batting)} FanGraphs batting rows")

    print("Loading FanGraphs pitching leaderboard...")
    fg_pitching = load_fangraphs_pitching()
    print(f"  {len(fg_pitching)} FanGraphs pitching rows")

    print("Enriching hitters...")
    hitters = enrich_hitters(base, savant_batting, fg_batting)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    hitters_path = DATA_DIR / "statcast_hitters.parquet"
    hitters.to_parquet(hitters_path, index=False)
    print(f"  {len(hitters)} hitters saved to {hitters_path}")

    print("Enriching pitchers...")
    pitchers = enrich_pitchers(base, savant_pitching, fg_pitching)
    pitchers_path = DATA_DIR / "statcast_pitchers.parquet"
    pitchers.to_parquet(pitchers_path, index=False)
    print(f"  {len(pitchers)} pitchers saved to {pitchers_path}")

    # Top 10 hitters by xwOBA - wOBA (biggest underperformers)
    print("\n-- Top 10 hitters by xwOBA minus wOBA (buy-low) --")
    display_cols_h = ["player_name", "team", "xwoba_minus_woba", "xba_minus_ba",
                      "hard_hit_percentile", "barrel_percentile"]
    cols_h = [c for c in display_cols_h if c in hitters.columns]
    print(hitters[cols_h].head(10).to_string(index=False))

    # Top 10 pitchers by xERA - ERA (ERA should drop)
    print("\n-- Top 10 pitchers by xERA minus ERA (buy-low) --")
    display_cols_p = ["player_name", "team", "xera_minus_era",
                      "barrel_percentile", "hard_hit_percentile"]
    cols_p = [c for c in display_cols_p if c in pitchers.columns]
    print(pitchers[cols_p].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
