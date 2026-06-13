"""Add/drop suggestion engine comparing roster weaknesses to FA upgrades.

For each rostered position, identifies the weakest player by composite score
and compares against the best available free agents.  A swap is flagged when
the FA's composite score exceeds the roster player by 10+ percentile points.

Availability comes from resolved identity, not names: the full Statcast
population is scored once (so roster and FA composites share one percentile
basis) and ``status`` 'owned'/'fa' is attached from the silver ID map by
vendor id.  The add pool is everyone with ``status == "fa"`` — which
permanently excludes rostered two-way players (Ohtani's "-H"/"-P" Fantrax
rows) and the accent-mismatch class of name-matching bugs.

Inputs:
    bronze/data/fantrax/my_roster_*.csv
    silver/data/statcast_hitters.parquet
    silver/data/statcast_pitchers.parquet
    silver/data/player_id_map.parquet   (ownership: status owned/fa)

Outputs:
    gold/data/add_drop_suggestions.csv
"""

import pathlib

import pandas as pd

from gold import waiver_ranker
from gold.ownership import attach_status

# ── paths ────────────────────────────────────────────────────────────

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fantrax"

# ── constants ────────────────────────────────────────────────────────

HITTER_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF"]
PITCHER_POSITIONS = ["SP"]
ALL_POSITIONS = HITTER_POSITIONS + PITCHER_POSITIONS

UPGRADE_THRESHOLD = 10  # minimum composite gap to flag a swap
TOP_FA_PER_POSITION = 3
YOUNG_ASSET_AGE = 26

MY_TEAM = "Rutsch Hour"


# ── loaders ──────────────────────────────────────────────────────────


def _latest_fantrax(prefix: str) -> pd.DataFrame | None:
    """Load the most recent date-stamped Fantrax CSV matching *prefix*.

    Args:
        prefix: File prefix to glob (e.g. ``"my_roster"``).

    Returns:
        DataFrame from the latest matching CSV, or None if no files found.
    """
    files = sorted(FANTRAX_DIR.glob(f"{prefix}_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])


def load_my_roster() -> pd.DataFrame:
    """Load my Fantrax roster with player names and positions.

    Returns:
        DataFrame with ``player_name`` and ``position`` columns.

    Raises:
        FileNotFoundError: If no my_roster CSV exists.
    """
    df = _latest_fantrax("my_roster")
    if df is None:
        raise FileNotFoundError(
            f"No my_roster_*.csv found in {FANTRAX_DIR}"
        )
    df = df[df["player_name"].notna() & (df["player_name"] != "None")].copy()
    df["player_name"] = df["player_name"].str.strip()
    return df[["player_name", "position"]]


def load_scored_hitters() -> pd.DataFrame:
    """Score the full Statcast hitter population and attach ownership.

    Scoring runs over every hitter (not the FA-only waiver pool) so roster
    and free-agent composites share one percentile basis and remain
    comparable for the upgrade-threshold check.  ``status`` is then
    attached by vendor id from the silver ID map.

    Returns:
        DataFrame with composite_hitter_score, Statcast metrics, and a
        ``status`` column ('owned'/'fa', or <NA> when unresolved).
    """
    df = pd.read_parquet(SILVER_DIR / "statcast_hitters.parquet")
    return attach_status(waiver_ranker.score_hitters(df))


def load_scored_pitchers() -> pd.DataFrame:
    """Score the full Statcast pitcher population and attach ownership.

    Same one-population scoring as :func:`load_scored_hitters` so roster and
    FA pitcher composites stay comparable.

    Returns:
        DataFrame with composite_pitcher_score, Statcast metrics, and a
        ``status`` column ('owned'/'fa', or <NA> when unresolved).
    """
    df = pd.read_parquet(SILVER_DIR / "statcast_pitchers.parquet")
    return attach_status(waiver_ranker.score_pitchers(df))


# ── engine ───────────────────────────────────────────────────────────


def _xwoba_gap_col(is_pitcher: bool) -> str:
    """Return the appropriate expected-vs-actual gap column name.

    Args:
        is_pitcher: True for pitchers, False for hitters.

    Returns:
        Column name string.
    """
    return "xera_minus_era" if is_pitcher else "xwoba_minus_woba"


def _composite_col(is_pitcher: bool) -> str:
    """Return the appropriate composite score column name.

    Args:
        is_pitcher: True for pitchers, False for hitters.

    Returns:
        Column name string.
    """
    return "composite_pitcher_score" if is_pitcher else "composite_hitter_score"


def find_suggestions(
    my_roster: pd.DataFrame,
    waiver_hitters: pd.DataFrame,
    waiver_pitchers: pd.DataFrame,
) -> pd.DataFrame:
    """Compare roster weaknesses against free-agent upgrades by position.

    For each position, finds the weakest rostered player (lowest composite
    score) and the top free agents.  A swap is suggested when the FA's
    composite score exceeds the roster player's by at least
    ``UPGRADE_THRESHOLD`` percentile points.

    The add pool is drawn from ``status == "fa"`` rather than a name-based
    exclusion of owned players, so resolved-but-rostered players (Ohtani's
    two-way "-H"/"-P" rows, accented names) can never appear as adds.

    Args:
        my_roster: DataFrame with ``player_name`` and ``position``.
        waiver_hitters: Scored hitter DataFrame with composite scores and a
            ``status`` column.
        waiver_pitchers: Scored pitcher DataFrame with composite scores and a
            ``status`` column.

    Returns:
        DataFrame of suggested add/drop moves.
    """
    suggestions = []

    for pos in ALL_POSITIONS:
        is_pitcher = pos in PITCHER_POSITIONS
        waiver_df = waiver_pitchers if is_pitcher else waiver_hitters
        score_col = _composite_col(is_pitcher)
        gap_col = _xwoba_gap_col(is_pitcher)

        # Players at this position on my roster
        roster_at_pos = my_roster[my_roster["position"] == pos]
        if roster_at_pos.empty:
            continue

        # Match roster players to waiver data for scores
        roster_names = set(roster_at_pos["player_name"])
        roster_scored = waiver_df[waiver_df["player_name"].isin(roster_names)].copy()
        if roster_scored.empty:
            continue

        # Weakest rostered player at this position
        weakest = roster_scored.sort_values(score_col).iloc[0]
        drop_name = weakest["player_name"]
        drop_score = weakest[score_col]
        drop_gap = weakest.get(gap_col, 0.0)
        drop_age = weakest.get("age", None)

        # Free agents at this position — availability is the resolved
        # ownership status, never a name match.
        fa_pool = waiver_df[waiver_df["status"] == "fa"].copy()
        # For hitters, match position; pitchers are already a pitcher pool
        if not is_pitcher and "position" in fa_pool.columns:
            fa_pool = fa_pool[fa_pool["position"] == pos]

        top_fas = fa_pool.nlargest(TOP_FA_PER_POSITION, score_col)

        for _, fa in top_fas.iterrows():
            add_score = fa[score_col]
            net_upgrade = round(add_score - drop_score, 1)

            if net_upgrade < UPGRADE_THRESHOLD:
                continue

            # Dynasty warning for young assets
            dynasty_warning = ""
            if drop_age is not None and drop_age < YOUNG_ASSET_AGE:
                dynasty_warning = "young asset - consider trading instead of dropping"

            suggestions.append({
                "position": pos,
                "drop_candidate": drop_name,
                "drop_score": drop_score,
                "drop_xwoba_gap": round(drop_gap, 3) if pd.notna(drop_gap) else 0.0,
                "add_candidate": fa["player_name"],
                "add_score": add_score,
                "add_xwoba_gap": round(fa.get(gap_col, 0.0), 3)
                if pd.notna(fa.get(gap_col, 0.0))
                else 0.0,
                "net_upgrade": net_upgrade,
                "dynasty_warning": dynasty_warning,
            })

    result = pd.DataFrame(suggestions)
    if not result.empty:
        result = result.sort_values("net_upgrade", ascending=False).reset_index(
            drop=True
        )
    return result


# ── display ──────────────────────────────────────────────────────────


def print_suggestions(df: pd.DataFrame) -> None:
    """Print the add/drop suggestions table to the console.

    Args:
        df: Suggestions DataFrame from ``find_suggestions()``.
    """
    if df.empty:
        print("  No suggested moves — your roster is solid at every position.")
        return

    cols = [
        "position",
        "drop_candidate",
        "drop_score",
        "add_candidate",
        "add_score",
        "net_upgrade",
        "dynasty_warning",
    ]
    names = {
        "position": "Pos",
        "drop_candidate": "Drop",
        "drop_score": "Drop Score",
        "add_candidate": "Add",
        "add_score": "Add Score",
        "net_upgrade": "Net +/-",
        "dynasty_warning": "Dynasty",
    }
    display = df[cols].rename(columns=names)
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Run the add/drop engine and save suggestions to gold layer."""
    print("Loading my roster...")
    my_roster = load_my_roster()
    print(f"  {len(my_roster)} players on roster")

    print("Scoring full player pool and attaching ownership...")
    waiver_h = load_scored_hitters()
    waiver_p = load_scored_pitchers()
    fa_h = int((waiver_h["status"] == "fa").sum())
    fa_p = int((waiver_p["status"] == "fa").sum())
    no_idmap = int(waiver_h["status"].isna().sum() + waiver_p["status"].isna().sum())
    print(
        f"  {len(waiver_h)} hitters, {len(waiver_p)} pitchers scored "
        f"({fa_h} FA hitters, {fa_p} FA pitchers; "
        f"{no_idmap} with no id_map row)"
    )

    print("Finding add/drop suggestions...")
    suggestions = find_suggestions(my_roster, waiver_h, waiver_p)

    if not suggestions.empty:
        print(f"\n  {len(suggestions)} suggested moves:\n")
        print_suggestions(suggestions)
    else:
        print("  No suggested moves.")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "add_drop_suggestions.csv"
    suggestions.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
