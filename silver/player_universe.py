"""Unified player universe with cross-source ID mapping and metadata.

Builds a master player table by fuzzy-matching names from Baseball Savant
and FanGraphs bronze-layer CSVs.  The output is a single Parquet file
containing one row per player with IDs from both sources.
"""

import pathlib
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

BRONZE_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"

MATCH_THRESHOLD = 90


# ── helpers ──────────────────────────────────────────────────────────


def _latest_csv(directory: pathlib.Path, suffix: str) -> pathlib.Path:
    """Return the most recent CSV in *directory* whose name ends with *suffix*.

    Files are expected to follow the ``YYYY-MM-DD_<suffix>.csv`` naming
    convention used by the bronze layer.

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


def _normalize_name(name: str) -> str:
    """Strip accents and normalize whitespace for fuzzy comparison.

    Args:
        name: Raw player name string.

    Returns:
        ASCII-folded, whitespace-collapsed name.
    """
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(stripped.split())


# ── loaders ──────────────────────────────────────────────────────────


def load_savant_players() -> pd.DataFrame:
    """Load and deduplicate players from the latest Savant batting and pitching CSVs.

    Savant CSVs have columns ``"last_name, first_name"``, ``player_id``,
    and ``year``.  The name column is reformatted to ``"First Last"``
    order for downstream matching.

    Returns:
        DataFrame with columns ``[player_name, savant_player_id]``.
    """
    frames = []
    for stat_type in ("batting", "pitching"):
        path = _latest_csv(BRONZE_DIR / "savant", stat_type)
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id"])

    # Convert "Last, First" → "First Last"
    combined["player_name"] = combined["last_name, first_name"].apply(
        lambda x: " ".join(reversed(x.split(", ")))
    )
    combined = combined.rename(columns={"player_id": "savant_player_id"})
    return combined[["player_name", "savant_player_id"]].reset_index(drop=True)


def load_fangraphs_players() -> pd.DataFrame:
    """Load and deduplicate players from the latest FanGraphs batting and pitching CSVs.

    Batting players keep whatever position FanGraphs reports.  Pitching-only
    players are assigned position ``"P"``.

    Returns:
        DataFrame with columns
        ``[player_name, fangraphs_id, team, position, age]``.
    """
    bat_path = _latest_csv(BRONZE_DIR / "fangraphs", "batting")
    bat = pd.read_csv(bat_path, usecols=["IDfg", "Name", "Team", "Age"])
    bat = bat.rename(columns={"IDfg": "fangraphs_id", "Name": "player_name",
                              "Team": "team", "Age": "age"})
    bat["position"] = None

    pit_path = _latest_csv(BRONZE_DIR / "fangraphs", "pitching")
    pit = pd.read_csv(pit_path, usecols=["IDfg", "Name", "Team", "Age"])
    pit = pit.rename(columns={"IDfg": "fangraphs_id", "Name": "player_name",
                              "Team": "team", "Age": "age"})
    pit["position"] = "P"

    combined = pd.concat([bat, pit], ignore_index=True)
    # Keep batting row when a player appears in both (has richer position info).
    combined = combined.drop_duplicates(subset=["fangraphs_id"], keep="first")
    return combined.reset_index(drop=True)


# ── matching ─────────────────────────────────────────────────────────


def build_player_universe(
    savant: pd.DataFrame,
    fangraphs: pd.DataFrame,
    threshold: int = MATCH_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fuzzy-match Savant and FanGraphs players into a unified table.

    For each Savant player, the best FanGraphs name match (by
    ``rapidfuzz.fuzz.ratio`` on accent-stripped names) is accepted when
    the score meets *threshold*.

    Args:
        savant: Output of :func:`load_savant_players`.
        fangraphs: Output of :func:`load_fangraphs_players`.
        threshold: Minimum ``fuzz.ratio`` score to accept a match.

    Returns:
        A tuple of ``(matched, unmatched_savant, unmatched_fangraphs)``.
        *matched* contains one row per successfully linked player.
    """
    fg_names = fangraphs["player_name"].tolist()
    fg_names_norm = [_normalize_name(n) for n in fg_names]

    matched_rows: list[dict] = []
    matched_fg_indices: set[int] = set()

    for _, sav_row in savant.iterrows():
        sav_norm = _normalize_name(sav_row["player_name"])
        result = process.extractOne(
            sav_norm, fg_names_norm, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if result is None:
            continue

        _, score, fg_idx = result
        if fg_idx in matched_fg_indices:
            continue

        matched_fg_indices.add(fg_idx)
        fg_row = fangraphs.iloc[fg_idx]
        matched_rows.append({
            "player_name": fg_row["player_name"],
            "savant_player_id": int(sav_row["savant_player_id"]),
            "fangraphs_id": int(fg_row["fangraphs_id"]),
            "team": fg_row["team"],
            "position": fg_row["position"],
            "age": int(fg_row["age"]),
        })

    matched = pd.DataFrame(matched_rows)

    matched_savant_ids = {r["savant_player_id"] for r in matched_rows}
    unmatched_savant = savant[
        ~savant["savant_player_id"].isin(matched_savant_ids)
    ].reset_index(drop=True)

    unmatched_fg = fangraphs[
        ~fangraphs.index.isin(matched_fg_indices)
    ].reset_index(drop=True)

    return matched, unmatched_savant, unmatched_fg


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Build the player universe and save to Parquet."""
    print("Loading Savant players...")
    savant = load_savant_players()
    print(f"  {len(savant)} unique Savant players")

    print("Loading FanGraphs players...")
    fangraphs = load_fangraphs_players()
    print(f"  {len(fangraphs)} unique FanGraphs players")

    print("Matching players across sources...")
    matched, unmatched_sav, unmatched_fg = build_player_universe(
        savant, fangraphs
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "player_universe.parquet"
    matched.to_parquet(out_path, index=False)

    print(f"\nPlayer universe saved to {out_path}")
    print(f"  Matched:              {len(matched)}")
    print(f"  Unmatched (Savant):   {len(unmatched_sav)}")
    print(f"  Unmatched (FanGraphs):{len(unmatched_fg)}")


if __name__ == "__main__":
    main()
