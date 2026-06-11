"""Master player identity resolution for the fantasy baseball pipeline.

Builds a single ID map that links every Fantrax roster player to their
Baseball Savant and FanGraphs IDs via fuzzy name matching.  All other
modules should use this map instead of doing their own matching.

Two-way players (e.g. Ohtani) who appear in Fantrax rosters as BOTH a
hitter and a pitcher get TWO separate rows — one per player_type — so
they are never accidentally merged.

Outputs:
    silver/data/player_id_map.parquet
"""

import hashlib
import pathlib
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process

from silver.freshness import warn_if_stale_fangraphs

# ── paths ──────────────────────────────────────────────────────────────

BRONZE_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fantrax"

MATCH_THRESHOLD = 90
_PITCHER_POSITIONS = {"SP", "RP", "P"}


# ── helpers ────────────────────────────────────────────────────────────


def _latest_csv(directory: pathlib.Path, prefix: str) -> pathlib.Path | None:
    """Return the most recent CSV matching ``<prefix>_*.csv``, or None."""
    matches = sorted(directory.glob(f"{prefix}_*.csv"))
    return matches[-1] if matches else None


def _normalize_name(name: str) -> str:
    """Strip accents and normalize whitespace for fuzzy comparison."""
    decomposed = unicodedata.normalize("NFD", name)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(stripped.split())


def _make_fantrax_id(player_name: str, player_type: str) -> str:
    """Generate a deterministic Fantrax ID from name + type."""
    key = f"{player_name.strip().lower()}|{player_type}"
    return "ftx_" + hashlib.md5(key.encode()).hexdigest()[:12]


# ── loaders ────────────────────────────────────────────────────────────


def _load_fantrax_players() -> pd.DataFrame:
    """Load all Fantrax roster players, detecting two-way players.

    Returns DataFrame with columns:
        fantrax_id, player_name, position, player_type, fantrax_team_name
    """
    my_path = _latest_csv(FANTRAX_DIR, "my_roster")
    all_path = _latest_csv(FANTRAX_DIR, "all_rosters")

    frames = []
    if my_path is not None:
        df = pd.read_csv(my_path)
        if "team_name" not in df.columns:
            df["team_name"] = "Rutsch Hour"
        frames.append(df)
    if all_path is not None:
        frames.append(pd.read_csv(all_path))

    if not frames:
        raise FileNotFoundError(f"No Fantrax roster CSVs found in {FANTRAX_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined["player_name"] = combined["player_name"].str.strip()
    combined["position"] = combined["position"].str.strip()
    combined["team_name"] = combined["team_name"].str.strip()

    # Drop empty slots
    combined = combined[
        combined["player_name"].notna()
        & (combined["player_name"] != "None")
        & (combined["player_name"] != "")
    ].copy()

    # Assign player_type
    combined["player_type"] = combined["position"].apply(
        lambda p: "Pitcher" if p in _PITCHER_POSITIONS else "Hitter"
    )

    # Deduplicate: keep one row per (player_name, player_type, team_name)
    # This naturally preserves two-way players as separate rows
    combined = combined.drop_duplicates(
        subset=["player_name", "player_type", "team_name"], keep="first"
    )

    # If a player appears on multiple teams for the SAME type, keep the
    # first (shouldn't happen in a well-formed league, but be safe)
    combined = combined.drop_duplicates(
        subset=["player_name", "player_type"], keep="first"
    )

    combined["fantrax_id"] = combined.apply(
        lambda r: _make_fantrax_id(r["player_name"], r["player_type"]), axis=1
    )
    combined = combined.rename(columns={"team_name": "fantrax_team_name"})

    return combined[
        ["fantrax_id", "player_name", "position", "player_type", "fantrax_team_name"]
    ].reset_index(drop=True)


def _load_savant_players() -> pd.DataFrame:
    """Load deduplicated Savant players from batting + pitching CSVs.

    Returns DataFrame with [player_name, savant_player_id].
    """
    frames = []
    for stat_type in ("batting", "pitching"):
        csv_dir = BRONZE_DIR / "savant"
        matches = sorted(csv_dir.glob(f"*_{stat_type}.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[-1])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_name", "savant_player_id"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["player_id"])

    # "Last, First" → "First Last"
    combined["player_name"] = combined["last_name, first_name"].apply(
        lambda x: " ".join(reversed(x.split(", ")))
    )
    combined = combined.rename(columns={"player_id": "savant_player_id"})
    return combined[["player_name", "savant_player_id"]].reset_index(drop=True)


def _load_fangraphs_players() -> pd.DataFrame:
    """Load deduplicated FanGraphs players from batting + pitching CSVs.

    Returns DataFrame with [player_name, fangraphs_id, fg_team].
    """
    frames = []
    fg_dir = BRONZE_DIR / "fangraphs"
    for stat_type in ("batting", "pitching"):
        matches = sorted(fg_dir.glob(f"*_{stat_type}.csv"))
        if not matches:
            continue
        warn_if_stale_fangraphs(matches[-1])
        df = pd.read_csv(matches[-1], usecols=["IDfg", "Name", "Team"])
        df = df.rename(columns={"IDfg": "fangraphs_id", "Name": "player_name", "Team": "fg_team"})
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_name", "fangraphs_id", "fg_team"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["fangraphs_id"], keep="first")
    return combined.reset_index(drop=True)


# ── matching ───────────────────────────────────────────────────────────


def build_player_id_map() -> pd.DataFrame:
    """Build the master player ID map.

    For every Fantrax player, fuzzy-match against Savant and FanGraphs
    player lists to find cross-source IDs.  Two-way players keep
    separate rows per player_type.

    Returns:
        DataFrame with columns: fantrax_id, player_name, team, position,
        player_type, fantrax_team_name, savant_player_id, fangraphs_id,
        match_quality
    """
    print("Loading Fantrax rosters...")
    fantrax = _load_fantrax_players()
    print(f"  {len(fantrax)} Fantrax roster entries")

    print("Loading Savant players...")
    savant = _load_savant_players()
    print(f"  {len(savant)} Savant players")

    print("Loading FanGraphs players...")
    try:
        fangraphs = _load_fangraphs_players()
        print(f"  {len(fangraphs)} FanGraphs players")
    except Exception:
        print("  FanGraphs data not available, skipping")
        fangraphs = pd.DataFrame(columns=["player_name", "fangraphs_id", "fg_team"])

    # Prep Savant lookup
    sav_names = savant["player_name"].tolist()
    sav_names_norm = [_normalize_name(n) for n in sav_names]

    # Prep FanGraphs lookup
    fg_names = fangraphs["player_name"].tolist()
    fg_names_norm = [_normalize_name(n) for n in fg_names]

    rows: list[dict] = []
    for _, ftx_row in fantrax.iterrows():
        name = ftx_row["player_name"]
        name_norm = _normalize_name(name)

        # --- Match to Savant ---
        savant_id = None
        savant_quality = "unmatched"
        sav_result = process.extractOne(
            name_norm, sav_names_norm, scorer=fuzz.token_sort_ratio,
            score_cutoff=MATCH_THRESHOLD,
        )
        if sav_result is not None:
            _, score, sav_idx = sav_result
            savant_id = int(savant.iloc[sav_idx]["savant_player_id"])
            savant_quality = "exact" if score == 100 else "fuzzy"

        # --- Match to FanGraphs ---
        fg_id = None
        fg_team = None
        fg_quality = "unmatched"
        if fg_names:
            fg_result = process.extractOne(
                name_norm, fg_names_norm, scorer=fuzz.token_sort_ratio,
                score_cutoff=MATCH_THRESHOLD,
            )
            if fg_result is not None:
                _, score, fg_idx = fg_result
                fg_id = int(fangraphs.iloc[fg_idx]["fangraphs_id"])
                fg_team = fangraphs.iloc[fg_idx]["fg_team"]
                fg_quality = "exact" if score == 100 else "fuzzy"

        # Overall match quality
        if savant_quality != "unmatched" or fg_quality != "unmatched":
            best = savant_quality if savant_quality != "unmatched" else fg_quality
            if best == "exact" or fg_quality == "exact":
                match_quality = "exact"
            else:
                match_quality = "fuzzy"
        else:
            match_quality = "unmatched"

        # Team comes from FanGraphs (MLB team), not Fantrax
        team = fg_team if fg_team else None

        rows.append({
            "fantrax_id": ftx_row["fantrax_id"],
            "player_name": name,
            "team": team,
            "position": ftx_row["position"],
            "player_type": ftx_row["player_type"],
            "fantrax_team_name": ftx_row["fantrax_team_name"],
            "savant_player_id": savant_id,
            "fangraphs_id": fg_id,
            "match_quality": match_quality,
        })

    id_map = pd.DataFrame(rows)

    # Coerce ID columns to nullable int
    for col in ("savant_player_id", "fangraphs_id"):
        id_map[col] = pd.array(id_map[col], dtype=pd.Int64Dtype())

    # Fill missing team from player_universe if available
    pu_path = DATA_DIR / "player_universe.parquet"
    if pu_path.exists():
        pu = pd.read_parquet(pu_path, columns=["player_name", "team"])
        pu = pu.drop_duplicates(subset=["player_name"], keep="first")
        missing_team = id_map["team"].isna()
        if missing_team.any():
            fill = id_map.loc[missing_team, ["player_name"]].merge(
                pu, on="player_name", how="left", suffixes=("", "_pu")
            )
            id_map.loc[missing_team, "team"] = fill["team"].values

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "player_id_map.parquet"
    id_map.to_parquet(out_path, index=False)
    print(f"\nPlayer ID map saved to {out_path}")

    return id_map


# ── lookup helpers ─────────────────────────────────────────────────────


def load_player_id_map() -> pd.DataFrame:
    """Load the pre-built player ID map from Parquet."""
    path = DATA_DIR / "player_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m silver.player_id_map` first."
        )
    return pd.read_parquet(path)


def get_player_data(
    player_name: str,
    player_type: str | None = None,
) -> pd.Series | None:
    """Look up a single player by name (and optional type) from the ID map.

    Returns the full row as a Series, or None if not found.
    """
    id_map = load_player_id_map()
    mask = id_map["player_name"].str.lower() == player_name.strip().lower()
    if player_type:
        mask = mask & (id_map["player_type"] == player_type)
    matches = id_map.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def enrich_with_fantrax(
    df: pd.DataFrame,
    name_col: str = "player_name",
) -> pd.DataFrame:
    """LEFT JOIN the ID map onto *df* to add team, position, fantrax_team_name.

    Joins on player name.  Existing columns in *df* are NOT overwritten.
    """
    try:
        id_map = load_player_id_map()
    except FileNotFoundError:
        return df

    # Deduplicate the map — for enrichment, prefer Hitter rows (more common)
    lookup = id_map.drop_duplicates(subset=["player_name"], keep="first")

    cols_to_add = ["team", "position", "fantrax_team_name", "savant_player_id", "fangraphs_id"]
    cols_to_add = [c for c in cols_to_add if c not in df.columns]
    if not cols_to_add:
        return df

    merged = df.merge(
        lookup[["player_name"] + cols_to_add],
        left_on=name_col,
        right_on="player_name",
        how="left",
        suffixes=("", "_idmap"),
    )

    # Clean up duplicate player_name column if name_col differs
    if name_col != "player_name" and "player_name_idmap" in merged.columns:
        merged = merged.drop(columns=["player_name_idmap"])
    elif name_col != "player_name" and "player_name" in merged.columns:
        extra_cols = [c for c in merged.columns if c.endswith("_idmap")]
        merged = merged.drop(columns=extra_cols, errors="ignore")

    return merged


# ── quality report ─────────────────────────────────────────────────────


def print_quality_report(id_map: pd.DataFrame) -> None:
    """Print a summary of match quality for debugging."""
    total = len(id_map)
    savant_matched = id_map["savant_player_id"].notna().sum()
    fg_matched = id_map["fangraphs_id"].notna().sum()
    unmatched = id_map[
        id_map["savant_player_id"].isna() & id_map["fangraphs_id"].isna()
    ]

    # Two-way players: names that appear with both Hitter and Pitcher types
    type_counts = id_map.groupby("player_name")["player_type"].nunique()
    two_way = type_counts[type_counts > 1].index.tolist()

    print(f"\n{'=' * 60}")
    print("  Player ID Map — Quality Report")
    print(f"{'=' * 60}")
    print(f"  Total Fantrax players:    {total}")
    print(f"  Matched to Savant:        {savant_matched} ({100 * savant_matched / total:.0f}%)")
    print(f"  Matched to FanGraphs:     {fg_matched} ({100 * fg_matched / total:.0f}%)")
    print(f"  Fully unmatched:          {len(unmatched)}")

    if not unmatched.empty:
        print(f"\n  Unmatched players:")
        for _, row in unmatched.iterrows():
            print(f"    - {row['player_name']} ({row['position']}, {row['fantrax_team_name']})")

    if two_way:
        print(f"\n  Two-way players found ({len(two_way)}):")
        for name in two_way:
            entries = id_map[id_map["player_name"] == name]
            owners = entries["fantrax_team_name"].tolist()
            print(f"    - {name}: {', '.join(f'{t} ({o})' for t, o in zip(entries['player_type'], owners))}")

    print(f"{'=' * 60}")


# ── entry point ────────────────────────────────────────────────────────


def main() -> None:
    """Build the player ID map and print quality report."""
    id_map = build_player_id_map()
    print_quality_report(id_map)


if __name__ == "__main__":
    main()
