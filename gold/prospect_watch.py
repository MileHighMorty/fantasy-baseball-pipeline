"""Prospect watch surfacing call-up candidates and stash-worthy minor leaguers.

Loads MiLB game-log data for tracked prospects, aggregates season stats,
checks 40-man roster status, and flags players performing above their
current level as call-up candidates.

Inputs:
    bronze/data/milb/{today}_batting.csv
    bronze/data/milb/{today}_pitching.csv
    config/prospect_watchlist.yaml
    MLB Stats API 40-man roster (live)

Outputs:
    gold/data/prospect_alerts.csv
"""

import pathlib
from datetime import date

import pandas as pd
import yaml

# ── paths ────────────────────────────────────────────────────────────

BRONZE_MILB = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "milb"
BRONZE_FANTRAX = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fantrax"
CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"
GOLD_DIR = pathlib.Path(__file__).resolve().parent / "data"

MY_TEAM = "Rutsch Hour"

# ── level hierarchy (higher number = closer to majors) ───────────────

LEVEL_RANK = {
    "Single-A": 1,
    "High-A": 2,
    "Double-A": 3,
    "Triple-A": 4,
}

# ── call-up thresholds (performing above level) ─────────────────────

CALLUP_BATTING = {
    "avg": 0.280,
    "obp": 0.350,
    "slg": 0.470,
    "min_level": 3,   # Double-A or higher
}

CALLUP_PITCHING = {
    "era": 3.50,
    "whip": 1.25,
    "min_level": 3,
}

# ── MLB team IDs (used for 40-man roster lookups) ────────────────────

TEAM_IDS = {
    "AZ": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CIN": 113, "CLE": 114, "COL": 115, "DET": 116,
    "HOU": 117, "KC": 118, "LAA": 108, "LAD": 119, "MIA": 146,
    "MIL": 158, "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
    "PHI": 143, "PIT": 134, "SD": 135, "SF": 137, "SEA": 136,
    "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "WSH": 120,
}


# ── loaders ──────────────────────────────────────────────────────────


def load_watchlist() -> list[dict]:
    """Load tracked prospects from config/prospect_watchlist.yaml.

    Returns:
        List of prospect dictionaries with name, player_id, team,
        position, level, and notes fields.

    Raises:
        FileNotFoundError: If the watchlist file does not exist.
    """
    watchlist_path = CONFIG_DIR / "prospect_watchlist.yaml"
    with open(watchlist_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("prospects", [])


def load_milb_batting(today: str) -> pd.DataFrame | None:
    """Load MiLB batting game logs from the bronze layer.

    Tries today's file first, then falls back to the most recent
    available file in the milb data directory.

    Args:
        today: ISO-format date string (e.g. '2026-04-03').

    Returns:
        DataFrame of batting game logs, or None if no data exists.
    """
    target = BRONZE_MILB / f"{today}_batting.csv"
    if target.exists():
        return pd.read_csv(target)

    candidates = sorted(BRONZE_MILB.glob("*_batting.csv"), reverse=True)
    if candidates:
        print(f"  No batting data for {today}, using {candidates[0].name}")
        return pd.read_csv(candidates[0])

    return None


def load_milb_pitching(today: str) -> pd.DataFrame | None:
    """Load MiLB pitching game logs from the bronze layer.

    Tries today's file first, then falls back to the most recent
    available file in the milb data directory.

    Args:
        today: ISO-format date string (e.g. '2026-04-03').

    Returns:
        DataFrame of pitching game logs, or None if no data exists.
    """
    target = BRONZE_MILB / f"{today}_pitching.csv"
    if target.exists():
        return pd.read_csv(target)

    candidates = sorted(BRONZE_MILB.glob("*_pitching.csv"), reverse=True)
    if candidates:
        print(f"  No pitching data for {today}, using {candidates[0].name}")
        return pd.read_csv(candidates[0])

    return None


def load_all_rosters() -> pd.DataFrame | None:
    """Load the most recent all_rosters CSV from Fantrax bronze data.

    Returns:
        DataFrame with team_name and player_name columns, or None.
    """
    files = sorted(BRONZE_FANTRAX.glob("all_rosters_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])


def load_40_man_roster(team_abbrev: str) -> set[int]:
    """Fetch the 40-man roster player IDs for a team from the MLB Stats API.

    Args:
        team_abbrev: Three-letter team abbreviation (e.g. 'TEX').

    Returns:
        Set of player IDs on the 40-man roster. Returns an empty set
        if the team abbreviation is unknown or the API call fails.
    """
    team_id = TEAM_IDS.get(team_abbrev)
    if team_id is None:
        return set()

    try:
        from bronze.milb_client import get_40_man_roster
        roster_df = get_40_man_roster(team_id)
        return set(roster_df["player_id"].dropna().astype(int))
    except Exception as exc:
        print(f"  Warning: could not fetch 40-man roster for {team_abbrev}: {exc}")
        return set()


# ── aggregation ──────────────────────────────────────────────────────


def aggregate_batting(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-log batting data into season totals per player.

    Computes AVG, OBP, SLG, K%, and BB% from raw counting stats.

    Args:
        df: Raw batting game-log DataFrame.

    Returns:
        DataFrame with one row per player and season summary stats.
    """
    numeric_cols = ["ab", "h", "bb", "so", "hr", "doubles", "triples"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = df.groupby(["player_id", "player_name"]).agg(
        team=("team", "last"),
        league_level=("league_level", "last"),
        games=("date", "nunique"),
        ab=("ab", "sum"),
        h=("h", "sum"),
        bb=("bb", "sum"),
        so=("so", "sum"),
        hr=("hr", "sum"),
    ).reset_index()

    pa = grouped["ab"] + grouped["bb"]
    grouped["avg"] = (grouped["h"] / grouped["ab"]).round(3)
    grouped["obp"] = ((grouped["h"] + grouped["bb"]) / pa).round(3)

    # SLG requires singles estimate (h - hr - doubles - triples)
    if "doubles" in df.columns and "triples" in df.columns:
        doubles = df.groupby("player_id")["doubles"].sum().reset_index()
        triples = df.groupby("player_id")["triples"].sum().reset_index()
        grouped = grouped.merge(doubles, on="player_id", how="left")
        grouped = grouped.merge(triples, on="player_id", how="left")
        grouped["doubles"] = grouped["doubles"].fillna(0)
        grouped["triples"] = grouped["triples"].fillna(0)
        singles = grouped["h"] - grouped["hr"] - grouped["doubles"] - grouped["triples"]
        total_bases = singles + 2 * grouped["doubles"] + 3 * grouped["triples"] + 4 * grouped["hr"]
        grouped["slg"] = (total_bases / grouped["ab"]).round(3)
        grouped.drop(columns=["doubles", "triples"], inplace=True)
    else:
        grouped["slg"] = None

    grouped["k_pct"] = ((grouped["so"] / pa) * 100).round(1)
    grouped["bb_pct"] = ((grouped["bb"] / pa) * 100).round(1)

    # Replace inf/NaN from zero-PA edge cases
    grouped = grouped.fillna(0)
    grouped = grouped.replace([float("inf"), float("-inf")], 0)

    return grouped


def aggregate_pitching(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-log pitching data into season totals per player.

    Computes ERA, WHIP, K/9, and BB/9 from raw counting stats.

    Args:
        df: Raw pitching game-log DataFrame.

    Returns:
        DataFrame with one row per player and season summary stats.
    """
    for col in ["ip", "er", "h", "bb", "so", "hr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = df.groupby(["player_id", "player_name"]).agg(
        team=("team", "last"),
        league_level=("league_level", "last"),
        games=("date", "nunique"),
        ip=("ip", "sum"),
        er=("er", "sum"),
        h=("h", "sum"),
        bb=("bb", "sum"),
        so=("so", "sum"),
    ).reset_index()

    grouped["era"] = ((grouped["er"] * 9) / grouped["ip"]).round(2)
    grouped["whip"] = ((grouped["h"] + grouped["bb"]) / grouped["ip"]).round(2)
    grouped["k_per_9"] = ((grouped["so"] * 9) / grouped["ip"]).round(1)
    grouped["bb_per_9"] = ((grouped["bb"] * 9) / grouped["ip"]).round(1)

    grouped = grouped.fillna(0)
    grouped = grouped.replace([float("inf"), float("-inf")], 0)

    return grouped


# ── call-up detection ────────────────────────────────────────────────


def flag_callup_hitter(row: pd.Series) -> bool:
    """Check if a hitter is performing above their current level.

    A hitter is a call-up candidate when at Double-A or above with
    AVG >= .280, OBP >= .350, and SLG >= .470.

    Args:
        row: A single-row Series from the aggregated batting DataFrame.

    Returns:
        True if the player meets call-up thresholds.
    """
    level = LEVEL_RANK.get(row.get("league_level", ""), 0)
    if level < CALLUP_BATTING["min_level"]:
        return False
    return (
        row.get("avg", 0) >= CALLUP_BATTING["avg"]
        and row.get("obp", 0) >= CALLUP_BATTING["obp"]
        and row.get("slg", 0) >= CALLUP_BATTING["slg"]
    )


def flag_callup_pitcher(row: pd.Series) -> bool:
    """Check if a pitcher is performing above their current level.

    A pitcher is a call-up candidate when at Double-A or above with
    ERA <= 3.50 and WHIP <= 1.25.

    Args:
        row: A single-row Series from the aggregated pitching DataFrame.

    Returns:
        True if the player meets call-up thresholds.
    """
    level = LEVEL_RANK.get(row.get("league_level", ""), 0)
    if level < CALLUP_PITCHING["min_level"]:
        return False
    return (
        row.get("era", 99) <= CALLUP_PITCHING["era"]
        and row.get("whip", 99) <= CALLUP_PITCHING["whip"]
    )


# ── assembly ─────────────────────────────────────────────────────────


def _lookup_ownership(player_name: str, all_rosters: pd.DataFrame | None) -> str:
    """Return fantasy ownership status for a prospect.

    Returns MY_TEAM name, other owner's team name, or 'FA'.
    """
    if all_rosters is None:
        return "FA"
    match = all_rosters.loc[
        all_rosters["player_name"].str.strip().str.lower() == player_name.strip().lower()
    ]
    if match.empty:
        return "FA"
    owner = match.iloc[0]["team_name"]
    return owner if owner else "FA"


def _calc_heat_score(row: dict) -> float:
    """Calculate a heat score for a prospect based on available stats.

    Hitters: HOT if AVG > .300 and OBP > .380
    Pitchers: HOT if ERA < 3.00 and K/9 > 9.0
    Returns a numeric score (0-100) for sorting.
    """
    ptype = row.get("player_type", "unknown")
    if ptype == "hitter" or ptype == "two-way":
        avg = row.get("avg")
        obp = row.get("obp")
        if avg is not None and obp is not None and avg > 0:
            score = 0.0
            score += min(avg / 0.300, 1.5) * 40  # AVG contribution
            score += min(obp / 0.380, 1.5) * 30  # OBP contribution
            slg = row.get("slg")
            if slg is not None and slg > 0:
                score += min(slg / 0.500, 1.5) * 30
            return round(score, 1)
    if ptype == "pitcher" or ptype == "two-way":
        era = row.get("era")
        k9 = row.get("k_per_9")
        if era is not None and k9 is not None:
            era_score = max(0, (5.0 - era) / 5.0) * 50
            k_score = min(k9 / 12.0, 1.5) * 50
            return round(era_score + k_score, 1)
    return 0.0


def _is_hot(row: dict) -> bool:
    """Check if a prospect qualifies as HOT based on stat thresholds."""
    ptype = row.get("player_type", "unknown")
    if ptype in ("hitter", "two-way"):
        avg = row.get("avg")
        obp = row.get("obp")
        if avg is not None and obp is not None:
            if avg > 0.300 and obp > 0.380:
                return True
    if ptype in ("pitcher", "two-way"):
        era = row.get("era")
        k9 = row.get("k_per_9")
        if era is not None and k9 is not None:
            if era < 3.00 and k9 > 9.0:
                return True
    return False


def build_prospect_table(
    watchlist: list[dict],
    batting: pd.DataFrame | None,
    pitching: pd.DataFrame | None,
    roster_cache: dict[str, set[int]],
    all_rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a unified prospect alert table from all available data sources.

    Merges watchlist metadata with aggregated MiLB stats, 40-man
    roster status, fantasy ownership, and heat scores into a single
    DataFrame ready for display and export.

    Args:
        watchlist: List of prospect dicts from the YAML watchlist.
        batting: Aggregated batting stats, or None if unavailable.
        pitching: Aggregated pitching stats, or None if unavailable.
        roster_cache: Dict mapping team abbreviation to set of 40-man
            roster player IDs.
        all_rosters: Fantrax all_rosters DataFrame for ownership lookup.

    Returns:
        DataFrame with one row per tracked prospect and all available
        stats, flags, and metadata.
    """
    rows = []
    for prospect in watchlist:
        pid = prospect["player_id"]
        team = prospect.get("team", "")
        row = {
            "name": prospect["name"],
            "team": team,
            "position": prospect.get("position", ""),
            "level": prospect.get("level", ""),
            "age": prospect.get("age", ""),
            "notes": prospect.get("notes", ""),
            "on_40_man": pid in roster_cache.get(team, set()),
            "callup_candidate": False,
            "player_type": "unknown",
            "has_stats": False,
        }

        # Merge batting stats
        if batting is not None and pid in batting["player_id"].values:
            b = batting.loc[batting["player_id"] == pid].iloc[0]
            row["level"] = b.get("league_level", row["level"])
            row["games"] = int(b.get("games", 0))
            row["avg"] = b.get("avg", None)
            row["obp"] = b.get("obp", None)
            row["slg"] = b.get("slg", None)
            row["k_pct"] = b.get("k_pct", None)
            row["bb_pct"] = b.get("bb_pct", None)
            row["hr"] = int(b.get("hr", 0))
            row["player_type"] = "hitter"
            row["callup_candidate"] = flag_callup_hitter(b)
            row["has_stats"] = True

        # Merge pitching stats
        if pitching is not None and pid in pitching["player_id"].values:
            p = pitching.loc[pitching["player_id"] == pid].iloc[0]
            row["level"] = p.get("league_level", row["level"])
            row["games"] = int(p.get("games", 0))
            row["era"] = p.get("era", None)
            row["whip"] = p.get("whip", None)
            row["k_per_9"] = p.get("k_per_9", None)
            row["bb_per_9"] = p.get("bb_per_9", None)
            row["ip"] = p.get("ip", None)
            if row["player_type"] == "hitter":
                row["player_type"] = "two-way"
            else:
                row["player_type"] = "pitcher"
                row["callup_candidate"] = flag_callup_pitcher(p)
            row["has_stats"] = True

        # Ownership from Fantrax
        row["ownership"] = _lookup_ownership(prospect["name"], all_rosters)

        # Heat score
        row["heat_score"] = _calc_heat_score(row)
        row["is_hot"] = _is_hot(row)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by heat_score descending so hottest prospects are at top
    if "heat_score" in df.columns:
        df = df.sort_values("heat_score", ascending=False).reset_index(drop=True)
    return df


# ── display ──────────────────────────────────────────────────────────


def print_prospect_table(df: pd.DataFrame) -> None:
    """Print a formatted prospect alerts table to the console.

    Args:
        df: Prospect alerts DataFrame from build_prospect_table.
    """
    if df.empty:
        print("  No prospects to display.\n")
        return

    display_cols = ["name", "team", "position", "level", "on_40_man", "callup_candidate"]
    col_names = {
        "name": "Name",
        "team": "Team",
        "position": "Pos",
        "level": "Level",
        "on_40_man": "40-Man",
        "callup_candidate": "Call-Up?",
    }

    # Add stat columns that exist
    batting_cols = {"avg": "AVG", "obp": "OBP", "slg": "SLG", "k_pct": "K%", "bb_pct": "BB%"}
    pitching_cols = {"era": "ERA", "whip": "WHIP", "k_per_9": "K/9", "bb_per_9": "BB/9"}

    for col, label in {**batting_cols, **pitching_cols}.items():
        if col in df.columns and df[col].notna().any():
            display_cols.append(col)
            col_names[col] = label

    available = [c for c in display_cols if c in df.columns]
    display = df[available].rename(columns=col_names)
    print(display.to_string(index=False))


# ── entry point ──────────────────────────────────────────────────────


def main() -> None:
    """Load prospect data, flag call-up candidates, and save alerts."""
    today = date.today().isoformat()

    print("Loading prospect watchlist...")
    watchlist = load_watchlist()
    print(f"  {len(watchlist)} prospects tracked\n")

    if not watchlist:
        print("  No prospects in watchlist. Add entries to config/prospect_watchlist.yaml")
        return

    for p in watchlist:
        print(f"  - {p['name']} ({p['team']}, {p.get('level', 'N/A')})")
    print()

    # Load MiLB game-log data
    print("Loading MiLB batting data...")
    batting_raw = load_milb_batting(today)
    if batting_raw is not None:
        batting = aggregate_batting(batting_raw)
        print(f"  {len(batting)} hitters aggregated")
    else:
        batting = None
        print("  No MiLB batting data available (season may not have started)")

    print("Loading MiLB pitching data...")
    pitching_raw = load_milb_pitching(today)
    if pitching_raw is not None:
        pitching = aggregate_pitching(pitching_raw)
        print(f"  {len(pitching)} pitchers aggregated")
    else:
        pitching = None
        print("  No MiLB pitching data available (season may not have started)")
    print()

    # Load 40-man roster data for each unique team
    teams = {p.get("team", "") for p in watchlist if p.get("team")}
    print(f"Checking 40-man roster status for {len(teams)} team(s)...")
    roster_cache: dict[str, set[int]] = {}
    for team in sorted(teams):
        roster_cache[team] = load_40_man_roster(team)
        roster_size = len(roster_cache[team])
        if roster_size > 0:
            print(f"  {team}: {roster_size} players on 40-man")
        else:
            print(f"  {team}: roster unavailable")
    print()

    # Load Fantrax all_rosters for ownership lookup
    print("Loading Fantrax rosters for ownership lookup...")
    all_rosters = load_all_rosters()
    if all_rosters is not None:
        print(f"  {len(all_rosters)} roster entries loaded")
    else:
        print("  No Fantrax roster data available")
    print()

    # Build unified prospect table
    print("Building prospect alerts...")
    alerts = build_prospect_table(watchlist, batting, pitching, roster_cache, all_rosters)

    callup_count = alerts["callup_candidate"].sum()
    print(f"  {len(alerts)} prospects evaluated, {callup_count} call-up candidate(s)\n")

    print_prospect_table(alerts)
    print()

    # Save output
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "prospect_alerts.csv"
    alerts.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
