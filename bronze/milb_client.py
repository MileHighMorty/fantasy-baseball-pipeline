"""MiLB client for minor league stats, rosters, and prospect data.

Pulls minor league batting and pitching game logs from the MLB Stats API
(statsapi.mlb.com) for tracked prospects, checks 40-man roster status,
and persists results as date-stamped CSVs in bronze/data/milb/.
"""

import pathlib
import time
from datetime import date

import pandas as pd
import requests
import yaml

BASE_URL = "https://statsapi.mlb.com/api/v1"

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "milb"

CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"

CURRENT_SEASON = date.today().year

REQUEST_TIMEOUT = 30

SPORT_IDS = {
    "Triple-A": 11,
    "Double-A": 12,
    "High-A": 13,
    "Single-A": 14,
}


def _get_json(endpoint: str, params: dict | None = None) -> dict:
    """Fetch JSON from the MLB Stats API.

    Args:
        endpoint: API path relative to the base URL
            (e.g. '/people/695370/stats').
        params: Optional query-string parameters.

    Returns:
        Parsed JSON response as a dictionary.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
    """
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _save_df(df: pd.DataFrame, filename: str) -> pathlib.Path:
    """Write a DataFrame to a CSV file inside DATA_DIR.

    Args:
        df: The DataFrame to save.
        filename: Target filename (e.g. '2026-04-03_milb_batting.csv').

    Returns:
        The full path to the written file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_prospect_watchlist() -> list[dict]:
    """Load tracked prospects from config/prospect_watchlist.yaml.

    Returns:
        List of prospect dictionaries, each containing at minimum
        'name', 'player_id', 'team', and 'position' keys.

    Raises:
        FileNotFoundError: If the watchlist file does not exist.
    """
    watchlist_path = CONFIG_DIR / "prospect_watchlist.yaml"
    with open(watchlist_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("prospects", [])


def get_milb_batting_stats(
    player_ids: list[int], season: int = CURRENT_SEASON
) -> pd.DataFrame:
    """Fetch minor league batting game logs for a list of prospect IDs.

    Queries the MLB Stats API game-log endpoint for each player and
    collects hitting stats across all minor league levels.

    Args:
        player_ids: List of MLB player IDs to query.
        season: Season year to pull stats for. Defaults to current year.

    Returns:
        DataFrame with batting stats including: player_id, player_name,
        team, league_level, date, ab, r, h, doubles, triples, hr, rbi,
        bb, so, sb, cs, avg, obp, slg, ops.
    """
    rows = []
    for player_id in player_ids:
        try:
            data = _get_json(f"/people/{player_id}/stats", params={
                "stats": "gameLog",
                "group": "hitting",
                "gameType": "R",
                "season": season,
            })
            person = _get_json(f"/people/{player_id}")
            player_name = (
                person.get("people", [{}])[0].get("fullName", str(player_id))
            )

            for split_group in data.get("stats", []):
                for split in split_group.get("splits", []):
                    stat = split.get("stat", {})
                    team_info = split.get("team", {})
                    sport = split.get("sport", {})
                    sport_id = sport.get("id")

                    level = next(
                        (k for k, v in SPORT_IDS.items() if v == sport_id),
                        sport.get("name", "Unknown"),
                    )

                    rows.append({
                        "player_id": player_id,
                        "player_name": player_name,
                        "team": team_info.get("name"),
                        "league_level": level,
                        "date": split.get("date"),
                        "ab": stat.get("atBats"),
                        "r": stat.get("runs"),
                        "h": stat.get("hits"),
                        "doubles": stat.get("doubles"),
                        "triples": stat.get("triples"),
                        "hr": stat.get("homeRuns"),
                        "rbi": stat.get("rbi"),
                        "bb": stat.get("baseOnBalls"),
                        "so": stat.get("strikeOuts"),
                        "sb": stat.get("stolenBases"),
                        "cs": stat.get("caughtStealing"),
                        "avg": stat.get("avg"),
                        "obp": stat.get("obp"),
                        "slg": stat.get("slg"),
                        "ops": stat.get("ops"),
                    })
        except requests.HTTPError as exc:
            print(f"  Warning: could not fetch batting stats for "
                  f"player {player_id}: {exc}")

        time.sleep(0.5)

    return pd.DataFrame(rows)


def get_milb_pitching_stats(
    player_ids: list[int], season: int = CURRENT_SEASON
) -> pd.DataFrame:
    """Fetch minor league pitching game logs for a list of prospect IDs.

    Queries the MLB Stats API game-log endpoint for each player and
    collects pitching stats across all minor league levels.

    Args:
        player_ids: List of MLB player IDs to query.
        season: Season year to pull stats for. Defaults to current year.

    Returns:
        DataFrame with pitching stats including: player_id, player_name,
        team, league_level, date, w, l, era, g, gs, sv, ip, h, r, er,
        bb, so, hr, whip, batting_avg_against.
    """
    rows = []
    for player_id in player_ids:
        try:
            data = _get_json(f"/people/{player_id}/stats", params={
                "stats": "gameLog",
                "group": "pitching",
                "gameType": "R",
                "season": season,
            })
            person = _get_json(f"/people/{player_id}")
            player_name = (
                person.get("people", [{}])[0].get("fullName", str(player_id))
            )

            for split_group in data.get("stats", []):
                for split in split_group.get("splits", []):
                    stat = split.get("stat", {})
                    team_info = split.get("team", {})
                    sport = split.get("sport", {})
                    sport_id = sport.get("id")

                    level = next(
                        (k for k, v in SPORT_IDS.items() if v == sport_id),
                        sport.get("name", "Unknown"),
                    )

                    rows.append({
                        "player_id": player_id,
                        "player_name": player_name,
                        "team": team_info.get("name"),
                        "league_level": level,
                        "date": split.get("date"),
                        "w": stat.get("wins"),
                        "l": stat.get("losses"),
                        "era": stat.get("era"),
                        "g": stat.get("gamesPlayed"),
                        "gs": stat.get("gamesStarted"),
                        "sv": stat.get("saves"),
                        "ip": stat.get("inningsPitched"),
                        "h": stat.get("hits"),
                        "r": stat.get("runs"),
                        "er": stat.get("earnedRuns"),
                        "bb": stat.get("baseOnBalls"),
                        "so": stat.get("strikeOuts"),
                        "hr": stat.get("homeRuns"),
                        "whip": stat.get("whip"),
                        "batting_avg_against": stat.get("avg"),
                    })
        except requests.HTTPError as exc:
            print(f"  Warning: could not fetch pitching stats for "
                  f"player {player_id}: {exc}")

        time.sleep(0.5)

    return pd.DataFrame(rows)


def get_40_man_roster(team_id: int) -> pd.DataFrame:
    """Fetch the 40-man roster for an MLB team to check call-up eligibility.

    Args:
        team_id: The MLB team ID (e.g. 140 for the Rangers).

    Returns:
        DataFrame with columns: player_id, full_name, jersey_number,
        position, status, bat_side, pitch_hand.
    """
    data = _get_json(f"/teams/{team_id}/roster", params={
        "rosterType": "40Man",
        "hydrate": "person",
    })

    rows = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        rows.append({
            "player_id": person.get("id"),
            "full_name": person.get("fullName"),
            "jersey_number": entry.get("jerseyNumber"),
            "position": entry.get("position", {}).get("abbreviation"),
            "status": entry.get("status", {}).get("description"),
            "bat_side": person.get("batSide", {}).get("code"),
            "pitch_hand": person.get("pitchHand", {}).get("code"),
        })

    return pd.DataFrame(rows)


def save_all(today: date | None = None,
             season: int = CURRENT_SEASON) -> dict[str, pathlib.Path]:
    """Fetch and save all MiLB prospect data for today.

    Loads the prospect watchlist, pulls batting and pitching game logs
    for each tracked prospect, and saves the results as dated CSVs.

    Args:
        today: Date used for the filename stamp. Defaults to today.
        season: Season year to pull stats for. Defaults to current year.

    Returns:
        Dictionary mapping data type names to their saved file paths.
    """
    today = today or date.today()
    date_str = today.isoformat()

    prospects = load_prospect_watchlist()
    player_ids = [p["player_id"] for p in prospects]

    paths = {}

    batting_df = get_milb_batting_stats(player_ids, season=season)
    if not batting_df.empty:
        paths["batting"] = _save_df(batting_df, f"{date_str}_batting.csv")

    pitching_df = get_milb_pitching_stats(player_ids, season=season)
    if not pitching_df.empty:
        paths["pitching"] = _save_df(pitching_df, f"{date_str}_pitching.csv")

    return paths


def main() -> None:
    """Pull MiLB stats for all watched prospects and print output paths."""
    print("Fetching MiLB prospect stats...")
    try:
        prospects = load_prospect_watchlist()
        print(f"  Loaded {len(prospects)} prospects from watchlist")
        for p in prospects:
            print(f"    - {p['name']} ({p['team']}, {p.get('level', 'N/A')})")

        paths = save_all()
        for name, path in paths.items():
            print(f"  {name}: {path}")

        if not paths:
            print("  No game log data found for tracked prospects.")

    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}")
    except requests.ConnectionError as exc:
        print(f"Connection error: {exc}")


if __name__ == "__main__":
    main()
