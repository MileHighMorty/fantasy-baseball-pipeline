"""Baseball Savant client for Statcast pitch-level and player-level data.

Pulls leaderboard CSVs and individual player Statcast data from
baseballsavant.mlb.com and persists them as date-stamped files in
bronze/data/savant/.
"""

import pathlib
from datetime import date

import requests

BASE_URL = "https://baseballsavant.mlb.com"
LEADERBOARD_URL = f"{BASE_URL}/leaderboard/custom"
PLAYER_URL = f"{BASE_URL}/statcast_search/csv"

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "savant"

CURRENT_SEASON = date.today().year

# Baseball Savant blocks default python-requests UA.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

REQUEST_TIMEOUT = 60


def _get_csv(url: str, params: dict | None = None) -> str:
    """Fetch CSV text from Baseball Savant.

    Args:
        url: The endpoint URL.
        params: Optional query-string parameters.

    Returns:
        The response body as a string.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
    """
    response = requests.get(
        url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    return response.text


def _save_csv(text: str, filename: str) -> pathlib.Path:
    """Write CSV text to a file inside DATA_DIR.

    Args:
        text: Raw CSV content.
        filename: Target filename (e.g. '2026-04-03_batting.csv').

    Returns:
        The full path to the written file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    path.write_text(text, encoding="utf-8")
    return path


def fetch_batting_leaderboard(
    season: int = CURRENT_SEASON, month: int = 0
) -> str:
    """Pull the Statcast expected-stats batting leaderboard CSV.

    Baseball Savant's 'expected' stat group includes xBA, xSLG, xwOBA,
    and barrel metrics -- the core Statcast quality-of-contact numbers
    used for evaluating hitter talent independent of BABIP luck.

    Args:
        season: MLB season year. Defaults to the current year.
        month: Month filter (0 = full season). Defaults to 0.

    Returns:
        Raw CSV text of the batting leaderboard.
    """
    params = {
        "view": "Batter",
        "n": "qual",
        "filteredStatGroup": "expected",
        "season": season,
        "month": month,
        "csv": "true",
    }
    return _get_csv(LEADERBOARD_URL, params=params)


def fetch_pitching_leaderboard(
    season: int = CURRENT_SEASON, month: int = 0
) -> str:
    """Pull the Statcast expected-stats pitching leaderboard CSV.

    Mirrors the batting leaderboard but from the pitcher perspective --
    xBA-against, xSLG-against, xwOBA-against, and barrel-rate-against.
    Useful for identifying pitchers whose surface ERA is masking poor
    underlying contact quality.

    Args:
        season: MLB season year. Defaults to the current year.
        month: Month filter (0 = full season). Defaults to 0.

    Returns:
        Raw CSV text of the pitching leaderboard.
    """
    params = {
        "view": "Pitcher",
        "n": "qual",
        "filteredStatGroup": "expected",
        "season": season,
        "month": month,
        "csv": "true",
    }
    return _get_csv(LEADERBOARD_URL, params=params)


def fetch_player_statcast(
    player_id: int,
    season: int = CURRENT_SEASON,
    player_type: str = "batter",
) -> str:
    """Pull pitch-level Statcast data for an individual player.

    Returns every pitch the player saw (as a batter) or threw (as a
    pitcher) during the given season. Each row is a single pitch with
    launch speed, launch angle, spin rate, pitch velocity, etc. This
    is the most granular Statcast data available and powers downstream
    rolling-window and percentile calculations.

    Args:
        player_id: The MLB player ID (e.g. 660271 for Shohei Ohtani).
        season: MLB season year. Defaults to the current year.
        player_type: 'batter' or 'pitcher'. Defaults to 'batter'.

    Returns:
        Raw CSV text of pitch-level Statcast data for the player.

    Raises:
        ValueError: If player_type is not 'batter' or 'pitcher'.
    """
    if player_type not in ("batter", "pitcher"):
        raise ValueError(
            f"player_type must be 'batter' or 'pitcher', got {player_type!r}"
        )
    params = {
        "all": "true",
        "type": "detail",
        f"player_id_{player_type}": player_id,
        "season": season,
    }
    return _get_csv(PLAYER_URL, params=params)


def save_leaderboards(
    today: date | None = None, season: int = CURRENT_SEASON
) -> tuple[pathlib.Path, pathlib.Path]:
    """Fetch and save both batting and pitching leaderboards.

    This is the primary entry point for the daily bronze-layer Savant
    ingestion. It pulls the current season's expected-stats leaderboards
    for both batters and pitchers, then writes them to date-stamped CSVs
    so downstream silver/gold layers always have a historical trail.

    Args:
        today: Date used for the filename stamp. Defaults to today.
        season: MLB season year. Defaults to the current year.

    Returns:
        A tuple of (batting_path, pitching_path) for the saved files.
    """
    today = today or date.today()
    date_str = today.isoformat()

    batting_csv = fetch_batting_leaderboard(season=season)
    batting_path = _save_csv(batting_csv, f"{date_str}_batting.csv")

    pitching_csv = fetch_pitching_leaderboard(season=season)
    pitching_path = _save_csv(pitching_csv, f"{date_str}_pitching.csv")

    return batting_path, pitching_path


def save_player_statcast(
    player_id: int,
    season: int = CURRENT_SEASON,
    player_type: str = "batter",
    today: date | None = None,
) -> pathlib.Path:
    """Fetch and save pitch-level Statcast data for one player.

    Args:
        player_id: The MLB player ID.
        season: MLB season year. Defaults to the current year.
        player_type: 'batter' or 'pitcher'. Defaults to 'batter'.
        today: Date used for the filename stamp. Defaults to today.

    Returns:
        Path to the saved CSV file.
    """
    today = today or date.today()
    date_str = today.isoformat()
    csv_text = fetch_player_statcast(
        player_id, season=season, player_type=player_type
    )
    filename = f"{date_str}_{player_type}_{player_id}.csv"
    return _save_csv(csv_text, filename)


def main() -> None:
    """Pull today's Savant leaderboards and print the output paths."""
    print("Fetching Statcast leaderboards...")
    try:
        batting_path, pitching_path = save_leaderboards()
        print(f"Batting:  {batting_path}")
        print(f"Pitching: {pitching_path}")
    except requests.HTTPError as exc:
        print(f"HTTP error fetching leaderboards: {exc}")
    except requests.ConnectionError as exc:
        print(f"Connection error: {exc}")


if __name__ == "__main__":
    main()
