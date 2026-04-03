"""Baseball Savant client for Statcast pitch-level and player-level data.

Pulls leaderboard CSVs and individual player Statcast data from
baseballsavant.mlb.com and persists them as date-stamped files in
bronze/data/savant/.
"""

import pathlib
from datetime import date

import requests

BASE_URL = "https://baseballsavant.mlb.com"
EXPECTED_STATS_URL = f"{BASE_URL}/leaderboard/expected_statistics"
STATCAST_URL = f"{BASE_URL}/leaderboard/statcast"
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


def fetch_batting_leaderboard(season: int = CURRENT_SEASON) -> str:
    """Pull the Statcast expected-stats batting leaderboard CSV.

    Uses the ``expected_statistics`` endpoint which returns xBA, xSLG,
    xwOBA, and their actual-vs-expected differentials for qualified
    batters.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        Raw CSV text of the batting expected-stats leaderboard.
    """
    params = {
        "type": "batter",
        "year": season,
        "position": "",
        "team": "",
        "min": "q",
        "csv": "true",
    }
    return _get_csv(EXPECTED_STATS_URL, params=params)


def fetch_pitching_leaderboard(season: int = CURRENT_SEASON) -> str:
    """Pull the Statcast expected-stats pitching leaderboard CSV.

    Uses the ``expected_statistics`` endpoint which returns xBA-against,
    xSLG-against, xwOBA-against, xERA, and their differentials for
    qualified pitchers.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        Raw CSV text of the pitching expected-stats leaderboard.
    """
    params = {
        "type": "pitcher",
        "year": season,
        "position": "",
        "team": "",
        "min": "q",
        "csv": "true",
    }
    return _get_csv(EXPECTED_STATS_URL, params=params)


def fetch_statcast_leaderboard(
    player_type: str = "batter", season: int = CURRENT_SEASON
) -> str:
    """Pull the Statcast batted-ball leaderboard CSV.

    Returns exit velocity, launch angle, barrel rate, and hard-hit
    metrics for qualified batters or pitchers.

    Args:
        player_type: ``'batter'`` or ``'pitcher'``.
        season: MLB season year. Defaults to the current year.

    Returns:
        Raw CSV text of the Statcast batted-ball leaderboard.

    Raises:
        ValueError: If *player_type* is not ``'batter'`` or ``'pitcher'``.
    """
    if player_type not in ("batter", "pitcher"):
        raise ValueError(
            f"player_type must be 'batter' or 'pitcher', got {player_type!r}"
        )
    params = {
        "type": player_type,
        "year": season,
        "position": "",
        "team": "",
        "min": "q",
        "csv": "true",
    }
    return _get_csv(STATCAST_URL, params=params)


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
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    """Fetch and save expected-stats and batted-ball leaderboards.

    This is the primary entry point for the daily bronze-layer Savant
    ingestion. It pulls the current season's expected-stats and Statcast
    batted-ball leaderboards for both batters and pitchers, then writes
    them to date-stamped CSVs so downstream silver/gold layers always
    have a historical trail.

    Args:
        today: Date used for the filename stamp. Defaults to today.
        season: MLB season year. Defaults to the current year.

    Returns:
        A tuple of ``(batting_path, pitching_path,
        batting_statcast_path, pitching_statcast_path)``.
    """
    today = today or date.today()
    date_str = today.isoformat()

    batting_csv = fetch_batting_leaderboard(season=season)
    batting_path = _save_csv(batting_csv, f"{date_str}_batting.csv")

    pitching_csv = fetch_pitching_leaderboard(season=season)
    pitching_path = _save_csv(pitching_csv, f"{date_str}_pitching.csv")

    batting_sc = fetch_statcast_leaderboard("batter", season=season)
    batting_sc_path = _save_csv(batting_sc, f"{date_str}_batting_statcast.csv")

    pitching_sc = fetch_statcast_leaderboard("pitcher", season=season)
    pitching_sc_path = _save_csv(pitching_sc, f"{date_str}_pitching_statcast.csv")

    return batting_path, pitching_path, batting_sc_path, pitching_sc_path


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
        bat, pit, bat_sc, pit_sc = save_leaderboards()
        print(f"Batting expected:    {bat}")
        print(f"Pitching expected:   {pit}")
        print(f"Batting statcast:    {bat_sc}")
        print(f"Pitching statcast:   {pit_sc}")
    except requests.HTTPError as exc:
        print(f"HTTP error fetching leaderboards: {exc}")
    except requests.ConnectionError as exc:
        print(f"Connection error: {exc}")


if __name__ == "__main__":
    main()
