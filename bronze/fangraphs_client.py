"""FanGraphs client for leaderboards and advanced stats.

Pulls batting and pitching leaderboard data via the pybaseball library
and persists them as date-stamped CSVs in bronze/data/fangraphs/.
"""

import pathlib
from datetime import date

import pandas as pd
from pybaseball import batting_stats, pitching_stats

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "fangraphs"

CURRENT_SEASON = date.today().year


def get_batting_leaderboard(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull the FanGraphs batting leaderboard for all qualified batters.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        DataFrame of qualified batter stats from FanGraphs.
    """
    return batting_stats(season, qual="y")


def get_pitching_leaderboard(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull the FanGraphs pitching leaderboard for all qualified pitchers.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        DataFrame of qualified pitcher stats from FanGraphs.
    """
    return pitching_stats(season, qual="y")


def _save_df(df: pd.DataFrame, filename: str) -> pathlib.Path:
    """Write a DataFrame to a CSV file inside DATA_DIR.

    Args:
        df: The DataFrame to save.
        filename: Target filename (e.g. '2026-04-03_batting.csv').

    Returns:
        The full path to the written file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def save_leaderboards(
    today: date | None = None, season: int = CURRENT_SEASON
) -> tuple[pathlib.Path, pathlib.Path]:
    """Fetch and save both batting and pitching leaderboards.

    Args:
        today: Date used for the filename stamp. Defaults to today.
        season: MLB season year. Defaults to the current year.

    Returns:
        A tuple of (batting_path, pitching_path) for the saved files.
    """
    today = today or date.today()
    date_str = today.isoformat()

    batting_df = get_batting_leaderboard(season=season)
    batting_path = _save_df(batting_df, f"{date_str}_batting.csv")

    pitching_df = get_pitching_leaderboard(season=season)
    pitching_path = _save_df(pitching_df, f"{date_str}_pitching.csv")

    return batting_path, pitching_path


def main() -> None:
    """Pull today's FanGraphs leaderboards and print the output paths."""
    print("Fetching FanGraphs leaderboards...")
    batting_path, pitching_path = save_leaderboards()
    print(f"Batting:  {batting_path}")
    print(f"Pitching: {pitching_path}")


if __name__ == "__main__":
    main()
