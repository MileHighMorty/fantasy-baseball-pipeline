"""MLB Stats API client for schedules, rosters, transactions, and standings.

Pulls structured game, roster, transaction, and standings data from the
free MLB Stats API (statsapi.mlb.com) and persists them as date-stamped
CSVs in bronze/data/mlb/.
"""

import pathlib
from datetime import date, timedelta

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "mlb"

REQUEST_TIMEOUT = 30


def _get_json(endpoint: str, params: dict | None = None) -> dict:
    """Fetch JSON from the MLB Stats API.

    Args:
        endpoint: API path relative to the base URL (e.g. '/schedule').
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
        filename: Target filename (e.g. '2026-04-03_games.csv').

    Returns:
        The full path to the written file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def get_todays_games(game_date: date | None = None) -> pd.DataFrame:
    """Get today's MLB games with probable pitchers and start times.

    Args:
        game_date: Date to query. Defaults to today.

    Returns:
        DataFrame with columns: game_id, away_team, home_team,
        away_pitcher, home_pitcher, game_time, status, venue.
    """
    game_date = game_date or date.today()
    data = _get_json("/schedule", params={
        "date": game_date.isoformat(),
        "sportId": 1,
        "hydrate": "probablePitcher,team,venue",
    })

    rows = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            rows.append({
                "game_id": game.get("gamePk"),
                "away_team": away.get("team", {}).get("name"),
                "home_team": home.get("team", {}).get("name"),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName"),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName"),
                "game_time": game.get("gameDate"),
                "status": game.get("status", {}).get("detailedState"),
                "venue": game.get("venue", {}).get("name"),
            })

    return pd.DataFrame(rows)


def get_transactions(days: int = 7) -> pd.DataFrame:
    """Get recent MLB transactions (DFA, option, call-up, trade, etc.).

    Args:
        days: Number of days to look back. Defaults to 7.

    Returns:
        DataFrame with columns: transaction_id, date, type,
        description, player, from_team, to_team.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    data = _get_json("/transactions", params={
        "startDate": start_date.strftime("%m/%d/%Y"),
        "endDate": end_date.strftime("%m/%d/%Y"),
    })

    rows = []
    for txn in data.get("transactions", []):
        rows.append({
            "transaction_id": txn.get("id"),
            "date": txn.get("date"),
            "type": txn.get("typeDesc"),
            "description": txn.get("description"),
            "player": txn.get("person", {}).get("fullName"),
            "from_team": txn.get("fromTeam", {}).get("name"),
            "to_team": txn.get("toTeam", {}).get("name"),
        })

    return pd.DataFrame(rows)


def get_team_roster(team_id: int) -> pd.DataFrame:
    """Get the active roster for a specific MLB team.

    Args:
        team_id: The MLB team ID (e.g. 147 for the Yankees).

    Returns:
        DataFrame with columns: player_id, full_name, jersey_number,
        position, status, bat_side, pitch_hand.
    """
    data = _get_json(f"/teams/{team_id}/roster", params={
        "rosterType": "active",
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


def get_standings() -> pd.DataFrame:
    """Get current MLB standings for both leagues.

    Returns:
        DataFrame with columns: team, league, division, wins, losses,
        win_pct, games_back, division_rank, wildcard_rank.
    """
    data = _get_json("/standings", params={
        "leagueId": "103,104",
        "hydrate": "team,division",
    })

    rows = []
    for record in data.get("records", []):
        division = record.get("division", {}).get("name")
        league = record.get("league", {}).get("name")
        for team_record in record.get("teamRecords", []):
            rows.append({
                "team": team_record.get("team", {}).get("name"),
                "league": league,
                "division": division,
                "wins": team_record.get("wins"),
                "losses": team_record.get("losses"),
                "win_pct": team_record.get("winningPercentage"),
                "games_back": team_record.get("gamesBack"),
                "division_rank": team_record.get("divisionRank"),
                "wildcard_rank": team_record.get("wildCardRank"),
            })

    return pd.DataFrame(rows)


def save_all(today: date | None = None) -> dict[str, pathlib.Path]:
    """Fetch and save all MLB Stats data for today.

    Args:
        today: Date used for the filename stamp. Defaults to today.

    Returns:
        Dictionary mapping data type names to their saved file paths.
    """
    today = today or date.today()
    date_str = today.isoformat()

    paths = {}

    games_df = get_todays_games(game_date=today)
    paths["games"] = _save_df(games_df, f"{date_str}_games.csv")

    transactions_df = get_transactions()
    paths["transactions"] = _save_df(
        transactions_df, f"{date_str}_transactions.csv"
    )

    standings_df = get_standings()
    paths["standings"] = _save_df(standings_df, f"{date_str}_standings.csv")

    return paths


def main() -> None:
    """Pull all MLB Stats data and print the output paths."""
    print("Fetching MLB Stats API data...")
    try:
        paths = save_all()
        for name, path in paths.items():
            print(f"  {name}: {path}")
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}")
    except requests.ConnectionError as exc:
        print(f"Connection error: {exc}")


if __name__ == "__main__":
    main()
