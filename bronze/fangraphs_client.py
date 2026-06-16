"""FanGraphs client for leaderboards and advanced stats.

Pulls batting and pitching leaderboard data from the modern FanGraphs
leaders data API and persists them as date-stamped CSVs in
bronze/data/fangraphs/.

FanGraphs sits behind a Cloudflare managed challenge that blocks plain
HTTP clients -- including pybaseball's legacy leaders-legacy.aspx scrape,
which now returns 403. We use curl_cffi's Chrome TLS impersonation, which
passes the challenge where a spoofed User-Agent alone does not: a warm-up
GET on the homepage collects the Cloudflare clearance cookie, then the
JSON data API responds. If FanGraphs ever tightens the challenge the
fetch fails loudly (raises) rather than writing a partial leaderboard.

Output contract: the saved CSVs use the historical pybaseball column
names that downstream consumers already read (IDfg, Name, Team, Age,
Spd, K%, K/9, BB%, wRC+, ...). The clean API columns are used for the
identity fields (PlayerName/playerid/TeamNameAbb), never the HTML-anchor
``Name``/``Team`` blobs the API returns alongside them.
"""

import pathlib
from datetime import date

import pandas as pd
from curl_cffi import requests as cffi_requests

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data" / "fangraphs"

CURRENT_SEASON = date.today().year

_ROOT_URL = "https://www.fangraphs.com/"
_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
# curl_cffi browser profile whose TLS/JA3 fingerprint clears Cloudflare.
_IMPERSONATE = "chrome"

# API field -> output CSV column. Output names match the historical
# pybaseball contract that downstream (player_id_map, player_universe,
# statcast_enriched, sp_streamer, dashboard) already reads. K%/BB% arrive
# as decimal fractions (0.174 == 17.4%), matching the historical CSVs, so
# no rescaling is applied.
_BATTING_COLUMN_MAP = {
    "playerid": "IDfg",
    "Season": "Season",
    "PlayerName": "Name",
    "TeamNameAbb": "Team",
    "Age": "Age",
    "G": "G",
    "PA": "PA",
    "AB": "AB",
    "H": "H",
    "HR": "HR",
    "R": "R",
    "RBI": "RBI",
    "SB": "SB",
    "BB": "BB",
    "SO": "SO",
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "wOBA": "wOBA",
    "BB%": "BB%",
    "K%": "K%",
    "wRC+": "wRC+",
    "Spd": "Spd",
}

_PITCHING_COLUMN_MAP = {
    "playerid": "IDfg",
    "Season": "Season",
    "PlayerName": "Name",
    "TeamNameAbb": "Team",
    "Age": "Age",
    "W": "W",
    "L": "L",
    "SV": "SV",
    "G": "G",
    "GS": "GS",
    "IP": "IP",
    "SO": "SO",
    "ERA": "ERA",
    "WHIP": "WHIP",
    "K/9": "K/9",
    "BB/9": "BB/9",
    "K%": "K%",
    "BB%": "BB%",
}

# Output columns downstream reads via strict ``usecols=`` (or relies on by
# name). If the API stops returning the source field, a consumer would
# crash -- so a missing one is fatal here and we refuse to write.
_BATTING_REQUIRED = ("IDfg", "Name", "Team", "Age", "Spd", "K%", "wRC+")
_PITCHING_REQUIRED = ("IDfg", "Name", "Team", "Age", "K%", "K/9", "BB%")

# Counting/index stats stored as integers in the historical contract; the
# API returns them as floats (9.0, 121.7). Round back to nullable ints for
# fidelity with the old pybaseball output.
_BATTING_INT_COLS = (
    "Age", "G", "PA", "AB", "H", "HR", "R", "RBI", "SB", "BB", "SO", "wRC+",
)
_PITCHING_INT_COLS = ("Age", "W", "L", "SV", "G", "GS", "SO")


def _fetch_leaderboard(stats: str, season: int) -> list[dict]:
    """GET the modern FanGraphs leaders data API for one stat type.

    Opens a curl_cffi session with Chrome TLS impersonation, warms up on
    the FanGraphs homepage to clear the Cloudflare challenge, then hits
    the qualified leaderboard data endpoint.

    Args:
        stats: ``"bat"`` or ``"pit"`` -- the FanGraphs stat category.
        season: MLB season year.

    Returns:
        The raw ``data`` array of per-player dicts from the API.

    Raises:
        RuntimeError: On any non-200 response or an empty payload, so a
            tightened Cloudflare challenge fails the run loudly instead of
            writing garbage.
    """
    session = cffi_requests.Session(impersonate=_IMPERSONATE)

    # Warm-up: the homepage hands back the Cloudflare clearance cookie that
    # the API request below needs.
    warmup = session.get(_ROOT_URL, timeout=45)
    if warmup.status_code != 200:
        raise RuntimeError(
            f"FanGraphs warm-up GET failed (status {warmup.status_code}); "
            "Cloudflare challenge was not cleared."
        )

    params = {
        "pos": "all",
        "stats": stats,
        "lg": "all",
        "qual": "0",
        "season": str(season),
        "season1": str(season),
        "ind": "0",
        "type": "8",
        "month": "0",
        "pageitems": "2000000000",
        "pagenum": "1",
    }
    resp = session.get(_API_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"FanGraphs {stats} API returned status {resp.status_code} "
            "(expected 200); Cloudflare may have tightened the challenge."
        )

    payload = resp.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not rows:
        raise RuntimeError(
            f"FanGraphs {stats} API returned no data rows; refusing to "
            "write an empty leaderboard."
        )
    return rows


def _shape_leaderboard(
    rows: list[dict],
    column_map: dict[str, str],
    required: tuple[str, ...],
    int_cols: tuple[str, ...],
    stats: str,
) -> pd.DataFrame:
    """Project the raw API rows onto the historical output contract.

    Selects the mapped clean columns, renames them to the pybaseball-era
    names, casts the FanGraphs id and integer counting stats, and verifies
    every required column survived.

    Args:
        rows: Raw ``data`` dicts from :func:`_fetch_leaderboard`.
        column_map: API field -> output column name.
        required: Output columns that must be present (fatal if absent).
        int_cols: Output columns to round to nullable ints.
        stats: ``"bat"`` / ``"pit"`` label for error messages.

    Returns:
        DataFrame matching the historical CSV column contract.

    Raises:
        RuntimeError: If a required source field is missing from the API
            response (schema drift).
    """
    df = pd.DataFrame(rows)

    # A missing source field is only fatal when it backs a required output.
    missing_required = [
        api for api, out in column_map.items()
        if out in required and api not in df.columns
    ]
    if missing_required:
        raise RuntimeError(
            f"FanGraphs {stats} response missing required fields "
            f"{missing_required}; schema drift -- refusing to write."
        )

    present = {api: out for api, out in column_map.items() if api in df.columns}
    out = df[list(present)].rename(columns=present)

    # IDfg is the join key into the id map; keep it a plain int.
    out["IDfg"] = out["IDfg"].astype(int)
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].round().astype("Int64")

    return out


def get_batting_leaderboard(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull the FanGraphs batting leaderboard for all qualified batters.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        DataFrame of qualified batter stats shaped to the historical
        output contract.
    """
    rows = _fetch_leaderboard("bat", season)
    return _shape_leaderboard(
        rows, _BATTING_COLUMN_MAP, _BATTING_REQUIRED, _BATTING_INT_COLS, "bat"
    )


def get_pitching_leaderboard(season: int = CURRENT_SEASON) -> pd.DataFrame:
    """Pull the FanGraphs pitching leaderboard for all qualified pitchers.

    Args:
        season: MLB season year. Defaults to the current year.

    Returns:
        DataFrame of qualified pitcher stats shaped to the historical
        output contract.
    """
    rows = _fetch_leaderboard("pit", season)
    return _shape_leaderboard(
        rows, _PITCHING_COLUMN_MAP, _PITCHING_REQUIRED, _PITCHING_INT_COLS, "pit"
    )


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
