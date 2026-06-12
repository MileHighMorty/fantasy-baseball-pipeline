"""Pull live Fantrax rosters for every league team into the bronze layer.

Live counterpart to bronze/fantrax_csv_import.py: hits the Fantrax API
with a browser session cookie and writes the exact same output contract,
so downstream consumers cannot tell which path produced the files.

One contract difference to know about: the live API reports fantasy TEAM
NAMES (e.g. "Rutsch Hour") in team_name, while the manual export reports
owner labels (e.g. "Matt").  my_roster filtering therefore uses
``fantrax.my_team_name`` here and ``fantrax.my_team_label`` in the
importer — see the comment in config/settings.yaml.

Also pulls the full available-player pool (free agents) via the raw
fxpa getPlayerStats endpoint — fantraxapi has no free-agent support, so
that pull speaks to the endpoint directly with the same session cookie.

Inputs:
    FANTRAX_COOKIE in .env (browser session cookie)
    config/settings.yaml (fantrax.league_id, fantrax.my_team_name)

Outputs:
    bronze/data/fantrax/all_rosters_YYYY-MM-DD.csv
    bronze/data/fantrax/my_roster_YYYY-MM-DD.csv
    bronze/data/fantrax/free_agents_YYYY-MM-DD.csv

Usage:
    python -m bronze.fantrax_client                 # roster pull
    python -m bronze.fantrax_client --free-agents   # available-player pool

The free-agent pull runs in weekly_refresh (the pipeline is a scheduled
job, so its ~3 minutes of paginated calls — the server caps pagination
at 50/page — is acceptable); the --free-agents flag also allows an
ad-hoc standalone refresh.
"""

import argparse
import json
import os
import pathlib
import sys
import time
from datetime import date

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from fantraxapi import FantraxAPI, NotLoggedIn
from fantraxapi import api as fantrax_api

# ── paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent / "data" / "fantrax"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# ── output contract (must match bronze/fantrax_csv_import.py) ────────

OUTPUT_COLUMNS = [
    "team_name", "player_name", "position", "fantasy_points",
    "points_per_game", "fantrax_id", "mlb_team", "status", "age",
]

FA_OUTPUT_COLUMNS = [
    "player_name", "position", "fantrax_id", "mlb_team", "status", "minors_eligible",
]

EXPECTED_TEAM_COUNT = 12
ROSTER_MIN = 20
ROSTER_MAX = 30

FXPA_URL = "https://www.fantrax.com/fxpa/req"
# The server caps page size at 50 no matter what is requested (probed with
# 200); asking for more is harmless and future-proofs a raised cap.
FA_PAGE_SIZE = 200
FA_MAX_PAGES = 250          # safety valve; the ~9.5k pool is ~191 pages at 50/page
FA_PAGE_SLEEP_SECONDS = 0.2  # be polite between paginated calls

# Cloudflare rejects requests carrying the default python-requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)


# ── config ───────────────────────────────────────────────────────────


def load_fantrax_config() -> tuple[str, str]:
    """Read the live-client settings from config/settings.yaml.

    Returns:
        Tuple of (league_id, my_team_name) from the ``fantrax:`` block.

    Raises:
        KeyError: If either key is missing from settings.yaml.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    try:
        fantrax = settings["fantrax"]
        return fantrax["league_id"], fantrax["my_team_name"]
    except (KeyError, TypeError):
        raise KeyError(
            f"fantrax.league_id / fantrax.my_team_name not found in {CONFIG_PATH} — "
            "both are required for the live client."
        ) from None


# ── fetch ────────────────────────────────────────────────────────────


def build_session(cookie: str) -> requests.Session:
    """Build a requests Session authenticated with a browser cookie.

    Args:
        cookie: Full Cookie header value copied from a logged-in browser.

    Returns:
        Session with the Cookie and a browser User-Agent set.
    """
    session = requests.Session()
    session.headers.update({"Cookie": cookie, "User-Agent": USER_AGENT})
    return session


def fetch_all_rosters(api: FantraxAPI) -> pd.DataFrame:
    """Pull every team's roster from the live API.

    Empty roster slots are skipped — the contract is one row per owned
    player.  Columns the API does not provide (age) are empty strings.

    Args:
        api: Authenticated FantraxAPI (League) instance.

    Returns:
        DataFrame with :data:`OUTPUT_COLUMNS`, one row per owned player.
    """
    # fantraxapi 1.0.1's Roster object also parses the SCHEDULE_FULL view and
    # its Game parser crashes (IndexError) on schedule cells without a game
    # time.  We only need the STATS view, so we take the raw response from the
    # library's request layer and parse the roster rows ourselves.
    records = []
    for team in api.teams:
        stats = fantrax_api.get_team_roster_info(api, team.id)[0]
        for table in stats["tables"]:
            header = table["header"]["cells"]
            for row in table["rows"]:
                # Empty slots either have no scorer at all or a stub scorer
                # dict with no player identity (seen on open MiLB slots).
                scorer = row.get("scorer", {})
                if "scorerId" not in scorer:
                    continue
                points = {"SCORE": "", "FPTS_PER_GAME": ""}
                for head, cell in zip(header, row["cells"]):
                    key = head.get("sortKey")
                    if key in points and cell.get("content") not in (None, ""):
                        points[key] = cell["content"]
                records.append({
                    "team_name": team.name,
                    "player_name": scorer["name"],
                    "position": api.positions[row["posId"]].short_name,
                    "fantasy_points": points["SCORE"],
                    "points_per_game": points["FPTS_PER_GAME"],
                    "fantrax_id": scorer["scorerId"],
                    "mlb_team": scorer.get("teamShortName", scorer.get("teamName", "")),
                    "status": "owned",
                    "age": "",
                })
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


# ── free agents ──────────────────────────────────────────────────────


def _fxpa_player_stats_page(session: requests.Session, league_id: str, page_number: int) -> dict:
    """Call the fxpa getPlayerStats endpoint for one page of the player pool."""
    body = {
        "msgs": [{
            "method": "getPlayerStats",
            "data": {
                "reload": "1",
                "pageNumber": str(page_number),
                "maxResultsPerPage": str(FA_PAGE_SIZE),
            },
        }],
        "uiv": 3,
        "refUrl": f"https://www.fantrax.com/fantasy/league/{league_id}/players",
        "dt": 2, "at": 0, "av": None, "tz": "America/Denver", "v": "183.1.3",
    }
    r = session.post(f"{FXPA_URL}?leagueId={league_id}", data=json.dumps(body), timeout=60)
    r.raise_for_status()
    return r.json()["responses"][0]["data"]


def _name_from_url_slug(slug: str, fallback: str) -> str:
    """Reconstruct a full player name from the Fantrax URL slug.

    The players-list scorer object only carries abbreviated names
    ("K. Kelly"), which would wreck downstream fuzzy matching; the slug
    ("kevin-kelly") preserves the full name, minus case and accents —
    both of which the matcher's normalization discards anyway.
    """
    if not slug:
        return fallback
    return " ".join(part.capitalize() for part in slug.split("-"))


def fetch_free_agents(session: requests.Session, league_id: str) -> pd.DataFrame:
    """Pull the full available-player pool via paginated getPlayerStats.

    The endpoint's default view is ALL_AVAILABLE (free agents); the status
    cell is read anyway, so any row showing a fantasy team instead of "FA"
    is tagged "owned" rather than trusted blindly.

    Returns:
        DataFrame with :data:`FA_OUTPUT_COLUMNS`, one row per player.
    """
    records = []
    seen_ids: set[str] = set()
    page = 1
    total_pages = 1
    while page <= total_pages and page <= FA_MAX_PAGES:
        data = _fxpa_player_stats_page(session, league_id, page)
        paging = data.get("paginatedResultSet", {})
        total_pages = int(paging.get("totalNumPages", 1))
        if page == 1:
            print(
                f"  Player pool: {paging.get('totalNumResults', '?')} players across "
                f"{total_pages} pages ({paging.get('maxResultsPerPage', '?')}/page)"
            )
            if total_pages > FA_MAX_PAGES:
                print(
                    f"  WARNING: {total_pages} pages exceeds the safety cap of "
                    f"{FA_MAX_PAGES}; pulling the first {FA_MAX_PAGES} pages only"
                )
        for row in data.get("statsTable", []):
            scorer = row.get("scorer", {})
            scorer_id = scorer.get("scorerId")
            # The pool can shift between paginated calls; dedupe on scorerId
            if not scorer_id or scorer_id in seen_ids:
                continue
            seen_ids.add(scorer_id)
            cells = row.get("cells", [])
            # Cell 1 is the status column: "FA" when available, a fantasy
            # team name when owned (verified in the endpoint spike).
            raw_status = cells[1].get("content", "") if len(cells) > 1 else ""
            records.append({
                "player_name": _name_from_url_slug(
                    scorer.get("urlName", ""), scorer.get("name", "")
                ),
                "position": scorer.get("posShortNames", ""),
                "fantrax_id": scorer_id,
                "mlb_team": scorer.get("teamShortName", ""),
                "status": "fa" if raw_status == "FA" else "owned",
                "minors_eligible": bool(scorer.get("minorsEligible", False)),
            })
        page += 1
        if page <= total_pages:
            time.sleep(FA_PAGE_SLEEP_SECONDS)
    return pd.DataFrame(records, columns=FA_OUTPUT_COLUMNS)


def run_free_agent_pull(cookie: str, stamp: str) -> pathlib.Path:
    """Pull the available-player pool and write free_agents_<stamp>.csv.

    Args:
        cookie: Browser session cookie for fantrax.com.
        stamp: ISO date string used in the output filename.

    Returns:
        Path to the CSV that was written.
    """
    league_id, _ = load_fantrax_config()
    print(f"Pulling free-agent pool for league {league_id}")
    session = build_session(cookie)
    session.headers.update({"Content-Type": "application/json"})

    started = time.monotonic()
    free_agents = fetch_free_agents(session, league_id)
    elapsed = time.monotonic() - started

    FANTRAX_DIR.mkdir(parents=True, exist_ok=True)
    path = FANTRAX_DIR / f"free_agents_{stamp}.csv"
    free_agents.to_csv(path, index=False, encoding="utf-8")
    available = int((free_agents["status"] == "fa").sum())
    print(
        f"  Wrote {path} ({len(free_agents)} rows, {available} available) "
        f"in {elapsed:.0f}s"
    )
    return path


# ── validation warnings / summary ────────────────────────────────────


def run_soft_checks(all_rosters: pd.DataFrame, my_roster: pd.DataFrame, my_team: str) -> None:
    """Print warnings for league-shape anomalies that should not fail bronze.

    Args:
        all_rosters: Full roster table from :func:`fetch_all_rosters`.
        my_roster: The filtered my_roster table.
        my_team: Fantasy team name being filtered on.
    """
    team_counts = all_rosters["team_name"].value_counts()
    if len(team_counts) != EXPECTED_TEAM_COUNT:
        print(
            f"  WARNING: expected {EXPECTED_TEAM_COUNT} teams, "
            f"found {len(team_counts)}: {sorted(team_counts.index)}"
        )
    for team, n in sorted(team_counts.items()):
        if not ROSTER_MIN <= n <= ROSTER_MAX:
            print(
                f"  WARNING: {team} has {n} players — "
                f"expected between {ROSTER_MIN} and {ROSTER_MAX}"
            )
    if my_roster.empty:
        print(
            f"  WARNING: my_roster is empty — no team named {my_team!r}. "
            "Check fantrax.my_team_name in config/settings.yaml."
        )


def print_summary(all_rosters: pd.DataFrame) -> None:
    """Print total and per-team roster counts.

    Args:
        all_rosters: Full roster table from :func:`fetch_all_rosters`.
    """
    print(f"\n  Total owned players: {len(all_rosters)}")
    print("\n  Per-team counts:")
    team_counts = all_rosters["team_name"].value_counts()
    for team, n in sorted(team_counts.items()):
        print(f"    {team}: {n}")


# ── entry point ──────────────────────────────────────────────────────


def run_live_pull(cookie: str, stamp: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Run the full live pull: fetch, validate, split, and write both CSVs.

    Args:
        cookie: Browser session cookie for fantrax.com.
        stamp: ISO date string used in the output filenames.

    Returns:
        Tuple of (all_rosters path, my_roster path) that were written.

    Raises:
        NotLoggedIn: If the cookie is expired or invalid.
    """
    league_id, my_team = load_fantrax_config()

    print(f"Pulling live rosters for league {league_id}")
    api = FantraxAPI(league_id, session=build_session(cookie))

    all_rosters = fetch_all_rosters(api)
    my_roster = all_rosters[all_rosters["team_name"] == my_team].reset_index(drop=True)
    run_soft_checks(all_rosters, my_roster, my_team)

    FANTRAX_DIR.mkdir(parents=True, exist_ok=True)
    all_path = FANTRAX_DIR / f"all_rosters_{stamp}.csv"
    my_path = FANTRAX_DIR / f"my_roster_{stamp}.csv"
    all_rosters.to_csv(all_path, index=False, encoding="utf-8")
    my_roster.to_csv(my_path, index=False, encoding="utf-8")

    print_summary(all_rosters)
    print(f"\n  Wrote {all_path} ({len(all_rosters)} rows)")
    print(f"  Wrote {my_path} ({len(my_roster)} rows)")
    return all_path, my_path


def main() -> None:
    """CLI entry point: pull rosters (default) or the free-agent pool."""
    # Team names can contain emoji a cp1252 Windows console can't encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Live Fantrax bronze pulls")
    parser.add_argument(
        "--free-agents",
        action="store_true",
        help="pull the full available-player pool instead of team rosters",
    )
    args = parser.parse_args()

    load_dotenv(ENV_PATH)
    cookie = os.environ.get("FANTRAX_COOKIE", "").strip()
    if not cookie:
        print(
            "FANTRAX_COOKIE is missing or empty - add your browser session "
            "cookie to .env (see .env.example) to enable live pulls."
        )
        sys.exit(1)

    try:
        if args.free_agents:
            run_free_agent_pull(cookie, date.today().isoformat())
        else:
            run_live_pull(cookie, date.today().isoformat())
    except NotLoggedIn:
        print("Fantrax cookie expired or invalid - refresh FANTRAX_COOKIE in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
