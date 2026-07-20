"""Pull live Fantrax rosters for the ShadyNasty football league into bronze.

Football fork of the baseball bronze/fantrax_client.py. Same platform, same
auth, same fxpa/req free-agent mechanism — Fantrax exposes one API shape across
sports, so the transport layer transfers unchanged. What changes is football:
the league id, the 12-team / 23-man league shape, and the position eligibility
string, which now yields FOOTBALL positions INCLUDING defensive ones (DL/LB/DB).
This is an IDP league; the pull is never filtered to offense.

Auth is identical to baseball: a browser session cookie in FANTRAX_COOKIE
(.env). No Selenium, no username/password — those are vestigial.

Also pulls the full available-player pool (free agents) via the raw fxpa
getPlayerStats endpoint — fantraxapi has no free-agent support, so that pull
speaks to the endpoint directly with the same session cookie.

FIELD VERIFICATION: the roster/points/eligibility field names below are carried
over from the baseball baseline and have NOT yet been confirmed against a live
football response. Run with --inspect to print the raw response structure and
verify field names before trusting derived columns downstream.

Inputs:
    FANTRAX_COOKIE in root .env (browser session cookie)
    football/config/settings.yaml (league.league_id, fantrax.my_team_name)

Outputs:
    football/bronze/data/fantrax/all_rosters_YYYY-MM-DD.csv
    football/bronze/data/fantrax/my_roster_YYYY-MM-DD.csv
    football/bronze/data/fantrax/free_agents_YYYY-MM-DD.csv

Usage:
    python -m football.bronze.fantrax_client                 # roster pull
    python -m football.bronze.fantrax_client --free-agents   # available pool
    python -m football.bronze.fantrax_client --inspect       # raw shape probe
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

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent / "data" / "fantrax"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# ── output contract (must match football/bronze/fantrax_csv_import.py) ──

OUTPUT_COLUMNS = [
    "team_name", "player_name", "position", "fantasy_points",
    "points_per_game", "fantrax_id", "nfl_team", "status", "age",
]

FA_OUTPUT_COLUMNS = [
    "player_name", "position", "fantrax_id", "nfl_team", "status",
]

EXPECTED_TEAM_COUNT = 12
# 23 counts against the roster (13 active + 10 reserve) and up to 2 IR sit
# outside it, so a full team can show up to ~25 rostered players. Soft-check
# bounds only — these WARN, they never fail the pull. Widen once the live pull
# confirms how IR players surface in the roster response.
ROSTER_MIN = 18
ROSTER_MAX = 25

FXPA_URL = "https://www.fantrax.com/fxpa/req"
# The baseball league's server capped page size at 50 regardless of request;
# asking for more is harmless and future-proofs a raised cap.
FA_PAGE_SIZE = 200
FA_MAX_PAGES = 250          # safety valve
FA_PAGE_SLEEP_SECONDS = 0.2  # be polite between paginated calls

# Cloudflare rejects requests carrying the default python-requests UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)


# ── config ───────────────────────────────────────────────────────────


def load_fantrax_config() -> tuple[str, str]:
    """Read the live-client settings from football/config/settings.yaml.

    Returns:
        Tuple of (league_id, my_team_name). league_id comes from the
        ``league:`` block (the canonical id); my_team_name from ``fantrax:``.

    Raises:
        KeyError: If a required key is missing from settings.yaml.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    try:
        league_id = settings["league"]["league_id"]
        my_team_name = settings["fantrax"]["my_team_name"]
        return league_id, my_team_name
    except (KeyError, TypeError):
        raise KeyError(
            f"league.league_id / fantrax.my_team_name not found in {CONFIG_PATH} — "
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

    Empty roster slots are skipped — the contract is one row per owned player.
    Columns the API does not provide (age) are empty strings.

    IDP NOTE: no position filter is applied. Every scorer with an identity is
    kept, so defensive players (DL/LB/DB) come through exactly like offensive
    ones. The ``position`` column carries Fantrax's multi-position eligibility
    string verbatim; classifying it into offense/defense happens in the silver
    identity layer, not here.

    Args:
        api: Authenticated FantraxAPI (League) instance.

    Returns:
        DataFrame with :data:`OUTPUT_COLUMNS`, one row per owned player.
    """
    # As in the baseball baseline, take the raw STATS-view response from the
    # library's request layer and parse rows ourselves (the library's full
    # Roster object also parses the schedule view, whose Game parser can crash
    # on cells without a game time). The STATS view is all we need.
    records = []
    for team in api.teams:
        stats = fantrax_api.get_team_roster_info(api, team.id)[0]
        for table in stats["tables"]:
            header = table["header"]["cells"]
            for row in table["rows"]:
                scorer = row.get("scorer", {})
                if "scorerId" not in scorer:
                    continue
                points = {"SCORE": "", "FPTS_PER_GAME": ""}
                for head, cell in zip(header, row["cells"]):
                    key = head.get("sortKey")
                    if key in points and cell.get("content") not in (None, ""):
                        points[key] = cell["content"]
                # posShortNames is the multi-position ELIGIBILITY string, the
                # authority on what positions a player qualifies at. Prefer it
                # over the roster slot (today's lineup spot). Fall back to the
                # slot only when Fantrax omits eligibility, so position is never
                # blank. (Field names verified against baseball; re-verify for
                # football with --inspect.)
                eligibility = (
                    scorer.get("posShortNames")
                    or api.positions[row["posId"]].short_name
                )
                records.append({
                    "team_name": team.name,
                    "player_name": scorer["name"],
                    "position": eligibility,
                    "fantasy_points": points["SCORE"],
                    "points_per_game": points["FPTS_PER_GAME"],
                    "fantrax_id": scorer["scorerId"],
                    "nfl_team": scorer.get("teamShortName", scorer.get("teamName", "")),
                    "status": "owned",
                    "age": "",
                })
    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def inspect_live_response(api: FantraxAPI) -> None:
    """Print the RAW structure of the first team's roster response.

    Verification aid for the football fork: the field names read in
    :func:`fetch_all_rosters` (posShortNames, teamShortName, the SCORE /
    FPTS_PER_GAME sort keys) are inherited from baseball and must be confirmed
    against a real football response before downstream code trusts them. This
    prints exactly what the API returns — no derived columns, no assumptions —
    so the contract can be verified from live data.
    """
    team = api.teams[0]
    stats = fantrax_api.get_team_roster_info(api, team.id)[0]
    print(f"\n=== RAW roster response for team: {team.name} ({team.id}) ===")
    print(f"Top-level keys: {list(stats.keys())}")
    for t_idx, table in enumerate(stats.get("tables", [])):
        header_keys = [c.get("sortKey") for c in table["header"]["cells"]]
        print(f"\n-- table[{t_idx}] header sortKeys: {header_keys}")
        for row in table.get("rows", [])[:3]:
            scorer = row.get("scorer", {})
            print(f"   scorer keys: {list(scorer.keys())}")
            print(f"     name={scorer.get('name')!r} "
                  f"posShortNames={scorer.get('posShortNames')!r} "
                  f"teamShortName={scorer.get('teamShortName')!r} "
                  f"scorerId={scorer.get('scorerId')!r}")
    print("\n=== end raw response ===\n")


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

    The players-list scorer object carries abbreviated names ("J. Allen"), which
    would wreck downstream fuzzy matching; the slug ("josh-allen") preserves the
    full name, minus case and accents — both discarded by the matcher's
    normalization anyway.
    """
    if not slug:
        return fallback
    return " ".join(part.capitalize() for part in slug.split("-"))


def fetch_free_agents(session: requests.Session, league_id: str) -> pd.DataFrame:
    """Pull the full available-player pool via paginated getPlayerStats.

    The endpoint's default view is ALL_AVAILABLE (free agents); the status cell
    is read anyway, so any row showing a fantasy team instead of "FA" is tagged
    "owned" rather than trusted blindly.

    IDP NOTE: no position filter — defensive free agents come through. The
    ``position`` column carries Fantrax eligibility (offense or defense);
    classification happens in silver.

    CONFIRM ON LIVE DATA: the status cell index (cells[1] == "FA") and the
    posShortNames / urlName / teamShortName field names are inherited from the
    baseball spike and must be re-verified for football via --inspect.

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
            # The pool can shift between paginated calls; dedupe on scorerId.
            if not scorer_id or scorer_id in seen_ids:
                continue
            seen_ids.add(scorer_id)
            cells = row.get("cells", [])
            # Cell 1 is the status column: "FA" when available, a fantasy team
            # name when owned (baseball-verified; confirm for football).
            raw_status = cells[1].get("content", "") if len(cells) > 1 else ""
            records.append({
                "player_name": _name_from_url_slug(
                    scorer.get("urlName", ""), scorer.get("name", "")
                ),
                "position": scorer.get("posShortNames", ""),
                "fantrax_id": scorer_id,
                "nfl_team": scorer.get("teamShortName", ""),
                "status": "fa" if raw_status == "FA" else "owned",
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
            "Check fantrax.my_team_name in football/config/settings.yaml."
        )


def summarize_position_groups(all_rosters: pd.DataFrame) -> None:
    """Print a rough offense/defense/unknown split as an IDP sanity check.

    This is the quickest confirmation that defensive players actually came
    through the pull: if the defense bucket is empty, the IDP roster did not
    load. Uses a lightweight token check independent of the silver classifier.
    """
    offense = {"QB", "RB", "WR", "TE"}
    defense = {"DL", "DE", "DT", "EDGE", "LB", "ILB", "OLB", "DB", "CB", "S", "SS", "FS"}

    def bucket(pos: str) -> str:
        tokens = {t.strip().upper() for t in str(pos).replace("/", ",").split(",") if t.strip()}
        is_off = bool(tokens & offense)
        is_def = bool(tokens & defense)
        if is_off and not is_def:
            return "offense"
        if is_def and not is_off:
            return "defense"
        return "unknown"

    counts = all_rosters["position"].apply(bucket).value_counts()
    print("\n  Position-group split (IDP sanity check):")
    for group in ("offense", "defense", "unknown"):
        print(f"    {group}: {int(counts.get(group, 0))}")
    if int(counts.get("defense", 0)) == 0:
        print("    WARNING: zero defensive players — IDP roster may not have loaded!")
    if int(counts.get("unknown", 0)) > 0:
        unknown_positions = sorted(
            all_rosters.loc[all_rosters["position"].apply(bucket) == "unknown", "position"]
            .dropna().unique()
        )
        print(f"    Unrecognized position strings (verify vocabulary): {unknown_positions}")


def print_summary(all_rosters: pd.DataFrame) -> None:
    """Print total and per-team roster counts."""
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
    summarize_position_groups(all_rosters)
    print(f"\n  Wrote {all_path} ({len(all_rosters)} rows)")
    print(f"  Wrote {my_path} ({len(my_roster)} rows)")
    return all_path, my_path


def main() -> None:
    """CLI entry point: pull rosters (default), the FA pool, or probe raw shape."""
    # Team names can contain emoji a cp1252 Windows console can't encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Live Fantrax bronze pulls (football)")
    parser.add_argument(
        "--free-agents",
        action="store_true",
        help="pull the full available-player pool instead of team rosters",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="print the RAW live response structure to verify field names",
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
        if args.inspect:
            league_id, _ = load_fantrax_config()
            api = FantraxAPI(league_id, session=build_session(cookie))
            inspect_live_response(api)
        elif args.free_agents:
            run_free_agent_pull(cookie, date.today().isoformat())
        else:
            run_live_pull(cookie, date.today().isoformat())
    except NotLoggedIn:
        print("Fantrax cookie expired or invalid - refresh FANTRAX_COOKIE in .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
