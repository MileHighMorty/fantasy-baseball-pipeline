# Fantasy Baseball → Football Port Plan

Read-only survey of the baseball pipeline plus a plan for forking it into
fantasy football (Fantrax dynasty + Yahoo redraft). Nothing in the baseball
pipeline is modified by this document.

## 1. Directory tree & file inventory

```
fantasy-baseball-pipeline/
├── CLAUDE.md                      # project instructions (see §note on drift)
├── README.md                      # identity-resolution is the centerpiece
├── requirements.txt               # 13 runtime deps
├── requirements-dev.txt           # pytest only
├── .env.example                   # FANTRAX_USERNAME/PASSWORD/LEAGUE_ID/COOKIE
├── .gitignore                     # ignores venv/, .env, */data/, *.duckdb, *.cookie
├── .streamlit/config.toml
│
├── config/
│   ├── settings.yaml              # league shape, scoring cats, fantrax ids, paths
│   └── prospect_watchlist.yaml    # hand-maintained prospect list
│
├── bronze/                        # RAW pulls → date-stamped files in bronze/data/ (gitignored)
│   ├── fantrax_client.py          # LIVE roster + free-agent pull (cookie auth)
│   ├── fantrax_csv_import.py      # manual CSV-export fallback (same output contract)
│   ├── savant_client.py           # Baseball Savant Statcast leaderboards
│   ├── fangraphs_client.py        # FanGraphs leaders API (Cloudflare bypass)
│   ├── mlb_stats_client.py        # MLB Stats API: schedule/txns/roster/standings
│   └── milb_client.py             # MiLB game logs + 40-man for prospects
│
├── silver/                        # CLEANED + identity-resolved → silver/data/
│   ├── player_id_map.py           # THE CENTERPIECE: fuzzy match + res_key master
│   ├── player_universe.py         # simpler Savant↔FanGraphs match → parquet
│   ├── statcast_enriched.py       # joins expected-vs-actual onto resolved ids
│   ├── freshness.py               # stale-FanGraphs warning helper
│   ├── prospect_tracker.py        # STUB (1 line)
│   └── roster_context.py          # STUB (1 line)
│
├── gold/                          # DECISIONS → gold/data/*.csv
│   ├── breakout_detector.py       # xwOBA vs wOBA buy-low
│   ├── regression_alerts.py       # overperformers to sell/hold
│   ├── waiver_ranker.py           # free-agent ranking
│   ├── sp_streamer.py             # daily SP streaming by matchup
│   ├── add_drop_engine.py         # weakest-rostered vs best-available
│   ├── prospect_watch.py          # MiLB call-up watch
│   ├── ownership.py               # owned/FA helper (shared by gold modules)
│   └── trade_evaluator.py         # STUB (1 line)
│
├── dashboard/
│   ├── app.py                     # Streamlit, 6 pages
│   └── theme.py                   # styling
│
├── scripts/
│   ├── weekly_refresh.py          # orchestrates bronze→silver→gold
│   └── daily_check.py             # STUB (pass)
│
├── overrides/player_name_overrides.csv   # manual force/block match corrections
├── tests/                         # test_player_id_map (48 tests), test_regression_breakout
└── docs/screenshots/*.png         # 6 dashboard screenshots
```

Stub files (not implemented): `gold/trade_evaluator.py`,
`silver/prospect_tracker.py`, `silver/roster_context.py`,
`scripts/daily_check.py`. Nothing there transfers.

## 2. The Fantrax auth module (`bronze/fantrax_client.py`)

**There is no Selenium.** Despite `.env.example` listing
`FANTRAX_USERNAME`/`FANTRAX_PASSWORD`, the live client never does a
username/password login. Authentication is a **manually-harvested browser
session cookie**:

- **Mechanism** (`build_session`): copy the full `Cookie` header from a
  logged-in fantrax.com tab into `FANTRAX_COOKIE` in `.env`. The client builds
  a `requests.Session`, sets that cookie plus a spoofed Chrome `User-Agent`
  (Cloudflare rejects the default `python-requests` UA), and hands the session
  to `fantraxapi`: `FantraxAPI(league_id, session=build_session(cookie))`.
- **`.env` vars actually read:** only `FANTRAX_COOKIE`. `USERNAME`/`PASSWORD`/
  `LEAGUE_ID` in `.env.example` are **vestigial** — the league id comes from
  `config/settings.yaml` (`fantrax.league_id`).
- **Session handling / expiry:** no refresh. An expired cookie surfaces as
  `fantraxapi.NotLoggedIn` → prints "refresh FANTRAX_COOKIE" and exits. In the
  orchestrator the whole live pull is wrapped in a bare `except` and swallowed,
  falling back to the manual CSV path — the pipeline never dies on a bad cookie.
- **Two feeds, one contract:** live API and CSV importer both emit
  `all_rosters_YYYY-MM-DD.csv` and `my_roster_YYYY-MM-DD.csv` with identical
  columns. Wrinkle: the live API reports **fantasy team names** while the CSV
  export reports **owner labels**; both live in `settings.yaml`
  (`my_team_name` vs `my_team_label`).

## 3. Every data-pull function

**Fantrax** — `bronze/fantrax_client.py` (cookie auth):
- `fetch_all_rosters(api)` → `fantraxapi.get_team_roster_info` (STATS view only;
  the library's Game parser crashes on schedule cells, so the raw response is
  parsed by hand). One row/owned player.
- `fetch_free_agents(session, league_id)` → raw
  `POST https://www.fantrax.com/fxpa/req?leagueId=<id>`, method `getPlayerStats`,
  paginated (server caps 50/page). The only free-agent source.

**Baseball Savant** — `bronze/savant_client.py` (spoofed-UA GET, CSV):
`/leaderboard/expected_statistics`, `/leaderboard/statcast`,
`/statcast_search/csv` (pitch-level detail).

**FanGraphs** — `bronze/fangraphs_client.py`: `curl_cffi` Chrome TLS
impersonation to clear Cloudflare (spoofed UA alone is not enough): warm-up GET
on the homepage, then `/api/leaders/major-league/data`.

**MLB Stats API** — `bronze/mlb_stats_client.py` (`statsapi.mlb.com/api/v1`, no
auth): `/schedule` (probable pitchers w/ MLBAM ids), `/transactions`,
`/teams/{id}/roster`, `/standings`.

**MiLB** — `bronze/milb_client.py` (same base): `/people/{id}/stats?stats=gameLog`,
`/people/{id}`, `/teams/{id}/roster?rosterType=40Man`.

## 4. Data flow / architecture

Medallion, all local disk. **No Drive/cloud-write logic anywhere** — outputs
land in gitignored per-layer `*/data/` directories.

```
BRONZE  bronze/data/{fantrax,savant,fangraphs,mlb,milb}/  ← date-stamped CSVs
   ▼    (consumers glob "*_<suffix>.csv" and take latest by filename sort)
SILVER  silver/data/  ← identity resolution
   │   player_id_map.py  → player_id_map.parquet + player_master.csv (res_key)
   │   statcast_enriched.py → statcast_hitters.parquet, statcast_pitchers.parquet
   ▼
GOLD    gold/data/*.csv  ← decisions
   ▼
DASHBOARD  dashboard/app.py (Streamlit, 6 pages)
```

Orchestration (`scripts/weekly_refresh.py`): runs bronze→silver→gold via
`__import__(...).main()`; each module wrapped so one failure logs and continues.
Bronze first attempts live Fantrax pulls if `FANTRAX_COOKIE` is set, then a
freshness check warns if the newest export is >7 days old.

**Identity layer** (`player_id_map.py`): resolves every Fantrax player to Savant
+ FanGraphs via `rapidfuzz` `token_sort_ratio` on accent-stripped names
(threshold 90), a **hard player-type gate** (hitters only match batting sources —
this is what makes two-way Ohtani correct), **team as tiebreaker only**, and
manual force/block overrides. `player_master.csv` assigns an immutable monotonic
`res_key`; established vendor IDs are never overwritten by a later fuzzy match
(the "Joscar/Teoscar" collision fix at 9.5k-player scale).

## 5. Dependencies

`requirements.txt`: `fantraxapi`, `pybaseball`, `pandas`, `pyarrow`, `requests`,
`curl_cffi`, `pyyaml`, `python-dotenv`, `streamlit`, `duckdb`, `beautifulsoup4`,
`lxml`, `rapidfuzz`. `requirements-dev.txt`: `pytest`. Python 3.10+, Windows/
PowerShell, venv is `venv` (not `.venv`).

## 6. Baseball-specific vs. reusable for football

| Reusable ~as-is (domain-agnostic) | Baseball-specific (rewrite) |
|---|---|
| Identity-resolution architecture (fuzzy + type gate + team tiebreak + override CSV + immutable `res_key` master) — the portfolio centerpiece; sport-agnostic pattern | The type-gate *values* (hitter/pitcher → football position groups) |
| Medallion layout + orchestrator (glob-latest convention, freshness warnings) | Every bronze data source (Savant/FanGraphs/MLB Stats/MiLB are all MLB) |
| Fantrax cookie-auth pattern + `fxpa/req` paginated FA pull | MLBAM id as the join spine (football has no universal equivalent) |
| CSV-import fallback contract | All gold analytics (xwOBA/xERA/OBP models) |
| Manual override force/block, staleness helpers, Streamlit shell/theme | Scoring config (5×5 categories, roster slots) |
| cp1252/emoji console handling, dated-file conventions | Prospect/MiLB pipeline (no football analog) |

## 7. Port plan

**Transfers as-is (the spine):** medallion structure, `weekly_refresh`
orchestrator, the `player_id_map` design, the Fantrax cookie-auth + `fxpa`
client, the CSV-fallback contract, freshness/console helpers, Streamlit shell.

**(a) Fantrax football** — *lowest lift.* Same platform, same cookie auth, same
`fxpa/req` shape. Point the client at a football `league_id`; adjust team/roster
bounds; the position eligibility string now yields football positions
**including IDP (DL/LB/DB)**. Retarget the type gate from hitter/pitcher to
**position groups** (offense {QB,RB,WR,TE} vs defense {DL,LB,DB}); everything
else in the matcher transfers unchanged.

**(b) Yahoo redraft** — *new auth model.* Yahoo uses **OAuth2** (Fantasy Sports
API; `yahoo-oauth`/`yfpy`), not a cookie scrape. A genuinely new bronze client:
token dance + refresh replaces copy-a-cookie. Map Yahoo `player_key`/`player_id`
into the same roster output contract so silver stays source-agnostic. Redraft
drops dynasty/prospect machinery and adds weekly cadence (byes, waivers,
start/sit).

**New bronze stat sources needed (both):** football feeds to replace Savant/
FanGraphs — e.g. `nfl-data-py`/nflverse (play-by-play, snap counts, targets,
PFR advanced), an injury/news feed, a Vegas lines/matchup source. Each to the
same date-stamped-CSV + glob-latest convention.

**New gold modules:** start/sit by matchup & projection, FAAB waiver targeting by
opportunity share, buy-low/sell-high on target-share vs production (the analog of
the xwOBA-vs-wOBA breakout board), trade evaluator (currently a stub either way).

## Two drift bugs to fix in the port (do not replicate)

1. **`CLAUDE.md` references `config/standing_rules.yaml`, which does not exist.**
   In reality thresholds are hardcoded inline in the gold modules. For football,
   pick ONE convention (thresholds in `settings.yaml`, single source of truth)
   and do not reference a phantom file.
2. **Scoring is described in two places that disagree** (`CLAUDE.md` says OBP 5×5
   / SVH punted; `settings.yaml` lists AVG/OPS + W/SV/K/ERA/WHIP/QS). Football
   scoring must live in exactly ONE place: `settings.yaml`.
