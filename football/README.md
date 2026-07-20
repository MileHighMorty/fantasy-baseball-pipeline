# ShadyNasty — Fantasy Football Pipeline

A fork of the fantasy-baseball-pipeline, adapted for the **ShadyNasty** dynasty
fantasy football league on Fantrax (12-team, single-QB, IDP). It reuses the part
of the baseball project that was always sport-agnostic — the identity-resolution
layer (`player_id_map`: fuzzy match + type gate + override CSV + immutable
`res_key` master) — and swaps in football data sources and league rules.

## Why a `football/` subtree (not a sibling repo)

The baseball pipeline lives at the repo root and is **left completely
untouched**: it still imports `bronze.*` / `silver.*` / `gold.*` and runs
exactly as before. This football package mirrors the same medallion layout one
level down, so the two:

- share the repo, the `venv`, and the single `.env` (one `FANTRAX_COOKIE`);
- stay import-isolated — baseball is `silver.player_id_map`, football is
  `football.silver.player_id_map`, no cross-imports, no name clashes;
- keep the identity-resolution portfolio story in one place instead of
  duplicating tooling across two repos.

A sibling repo would have duplicated the venv, the CI, and the `.env` handling
for no benefit. If football later needs to ship independently, this subtree
lifts out cleanly.

## Layout (mirrors the baseball medallion)

```
football/
├── config/settings.yaml       # SINGLE source of truth: ids, roster, scoring
├── bronze/
│   ├── fantrax_client.py       # live roster + FA pull (cookie auth, league vqlp0t1rmk1rgquz)
│   └── fantrax_csv_import.py    # manual export fallback, same output contract
├── silver/
│   └── player_id_map.py         # identity resolution; TYPE GATE = offense vs defense
├── overrides/player_name_overrides.csv   # manual force/block corrections
└── (bronze/data, silver/data, gold/data are gitignored, created at runtime)
```

## Build status

This is an **in-progress fork**, paused for review before the analytics layer.

Done:
- `config/settings.yaml` — league facts + custom scoring encoded (gaps flagged
  in a `to_confirm` block rather than guessed).
- `bronze/fantrax_client.py` — Fantrax football roster + free-agent pull.
- `bronze/fantrax_csv_import.py` — manual-export fallback for offline validation.
- `silver/player_id_map.py` — identity resolution with the type gate changed
  from hitter/pitcher to offense/defense position groups.

Not built yet (deliberately — awaiting review + a live-pull confirmation):
- Football bronze **stat** sources (nfl-data-py / nflverse, injuries, Vegas).
- The gold analytics layer and a dashboard.

Until a stat source exists, `player_id_map` runs **roster-only**: it registers
every Fantrax player into the `res_key` master keyed by their Fantrax id and
position group. External stat ids fill in later via the same immutable-master
rule the baseball pipeline uses.

## Running the Fantrax pull

Live pull needs a fresh `FANTRAX_COOKIE` in the repo-root `.env` (same mechanism
as baseball — a browser session cookie, no username/password, no Selenium):

```bash
python -m football.bronze.fantrax_client               # roster pull
python -m football.bronze.fantrax_client --free-agents # free-agent pool
python -m football.bronze.fantrax_client --inspect     # print the RAW live
                                                        # response shape to
                                                        # verify field names
```

`--inspect` prints the raw scorer/header structure from the live API so field
names (positions, points columns, IDP eligibility) are **verified against a real
response** before any code trusts them.

No cookie? Validate structure offline from a manual Players-page export:

```bash
python -m football.bronze.fantrax_csv_import --input <export.csv>
```
