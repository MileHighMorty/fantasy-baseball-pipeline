# football/ — ShadyNasty fantasy football pipeline

Fork of the root fantasy-baseball-pipeline for a Fantrax dynasty FOOTBALL league.
Medallion architecture: bronze (raw ingestion) → silver (cleaned/identity-
resolved) → gold (analytical outputs). The identity-resolution story
(silver/player_id_map.py) is the reused centerpiece; code quality matters
everywhere.

## Relationship to the baseball pipeline
- The baseball pipeline at the repo root is SEPARATE and must stay working.
  Never import baseball modules (`bronze.*`, `silver.*`, `gold.*`) from here, and
  never edit them for football reasons. Football code imports `football.*` only.
- Shared: the repo, the venv, and the single root `.env` (one FANTRAX_COOKIE).

## Config — single source of truth
- config/settings.yaml holds EVERYTHING league-specific: identifiers, roster
  shape, scoring, and (later) analytical thresholds. Do not describe scoring or
  thresholds anywhere else. There is no standing_rules.yaml — thresholds live in
  settings.yaml under `thresholds:`. (These two rules exist because the baseball
  baseline drifted: a phantom standing_rules.yaml reference and scoring described
  in two disagreeing places. Do not reintroduce either pattern.)

## Layer rules
- bronze/: raw data only, date-stamped CSVs, no transformations. Fail loudly on
  schema drift. Never silently coerce.
- silver/: cleaned + identity-resolved. The player_id_map TYPE GATE is by
  position GROUP (offense {QB,RB,WR,TE} vs defense {DL,LB,DB}) — an offensive
  player only matches offensive stat sources and vice versa. Never let a
  defensive player inherit an offensive player's stats or the reverse.
- gold/: decision-ready outputs. ALL thresholds come from
  config/settings.yaml (`thresholds:`). Never hardcode an analytical threshold.

## Auth (Fantrax)
- Browser session cookie in FANTRAX_COOKIE (root .env). No Selenium, no
  username/password — those .env vars from the baseball baseline are vestigial;
  do not reintroduce them.

## Football specifics
- IDP league: defensive players (DL/LB/DB) are rostered and scored. Never filter
  the roster/FA pull to offense only.
- Single QB, not superflex. Dynasty. 12 teams. 23-man roster (+2 IR).
- Verify Fantrax's football position vocabulary and points columns against a
  LIVE response (fantrax_client --inspect) before trusting field names. Never
  claim a field exists without printing it from a live response.

## Code style
- Python 3.10+, type hints on public functions, Google docstrings, 100-char
  lines, snake_case, one responsibility per function.
- Comments explain business logic to a peer, never AI-explaining-syntax. No
  commented-out code in commits.
- Any module printing user data (team names may contain emoji) reconfigures
  stdout with errors="replace" for cp1252 Windows consoles.

## Do not
- Commit secrets, cookies, or .env (only .env.example).
- Break the baseball pipeline.
- Add dependencies without flagging it.
