# fantasy-baseball-pipeline

Local Python pipeline for dynasty fantasy baseball research. Medallion architecture:
bronze (raw ingestion) -> silver (cleaned/matched) -> gold (analytical outputs).
Also a portfolio piece: the identity-resolution story (silver/player_universe.py,
silver/player_id_map.py) is the centerpiece. Code quality matters everywhere.

## Environment
- Windows, PowerShell. Python via .\venv\Scripts\python.exe (note: venv, not .venv)
- Run modules with python -m (e.g. python -m scripts.weekly_refresh)
- Git branch: master. Conventional commits (feat:, fix:, docs:, test:, chore:)

## Layer rules
- bronze/: raw data only, date-stamped CSVs, no transformations. Fail loudly on
  schema drift. Never silently coerce.
- silver/: cleaned + enriched, parquet with explicit schema. Fuzzy-match confidence
  captured as a column.
- gold/: decision-ready outputs. ALL thresholds come from config/standing_rules.yaml.
  Never hardcode an analytical threshold in a gold module.

## Data contracts
- Fantrax roster data: bronze/data/fantrax/all_rosters_YYYY-MM-DD.csv and
  my_roster_YYYY-MM-DD.csv. Consumers glob and take latest by filename sort.
  Fed by bronze/fantrax_client.py (live) when FANTRAX_COOKIE is set — this is the
  path scripts/weekly_refresh.py takes — falling back to python -m
  bronze.fantrax_csv_import --input <export> (manual) otherwise. Same output
  contract either way.
- bronze/data/ is gitignored. Never commit data files. Tests use fixtures in
  tests/fixtures/ instead.

## Code style
- Python 3.10+, type hints on public functions, Google docstrings, 100-char lines
- snake_case, one responsibility per function
- Comments are developer notes to a peer explaining business logic, never
  AI-explaining-syntax. No commented-out code in commits.
- Windows console is cp1252: any module printing user data (team names contain
  emoji) needs sys.stdout.reconfigure(errors="replace")

## League context (affects analytical logic)
- 12-team H2H 5x5 dynasty. Hitting: HR/RBI/R/SB/OBP. Pitching: ERA/WHIP/K/W/SVH.
- OBP league, not AVG: walk-rate signals carry extra weight
- SVH is intentionally punted: never flag closer/holds churn as actionable

## Do not
- Commit secrets, cookies, or .env (only .env.example)
- Reference paths outside this repo (no sibling-project dependencies)
- Add dependencies without flagging it
- Touch files outside the stated scope of a task
## Environment note (added 2026-08-09)
Also cloned to WSL2 Ubuntu at ~/lab/fantasy-baseball-pipeline for CI/docs work.
The Windows/venv instructions above remain canonical for RUNNING the pipeline.
Do not attempt to run pipeline modules from the WSL clone without setting up a
Linux venv first — and do not do that as a side effect of another task.

## Current sprint: Rockies application (deadline 2026-08-24)

### In scope
- GitHub Actions CI running the existing 54 tests
- README refinements
- master -> main rename (update this file's Environment section when done)
- Bug fixes surfaced by CI setup

### OUT of scope until after submission
- Orchestration (Dagster/Airflow/Prefect) — do not start
- Cloud migration
- player_master.csv -> database migration
- Any refactor of silver/player_id_map.py or silver/player_universe.py

### Rules
- Write a plan and get approval before changing pipeline code
- Tests must pass before any commit
- Small commits, one concern each
- Commit messages and README are read by hiring managers, not just tooling
