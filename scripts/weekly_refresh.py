"""Weekly full data refresh across all pipeline layers.

Orchestrates the bronze -> silver -> gold pipeline, running each layer's
modules in order and printing a timestamped summary after each layer
completes.

Usage:
    python -m scripts.weekly_refresh          # full pipeline (default)
    python -m scripts.weekly_refresh --bronze-only
    python -m scripts.weekly_refresh --gold-only
"""

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure bronze/, silver/, gold/ are importable regardless of how the
# script is invoked (direct run vs ``python -m scripts.weekly_refresh``).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
SILVER_DATA = PROJECT_ROOT / "silver" / "data"
GOLD_DATA = PROJECT_ROOT / "gold" / "data"


# ── layer runners ────────────────────────────────────────────────────


def _run_module(module_name: str, import_path: str) -> bool:
    """Import and call a module's main() function.

    Args:
        module_name: Human-readable name for logging.
        import_path: Dotted import path (e.g. ``"bronze.savant_client"``).

    Returns:
        True if the module ran successfully, False otherwise.
    """
    try:
        print(f"\n{'-' * 60}")
        print(f"  Running {module_name}...")
        print(f"{'-' * 60}")
        # __import__ with fromlist to get the actual submodule
        module = __import__(import_path, fromlist=["main"])
        module.main()
        return True
    except Exception:
        print(f"\n  ERROR in {module_name}:")
        traceback.print_exc()
        return False


def run_bronze() -> dict[str, bool]:
    """Run all bronze-layer data clients.

    Returns:
        Dict mapping module name to success/failure boolean.
    """
    modules = [
        ("Savant Client", "bronze.savant_client"),
        ("FanGraphs Client", "bronze.fangraphs_client"),
        ("MLB Stats Client", "bronze.mlb_stats_client"),
        ("MiLB Client", "bronze.milb_client"),
    ]
    results = {}
    for name, path in modules:
        results[name] = _run_module(name, path)
    return results


def run_silver() -> dict[str, bool]:
    """Run all silver-layer enrichment modules.

    Player universe must run before statcast enrichment because
    the enrichment step joins against the universe table.

    Returns:
        Dict mapping module name to success/failure boolean.
    """
    modules = [
        ("Player Universe", "silver.player_universe"),
        ("Statcast Enriched", "silver.statcast_enriched"),
    ]
    results = {}
    for name, path in modules:
        results[name] = _run_module(name, path)
    return results


def run_gold() -> dict[str, bool]:
    """Run all gold-layer analysis modules.

    Returns:
        Dict mapping module name to success/failure boolean.
    """
    modules = [
        ("Breakout Detector", "gold.breakout_detector"),
        ("Regression Alerts", "gold.regression_alerts"),
        ("Waiver Ranker", "gold.waiver_ranker"),
        ("SP Streamer", "gold.sp_streamer"),
        ("Prospect Watch", "gold.prospect_watch"),
    ]
    results = {}
    for name, path in modules:
        results[name] = _run_module(name, path)
    return results


# ── summary helpers ──────────────────────────────────────────────────


def _timestamp() -> str:
    """Return the current time formatted as ``YYYY-MM-DD HH:MM:SS``."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _count_csv_rows(path: Path) -> int:
    """Return the number of data rows in a CSV file, or 0 if missing."""
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def _count_parquet_rows(path: Path) -> int:
    """Return the number of rows in a Parquet file, or 0 if missing."""
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return 0


def print_bronze_summary(results: dict[str, bool]) -> None:
    """Print a timestamped summary of the bronze layer run.

    Args:
        results: Dict mapping module name to success/failure.
    """
    succeeded = sum(results.values())
    total = len(results)
    failed = [name for name, ok in results.items() if not ok]

    print(f"\n{'=' * 60}")
    print(f"  Bronze layer complete - data refreshed at {_timestamp()}")
    print(f"  {succeeded}/{total} modules succeeded")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")


def print_silver_summary(results: dict[str, bool]) -> None:
    """Print a timestamped summary of the silver layer run.

    Args:
        results: Dict mapping module name to success/failure.
    """
    player_count = _count_parquet_rows(SILVER_DATA / "player_universe.parquet")
    hitter_count = _count_parquet_rows(SILVER_DATA / "statcast_hitters.parquet")
    pitcher_count = _count_parquet_rows(
        SILVER_DATA / "statcast_pitchers.parquet"
    )
    enriched_total = hitter_count + pitcher_count

    failed = [name for name, ok in results.items() if not ok]

    print(f"\n{'=' * 60}")
    print(
        f"  Silver layer complete - "
        f"{enriched_total} players enriched"
    )
    print(f"  Player universe: {player_count} players")
    print(f"  Hitters: {hitter_count}  |  Pitchers: {pitcher_count}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")


def print_gold_summary(results: dict[str, bool]) -> None:
    """Print a timestamped summary of the gold layer run.

    Args:
        results: Dict mapping module name to success/failure.
    """
    breakout_count = (
        _count_csv_rows(GOLD_DATA / "breakout_hitters.csv")
        + _count_csv_rows(GOLD_DATA / "breakout_pitchers.csv")
    )
    regression_count = (
        _count_csv_rows(GOLD_DATA / "regression_hitters.csv")
        + _count_csv_rows(GOLD_DATA / "regression_pitchers.csv")
    )
    waiver_count = (
        _count_csv_rows(GOLD_DATA / "waiver_hitters_ranked.csv")
        + _count_csv_rows(GOLD_DATA / "waiver_pitchers_ranked.csv")
    )
    streamer_count = _count_csv_rows(GOLD_DATA / "sp_streaming_picks.csv")
    prospect_count = _count_csv_rows(GOLD_DATA / "prospect_alerts.csv")

    failed = [name for name, ok in results.items() if not ok]

    print(f"\n{'=' * 60}")
    print(
        f"  Gold layer complete - "
        f"{breakout_count} breakout candidates, "
        f"{regression_count} regression alerts"
    )
    print(
        f"  Waiver targets: {waiver_count}  |  "
        f"SP streamers: {streamer_count}  |  "
        f"Prospect alerts: {prospect_count}"
    )
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'=' * 60}")


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed namespace with the selected run mode.
    """
    parser = argparse.ArgumentParser(
        description="Weekly fantasy baseball pipeline refresh.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--bronze-only",
        action="store_true",
        help="Only refresh bronze (raw data) layer.",
    )
    group.add_argument(
        "--gold-only",
        action="store_true",
        help="Only rerun gold (analysis) layer on existing data.",
    )
    group.add_argument(
        "--full",
        action="store_true",
        default=True,
        help="Run the full pipeline (default).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the weekly refresh pipeline.

    Executes bronze -> silver -> gold layers in order, printing a
    timestamped summary after each layer. Individual module failures
    are caught and logged without stopping the rest of the pipeline.

    Args:
        argv: Optional argument list for testing; defaults to sys.argv.
    """
    args = parse_args(argv)

    print(f"\n{'#' * 60}")
    print(f"  Fantasy Baseball Pipeline - Weekly Refresh")
    print(f"  Started at {_timestamp()}")
    print(f"{'#' * 60}")

    if args.bronze_only:
        results = run_bronze()
        print_bronze_summary(results)

    elif args.gold_only:
        results = run_gold()
        print_gold_summary(results)

    else:
        # Full pipeline: bronze -> silver -> gold
        bronze_results = run_bronze()
        print_bronze_summary(bronze_results)

        silver_results = run_silver()
        print_silver_summary(silver_results)

        gold_results = run_gold()
        print_gold_summary(gold_results)

    print(f"\n  Pipeline finished at {_timestamp()}\n")


if __name__ == "__main__":
    main()
