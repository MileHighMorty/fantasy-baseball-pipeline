"""Import a manual Fantrax "Players" page export into the football bronze layer.

Football fork of the baseball bronze/fantrax_csv_import.py. Fantrax has no export
API for the full player pool, so the user can download the Players page CSV by
hand; this module validates that export and splits it into the two date-stamped
roster CSVs the rest of the football pipeline consumes. It exists mainly as the
offline validation path — structure can be checked from a real export without a
live cookie.

The export's Status column is exactly one of: "FA", an owner label (the fantasy
team), or a waiver tag containing literal HTML (e.g. ``W <small>(Tue)</small>``).
The status vocabulary is Fantrax-platform behavior, not sport-specific.

SCHEMA NOTE: only the columns actually consumed are required (ID, Player, Team,
Position, Status, Age). The baseball fork hard-pinned the full export column
list; the football export's extra columns are not yet confirmed, so extras are
passed over rather than treated as schema drift. Confirm the full football
export header against a real download.

Inputs:
    Fantrax Players export CSV (path given via --input)
    football/config/settings.yaml (fantrax.my_team_label)

Outputs:
    football/bronze/data/fantrax/all_rosters_YYYY-MM-DD.csv
    football/bronze/data/fantrax/my_roster_YYYY-MM-DD.csv

Usage:
    python -m football.bronze.fantrax_csv_import --input <export.csv> [--date YYYY-MM-DD]
"""

import argparse
import pathlib
import re
import sys
from datetime import date

import pandas as pd
import yaml

# ── paths ────────────────────────────────────────────────────────────

FANTRAX_DIR = pathlib.Path(__file__).resolve().parent / "data" / "fantrax"
CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

# ── export contract ──────────────────────────────────────────────────

# Only the columns the transform reads. Extra export columns are ignored, not
# rejected (the full football export header is not yet confirmed).
REQUIRED_COLUMNS = ["ID", "Player", "Team", "Position", "Status", "Age"]

FA_LABEL = "FA"
WAIVER_RE = re.compile(r"^W\s*<small>.*</small>$")

EXPECTED_OWNER_COUNT = 12
MY_ROSTER_MIN = 18
MY_ROSTER_MAX = 25

# Output contract shared with football/bronze/fantrax_client.py. fantasy_points
# and points_per_game are empty-string artifacts the export does not carry.
OUTPUT_COLUMNS = [
    "team_name", "player_name", "position", "fantasy_points",
    "points_per_game", "fantrax_id", "nfl_team", "status", "age",
]


class SchemaDriftError(ValueError):
    """Raised when the Fantrax export is missing a consumed column."""


# ── config ───────────────────────────────────────────────────────────


def load_my_team_label() -> str:
    """Read the user's Fantrax owner label from football/config/settings.yaml.

    Returns:
        The value of the ``fantrax.my_team_label`` key.

    Raises:
        KeyError: If the key is missing from settings.yaml.
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    try:
        return settings["fantrax"]["my_team_label"]
    except (KeyError, TypeError):
        raise KeyError(
            f"fantrax.my_team_label not found in {CONFIG_PATH} — add it "
            "(the Status value Fantrax shows for your own team)."
        ) from None


# ── load / validate ──────────────────────────────────────────────────


def classify_status(status: str) -> str:
    """Classify a raw Status value as ``'fa'``, ``'waiver'``, or ``'owner'``."""
    if status == FA_LABEL:
        return "fa"
    if WAIVER_RE.match(status):
        return "waiver"
    return "owner"


def read_export(input_path: pathlib.Path) -> pd.DataFrame:
    """Read and validate a Fantrax Players export CSV.

    Validates that the consumed columns are present, strips the asterisk
    wrapping from IDs, and checks fantrax_id integrity.

    Args:
        input_path: Path to the downloaded export CSV.

    Returns:
        DataFrame with all export columns (as strings) plus ``fantrax_id`` and
        ``status_class``.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        SchemaDriftError: If a consumed column is missing.
        ValueError: If any fantrax_id is null/empty or duplicated.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Fantrax export not found: {input_path}")

    df = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaDriftError(
            f"Fantrax export missing consumed columns {missing} in "
            f"{input_path.name}. Present columns: {list(df.columns)}. "
            f"Required (consumed): {REQUIRED_COLUMNS}"
        )

    # "*05ynb*" -> "05ynb"
    df["fantrax_id"] = df["ID"].str.strip().str.strip("*")

    null_ids = df["fantrax_id"].isna() | (df["fantrax_id"] == "")
    if null_ids.any():
        bad = df.loc[null_ids, "Player"].tolist()
        raise ValueError(f"{null_ids.sum()} rows have null/empty fantrax_id: {bad}")

    dupes = df.loc[df["fantrax_id"].duplicated(), "fantrax_id"].unique().tolist()
    if dupes:
        raise ValueError(f"Duplicate fantrax_id values after asterisk strip: {dupes}")

    df["status_class"] = df["Status"].fillna("").apply(classify_status)
    return df


# ── transform ────────────────────────────────────────────────────────


def build_all_rosters(df: pd.DataFrame) -> pd.DataFrame:
    """Build the all_rosters output table from a validated export.

    Keeps only rows whose Status is an owner label (FA and waiver rows are
    excluded) and maps export columns onto the shared output schema. No position
    filter — defensive (IDP) players are kept.
    """
    owned = df[df["status_class"] == "owner"]
    out = pd.DataFrame({
        "team_name": owned["Status"],
        "player_name": owned["Player"],
        "position": owned["Position"],
        "fantasy_points": "",
        "points_per_game": "",
        "fantrax_id": owned["fantrax_id"],
        "nfl_team": owned["Team"],
        "status": "owned",
        "age": owned["Age"],
    })
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def build_my_roster(all_rosters: pd.DataFrame, my_team: str) -> pd.DataFrame:
    """Filter the all_rosters table down to the user's own team."""
    mine = all_rosters[all_rosters["team_name"] == my_team]
    return mine.reset_index(drop=True)


# ── validation warnings / summary ────────────────────────────────────


def run_soft_checks(df: pd.DataFrame, my_roster: pd.DataFrame, my_team: str) -> None:
    """Print warnings for league-shape anomalies that should not fail bronze."""
    owners = sorted(df.loc[df["status_class"] == "owner", "Status"].unique())
    if len(owners) != EXPECTED_OWNER_COUNT:
        print(
            f"  WARNING: expected {EXPECTED_OWNER_COUNT} distinct owner labels, "
            f"found {len(owners)}: {owners}"
        )
    if not MY_ROSTER_MIN <= len(my_roster) <= MY_ROSTER_MAX:
        print(
            f"  WARNING: my_roster ({my_team}) has {len(my_roster)} rows — "
            f"expected between {MY_ROSTER_MIN} and {MY_ROSTER_MAX}"
        )


def print_summary(df: pd.DataFrame) -> None:
    """Print row counts by status class and per-team roster counts."""
    counts = df["status_class"].value_counts()
    print(f"\n  Total rows:   {len(df)}")
    print(f"  FA:           {counts.get('fa', 0)}")
    print(f"  Owned:        {counts.get('owner', 0)}")
    print(f"  Waiver:       {counts.get('waiver', 0)}")
    print("\n  Per-team counts:")
    team_counts = df.loc[df["status_class"] == "owner", "Status"].value_counts()
    for team, n in sorted(team_counts.items()):
        print(f"    {team}: {n}")


# ── entry point ──────────────────────────────────────────────────────


def run_import(input_path: pathlib.Path, stamp: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Run the full import: read, validate, split, and write both CSVs."""
    my_team = load_my_team_label()

    print(f"Reading Fantrax export: {input_path}")
    df = read_export(input_path)

    all_rosters = build_all_rosters(df)
    my_roster = build_my_roster(all_rosters, my_team)
    run_soft_checks(df, my_roster, my_team)

    FANTRAX_DIR.mkdir(parents=True, exist_ok=True)
    all_path = FANTRAX_DIR / f"all_rosters_{stamp}.csv"
    my_path = FANTRAX_DIR / f"my_roster_{stamp}.csv"
    all_rosters.to_csv(all_path, index=False, encoding="utf-8")
    my_roster.to_csv(my_path, index=False, encoding="utf-8")

    print_summary(df)
    print(f"\n  Wrote {all_path} ({len(all_rosters)} rows)")
    print(f"  Wrote {my_path} ({len(my_roster)} rows)")
    return all_path, my_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Import a manual Fantrax Players export into football bronze.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=pathlib.Path,
        help="Path to the downloaded Fantrax Players export CSV.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date stamp for output filenames (YYYY-MM-DD, default today).",
    )
    args = parser.parse_args(argv)
    try:
        date.fromisoformat(args.date)
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    return args


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: import the export given by --input."""
    # Owner labels can contain emoji a cp1252 Windows console can't encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(argv)
    run_import(args.input, args.date)


if __name__ == "__main__":
    main()
