"""Staleness warning for the bronze FanGraphs CSVs consumed by silver modules.

The FanGraphs bronze client can fail (Cloudflare 403s the leaderboard
endpoint) while older date-stamped CSVs remain on disk.  Silver loaders
fall back to the newest file present, so without this check stale data
flows downstream silently.  Warn only — staleness never fails the pipeline.
"""

import pathlib
import re
from datetime import date

MAX_AGE_DAYS = 7

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def warn_if_stale_fangraphs(path: pathlib.Path) -> None:
    """Print a one-line warning when a dated FanGraphs CSV is older than a week.

    Args:
        path: Bronze FanGraphs CSV following the ``YYYY-MM-DD_<suffix>.csv``
            naming convention.  Files without a parseable date are skipped.
    """
    match = _DATE_RE.search(path.name)
    if match is None:
        return
    try:
        stamp = date.fromisoformat(match.group())
    except ValueError:
        return
    age_days = (date.today() - stamp).days
    if age_days > MAX_AGE_DAYS:
        print(
            f"  WARNING: FanGraphs data ({path.name}) is {age_days} days old ({stamp}). "
            "Match rates and scores using FanGraphs columns may be stale; "
            "refresh the FanGraphs pull."
        )
