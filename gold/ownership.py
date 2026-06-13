"""Resolve Fantrax ownership status for gold-layer player pools.

``silver/player_id_map.py`` links every Fantrax player — rostered and the
live free-agent pool — to Baseball Savant and FanGraphs IDs and tags each
row ``status`` 'owned' or 'fa'.  Gold modules attach that status by joining
on the RESOLVED VENDOR IDS (Savant first, FanGraphs as fallback) rather
than on raw names, so Fantrax two-way suffixes ("Shohei Ohtani-P") and
accent mismatches can never leak a rostered player into an "available"
pool.

A player with no id_map row in either source is deliberately treated as
NOT available: an entry we cannot resolve to a known Fantrax identity is
not one we can confidently call a free agent.
"""

import pathlib

import pandas as pd

SILVER_DIR = pathlib.Path(__file__).resolve().parent.parent / "silver" / "data"
ID_MAP_PATH = SILVER_DIR / "player_id_map.parquet"

_ID_COLUMNS = ("savant_player_id", "fangraphs_id")


def load_id_map() -> pd.DataFrame:
    """Load the silver player ID map.

    Raises:
        FileNotFoundError: If the map has not been built yet.
    """
    if not ID_MAP_PATH.exists():
        raise FileNotFoundError(
            f"{ID_MAP_PATH} not found. Run `python -m silver.player_id_map` first."
        )
    return pd.read_parquet(ID_MAP_PATH)


def _status_by_id(
    id_map: pd.DataFrame, id_col: str
) -> tuple[dict[int, str], dict[int, str]]:
    """Map each resolved vendor id to its (status, fantrax_team_name).

    A single vendor id can appear on more than one id_map row: a two-way
    player (Ohtani) shares one MLBAM/FanGraphs id across his Hitter and
    Pitcher rows.  'owned' wins over 'fa' on collision so a player rostered
    under either role is never reported as available.

    Args:
        id_map: The silver player ID map.
        id_col: Vendor id column to key on.

    Returns:
        ``(status_by_id, team_by_id)`` keyed by the integer vendor id.
    """
    sub = id_map[id_map[id_col].notna()].copy()
    # owned sorts before fa, so drop_duplicates(keep="first") resolves a
    # shared id to the owned row whenever one exists.
    sub = sub.sort_values("status", key=lambda s: s.ne("owned"), kind="stable")
    sub = sub.drop_duplicates(subset=[id_col], keep="first")

    status_by_id = {int(k): v for k, v in zip(sub[id_col], sub["status"])}
    team_by_id = {
        int(k): (v if isinstance(v, str) else "")
        for k, v in zip(sub[id_col], sub["fantrax_team_name"])
    }
    return status_by_id, team_by_id


def attach_status(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``status`` and ``fantrax_team_name`` via resolved vendor ids.

    Joins on ``savant_player_id``, falling back to ``fangraphs_id`` where
    Savant is null or unmatched.  Rows with no id_map match in either source
    get ``status`` ``<NA>`` — callers treat those as not available.

    Args:
        df: A frame carrying ``savant_player_id`` and ``fangraphs_id``
            columns (e.g. a silver Statcast table).

    Returns:
        A copy of *df* with ``status`` and ``fantrax_team_name`` columns.
    """
    missing = [c for c in _ID_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"attach_status requires {_ID_COLUMNS}; missing {missing}"
        )

    id_map = load_id_map()
    sav_status, sav_team = _status_by_id(id_map, "savant_player_id")
    fg_status, fg_team = _status_by_id(id_map, "fangraphs_id")

    statuses: list[object] = []
    teams: list[object] = []
    for sav, fg in zip(df["savant_player_id"], df["fangraphs_id"]):
        if pd.notna(sav) and int(sav) in sav_status:
            statuses.append(sav_status[int(sav)])
            teams.append(sav_team[int(sav)])
        elif pd.notna(fg) and int(fg) in fg_status:
            statuses.append(fg_status[int(fg)])
            teams.append(fg_team[int(fg)])
        else:
            statuses.append(pd.NA)
            teams.append(pd.NA)

    out = df.copy()
    out["status"] = statuses
    out["fantrax_team_name"] = teams
    return out


def available_players(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return only the resolved free agents in *df*, logging what was cut.

    Args:
        df: A frame carrying the vendor id columns.
        label: Short name for the pool, used in the printed summary.

    Returns:
        The subset of *df* whose resolved ``status`` is 'fa', with the
        ``status`` and ``fantrax_team_name`` helper columns dropped so the
        caller's output schema is unchanged.
    """
    attached = attach_status(df)
    n_owned = int((attached["status"] == "owned").sum())
    n_no_idmap = int(attached["status"].isna().sum())
    fa = attached[attached["status"] == "fa"].drop(
        columns=["status", "fantrax_team_name"]
    )
    print(
        f"  Ownership filter ({label}): {len(fa)} available | "
        f"{n_owned} rostered excluded | {n_no_idmap} with no id_map row excluded"
    )
    return fa
