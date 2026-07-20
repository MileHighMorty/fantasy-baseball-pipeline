"""Master player identity resolution for the ShadyNasty football pipeline.

Football fork of the baseball silver/player_id_map.py. The resolution
ARCHITECTURE is preserved verbatim — it was always the sport-agnostic
centerpiece:

    * NAME is the only fuzzy-scored field (rapidfuzz token_sort_ratio on
      accent-stripped, whitespace-normalized names).
    * A hard TYPE GATE segregates candidate pools so a name can only match a
      same-type row.
    * TEAM breaks exact score ties only — never rejects or penalizes a match.
    * Manual overrides (overrides/player_name_overrides.csv) outrank the fuzzy
      matcher for known-bad names (force by id, or block with the NONE sentinel).
    * Output is keyed by res_key, a surrogate primary key assigned once at first
      sight, persisted in silver/data/player_master.csv, NEVER reused or
      renumbered, and — critically — an id already on a master row is NEVER
      overwritten by a later probabilistic match (the baseball "Joscar/Teoscar"
      collision fix; corrections go through the override CSV).

THE ONE CHANGE FROM BASEBALL — the type gate.
    Baseball gated on Hitter vs Pitcher (so two-way Ohtani matched each stat
    source only against its own type). Football gates on POSITION GROUP:

        Offense = {QB, RB, WR, TE}      Defense = {DL, LB, DB, ...}

    An offensive player is only ever matched against offensive stat sources and
    a defensive player only against defensive ones. This is the football
    equivalent of the two-way defense: an offensive "Josh Allen" (QB) can never
    fuzzy-match a defensive "Josh Allen" (LB) — they land in disjoint candidate
    pools and are mastered under separate res_keys.

BUILD PHASE — roster-only until the nflverse bronze layer exists.
    The external football stat sources (nflverse / PFR) are NOT built yet (that
    is the next phase, deliberately deferred for review). Their loaders below are
    honest stubs returning empty frames. Until they exist, this module runs
    roster-only: it fuzzy-matches nothing externally but still registers every
    Fantrax player into the res_key master keyed by their (stable) Fantrax id and
    position group. External stat ids fill in later via the same immutable-master
    rule — no id is invented, and no field is claimed that a live source has not
    produced.

Outputs:
    football/silver/data/player_id_map.parquet
    football/silver/data/player_master.csv       (persistent res_key registry)
    football/silver/data/match_review_queue.csv  (once stat sources exist)
"""

import hashlib
import pathlib
import re
import sys
import unicodedata
from datetime import date

import pandas as pd
from rapidfuzz import fuzz, process

# ── paths ──────────────────────────────────────────────────────────────

BRONZE_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
FANTRAX_DIR = BRONZE_DIR / "fantrax"
MASTER_PATH = DATA_DIR / "player_master.csv"
OVERRIDES_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "overrides" / "player_name_overrides.csv"
)

MATCH_THRESHOLD = 90
REVIEW_FLOOR = 75
HIGH_CONFIDENCE_FLOOR = 85

# ── the type gate: football position groups ─────────────────────────────
# The single hard gate that keeps an offensive player's row from ever matching
# a defensive stat candidate (and vice versa). Slot names in the league are
# DL/LB/DB, but Fantrax position ELIGIBILITY may use finer NFL tokens, so both
# sets are broad. Any token in neither set is surfaced (never silently bucketed)
# so the real Fantrax football vocabulary can be confirmed from a live pull and
# these sets refined — see _position_group and the unknown-token warning.
_OFFENSE_POSITIONS = {"QB", "RB", "FB", "WR", "TE"}
_DEFENSE_POSITIONS = {
    "DL", "DE", "DT", "NT", "EDGE",
    "LB", "ILB", "OLB", "MLB",
    "DB", "CB", "S", "SS", "FS",
}

# External stat sources, populated by the nflverse bronze phase (not yet built).
# Two independent sources, each contributing its own id — the structural analog
# of Savant + FanGraphs. Each source row MUST carry a position_group so the type
# gate applies to candidates too.
_SOURCES = ("nflverse", "pfr")
_SOURCE_ID_COL = {"nflverse": "nflverse_id", "pfr": "pfr_id"}

# Fantrax NFL-team abbreviations that may diverge from nflverse/PFR convention.
# Left empty until confirmed against a live pull — do not guess mappings.
_FANTRAX_TEAM_TO_STD: dict[str, str] = {}


def _normalize_fantrax_team(raw: str | None) -> str | None:
    """Return a Fantrax NFL-team abbreviation in the standard convention.

    Treats Fantrax's empty and "(N/A)" placeholders (a player with no NFL team,
    e.g. a free agent) as no team.
    """
    if not raw or raw == "(N/A)":
        return None
    return _FANTRAX_TEAM_TO_STD.get(raw, raw)


# ── helpers ────────────────────────────────────────────────────────────


def _latest_csv(directory: pathlib.Path, prefix: str) -> pathlib.Path | None:
    """Return the most recent CSV matching ``<prefix>_*.csv``, or None."""
    matches = sorted(directory.glob(f"{prefix}_*.csv"))
    return matches[-1] if matches else None


def _canonical_name(name: str) -> str:
    """Return the display name, trimmed. Accents preserved (human-facing)."""
    return " ".join(str(name).strip().split())


def _normalize_name(name: str) -> str:
    """Strip accents and collapse whitespace for fuzzy comparison."""
    decomposed = unicodedata.normalize("NFD", _canonical_name(name))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(stripped.split())


def _make_fantrax_id(player_name: str, position_group: str) -> str:
    """Deterministic fallback id from name + group.

    Only used when a Fantrax row has no real scorerId (should not happen on the
    live/FA/export feeds, which all carry one). The real Fantrax id is preferred
    as the spine because, unlike baseball, football has no two-way players, so a
    single stable id per player is correct.
    """
    key = f"{player_name.strip().lower()}|{position_group}"
    return "ftx_" + hashlib.md5(key.encode()).hexdigest()[:12]


# module-level accumulator for position tokens seen in neither group, so the
# live Fantrax football vocabulary can be reported and the sets refined.
_unknown_position_tokens: set[str] = set()


def _position_group(position: str) -> str:
    """Classify a (possibly multi-eligible) position string into a group.

    Returns "Offense", "Defense", or "Unknown". A player whose eligibility
    tokens fall purely in one set gets that group. Tokens in neither set (a
    vocabulary we have not confirmed for football yet) yield "Unknown" and are
    recorded — the module refuses to GUESS a side of the ball, because a wrong
    guess is exactly the failure the gate exists to prevent. An Unknown player
    matches no external source (empty candidate pool), so it can never inherit
    another player's stats; it is still registered in the master by its Fantrax
    id and surfaced for a human to resolve.
    """
    tokens = [t.strip().upper() for t in str(position).replace("/", ",").split(",") if t.strip()]
    is_off = any(t in _OFFENSE_POSITIONS for t in tokens)
    is_def = any(t in _DEFENSE_POSITIONS for t in tokens)
    if is_off and not is_def:
        return "Offense"
    if is_def and not is_off:
        return "Defense"
    # Contradictory (both) or unrecognized (neither): do not guess.
    for t in tokens:
        if t not in _OFFENSE_POSITIONS and t not in _DEFENSE_POSITIONS:
            _unknown_position_tokens.add(t)
    return "Unknown"


# ── loaders ────────────────────────────────────────────────────────────

_FANTRAX_COLUMNS = [
    "fantrax_id", "player_name", "position", "position_group",
    "fantrax_team_name", "nfl_team", "status",
]


def _load_fantrax_players() -> pd.DataFrame:
    """Load all Fantrax roster players with their position group.

    The real Fantrax scorerId (present on every feed) is kept as the identity
    spine. my_roster is a subset of all_rosters, so the two are concatenated and
    de-duplicated on fantrax_id.

    Returns DataFrame with :data:`_FANTRAX_COLUMNS`.
    """
    my_path = _latest_csv(FANTRAX_DIR, "my_roster")
    all_path = _latest_csv(FANTRAX_DIR, "all_rosters")

    frames = []
    if all_path is not None:
        frames.append(pd.read_csv(all_path, dtype={"fantrax_id": str}))
    if my_path is not None:
        frames.append(pd.read_csv(my_path, dtype={"fantrax_id": str}))
    if not frames:
        raise FileNotFoundError(f"No Fantrax roster CSVs found in {FANTRAX_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    for col in ("player_name", "position", "team_name"):
        if col in combined.columns:
            combined[col] = combined[col].astype(str).str.strip()

    if "nfl_team" not in combined.columns:
        combined["nfl_team"] = pd.NA
    combined["nfl_team"] = combined["nfl_team"].astype("string").str.strip()

    # Drop empty slots.
    combined = combined[
        combined["player_name"].notna()
        & (combined["player_name"] != "None")
        & (combined["player_name"] != "")
        & (combined["player_name"] != "nan")
    ].copy()

    combined["position_group"] = combined["position"].apply(_position_group)

    # Prefer the real Fantrax id; synthesize only if absent.
    if "fantrax_id" not in combined.columns:
        combined["fantrax_id"] = pd.NA
    missing_id = combined["fantrax_id"].isna() | (combined["fantrax_id"].astype(str) == "")
    combined.loc[missing_id, "fantrax_id"] = combined.loc[missing_id].apply(
        lambda r: _make_fantrax_id(r["player_name"], r["position_group"]), axis=1
    )

    # Rows WITH an nfl_team win the dedupe (my_roster export may lack it).
    combined = combined.sort_values(
        "nfl_team", na_position="last", kind="stable"
    ).drop_duplicates(subset=["fantrax_id"], keep="first")

    combined = combined.rename(columns={"team_name": "fantrax_team_name"})
    if "fantrax_team_name" not in combined.columns:
        combined["fantrax_team_name"] = ""
    combined["status"] = "owned"

    return combined[_FANTRAX_COLUMNS].reset_index(drop=True)


def _load_free_agents() -> pd.DataFrame:
    """Load the latest free-agent pull shaped like the roster frame.

    Returns an empty frame when no free_agents CSV exists — the matcher then
    runs roster-only.
    """
    fa_path = _latest_csv(FANTRAX_DIR, "free_agents")
    if fa_path is None:
        return pd.DataFrame(columns=_FANTRAX_COLUMNS)

    df = pd.read_csv(fa_path, dtype={"fantrax_id": str})
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df = df[(df["player_name"] != "") & (df["player_name"] != "nan")].copy()
    df = df[df["status"] == "fa"]
    df = df.drop_duplicates(subset=["fantrax_id"], keep="first")

    df["position_group"] = df["position"].apply(_position_group)
    df["fantrax_team_name"] = ""
    if "nfl_team" not in df.columns:
        df["nfl_team"] = ""
    df["nfl_team"] = df["nfl_team"].fillna("")
    return df[_FANTRAX_COLUMNS].reset_index(drop=True)


def _load_stat_source(source: str) -> pd.DataFrame:
    """Load an external football stat source (nflverse / PFR). STUB.

    Not yet built — the nflverse bronze layer is the next phase. Returns an empty
    frame with the expected shape so the matcher runs roster-only until then.
    When implemented, each row MUST carry:
        player_name, <source>_id, position_group, team
    with position_group already classified (Offense/Defense) so the type gate
    applies to candidates. No field is asserted here that a live source has not
    produced.
    """
    id_col = _SOURCE_ID_COL[source]
    return pd.DataFrame(columns=["player_name", id_col, "position_group", "team"])


def _load_overrides() -> dict[tuple[str, str, str], int | None]:
    """Load manual match overrides, creating an empty template if absent.

    CSV columns: fantrax_name, position_group, source, source_id, note.
    A numeric source_id FORCES that mapping; the literal "NONE" BLOCKS matching.
    source may be "all"/"*" to block every source at once (a non-entity is a
    non-entity everywhere); forcing with source=all is rejected (a forced id is
    per-source).

    Returns:
        {(normalized fantrax name, position_group, source): source_id or None}.
    """
    if not OVERRIDES_PATH.exists():
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDES_PATH.write_text(
            "fantrax_name,position_group,source,source_id,note\n", encoding="utf-8"
        )
        return {}

    df = pd.read_csv(OVERRIDES_PATH, dtype=str, keep_default_na=False)
    overrides: dict[tuple[str, str, str], int | None] = {}
    for _, row in df.iterrows():
        source = row["source"].strip().lower()
        raw_id = row["source_id"].strip()
        blocked = raw_id.upper() == "NONE"
        if source in ("all", "*"):
            if not blocked:
                print(f"  WARNING: override for {row['fantrax_name']} forces an ID "
                      f"with source 'all' — a forced ID is per-source; skipped")
                continue
            targets: tuple[str, ...] = _SOURCES
        elif source in _SOURCES:
            targets = (source,)
        else:
            print(f"  WARNING: override for {row['fantrax_name']} has unknown source "
                  f"'{row['source']}' — skipped")
            continue
        name_key = _normalize_name(row["fantrax_name"]).lower()
        position_group = row["position_group"].strip()
        for target in targets:
            overrides[(name_key, position_group, target)] = None if blocked else int(raw_id)
    return overrides


# ── matching ───────────────────────────────────────────────────────────


def _prep_pools(source: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Split a source frame into per-position-group candidate pools.

    The type gate: an Offense Fantrax row only ever sees the Offense candidate
    pool and a Defense row only the Defense pool, so a same-name offensive and
    defensive player can never collide. Unknown-group players get no pool and so
    match nothing (safe).

    Returns:
        {position_group: (subframe, normalized candidate names)}.
    """
    pools = {}
    for group in ("Offense", "Defense"):
        if "position_group" in source.columns:
            sub = source[source["position_group"] == group].reset_index(drop=True)
        else:
            sub = source.iloc[0:0]
        pools[group] = (sub, [_normalize_name(n) for n in sub["player_name"]])
    return pools


def _best_match(
    name_norm: str,
    pool: tuple[pd.DataFrame, list[str]],
    fantrax_team: str | None = None,
    tiebreak_log: list[str] | None = None,
) -> tuple[float, str | None, int | None]:
    """Score a normalized name against a candidate pool with NO cutoff.

    The sub-threshold best score is the near-miss signal. When several
    candidates TIE on the top score and the pool carries a ``team`` column, the
    candidate matching the player's Fantrax nfl_team wins the tie. Team is a
    tiebreaker ONLY — it never rejects or penalizes a match.

    Returns:
        (best_score, best_candidate_name, best_index), or (0.0, None, None) for
        an empty pool.
    """
    sub, names_norm = pool
    if not names_norm:
        return 0.0, None, None

    results = process.extract(
        name_norm, names_norm, scorer=fuzz.token_sort_ratio, limit=10
    )
    top_score = results[0][1]
    tied = [r for r in results if r[1] == top_score]
    idx = tied[0][2]

    if len(tied) > 1 and fantrax_team and "team" in sub.columns:
        for _, _, cand_idx in tied:
            if sub.iloc[cand_idx]["team"] == fantrax_team:
                if cand_idx != idx and tiebreak_log is not None:
                    tiebreak_log.append(
                        f"{name_norm}: {len(tied)}-way tie at {top_score:.1f}, "
                        f"chose '{sub.iloc[cand_idx]['player_name']}' ({fantrax_team}) "
                        f"over '{sub.iloc[idx]['player_name']}'"
                    )
                idx = cand_idx
                break

    return round(float(top_score), 1), sub.iloc[idx]["player_name"], idx


def _classify_score(score: float) -> str:
    """Map a best-candidate score to a match class."""
    if score >= 100:
        return "exact"
    if score >= MATCH_THRESHOLD:
        return "fuzzy"
    if score >= HIGH_CONFIDENCE_FLOOR:
        return "fuzzy_miss_high"
    if score >= REVIEW_FLOOR:
        return "fuzzy_miss_low"
    return "no_candidate"


def _apply_override(
    overrides: dict[tuple[str, str, str], int | None],
    fantrax_name: str,
    position_group: str,
    source: str,
    fuzzy_id: int | None,
    fuzzy_class: str,
) -> tuple[int | None, str]:
    """Apply a manual override on top of a fuzzy-match result.

    force-match (numeric id) -> (override_id, 'override'); block (None) ->
    (None, 'no_candidate'); no override -> the fuzzy result unchanged.
    """
    key = (_normalize_name(fantrax_name).lower(), position_group, source)
    if key not in overrides:
        return fuzzy_id, fuzzy_class
    forced = overrides[key]
    if forced is None:
        return None, "no_candidate"
    return forced, "override"


def build_player_id_map() -> pd.DataFrame:
    """Build the master player ID map.

    For every Fantrax player, fuzzy-match against the type-correct (position
    group) candidate pool of each external stat source. Until those sources are
    built, all external matches are empty and the map registers roster identity
    only, which still feeds the res_key master.

    Returns:
        id_map DataFrame with columns: fantrax_id, player_name, team, position,
        position_group, fantrax_team_name, status, nflverse_id, pfr_id,
        match_quality, and per-source best_score / best_candidate / match_class.
    """
    print("Loading Fantrax rosters...")
    fantrax = _load_fantrax_players()
    print(f"  {len(fantrax)} Fantrax roster entries")

    free_agents = _load_free_agents()
    if not free_agents.empty:
        owned_ids = set(fantrax["fantrax_id"])
        before = len(free_agents)
        free_agents = free_agents[~free_agents["fantrax_id"].isin(owned_ids)]
        print(
            f"  {len(free_agents)} free agents added to the match pool "
            f"({before - len(free_agents)} dropped as roster overlap)"
        )
        fantrax = pd.concat([fantrax, free_agents], ignore_index=True)

    # External stat sources (stubs until the nflverse bronze phase lands).
    sources = {name: _load_stat_source(name) for name in _SOURCES}
    have_sources = any(not df.empty for df in sources.values())
    if not have_sources:
        print("  No external stat sources yet (nflverse/PFR bronze not built) — "
              "running roster-only identity registration.")
    pools = {name: _prep_pools(df) for name, df in sources.items()}

    overrides = _load_overrides()
    if overrides:
        print(f"  {len(overrides)} manual override(s) loaded")

    tiebreak_log: list[str] = []
    rows: list[dict] = []
    for _, ftx_row in fantrax.iterrows():
        name = ftx_row["player_name"]
        group = ftx_row["position_group"]
        name_norm = _normalize_name(name)
        nfl_team = ftx_row["nfl_team"] if pd.notna(ftx_row["nfl_team"]) else None

        record = {
            "fantrax_id": ftx_row["fantrax_id"],
            "player_name": name,
            "team": _normalize_fantrax_team(nfl_team),
            "position": ftx_row["position"],
            "position_group": group,
            "fantrax_team_name": ftx_row["fantrax_team_name"],
            "status": ftx_row["status"],
        }

        matched_classes: set[str] = set()
        for source in _SOURCES:
            id_col = _SOURCE_ID_COL[source]
            # Unknown-group players have no pool for either group -> no match.
            pool = pools[source].get(group)
            if pool is None:
                score, candidate, idx = 0.0, None, None
            else:
                score, candidate, idx = _best_match(
                    name_norm, pool, fantrax_team=nfl_team, tiebreak_log=tiebreak_log
                )
            match_class = _classify_score(score)
            resolved_id = None
            if idx is not None and score >= MATCH_THRESHOLD:
                sub, _ = pool
                resolved_id = int(sub.iloc[idx][id_col])
            resolved_id, match_class = _apply_override(
                overrides, name, group, source, resolved_id, match_class
            )
            if match_class in ("exact", "fuzzy", "override"):
                matched_classes.add(match_class)
            record[id_col] = resolved_id
            record[f"{source}_best_score"] = score
            record[f"{source}_best_candidate"] = candidate
            record[f"{source}_match_class"] = match_class

        if "exact" in matched_classes:
            record["match_quality"] = "exact"
        elif "override" in matched_classes:
            record["match_quality"] = "override"
        elif "fuzzy" in matched_classes:
            record["match_quality"] = "fuzzy"
        else:
            record["match_quality"] = "unmatched"

        rows.append(record)

    id_map = pd.DataFrame(rows)
    for id_col in _SOURCE_ID_COL.values():
        id_map[id_col] = pd.array(id_map[id_col], dtype=pd.Int64Dtype())

    if tiebreak_log:
        print(f"\n  Team tiebreaker applied ({len(tiebreak_log)}):")
        for entry in tiebreak_log:
            print(f"    - {entry}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "player_id_map.parquet"
    id_map.to_parquet(out_path, index=False)
    print(f"\nPlayer ID map saved to {out_path}")

    if have_sources:
        _write_review_queue(id_map)
    else:
        print("Match review queue skipped (no external stat sources loaded yet).")
    _update_player_master(id_map)

    return id_map


def _write_review_queue(id_map: pd.DataFrame) -> None:
    """Write the human-in-the-loop review queue CSV (fuzzy misses to inspect)."""
    review_classes = ("fuzzy_miss_high", "fuzzy_miss_low", "no_candidate")
    fuzzy_classes = ("fuzzy_miss_high", "fuzzy_miss_low")

    class_cols = [f"{s}_match_class" for s in _SOURCES]
    owned_mask = id_map["status"] == "owned"
    fa_mask = id_map["status"] == "fa"
    owned_review = owned_mask & pd.concat(
        [id_map[c].isin(review_classes) for c in class_cols], axis=1
    ).any(axis=1)
    fa_review = fa_mask & pd.concat(
        [id_map[c].isin(fuzzy_classes) for c in class_cols], axis=1
    ).any(axis=1)

    keep_cols = ["player_name", "position_group", "fantrax_team_name", "status"]
    for s in _SOURCES:
        keep_cols += [f"{s}_best_score", f"{s}_best_candidate", f"{s}_match_class"]
    queue = id_map.loc[owned_review | fa_review, keep_cols].copy()

    queue_path = DATA_DIR / "match_review_queue.csv"
    queue.to_csv(queue_path, index=False)
    print(f"Match review queue saved to {queue_path} ({len(queue)} players)")


# ── player master (res_key) ────────────────────────────────────────────

_MASTER_COLUMNS = [
    "res_key", "canonical_name", "position_group", "position",
    "nflverse_id", "pfr_id", "fantrax_id", "first_seen_date", "last_seen_date",
]
_MASTER_ID_COLUMNS = ("nflverse_id", "pfr_id", "fantrax_id")


def _update_player_master(id_map: pd.DataFrame) -> pd.DataFrame:
    """Upsert resolved players into the persistent res_key master.

    res_key is a surrogate key assigned once at first sight and never reused or
    renumbered. Identity is recognized by (position_group + any known source id).
    In the roster-only phase the Fantrax id is the spine (stable, one per player
    — football has no two-way players); external ids fill in later where a row is
    blank. An id ALREADY on a row is NEVER overwritten (corrections go through the
    override CSV). first_seen_date and res_key never change.
    """
    today = date.today().isoformat()

    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH, dtype={"fantrax_id": str})
        for col in ("nflverse_id", "pfr_id"):
            master[col] = pd.array(master[col], dtype=pd.Int64Dtype())
        records = master.to_dict("records")
    else:
        records = []

    id_index: dict[tuple[str, str, object], int] = {}

    def _index_record(pos: int) -> None:
        rec = records[pos]
        for col in _MASTER_ID_COLUMNS:
            val = rec.get(col)
            if val is not None and pd.notna(val):
                id_index[(rec["position_group"], col, val)] = pos

    for pos in range(len(records)):
        _index_record(pos)

    next_key = max((int(r["res_key"]) for r in records), default=0) + 1
    new_count = 0

    # Every Fantrax player has an id, so every player is registered.
    for _, row in id_map.iterrows():
        ids = {
            col: (int(row[col]) if col != "fantrax_id" and pd.notna(row[col])
                  else row[col] if col == "fantrax_id" and pd.notna(row[col]) else None)
            for col in _MASTER_ID_COLUMNS
        }
        pos = next(
            (
                id_index[(row["position_group"], col, val)]
                for col, val in ids.items()
                if val is not None and (row["position_group"], col, val) in id_index
            ),
            None,
        )
        if pos is None:
            records.append({
                "res_key": next_key,
                "canonical_name": _canonical_name(row["player_name"]),
                "position_group": row["position_group"],
                "position": row["position"],
                **ids,
                "first_seen_date": today,
                "last_seen_date": today,
            })
            _index_record(len(records) - 1)
            next_key += 1
            new_count += 1
        else:
            rec = records[pos]
            rec["last_seen_date"] = today
            for col, val in ids.items():
                current = rec.get(col)
                if val is not None and (current is None or pd.isna(current)):
                    rec[col] = val
            _index_record(pos)

    master = pd.DataFrame(records, columns=_MASTER_COLUMNS).sort_values("res_key")
    for col in ("nflverse_id", "pfr_id"):
        master[col] = pd.array(master[col], dtype=pd.Int64Dtype())
    master.to_csv(MASTER_PATH, index=False)
    print(
        f"Player master saved to {MASTER_PATH} "
        f"({len(master)} players, {new_count} new this run)"
    )
    return master


# ── lookup helpers ─────────────────────────────────────────────────────


def load_player_id_map() -> pd.DataFrame:
    """Load the pre-built player ID map from Parquet."""
    path = DATA_DIR / "player_id_map.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m football.silver.player_id_map` first."
        )
    return pd.read_parquet(path)


def get_player_data(
    player_name: str,
    position_group: str | None = None,
) -> pd.Series | None:
    """Look up a single player by name (and optional position group)."""
    id_map = load_player_id_map()
    mask = id_map["player_name"].str.lower() == player_name.strip().lower()
    if position_group:
        mask = mask & (id_map["position_group"] == position_group)
    matches = id_map.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


# ── quality report ─────────────────────────────────────────────────────


def print_quality_report(id_map: pd.DataFrame) -> None:
    """Print a match/registration summary.

    In the roster-only phase this is dominated by the position-group split — the
    direct confirmation that IDP (Defense) players came through the pull — plus
    the res_key registration count. Per-source match rates activate once the
    nflverse/PFR bronze sources exist.
    """
    owned = id_map[id_map["status"] == "owned"]
    fa = id_map[id_map["status"] == "fa"]

    print(f"\n{'=' * 60}")
    print("  Player ID Map — Quality Report (ShadyNasty / football)")
    print(f"{'=' * 60}")
    print(f"  Rostered players:  {len(owned)}")
    print(f"  Free agents:       {len(fa)}")

    print("\n  Position-group split (IDP sanity check):")
    for group in ("Offense", "Defense", "Unknown"):
        n_owned = int((owned["position_group"] == group).sum())
        n_fa = int((fa["position_group"] == group).sum())
        print(f"    {group:8s}  rostered={n_owned:4d}  free_agents={n_fa}")
    if int((owned["position_group"] == "Defense").sum()) == 0:
        print("    WARNING: zero rostered defensive players — IDP may not have loaded!")

    if _unknown_position_tokens:
        print("\n  Unrecognized position tokens (confirm Fantrax football "
              "vocabulary, then extend _OFFENSE/_DEFENSE_POSITIONS):")
        print(f"    {sorted(_unknown_position_tokens)}")

    externally_matched = int(
        (id_map["match_quality"] != "unmatched").sum()
    )
    print(f"\n  Externally matched (nflverse/PFR): {externally_matched}")
    if externally_matched == 0:
        print("    (expected 0 until the stat-source bronze layer is built)")
    print(f"{'=' * 60}")


# ── entry point ────────────────────────────────────────────────────────


def main() -> None:
    """Build the player ID map and print the quality report."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    id_map = build_player_id_map()
    print_quality_report(id_map)


if __name__ == "__main__":
    main()
