"""Master player identity resolution for the fantasy baseball pipeline.

Builds a single ID map that links every Fantrax player — rostered, and
the full free-agent pool when a free_agents bronze pull exists — to
their Baseball Savant and FanGraphs IDs via fuzzy name matching.  All
other modules should use this map instead of doing their own matching.
Rows carry status 'owned' or 'fa'.

Two-way players (e.g. Ohtani) who appear in Fantrax rosters as BOTH a
hitter and a pitcher get TWO separate rows — one per player_type — and
matching is type-segregated: a Hitter row only sees batting-source
candidates, a Pitcher row only pitching-source candidates.

Every player records the best candidate score per source even below the
match threshold, so unmatched players split into "fuzzy_miss_high" (a
real counterpart almost certainly exists), "fuzzy_miss_low" (possible
but often coincidental), and "no_candidate" (nothing plausible in the
source — prospect, IL stash, sub-qualification PA/IP).

Matching design — one scored field, one gate, one tiebreaker:
    * NAME is the only fuzzy-scored field.  Concatenating name+team+type
      into one fuzzy string would blur the signals: a wrong-team exact
      name and a right-team wrong name could score identically, and the
      resulting number is uninterpretable for the review bands.
    * PLAYER_TYPE is a hard gate (separate candidate pools), because a
      two-way player is the same name with genuinely different stat rows.
    * TEAM breaks exact score ties only.  It can never reject or penalize
      a match: Savant carries no team column at all and FanGraphs team
      goes stale between pulls (trades, DFAs), so as a scored field it
      would punish correct matches.  As a tiebreaker it costs nothing.
    * Manual overrides (overrides/player_name_overrides.csv) outrank the
      fuzzy matcher entirely for known-bad names; see _load_overrides.

The resolved output is keyed by res_key, a surrogate primary key spanning
vendor ID systems (the securities-mastering pattern): assigned once at
first sight, persisted in silver/data/player_master.csv, never reused or
renumbered, so downstream tables can join on it across runs even when a
vendor ID arrives late or a name spelling changes.

Outputs:
    silver/data/player_id_map.parquet
    silver/data/player_master.csv       (persistent res_key registry)
    silver/data/match_review_queue.csv  (human-in-the-loop review of
        likely fuzzy misses and confirmed-absent players)
"""

import hashlib
import pathlib
import re
import sys
import unicodedata
from datetime import date

import pandas as pd
from rapidfuzz import fuzz, process

from silver.freshness import warn_if_stale_fangraphs

# ── paths ──────────────────────────────────────────────────────────────

BRONZE_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data"
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
FANTRAX_DIR = pathlib.Path(__file__).resolve().parent.parent / "bronze" / "data" / "fantrax"
MASTER_PATH = DATA_DIR / "player_master.csv"
OVERRIDES_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "overrides" / "player_name_overrides.csv"
)

MATCH_THRESHOLD = 90

# Unmatched players whose best candidate scores in [REVIEW_FLOOR, MATCH_THRESHOLD)
# may have a real counterpart the fuzzy matcher missed; below REVIEW_FLOOR nothing
# plausible exists in the source at all (prospect, IL stash, player under the
# leaderboard's qualification floor).  Heuristic band, not ground truth —
# borderline names land in the review queue.
REVIEW_FLOOR = 75

# Splits the fuzzy-miss band by confidence: at or above HIGH_CONFIDENCE_FLOOR the
# best candidate is almost always the same player obscured by an accent, middle
# initial, or nickname ("Jose Ferrer" vs "Jose A. Ferrer", "Cam Schlittler" vs
# "Cameron Schlittler"); below it the near-name is frequently a coincidence
# ("Carlos Correa" vs "Carlos Cortes").  Only the high band is treated as the
# actionable review queue in the quality report.
HIGH_CONFIDENCE_FLOOR = 85

_PITCHER_POSITIONS = {"SP", "RP", "P"}
_STAT_TYPE_TO_PLAYER_TYPE = {"batting": "Hitter", "pitching": "Pitcher"}

# Fantrax MLB-team abbreviations that differ from FanGraphs/Savant convention.
# Used when falling back to the Fantrax team for a FanGraphs-absent player, so
# the id_map's team column stays internally consistent (mirrors the dashboard's
# _MLB_TO_SAVANT_ABBR). Only the five abbreviations that actually diverge in the
# data are listed; every other team already agrees across both sources.
_FANTRAX_TEAM_TO_FG = {
    "KC": "KCR", "SD": "SDP", "SF": "SFG", "TB": "TBR", "WSH": "WSN",
}


def _normalize_fantrax_team(raw: str | None) -> str | None:
    """Return a Fantrax MLB-team abbreviation in FanGraphs/Savant convention.

    Maps the five divergent codes (TB->TBR, etc.) and treats Fantrax's empty
    and "(N/A)" placeholders (a player with no MLB team) as no team.

    Args:
        raw: The Fantrax ``mlb_team`` value, possibly empty or "(N/A)".

    Returns:
        The normalized team abbreviation, or ``None`` when there is no team.
    """
    if not raw or raw == "(N/A)":
        return None
    return _FANTRAX_TEAM_TO_FG.get(raw, raw)


# ── helpers ────────────────────────────────────────────────────────────


def _latest_csv(directory: pathlib.Path, prefix: str) -> pathlib.Path | None:
    """Return the most recent CSV matching ``<prefix>_*.csv``, or None."""
    matches = sorted(directory.glob(f"{prefix}_*.csv"))
    return matches[-1] if matches else None


def _canonical_name(name: str) -> str:
    """Return the display name with Fantrax two-way markers removed.

    Fantrax suffixes a two-way player's roster name per role ("Shohei
    Ohtani-H" / "Shohei Ohtani-P"), which no stats source uses.  Only a
    trailing bare "-H"/"-P" is removed — real hyphenated surnames and
    generational suffixes ("Lombard Jr.") have more after the hyphen or no
    hyphen at all, and are kept intact.  Accents are preserved here; this
    is the human-facing spelling used in the player master.
    """
    stripped = re.sub(r"-[HP]$", "", name.strip(), flags=re.IGNORECASE)
    return " ".join(stripped.split())


def _normalize_name(name: str) -> str:
    """Strip accents, Fantrax two-way markers, and extra whitespace.

    The two-way marker costs enough fuzzy points to demote an
    otherwise-exact match (see _canonical_name for the suffix rules).
    """
    decomposed = unicodedata.normalize("NFD", _canonical_name(name))
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return " ".join(stripped.split())


def _make_fantrax_id(player_name: str, player_type: str) -> str:
    """Generate a deterministic Fantrax ID from name + type."""
    key = f"{player_name.strip().lower()}|{player_type}"
    return "ftx_" + hashlib.md5(key.encode()).hexdigest()[:12]


# ── loaders ────────────────────────────────────────────────────────────


def _load_fantrax_players() -> pd.DataFrame:
    """Load all Fantrax roster players, detecting two-way players.

    Returns DataFrame with columns:
        fantrax_id, player_name, position, player_type, fantrax_team_name
    """
    my_path = _latest_csv(FANTRAX_DIR, "my_roster")
    all_path = _latest_csv(FANTRAX_DIR, "all_rosters")

    frames = []
    if my_path is not None:
        df = pd.read_csv(my_path)
        if "team_name" not in df.columns:
            df["team_name"] = "Rutsch Hour"
        frames.append(df)
    if all_path is not None:
        frames.append(pd.read_csv(all_path))

    if not frames:
        raise FileNotFoundError(f"No Fantrax roster CSVs found in {FANTRAX_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined["player_name"] = combined["player_name"].str.strip()
    combined["position"] = combined["position"].str.strip()
    combined["team_name"] = combined["team_name"].str.strip()

    # mlb_team exists on the live Fantrax pull but not on the manual CSV
    # importer path (or the my_roster file) — used as a match tiebreaker
    # only, so absence is fine.
    if "mlb_team" not in combined.columns:
        combined["mlb_team"] = pd.NA
    combined["mlb_team"] = combined["mlb_team"].str.strip()

    # Drop empty slots
    combined = combined[
        combined["player_name"].notna()
        & (combined["player_name"] != "None")
        & (combined["player_name"] != "")
    ].copy()

    # Assign player_type.  Roster positions are now multi-position eligibility
    # strings ("SP,RP") like the FA pool, so use the same comma/slash-aware
    # classifier the FA path uses — an exact-membership test would misread
    # "SP,RP" as a non-pitcher.
    combined["player_type"] = combined["position"].apply(_player_type_from_position)

    # Deduplicate: keep one row per (player_name, player_type, team_name).
    # This naturally preserves two-way players as separate rows.  Sort so
    # duplicates WITH an mlb_team win (my_roster rows lack the column).
    combined = combined.sort_values(
        "mlb_team", na_position="last", kind="stable"
    ).drop_duplicates(subset=["player_name", "player_type", "team_name"], keep="first")

    # If a player appears on multiple teams for the SAME type, keep the
    # first (shouldn't happen in a well-formed league, but be safe)
    combined = combined.drop_duplicates(
        subset=["player_name", "player_type"], keep="first"
    )

    combined["fantrax_id"] = combined.apply(
        lambda r: _make_fantrax_id(r["player_name"], r["player_type"]), axis=1
    )
    combined = combined.rename(columns={"team_name": "fantrax_team_name"})
    combined["status"] = "owned"

    return combined[
        [
            "fantrax_id", "player_name", "position", "player_type",
            "fantrax_team_name", "mlb_team", "status",
        ]
    ].reset_index(drop=True)


def _player_type_from_position(position: str) -> str:
    """Classify a (possibly multi-eligible) position string.

    Free-agent rows can list several positions ("SP,RP", "1B,OF"): only a
    pure pitching listing classifies as Pitcher, since any hitting
    eligibility means the player's offensive row is the one worth matching.
    """
    tokens = [t.strip() for t in str(position).replace("/", ",").split(",") if t.strip()]
    if tokens and all(t in _PITCHER_POSITIONS for t in tokens):
        return "Pitcher"
    return "Hitter"


def _load_free_agents() -> pd.DataFrame:
    """Load the latest free-agent pull shaped like the roster frame.

    Returns an empty frame when no free_agents CSV exists (manual-import
    workflows have no FA pull) — the matcher then runs roster-only.
    Unlike roster rows, fantrax_id here is the real Fantrax scorerId
    straight from the API, not a name-derived hash.
    """
    columns = [
        "fantrax_id", "player_name", "position", "player_type",
        "fantrax_team_name", "mlb_team", "status",
    ]
    fa_path = _latest_csv(FANTRAX_DIR, "free_agents")
    if fa_path is None:
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(fa_path)
    df["player_name"] = df["player_name"].astype(str).str.strip()
    df = df[(df["player_name"] != "") & (df["player_name"] != "nan")].copy()
    df = df[df["status"] == "fa"]
    df = df.drop_duplicates(subset=["fantrax_id"], keep="first")

    df["player_type"] = df["position"].apply(_player_type_from_position)
    df["fantrax_team_name"] = ""
    df["mlb_team"] = df["mlb_team"].fillna("")
    return df[columns].reset_index(drop=True)


def _load_savant_players() -> pd.DataFrame:
    """Load Savant players from batting + pitching CSVs, tagged by player_type.

    Deduplication happens WITHIN each stat type, never across: a two-way
    player (Shohei Ohtani) legitimately has both a batting row and a
    pitching row under an identical name, and both must survive so each
    Fantrax row can match the type-correct source row.

    The `pa` column is kept as evidence of the leaderboard's qualification
    floor — players below it have no row here to match.

    Returns DataFrame with [player_name, savant_player_id, player_type, pa].
    """
    frames = []
    for stat_type in ("batting", "pitching"):
        csv_dir = BRONZE_DIR / "savant"
        matches = sorted(csv_dir.glob(f"*_{stat_type}.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[-1])
        df = df.drop_duplicates(subset=["player_id"])
        df["player_type"] = _STAT_TYPE_TO_PLAYER_TYPE[stat_type]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_name", "savant_player_id", "player_type", "pa"])

    combined = pd.concat(frames, ignore_index=True)

    # "Last, First" → "First Last"
    combined["player_name"] = combined["last_name, first_name"].apply(
        lambda x: " ".join(reversed(x.split(", ")))
    )
    combined = combined.rename(columns={"player_id": "savant_player_id"})
    return combined[
        ["player_name", "savant_player_id", "player_type", "pa"]
    ].reset_index(drop=True)


def _load_fangraphs_players() -> pd.DataFrame:
    """Load FanGraphs players from batting + pitching CSVs, tagged by player_type.

    As with Savant, dedup happens within each stat type only, so a two-way
    player (Shohei Ohtani) keeps one row per type for type-segregated
    matching instead of collapsing to whichever file loaded first.

    Returns DataFrame with [player_name, fangraphs_id, fg_team, player_type].
    """
    frames = []
    fg_dir = BRONZE_DIR / "fangraphs"
    for stat_type in ("batting", "pitching"):
        matches = sorted(fg_dir.glob(f"*_{stat_type}.csv"))
        if not matches:
            continue
        warn_if_stale_fangraphs(matches[-1])
        df = pd.read_csv(matches[-1], usecols=["IDfg", "Name", "Team"])
        df = df.rename(columns={"IDfg": "fangraphs_id", "Name": "player_name", "Team": "fg_team"})
        df = df.drop_duplicates(subset=["fangraphs_id"], keep="first")
        df["player_type"] = _STAT_TYPE_TO_PLAYER_TYPE[stat_type]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["player_name", "fangraphs_id", "fg_team", "player_type"])

    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def _load_overrides() -> dict[tuple[str, str, str], int | None]:
    """Load manual match overrides, creating an empty template if absent.

    overrides/player_name_overrides.csv holds human-confirmed resolutions
    for names the fuzzy matcher gets wrong (accents with middle initials,
    nicknames).  Two kinds of row, distinguished by source_id:

    * a numeric source_id FORCES that mapping regardless of fuzzy score
      (match_class='override');
    * the literal "NONE" BLOCKS matching — this player has no counterpart
      in that source no matter how well a name scores (match_class
      'no_candidate'), which is how look-alike names that clear the match
      threshold (e.g. FA "Stanly Alcantara" at 90.3 vs Sandy Alcantara)
      are pinned down as different people.  source may be "all" (or "*")
      to block every source with one row — the usual case, since a
      non-entity is a non-entity everywhere.  Forcing with source=all is
      rejected: a forced ID is inherently per-source.

    Either way a fix made once persists across every future run — the
    review queue surfaces the problem, a human adds one CSV row, and it
    never returns.

    Returns:
        {(normalized fantrax name, player_type, source): source_id or None}
        where source is 'savant' or 'fangraphs' and None means "blocked".
    """
    if not OVERRIDES_PATH.exists():
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        OVERRIDES_PATH.write_text(
            "fantrax_name,player_type,source,source_id,note\n", encoding="utf-8"
        )
        return {}

    # dtype=str + keep_default_na=False so the "NONE" sentinel (and any
    # odd id) reaches us verbatim instead of becoming NaN.
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
            targets: tuple[str, ...] = ("savant", "fangraphs")
        elif source in ("savant", "fangraphs"):
            targets = (source,)
        else:
            print(f"  WARNING: override for {row['fantrax_name']} has unknown source "
                  f"'{row['source']}' — skipped")
            continue
        name_key = _normalize_name(row["fantrax_name"]).lower()
        player_type = row["player_type"].strip()
        for target in targets:
            overrides[(name_key, player_type, target)] = None if blocked else int(raw_id)
    return overrides


# ── matching ───────────────────────────────────────────────────────────


def _prep_pools(source: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, list[str]]]:
    """Split a source frame into per-player_type candidate pools.

    Type segregation fixes the two-way name collision: Shohei Ohtani
    appears in BOTH the batting and pitching source files under an
    identical name, so a single name-only pool let his Fantrax Hitter row
    match the pitching source row at score 100 — a silent wrong match,
    not a miss.  Restricting each Fantrax row to candidates of its own
    player_type makes that impossible.

    Returns:
        {player_type: (subframe, normalized candidate names)}.
    """
    pools = {}
    for player_type in ("Hitter", "Pitcher"):
        sub = source[source["player_type"] == player_type].reset_index(drop=True)
        pools[player_type] = (sub, [_normalize_name(n) for n in sub["player_name"]])
    return pools


def _best_match(
    name_norm: str,
    pool: tuple[pd.DataFrame, list[str]],
    fantrax_team: str | None = None,
    tiebreak_log: list[str] | None = None,
) -> tuple[float, str | None, int | None]:
    """Score a normalized name against a candidate pool with NO cutoff.

    The sub-threshold best score is the near-miss signal: it is what
    separates "the matcher missed a real candidate" from "no counterpart
    exists in this source."

    When several candidates TIE on the top score and the pool carries a
    team column, the candidate matching the player's Fantrax mlb_team
    wins the tie.  Team is a tiebreaker ONLY — it never rejects or
    penalizes a match, because Savant has no team column at all and
    FanGraphs team goes stale between pulls; as a scored or filtered
    field it would punish correct matches.

    Returns:
        (best_score, best_candidate_name, best_index), or (0.0, None, None)
        when the pool is empty.
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

    if len(tied) > 1 and fantrax_team and "fg_team" in sub.columns:
        for _, _, cand_idx in tied:
            if sub.iloc[cand_idx]["fg_team"] == fantrax_team:
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
    """Map a best-candidate score to a match class.

    exact/fuzzy keep the original bands (matched at >= MATCH_THRESHOLD);
    unmatched players split on HIGH_CONFIDENCE_FLOOR into fuzzy_miss_high
    (a real candidate almost certainly exists) vs fuzzy_miss_low (possible
    but often coincidental), and below REVIEW_FLOOR into no_candidate
    (genuine absence from the source).
    """
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
    player_type: str,
    source: str,
    fuzzy_id: int | None,
    fuzzy_class: str,
) -> tuple[int | None, str]:
    """Apply a manual override on top of a fuzzy-match result.

    Pure decision function: given the loaded overrides and the fuzzy
    outcome for one player+source, return the final (resolved_id,
    match_class).  Three cases:

    * force-match: a numeric override id wins regardless of fuzzy score
      -> (override_id, 'override')
    * block: a None override (CSV source_id "NONE"; source 'all' rows
      are expanded per-source at load time) means this player matches
      nothing here no matter how well a name scores
      -> (None, 'no_candidate')
    * pass-through: no override for this player+source
      -> the fuzzy result unchanged
    """
    key = (_normalize_name(fantrax_name).lower(), player_type, source)
    if key not in overrides:
        return fuzzy_id, fuzzy_class
    forced = overrides[key]
    if forced is None:
        return None, "no_candidate"
    return forced, "override"


def build_player_id_map() -> tuple[pd.DataFrame, dict[str, int | None]]:
    """Build the master player ID map.

    For every Fantrax player, fuzzy-match against the type-correct Savant
    and FanGraphs candidate pools to find cross-source IDs.  Two-way
    players keep separate rows per player_type.

    Returns:
        A tuple of (id_map, savant_pa_floor) where id_map has columns:
        fantrax_id, player_name, team, position, player_type,
        fantrax_team_name, status ('owned' or 'fa'),
        savant_player_id, fangraphs_id, match_quality,
        savant_best_score, savant_best_candidate, savant_match_class,
        fangraphs_best_score, fangraphs_best_candidate,
        fangraphs_match_class — and savant_pa_floor maps player_type to
        the minimum `pa` observed in the Savant source (the empirical
        qualification floor), or None when that source is missing.
    """
    print("Loading Fantrax rosters...")
    fantrax = _load_fantrax_players()
    print(f"  {len(fantrax)} Fantrax roster entries")

    free_agents = _load_free_agents()
    if not free_agents.empty:
        # Roster wins on overlap: a player added since the FA snapshot
        # must not appear twice with conflicting status.
        owned_keys = {
            (_normalize_name(n), t)
            for n, t in zip(fantrax["player_name"], fantrax["player_type"])
        }
        before = len(free_agents)
        keep = [
            (_normalize_name(n), t) not in owned_keys
            for n, t in zip(free_agents["player_name"], free_agents["player_type"])
        ]
        free_agents = free_agents[keep]
        print(
            f"  {len(free_agents)} free agents added to the match pool "
            f"({before - len(free_agents)} dropped as roster overlap)"
        )
        fantrax = pd.concat([fantrax, free_agents], ignore_index=True)

    print("Loading Savant players...")
    savant = _load_savant_players()
    print(f"  {len(savant)} Savant players")

    print("Loading FanGraphs players...")
    try:
        fangraphs = _load_fangraphs_players()
        print(f"  {len(fangraphs)} FanGraphs players")
    except Exception:
        print("  FanGraphs data not available, skipping")
        fangraphs = pd.DataFrame(columns=["player_name", "fangraphs_id", "fg_team"])

    overrides = _load_overrides()
    if overrides:
        print(f"  {len(overrides)} manual override(s) loaded")

    # Candidate pools are split by player_type so two-way players (Ohtani)
    # can only match the source row of the matching type — see _prep_pools.
    sav_pools = _prep_pools(savant)
    fg_pools = _prep_pools(fangraphs)
    tiebreak_log: list[str] = []

    # Empirical qualification floor: the Savant leaderboard only carries
    # players above a PA threshold, so the minimum observed `pa` is direct
    # evidence that low-PA players have no row here to match.
    savant_pa_floor = {
        ptype: (int(sub["pa"].min()) if not sub.empty else None)
        for ptype, (sub, _) in sav_pools.items()
    }

    rows: list[dict] = []
    for _, ftx_row in fantrax.iterrows():
        name = ftx_row["player_name"]
        player_type = ftx_row["player_type"]
        name_norm = _normalize_name(name)
        mlb_team = ftx_row["mlb_team"] if pd.notna(ftx_row["mlb_team"]) else None

        # --- Match to Savant (type-correct pool, no cutoff) ---
        sav_score, sav_candidate, sav_idx = _best_match(
            name_norm, sav_pools[player_type], fantrax_team=mlb_team, tiebreak_log=tiebreak_log
        )
        sav_class = _classify_score(sav_score)
        savant_id = None
        if sav_score >= MATCH_THRESHOLD:
            sav_sub, _ = sav_pools[player_type]
            savant_id = int(sav_sub.iloc[sav_idx]["savant_player_id"])

        # Manual override outranks the fuzzy result for this player+source
        savant_id, sav_class = _apply_override(
            overrides, name, player_type, "savant", savant_id, sav_class
        )

        # --- Match to FanGraphs (type-correct pool, no cutoff) ---
        fg_score, fg_candidate, fg_idx = _best_match(
            name_norm, fg_pools[player_type], fantrax_team=mlb_team, tiebreak_log=tiebreak_log
        )
        fg_class = _classify_score(fg_score)
        fg_id = None
        fg_team = None
        if fg_score >= MATCH_THRESHOLD:
            fg_sub, _ = fg_pools[player_type]
            fg_id = int(fg_sub.iloc[fg_idx]["fangraphs_id"])
            fg_team = fg_sub.iloc[fg_idx]["fg_team"]

        fg_id, fg_class = _apply_override(
            overrides, name, player_type, "fangraphs", fg_id, fg_class
        )
        if fg_class == "override":
            # A forced FanGraphs id needs its team looked up directly
            ov_row = fangraphs[fangraphs["fangraphs_id"] == fg_id]
            fg_team = ov_row.iloc[0]["fg_team"] if not ov_row.empty else None
        elif fg_id is None:
            fg_team = None

        # Overall match quality: best of the two sources (unchanged bands;
        # override ranks between exact and fuzzy — human-confirmed, but
        # worth distinguishing from a clean automatic match)
        matched_classes = {
            c for c in (sav_class, fg_class) if c in ("exact", "fuzzy", "override")
        }
        if "exact" in matched_classes:
            match_quality = "exact"
        elif "override" in matched_classes:
            match_quality = "override"
        elif "fuzzy" in matched_classes:
            match_quality = "fuzzy"
        else:
            match_quality = "unmatched"

        # Team comes from FanGraphs (MLB team) when matched; for a player with
        # no FanGraphs row (deep prospects, etc.) fall back to the Fantrax MLB
        # team, normalized to FanGraphs convention so the column is consistent.
        team = fg_team if fg_team else _normalize_fantrax_team(mlb_team)

        rows.append({
            "fantrax_id": ftx_row["fantrax_id"],
            "player_name": name,
            "team": team,
            "position": ftx_row["position"],
            "player_type": player_type,
            "fantrax_team_name": ftx_row["fantrax_team_name"],
            "status": ftx_row["status"],
            "savant_player_id": savant_id,
            "fangraphs_id": fg_id,
            "match_quality": match_quality,
            "savant_best_score": sav_score,
            "savant_best_candidate": sav_candidate,
            "savant_match_class": sav_class,
            "fangraphs_best_score": fg_score,
            "fangraphs_best_candidate": fg_candidate,
            "fangraphs_match_class": fg_class,
        })

    id_map = pd.DataFrame(rows)

    # Coerce ID columns to nullable int
    for col in ("savant_player_id", "fangraphs_id"):
        id_map[col] = pd.array(id_map[col], dtype=pd.Int64Dtype())

    # Fill missing team from player_universe if available
    pu_path = DATA_DIR / "player_universe.parquet"
    if pu_path.exists():
        pu = pd.read_parquet(pu_path, columns=["player_name", "team"])
        pu = pu.drop_duplicates(subset=["player_name"], keep="first")
        missing_team = id_map["team"].isna()
        if missing_team.any():
            fill = id_map.loc[missing_team, ["player_name"]].merge(
                pu, on="player_name", how="left", suffixes=("", "_pu")
            )
            id_map.loc[missing_team, "team"] = fill["team"].values

    if tiebreak_log:
        print(f"\n  Team tiebreaker applied ({len(tiebreak_log)}):")
        for entry in tiebreak_log:
            print(f"    - {entry}")
    else:
        print("\n  Team tiebreaker: no score ties required it this run")

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "player_id_map.parquet"
    id_map.to_parquet(out_path, index=False)
    print(f"\nPlayer ID map saved to {out_path}")

    _write_review_queue(id_map)
    _update_player_master(id_map)

    return id_map, savant_pa_floor


def _write_review_queue(id_map: pd.DataFrame) -> None:
    """Write the human-in-the-loop review queue CSV.

    Contains every player who is unmatched in at least one source.
    fuzzy_miss rows are fixable matcher gaps; no_candidate rows confirm
    genuine absence from the source.

    Sorted by the best score among the UNMATCHED sources, descending:
    a genuine fixable miss (high score in a source that failed to match)
    rises to the top, while a player matched in one source and merely
    absent from the other sinks — their only unmatched score is the
    low no_candidate one.

    Free agents enter the queue only on a fuzzy miss: listing every
    minor leaguer absent from the MLB leaderboards (thousands of
    no_candidate rows) would bury the actionable rows a human review
    queue exists to surface.
    """
    review_classes = ("fuzzy_miss_high", "fuzzy_miss_low", "no_candidate")
    fuzzy_classes = ("fuzzy_miss_high", "fuzzy_miss_low")
    owned_review = (id_map["status"] == "owned") & (
        id_map["savant_match_class"].isin(review_classes)
        | id_map["fangraphs_match_class"].isin(review_classes)
    )
    fa_review = (id_map["status"] == "fa") & (
        id_map["savant_match_class"].isin(fuzzy_classes)
        | id_map["fangraphs_match_class"].isin(fuzzy_classes)
    )
    queue = id_map.loc[
        owned_review | fa_review,
        [
            "player_name", "player_type", "fantrax_team_name", "status",
            "savant_best_score", "savant_best_candidate", "savant_match_class",
            "fangraphs_best_score", "fangraphs_best_candidate", "fangraphs_match_class",
        ],
    ].copy()

    unmatched_scores = pd.concat(
        [
            queue["savant_best_score"].where(
                queue["savant_match_class"].isin(review_classes)
            ),
            queue["fangraphs_best_score"].where(
                queue["fangraphs_match_class"].isin(review_classes)
            ),
        ],
        axis=1,
    )
    queue = (
        queue.assign(_unmatched_best=unmatched_scores.max(axis=1))
        .sort_values("_unmatched_best", ascending=False)
        .drop(columns="_unmatched_best")
    )

    queue_path = DATA_DIR / "match_review_queue.csv"
    queue.to_csv(queue_path, index=False)
    print(f"Match review queue saved to {queue_path} ({len(queue)} players)")


# ── player master (res_key) ────────────────────────────────────────────

_MASTER_COLUMNS = [
    "res_key", "canonical_name", "player_type", "savant_player_id",
    "fangraphs_id", "fantrax_id", "first_seen_date", "last_seen_date",
]
_MASTER_ID_COLUMNS = ("savant_player_id", "fangraphs_id", "fantrax_id")


def _update_player_master(id_map: pd.DataFrame) -> pd.DataFrame:
    """Upsert resolved players into the persistent res_key master.

    res_key is a surrogate primary key spanning vendor ID systems — the
    securities-mastering pattern: each resolved real player receives a
    monotonically increasing integer at first sight, and that key is
    NEVER reused or renumbered.  Downstream tables join on res_key so
    they survive vendor ID gaps, name respellings, and late-arriving
    sources.

    Identity is recognized by (player_type + any known source ID).
    player_type is part of the identity because a two-way player's
    hitter and pitcher records are mastered separately — their shared
    MLBAM/FanGraphs ids must not collapse them into one row.  On a
    re-sighting, last_seen_date advances and newly resolved source IDs
    fill in where missing; an ID already on the row is NEVER overwritten
    (a borderline fuzzy false positive — e.g. FA "Stanly Alcantara"
    scoring 90.3 against Sandy Alcantara — must not corrupt the real
    player's row; corrections go through the override CSV).
    first_seen_date and res_key never change.
    """
    today = date.today().isoformat()

    if MASTER_PATH.exists():
        master = pd.read_csv(MASTER_PATH)
        for col in ("savant_player_id", "fangraphs_id"):
            master[col] = pd.array(master[col], dtype=pd.Int64Dtype())
        records = master.to_dict("records")
    else:
        records = []

    # (player_type, id_column, id_value) -> position in records
    id_index: dict[tuple[str, str, object], int] = {}

    def _index_record(pos: int) -> None:
        rec = records[pos]
        for col in _MASTER_ID_COLUMNS:
            val = rec.get(col)
            if val is not None and pd.notna(val):
                id_index[(rec["player_type"], col, val)] = pos

    for pos in range(len(records)):
        _index_record(pos)

    next_key = max((int(r["res_key"]) for r in records), default=0) + 1
    new_count = 0

    resolved = id_map[
        id_map["savant_player_id"].notna() | id_map["fangraphs_id"].notna()
    ]
    for _, row in resolved.iterrows():
        ids = {
            col: (int(row[col]) if col != "fantrax_id" and pd.notna(row[col])
                  else row[col] if col == "fantrax_id" else None)
            for col in _MASTER_ID_COLUMNS
        }
        pos = next(
            (
                id_index[(row["player_type"], col, val)]
                for col, val in ids.items()
                if val is not None and (row["player_type"], col, val) in id_index
            ),
            None,
        )
        if pos is None:
            records.append({
                "res_key": next_key,
                "canonical_name": _canonical_name(row["player_name"]),
                "player_type": row["player_type"],
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
    for col in ("savant_player_id", "fangraphs_id"):
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
            f"{path} not found. Run `python -m silver.player_id_map` first."
        )
    return pd.read_parquet(path)


def get_player_data(
    player_name: str,
    player_type: str | None = None,
) -> pd.Series | None:
    """Look up a single player by name (and optional type) from the ID map.

    Returns the full row as a Series, or None if not found.
    """
    id_map = load_player_id_map()
    mask = id_map["player_name"].str.lower() == player_name.strip().lower()
    if player_type:
        mask = mask & (id_map["player_type"] == player_type)
    matches = id_map.loc[mask]
    if matches.empty:
        return None
    return matches.iloc[0]


def enrich_with_fantrax(
    df: pd.DataFrame,
    name_col: str = "player_name",
) -> pd.DataFrame:
    """LEFT JOIN the ID map onto *df* to add team, position, fantrax_team_name.

    Joins on player name.  Existing columns in *df* are NOT overwritten.
    """
    try:
        id_map = load_player_id_map()
    except FileNotFoundError:
        return df

    # Deduplicate the map — for enrichment, prefer Hitter rows (more common)
    lookup = id_map.drop_duplicates(subset=["player_name"], keep="first")

    cols_to_add = ["team", "position", "fantrax_team_name", "savant_player_id", "fangraphs_id"]
    cols_to_add = [c for c in cols_to_add if c not in df.columns]
    if not cols_to_add:
        return df

    merged = df.merge(
        lookup[["player_name"] + cols_to_add],
        left_on=name_col,
        right_on="player_name",
        how="left",
        suffixes=("", "_idmap"),
    )

    # Clean up duplicate player_name column if name_col differs
    if name_col != "player_name" and "player_name_idmap" in merged.columns:
        merged = merged.drop(columns=["player_name_idmap"])
    elif name_col != "player_name" and "player_name" in merged.columns:
        extra_cols = [c for c in merged.columns if c.endswith("_idmap")]
        merged = merged.drop(columns=extra_cols, errors="ignore")

    return merged


# ── quality report ─────────────────────────────────────────────────────


def _print_source_section(
    id_map: pd.DataFrame,
    label: str,
    prefix: str,
    pa_floor: dict[str, int | None] | None = None,
) -> None:
    """Print raw vs matchable match rates and bucket counts for one source.

    The matchable denominator excludes no_candidate players — those have
    no row in the source to match, so counting them against the matcher
    conflates source coverage with matcher accuracy.
    """
    total = len(id_map)
    classes = id_map[f"{prefix}_match_class"]
    buckets = classes.value_counts()
    matched = int(
        buckets.get("exact", 0) + buckets.get("fuzzy", 0) + buckets.get("override", 0)
    )
    no_candidate = int(buckets.get("no_candidate", 0))
    matchable = total - no_candidate
    matchable_rate = f"{100 * matched / matchable:.1f}%" if matchable else "n/a"

    print(f"\n  -- {label} --")
    print(f"  Raw match rate:           {matched}/{total} ({100 * matched / total:.0f}%)")
    print(f"  No candidate in source:   {no_candidate}")
    print(f"  Matchable denominator:    {matchable}")
    print(f"  MATCHABLE match rate:     {matched}/{matchable} ({matchable_rate})")
    print(
        f"  Buckets: exact={int(buckets.get('exact', 0))}"
        f"  fuzzy={int(buckets.get('fuzzy', 0))}"
        f"  override={int(buckets.get('override', 0))}"
        f"  fuzzy_miss_high={int(buckets.get('fuzzy_miss_high', 0))}"
        f"  fuzzy_miss_low={int(buckets.get('fuzzy_miss_low', 0))}"
        f"  no_candidate={no_candidate}"
    )
    if pa_floor is not None:
        floors = ", ".join(
            f"{ptype}: {floor if floor is not None else 'n/a'}"
            for ptype, floor in pa_floor.items()
        )
        print(f"  Qualification floor (min pa in source): {floors}")

    high_misses = id_map[classes == "fuzzy_miss_high"]
    if not high_misses.empty:
        print(f"  Actionable misses (score >= {HIGH_CONFIDENCE_FLOOR}, real candidate very likely):")
        for _, row in high_misses.iterrows():
            print(
                f"    - {row['player_name']} ({row['player_type']}): "
                f"score {row[f'{prefix}_best_score']:.1f} "
                f"vs '{row[f'{prefix}_best_candidate']}'"
            )
    low_count = int((classes == "fuzzy_miss_low").sum())
    if low_count:
        print(
            f"  Low-confidence misses ({REVIEW_FLOOR}-{HIGH_CONFIDENCE_FLOOR - 1}, "
            f"often coincidental near-names): {low_count} (see review queue)"
        )


def print_quality_report(
    id_map: pd.DataFrame,
    savant_pa_floor: dict[str, int | None] | None = None,
) -> None:
    """Print match-quality summary: raw and matchable rates per source.

    The headline number is the MATCHABLE match rate — matches among
    players who actually have a counterpart row in the source — because
    the raw rate conflates matcher misses with genuine source absence.

    The detailed sections cover rostered players only; free agents get a
    summary block, since the FA pool is dominated by minor leaguers with
    genuinely no leaderboard row (a flood of correct no_candidates).
    """
    fa = id_map[id_map["status"] == "fa"]
    id_map = id_map[id_map["status"] == "owned"]

    total = len(id_map)
    savant_matched = id_map["savant_player_id"].notna().sum()
    fg_matched = id_map["fangraphs_id"].notna().sum()
    unmatched = id_map[
        id_map["savant_player_id"].isna() & id_map["fangraphs_id"].isna()
    ]

    # Combined view: matched in either source; no_candidate only if absent
    # from BOTH sources (a fuzzy miss in either means a counterpart exists).
    # An unmatched row takes the higher-confidence class of its two sources.
    def _combined_class(r: pd.Series) -> str:
        if r["match_quality"] != "unmatched":
            return r["match_quality"]
        source_classes = (r["savant_match_class"], r["fangraphs_match_class"])
        for cls in ("fuzzy_miss_high", "fuzzy_miss_low"):
            if cls in source_classes:
                return cls
        return "no_candidate"

    combined_class = id_map.apply(_combined_class, axis=1)

    # Two-way players: Fantrax suffixes the name per role ("Shohei Ohtani-H"
    # vs "Shohei Ohtani-P"), so identical-name detection misses them.  A
    # shared source ID appearing under both player_types is the real signal.
    two_way_mask = pd.Series(False, index=id_map.index)
    for id_col in ("savant_player_id", "fangraphs_id"):
        with_id = id_map.dropna(subset=[id_col])
        types_per_id = with_id.groupby(id_col)["player_type"].nunique()
        two_way_mask |= id_map[id_col].isin(types_per_id[types_per_id > 1].index)
    type_counts = id_map.groupby("player_name")["player_type"].nunique()
    two_way_mask |= id_map["player_name"].isin(type_counts[type_counts > 1].index)
    two_way = id_map.loc[two_way_mask].sort_values(["player_name", "player_type"])

    print(f"\n{'=' * 60}")
    print("  Player ID Map — Quality Report")
    print(f"{'=' * 60}")
    print(f"  Total rostered players:   {total}")
    print(f"  Matched to Savant:        {savant_matched} ({100 * savant_matched / total:.0f}%)")
    print(f"  Matched to FanGraphs:     {fg_matched} ({100 * fg_matched / total:.0f}%)")
    print(f"  Fully unmatched:          {len(unmatched)}")

    _print_source_section(id_map, "Savant", "savant", pa_floor=savant_pa_floor)
    _print_source_section(id_map, "FanGraphs", "fangraphs")

    combined = id_map.assign(combined_match_class=combined_class)
    combined_buckets = combined["combined_match_class"].value_counts()
    combined_matched = int(
        combined_buckets.get("exact", 0)
        + combined_buckets.get("fuzzy", 0)
        + combined_buckets.get("override", 0)
    )
    combined_no_cand = int(combined_buckets.get("no_candidate", 0))
    combined_matchable = total - combined_no_cand
    combined_rate = (
        f"{100 * combined_matched / combined_matchable:.1f}%" if combined_matchable else "n/a"
    )
    print(f"\n  -- Combined (either source) --")
    print(f"  Raw match rate:           {combined_matched}/{total} ({100 * combined_matched / total:.0f}%)")
    print(f"  No candidate in either:   {combined_no_cand}")
    print(f"  MATCHABLE match rate:     {combined_matched}/{combined_matchable} ({combined_rate})")
    print(
        f"  Buckets: exact={int(combined_buckets.get('exact', 0))}"
        f"  fuzzy={int(combined_buckets.get('fuzzy', 0))}"
        f"  override={int(combined_buckets.get('override', 0))}"
        f"  fuzzy_miss_high={int(combined_buckets.get('fuzzy_miss_high', 0))}"
        f"  fuzzy_miss_low={int(combined_buckets.get('fuzzy_miss_low', 0))}"
        f"  no_candidate={combined_no_cand}"
    )

    if not fa.empty:
        print(f"\n  -- Free agents (same matching pipeline, summarized) --")
        print(f"  Pool size:                {len(fa)}")
        for label, prefix in (("Savant", "savant"), ("FanGraphs", "fangraphs")):
            classes = fa[f"{prefix}_match_class"]
            fa_matched = int(classes.isin(("exact", "fuzzy", "override")).sum())
            print(
                f"  {label}: matched={fa_matched}"
                f"  fuzzy_miss_high={int((classes == 'fuzzy_miss_high').sum())}"
                f"  fuzzy_miss_low={int((classes == 'fuzzy_miss_low').sum())}"
                f"  no_candidate={int((classes == 'no_candidate').sum())}"
            )
        print("  (a large no_candidate count is expected and correct — the FA")
        print("   pool is mostly minor leaguers below the qualification floors)")

    if not two_way.empty:
        print(f"\n  Two-way player rows ({len(two_way)}) — each matched ONLY against")
        print("  its own type's source pool (Hitter->batting, Pitcher->pitching):")
        for _, row in two_way.iterrows():
            print(
                f"    - {row['player_name']} [{row['player_type']}, {row['fantrax_team_name']}]:\n"
                f"        Savant    -> {row['savant_best_candidate']} "
                f"({row['savant_match_class']}, id={row['savant_player_id']})\n"
                f"        FanGraphs -> {row['fangraphs_best_candidate']} "
                f"({row['fangraphs_match_class']}, id={row['fangraphs_id']})"
            )

    print(f"{'=' * 60}")


# ── entry point ────────────────────────────────────────────────────────


def main() -> None:
    """Build the player ID map and print quality report."""
    # Fantrax team names can contain emoji the cp1252 console can't encode
    sys.stdout.reconfigure(errors="replace")
    id_map, savant_pa_floor = build_player_id_map()
    print_quality_report(id_map, savant_pa_floor=savant_pa_floor)


if __name__ == "__main__":
    main()
