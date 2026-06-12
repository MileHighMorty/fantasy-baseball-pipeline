"""Tests for the core matching logic in silver/player_id_map.py.

Golden cases come from real findings during the match-rate-honesty and
free-agent work: the Ohtani -H/-P suffix collision, the José A. Ferrer
accent miss, the Joscar/Stanly look-alike collisions at scale, and the
master res_key stability guarantees.
"""

import pathlib
from datetime import date

import pandas as pd
import pytest

from silver import player_id_map as pim

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── name normalization ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Accents stripped (the José A. Ferrer case)
        ("José A. Ferrer", "Jose A. Ferrer"),
        # Fantrax two-way suffixes stripped, either role, any case
        ("Shohei Ohtani-H", "Shohei Ohtani"),
        ("Shohei Ohtani-P", "Shohei Ohtani"),
        ("Shohei Ohtani-p", "Shohei Ohtani"),
        # Generational suffixes and real hyphenated surnames survive
        ("George Lombard Jr.", "George Lombard Jr."),
        ("Pete Crow-Armstrong", "Pete Crow-Armstrong"),
        # A trailing token longer than the bare marker is NOT stripped
        ("TJ Smith-Hall", "TJ Smith-Hall"),
        # Whitespace collapses; case is intentionally preserved
        ("  Mike   Trout  ", "Mike Trout"),
    ],
)
def test_normalize_name(raw, expected):
    assert pim._normalize_name(raw) == expected


def test_two_way_suffixes_normalize_to_same_base():
    assert pim._normalize_name("Shohei Ohtani-H") == pim._normalize_name(
        "Shohei Ohtani-P"
    )


def test_canonical_name_keeps_accents_but_strips_suffix():
    assert pim._canonical_name("José Ferrer-P") == "José Ferrer"
    assert pim._canonical_name("José Ferrer") == "José Ferrer"


# ── score classification band boundaries ───────────────────────────────


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (100, "exact"),
        (99.9, "fuzzy"),
        (90, "fuzzy"),            # MATCH_THRESHOLD is inclusive
        (89.9, "fuzzy_miss_high"),
        (89, "fuzzy_miss_high"),
        (85, "fuzzy_miss_high"),  # HIGH_CONFIDENCE_FLOOR is inclusive
        (84.9, "fuzzy_miss_low"),
        (84, "fuzzy_miss_low"),
        (75, "fuzzy_miss_low"),   # REVIEW_FLOOR is inclusive
        (74.9, "no_candidate"),
        (74, "no_candidate"),
        (0, "no_candidate"),
    ],
)
def test_classify_score_boundaries(score, expected):
    assert pim._classify_score(score) == expected


# ── fuzzy match: near-miss capture and team tiebreaker ─────────────────


def _pool(names, teams=None):
    """Build a (subframe, normalized names) candidate pool by hand."""
    sub = pd.DataFrame({"player_name": names})
    if teams is not None:
        sub["fg_team"] = teams
    return sub, [pim._normalize_name(n) for n in names]


def test_best_match_captures_sub_threshold_near_miss():
    # The real Bobby Miller / Hoby Milner case: no match, but the score
    # and candidate must still come back — that signal drives the
    # fuzzy-miss classification.
    score, candidate, idx = pim._best_match(
        pim._normalize_name("Bobby Miller"), _pool(["Hoby Milner", "Chris Sale"])
    )
    assert candidate == "Hoby Milner"
    assert idx == 0
    assert 0 < score < pim.MATCH_THRESHOLD


def test_best_match_empty_pool():
    assert pim._best_match("Anyone", _pool([])) == (0.0, None, None)


def test_team_tiebreaker_prefers_matching_team():
    # Two source candidates tie at 100 on name; the one whose team
    # matches the player's Fantrax mlb_team must win.
    pool = _pool(["Alex Garcia", "Alex Garcia"], teams=["NYM", "BOS"])
    log: list[str] = []
    score, candidate, idx = pim._best_match(
        "Alex Garcia", pool, fantrax_team="BOS", tiebreak_log=log
    )
    assert score == 100
    assert idx == 1  # the BOS row, not the first-listed NYM row
    assert len(log) == 1


def test_team_tiebreaker_inert_without_team():
    # No Fantrax team -> first tied candidate wins, nothing logged.
    pool = _pool(["Alex Garcia", "Alex Garcia"], teams=["NYM", "BOS"])
    log: list[str] = []
    _, _, idx = pim._best_match("Alex Garcia", pool, tiebreak_log=log)
    assert idx == 0
    assert log == []


# ── override application (pure decision function) ──────────────────────


def test_apply_override_force_match_wins_regardless_of_fuzzy():
    overrides = {("jose ferrer", "Pitcher", "savant"): 678606}
    # Fuzzy found nothing (sub-threshold) — the override still forces it.
    resolved, cls = pim._apply_override(
        overrides, "Jose Ferrer", "Pitcher", "savant", None, "fuzzy_miss_high"
    )
    assert (resolved, cls) == (678606, "override")


def test_apply_override_block_beats_high_fuzzy_score():
    # The Stanly Alcantara case: fuzzy matched Sandy Alcantara at 90.3,
    # but the human says it's a different person.
    overrides = {("stanly alcantara", "Pitcher", "savant"): None}
    resolved, cls = pim._apply_override(
        overrides, "Stanly Alcantara", "Pitcher", "savant", 645261, "fuzzy"
    )
    assert (resolved, cls) == (None, "no_candidate")


def test_apply_override_pass_through_when_no_override():
    resolved, cls = pim._apply_override(
        {}, "Mike Trout", "Hitter", "savant", 545361, "exact"
    )
    assert (resolved, cls) == (545361, "exact")


def test_load_overrides_expands_source_all_and_blocks_both(monkeypatch):
    monkeypatch.setattr(
        pim, "OVERRIDES_PATH", FIXTURES / "player_name_overrides_sample.csv"
    )
    overrides = pim._load_overrides()

    # 1 force row + 1 all-block row expanded to 2 sources; the
    # unknown-source row and the force-with-all row are both skipped.
    assert overrides == {
        ("jose ferrer", "Pitcher", "savant"): 678606,
        ("stanly alcantara", "Pitcher", "savant"): None,
        ("stanly alcantara", "Pitcher", "fangraphs"): None,
    }

    # And the expanded blocks veto a strong fuzzy match in BOTH sources.
    for source, wrong_id in (("savant", 645261), ("fangraphs", 18684)):
        resolved, cls = pim._apply_override(
            overrides, "Stanly Alcantara", "Pitcher", source, wrong_id, "fuzzy"
        )
        assert (resolved, cls) == (None, "no_candidate")


# ── player master: res_key assignment, stability, immutability ─────────


def _id_map(rows):
    """Hand-build the minimal id_map slice the master upsert consumes."""
    return pd.DataFrame(
        rows,
        columns=[
            "player_name", "player_type",
            "savant_player_id", "fangraphs_id", "fantrax_id",
        ],
    )


@pytest.fixture
def master_path(tmp_path, monkeypatch):
    path = tmp_path / "player_master.csv"
    monkeypatch.setattr(pim, "MASTER_PATH", path)
    return path


def test_res_keys_monotonic_and_assigned_once(master_path):
    first = pim._update_player_master(_id_map([
        ("Corbin Carroll", "Hitter", 682998, 25878, "ftx_aaa"),
        ("Merrill Kelly", "Pitcher", 518876, None, "ftx_bbb"),
    ]))
    assert list(first["res_key"]) == [1, 2]

    # A later run with one NEW player extends the sequence; existing
    # players keep their keys.
    second = pim._update_player_master(_id_map([
        ("Corbin Carroll", "Hitter", 682998, 25878, "ftx_aaa"),
        ("Merrill Kelly", "Pitcher", 518876, None, "ftx_bbb"),
        ("Spencer Steer", "Hitter", 668715, 26323, "ftx_ccc"),
    ]))
    assert list(second["res_key"]) == [1, 2, 3]
    assert second.loc[second.canonical_name == "Corbin Carroll", "res_key"].item() == 1


def test_rerun_with_same_players_creates_zero_new_keys(master_path):
    rows = [
        ("Corbin Carroll", "Hitter", 682998, 25878, "ftx_aaa"),
        ("Merrill Kelly", "Pitcher", 518876, None, "ftx_bbb"),
    ]
    first = pim._update_player_master(_id_map(rows))
    second = pim._update_player_master(_id_map(rows))
    assert list(second["res_key"]) == list(first["res_key"])
    assert len(second) == len(first) == 2


def test_two_way_player_gets_two_distinct_res_keys(master_path):
    # Ohtani: same person, same vendor ids, but Hitter and Pitcher are
    # mastered separately — the shared MLBAM id must not collapse them.
    master = pim._update_player_master(_id_map([
        ("Shohei Ohtani-H", "Hitter", 660271, 19755, "ftx_hhh"),
        ("Shohei Ohtani-P", "Pitcher", 660271, 19755, "ftx_ppp"),
    ]))
    assert len(master) == 2
    assert master["res_key"].nunique() == 2
    assert set(master["canonical_name"]) == {"Shohei Ohtani"}
    assert set(master["player_type"]) == {"Hitter", "Pitcher"}


def test_established_id_never_overwritten(master_path):
    # The scale-collision guard (Joscar -> Teoscar): a later row merging
    # via a shared source id must FILL missing ids but never replace an
    # established one.
    today = date.today().isoformat()

    first = pim._update_player_master(_id_map([
        ("Teoscar Hernandez", "Hitter", 606192, None, "ftx_teo"),
    ]))
    assert first.loc[0, "fangraphs_id"] is pd.NA or pd.isna(first.loc[0, "fangraphs_id"])

    # Same savant id, but a different fantrax_id and a newly resolved
    # fangraphs_id (the FA look-alike that fuzzy-matched at 90.9).
    second = pim._update_player_master(_id_map([
        ("Joscar Hernandez", "Hitter", 606192, 13066, "0737l"),
    ]))

    assert len(second) == 1, "must merge by shared source id, not add a row"
    row = second.iloc[0]
    assert row["res_key"] == 1
    assert row["fantrax_id"] == "ftx_teo", "established id was overwritten"
    assert row["fangraphs_id"] == 13066, "missing id should be filled in"
    assert row["canonical_name"] == "Teoscar Hernandez"
    assert row["first_seen_date"] == today
    assert row["last_seen_date"] == today
