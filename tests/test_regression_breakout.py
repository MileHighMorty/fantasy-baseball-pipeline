"""Tests locking the gold breakout/regression sign convention and quality guard.

These come from real defects found while cleaning up the pitcher buy/sell board:

* the pitcher gap sign was inverted (breakout vs regression swapped);
* a ``hard_hit_percentile`` OR-clause flagged directional BUYS as sells
  (positive-gap Mookie Betts, negative-gap Max Fried);
* gap-only logic flagged "regression to a still-elite level" as a sell
  (Ohtani/Skubal/Sale), with no league-quality floor.

The invariants below are deliberately data-light: hand-built frames with only the
columns the detectors read, so a future refactor that re-inverts a sign or drops
the quality floor fails here instead of silently shipping a wrong sell list.
"""

import pandas as pd

from gold import breakout_detector as bd
from gold import regression_alerts as ra


def _pitchers(rows: list[dict]) -> pd.DataFrame:
    """Build a pitcher frame with the columns the detectors read.

    ``xera_minus_era`` is derived as ``xera - era`` so the gap sign always
    matches the era/xera the test states, the same way the silver layer computes
    it.
    """
    df = pd.DataFrame(rows)
    df["xera_minus_era"] = df["xera"] - df["era"]
    if "hard_hit_percentile" not in df.columns:
        df["hard_hit_percentile"] = 50.0
    return df


def _hitters(rows: list[dict]) -> pd.DataFrame:
    """Build a hitter frame with the columns the detectors read.

    ``xwoba_minus_woba`` is derived as ``est_woba - woba`` to keep the gap sign
    consistent with the stated wOBA/xwOBA.
    """
    df = pd.DataFrame(rows)
    df["xwoba_minus_woba"] = df["est_woba"] - df["woba"]
    if "hard_hit_percentile" not in df.columns:
        df["hard_hit_percentile"] = 50.0
    return df


# ── invariant (a): a still-elite lucky pitcher is NOT a sell ─────────────


def test_still_elite_lucky_pitcher_is_not_a_sell():
    # era 1.06 on a 2.60 xERA: lucky (positive gap) but xERA is elite (< 3.50).
    # This is the Ohtani/Skubal/Sale case the quality floor must exclude.
    df = _pitchers([
        {"player_name": "Elite Lucky", "era": 1.06, "xera": 2.60},
        {"player_name": "Lucky Mediocre", "era": 3.60, "xera": 5.50},  # genuine sell
    ])
    sells = set(ra.detect_regression_pitchers(df)["player_name"])
    assert "Elite Lucky" not in sells
    assert "Lucky Mediocre" in sells  # positive control: the real sell survives


# ── invariant (b): a negative-gap (unlucky) pitcher is NEVER a sell ──────


def test_unlucky_pitcher_is_never_a_sell_even_with_high_hard_hit():
    # xERA 3.44 below ERA 4.35 = unlucky = a BUY. High hard-hit must NOT drag him
    # onto the sell list (the removed OR-clause did exactly that to Fried/Luzardo).
    df = _pitchers([
        {"player_name": "Unlucky Hard Hit", "era": 4.35, "xera": 3.44,
         "hard_hit_percentile": 100.0},
    ])
    sells = set(ra.detect_regression_pitchers(df)["player_name"])
    assert "Unlucky Hard Hit" not in sells


# ── invariant (c): a positive-gap / weak-contact hitter is NEVER a sell ──


def test_positive_gap_or_low_hard_hit_hitter_is_never_a_sell():
    # Betts: underperforming his xwOBA (positive gap) AND low hard-hit. The old
    # OR-clause flagged him as a sell; gap direction now governs, so he is not.
    df = _hitters([
        {"player_name": "Betts", "woba": 0.269, "est_woba": 0.330,
         "hard_hit_percentile": 20.0},
        {"player_name": "Real Sell", "woba": 0.350, "est_woba": 0.300,
         "hard_hit_percentile": 30.0},  # overperforming + below-avg xwOBA
    ])
    sells = set(ra.detect_regression_hitters(df)["player_name"])
    assert "Betts" not in sells
    assert "Real Sell" in sells  # positive control


def test_still_above_average_hitter_is_not_a_sell():
    # Overperforming (negative gap) but xwOBA still above the .320 league anchor:
    # regression to a still-good level is not a sell (the Semien/Diaz case).
    df = _hitters([
        {"player_name": "Above Avg Overperformer", "woba": 0.360, "est_woba": 0.330},
    ])
    sells = set(ra.detect_regression_hitters(df)["player_name"])
    assert "Above Avg Overperformer" not in sells


# ── invariant (d): a still-bad unlucky pitcher is NOT a buy ──────────────


def test_still_bad_unlucky_pitcher_is_not_a_buy():
    # Unlucky (negative gap) but xERA 4.50 is still poor (>= 3.50): not a buy.
    # A genuinely good unlucky arm (xERA 2.54) is the positive control.
    df = _pitchers([
        {"player_name": "Unlucky Bad", "era": 6.00, "xera": 4.50},
        {"player_name": "Unlucky Good", "era": 6.00, "xera": 2.54},
    ])
    buys = set(bd.detect_breakout_pitchers(df)["player_name"])
    assert "Unlucky Bad" not in buys
    assert "Unlucky Good" in buys  # positive control


def test_below_average_underperformer_is_not_a_buy():
    # Underperforming (positive gap) with strong contact, but xwOBA still below
    # the .320 anchor: not a buy. Bichette (xwOBA .326 > .320) is the control and
    # must be restored to the buy list under the absolute floor.
    df = _hitters([
        {"player_name": "Weak Bat", "woba": 0.250, "est_woba": 0.310,
         "hard_hit_percentile": 80.0},
        {"player_name": "Bichette", "woba": 0.271, "est_woba": 0.326,
         "hard_hit_percentile": 74.0},
    ])
    buys = set(bd.detect_breakout_hitters(df)["player_name"])
    assert "Weak Bat" not in buys
    assert "Bichette" in buys  # absolute .320 floor keeps the league-average bat
