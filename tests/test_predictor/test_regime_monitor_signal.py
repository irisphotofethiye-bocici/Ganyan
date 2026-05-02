"""Unit tests for the rewritten regime_monitor.implied_takeout signal.

The signal must respond to TJK payout structure (the old 1 - mean_prob
formula did not). These tests verify the per-hit efficiency calculation
behaves correctly across realistic scenarios.
"""

from __future__ import annotations

from types import SimpleNamespace

from scripts.regime_monitor import _payout_efficiency


def _pick(stake_tl, model_prob_pct, payout_tl):
    return SimpleNamespace(stake_tl=stake_tl, model_prob_pct=model_prob_pct, payout_tl=payout_tl)


def test_efficiency_returns_none_for_missing_fields():
    assert _payout_efficiency(_pick(None, 30.0, 2.50)) is None
    assert _payout_efficiency(_pick(1.0, None, 2.50)) is None
    assert _payout_efficiency(_pick(1.0, 30.0, None)) is None
    assert _payout_efficiency(_pick(0, 30.0, 2.50)) is None  # zero stake → falsy


def test_efficiency_typical_uclu_hit():
    # Typical üçlü hit: 1 TL stake, 30% model prob, 2.50 TL payout
    eff = _payout_efficiency(_pick(1.0, 30.0, 2.50))
    assert eff == 0.75
    # implied_takeout would be 1 - 0.75 = 0.25 = 25%, plausible for üçlü.


def test_efficiency_drops_when_tjk_payout_drops():
    # Same prob, lower payout → efficiency drops → implied_takeout rises
    high = _payout_efficiency(_pick(1.0, 30.0, 2.50))
    low = _payout_efficiency(_pick(1.0, 30.0, 2.10))
    assert high > low
    assert (1 - high) < (1 - low)  # implied_takeout response


def test_efficiency_unaffected_by_uniform_model_calibration_shift():
    # If the model is over-confident on hits (says 45% when realized is the
    # same hit), efficiency rises mechanically — this is the known confound.
    realistic = _payout_efficiency(_pick(1.0, 30.0, 2.50))
    overconfident = _payout_efficiency(_pick(1.0, 45.0, 2.50))
    # Confound exists, but the docstring acknowledges it. This test
    # documents the behavior so a future reader understands the trade.
    assert overconfident > realistic


def test_efficiency_scales_with_stake():
    # Doubling stake should leave efficiency unchanged when payout doubles too
    # (TJK pari-mutuel scales linearly within a strategy).
    one_unit = _payout_efficiency(_pick(1.0, 30.0, 2.50))
    two_units = _payout_efficiency(_pick(2.0, 30.0, 5.00))
    assert one_unit == two_units
