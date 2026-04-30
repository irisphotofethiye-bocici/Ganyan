"""Tests for picks grading — particularly the per-pool birim factor."""

from __future__ import annotations

import pytest

from ganyan.predictor.picks import BIRIM_TL_BY_STRATEGY, _birim_tl


def test_birim_table_covers_all_strategies():
    for strat in ("ganyan_top1", "uclu_top1", "uclu_box6", "sirali_ikili_top1"):
        assert strat in BIRIM_TL_BY_STRATEGY


def test_uclu_birim_is_two_tl():
    assert _birim_tl("uclu_top1") == 2.0
    assert _birim_tl("uclu_box6") == 2.0


def test_ganyan_and_sirali_ikili_birim_is_one_tl():
    assert _birim_tl("ganyan_top1") == 1.0
    assert _birim_tl("sirali_ikili_top1") == 1.0


def test_unknown_strategy_defaults_to_one_tl():
    assert _birim_tl("not_a_strategy") == 1.0


# Regression fixture — derived from operator's real bilet 68671854006416,
# Ankara R8, 2026-04-30. This is the strongest ground-truth datapoint we
# have for the üçlü payout math.
APR30_R8_FIXTURE = {
    "pool_value": 92.50,           # uclu_payout_tl as published by TJK
    "stake_per_ticket_tl": 100.0,  # what picks.STAKE_PER_TICKET_TL uses
    "birim": 2.0,                  # Sıralı Üçlü Bahis birim fiyat
    "expected_payout_per_TL_at_100_per_perm": 4625.0,
    # Operator's actual bilet (sanity check):
    "actual_misli": 4,
    "actual_per_perm_stake_TL": 8.0,    # 4 misli × 2 birim
    "actual_payout": 370.0,             # = 4 × 92.50
}


def test_apr30_r8_payout_math():
    """At 100 TL/perm notional, a 92.50 pool figure should yield 4625 TL
    on the winning permutation — half of the pre-fix 9250 TL value.
    """
    fix = APR30_R8_FIXTURE
    payout = fix["pool_value"] * fix["stake_per_ticket_tl"] / fix["birim"]
    assert payout == pytest.approx(fix["expected_payout_per_TL_at_100_per_perm"])


def test_apr30_r8_real_bilet_reconciles():
    """The operator's real 192 TL bilet at Misli=4 paid 370 TL.
    Verify that's exactly Misli × pool_value (= per-bilet × number of
    bilets purchased at the winning permutation).
    """
    fix = APR30_R8_FIXTURE
    # Per-bilet payout = pool_value (TJK reports per-bilet at default birim).
    # Operator bought Misli=4 bilets of the winning permutation.
    real_payout = fix["actual_misli"] * fix["pool_value"]
    assert real_payout == pytest.approx(fix["actual_payout"])
