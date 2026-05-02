"""Regression test for OOS framework's minimum-window guard.

The V2 features failure (2026-05-02 memory: in-sample +3.6pp, OOS +0.07pp on
1500 races) settled the bar: any new feature must clear +1pp on a 1500+ race
2025+ OOS window. This test prevents a future 'just run --window 600' shortcut
that would replicate V2's failure mode.
"""

from datetime import date

import pytest


def test_oos_window_rejects_short_date_range():
    """A 90-day window must raise — would hide overfitting."""
    from logs import discordance_oos_backtest as oos
    with pytest.raises(AssertionError, match="365 days"):
        oos.assert_min_window(from_date=date(2026, 1, 1), cutoff=date(2026, 4, 1), n_races=2000)


def test_oos_window_rejects_low_race_count():
    """A 365-day window with only 800 races must raise."""
    from logs import discordance_oos_backtest as oos
    with pytest.raises(AssertionError, match="1500 races"):
        oos.assert_min_window(from_date=date(2025, 1, 1), cutoff=date(2026, 1, 31), n_races=800)


def test_oos_window_accepts_full_window():
    """The standard 2025-01-01 → 2026-01-30 window with 1500+ races passes."""
    from logs import discordance_oos_backtest as oos
    oos.assert_min_window(from_date=date(2025, 1, 1), cutoff=date(2026, 1, 30), n_races=1825)
