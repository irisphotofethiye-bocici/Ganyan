"""Daily regime monitor — writes per-strategy daily snapshots to ``regime_daily``
and halts via the shared halt flag when the rolling 7d-vs-30d implied_takeout
signal drifts more than the threshold from baseline.

Closes premortem failure mode 08 (TJK pool dynamics drift). The signal is
realized payout efficiency on hits, normalized by the model's win probability:
each hit contributes ``(payout_tl × prob_win) / stake_tl``, and ``implied_takeout``
is ``max(0, 1 - mean_efficiency)``.

Why this works: TJK takeout sets a hard cap on the average ``payout_tl /
stake_tl`` ratio. By multiplying by ``prob_win``, we control for model
calibration on the picks that hit (an over-confident model on hits would
inflate the ratio; an under-confident one would deflate it, but day-over-day
those drift on the timescale of model retrains, not within a 7-day window).
What remains is the TJK-determined payout structure — exactly the regime
signal we want to track.

Notes
-----
* ``mean_pool_proxy_tl`` is a synthesizing heuristic — there is no real pool
  size in the picks ledger. We approximate it as ``stake * 1000`` averaged
  per winning pick. The shape, not the absolute level, is what matters.
* The signal has a remaining model-calibration confound: if the model becomes
  systematically over- or under-confident *on hits* (not all picks) and that
  drifts over a 7-day window, this metric will drift too. In practice this
  drifts on the timescale of model retrains (weeks-to-months), not days, so
  the 7d-vs-30d window mostly captures TJK dynamics. If you retrain mid-week,
  expect a one-time step in this signal that is not real TJK drift.
* Filter is ``hit=True`` so the sample is sparse (≈3 winners/strategy/day).
  Day-over-day noise on small n means thresholds <4pp will fire on noise
  alone — see ``GANYAN_TAKEOUT_DRIFT_PP``, default 4.0.
* ``model_prob_pct`` is stored as percentage 0-100 in the picks ledger, hence
  the divide-by-100 to get a fraction.

Env overrides
-------------
* ``GANYAN_TAKEOUT_DRIFT_PP`` — float, default 4.0 (raised from the original
  2.0 plan-time prior; per the canary-tuning memory, 2pp on n≈3 fires on noise).
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from statistics import mean

from sqlalchemy import func

from ganyan.db.models import Pick, RegimeDaily
from ganyan.db.session import get_session
from ganyan.predictor import halt_flag


STRATEGIES = ("uclu_box6", "sirali_ikili_top1")
TAKEOUT_DRIFT_PP_THRESHOLD = float(os.environ.get("GANYAN_TAKEOUT_DRIFT_PP", "4.0"))


def _payout_efficiency(pick) -> float | None:
    """Per-hit signal: ``(payout_tl × prob_win) / stake_tl``.

    Returns None if any of stake_tl, model_prob_pct, payout_tl is missing
    or stake is zero.
    """
    if not pick.model_prob_pct or not pick.stake_tl or not pick.payout_tl:
        return None
    return float(pick.payout_tl) * (float(pick.model_prob_pct) / 100.0) / float(pick.stake_tl)


def _snapshot_one(session, today, strategy):
    today_picks = (
        session.query(Pick)
        .filter(
            Pick.strategy == strategy,
            Pick.graded.is_(True),
            Pick.hit.is_(True),
            func.date(Pick.graded_at) == today,
        )
        .all()
    )
    if not today_picks:
        return None

    payouts = [float(p.payout_tl or 0) for p in today_picks]
    n = len(today_picks)
    mean_payout = mean(payouts)
    pool_proxy = sum(float(p.stake_tl or 0) * 1000 for p in today_picks) / max(n, 1)

    efficiencies = [e for e in (_payout_efficiency(p) for p in today_picks) if e is not None]
    if efficiencies:
        mean_efficiency = mean(efficiencies)
        implied_takeout = max(0.0, 1.0 - mean_efficiency)
        realized_vs_expected = mean_efficiency
    else:
        implied_takeout = None
        realized_vs_expected = None

    return RegimeDaily(
        snapshot_date=today,
        strategy=strategy,
        n_winning=n,
        mean_payout_tl=mean_payout,
        mean_pool_proxy_tl=pool_proxy,
        implied_takeout=implied_takeout,
        realized_vs_expected=realized_vs_expected,
    )


def _check_takeout_drift(session, today, strategy):
    cutoff_recent = today - timedelta(days=7)
    cutoff_baseline = today - timedelta(days=30)

    recent = (
        session.query(func.avg(RegimeDaily.implied_takeout))
        .filter(
            RegimeDaily.strategy == strategy,
            RegimeDaily.snapshot_date >= cutoff_recent,
            RegimeDaily.implied_takeout.isnot(None),
        )
        .scalar()
    )
    baseline = (
        session.query(func.avg(RegimeDaily.implied_takeout))
        .filter(
            RegimeDaily.strategy == strategy,
            RegimeDaily.snapshot_date >= cutoff_baseline,
            RegimeDaily.snapshot_date < cutoff_recent,
            RegimeDaily.implied_takeout.isnot(None),
        )
        .scalar()
    )
    if recent is None or baseline is None:
        return None
    drift_pp = abs(float(recent) - float(baseline)) * 100
    if drift_pp > TAKEOUT_DRIFT_PP_THRESHOLD:
        return (
            f"regime_monitor: {strategy} 7d implied takeout "
            f"{float(recent):.1%} vs 30d baseline {float(baseline):.1%} "
            f"(drift {drift_pp:.1f}pp)"
        )
    return None


def main() -> int:
    today = date.today()
    session = get_session()
    try:
        for strat in STRATEGIES:
            row = _snapshot_one(session, today, strat)
            if row is None:
                continue
            session.query(RegimeDaily).filter(
                RegimeDaily.snapshot_date == today,
                RegimeDaily.strategy == strat,
            ).delete()
            session.add(row)
        session.commit()

        reasons = []
        for strat in STRATEGIES:
            reason = _check_takeout_drift(session, today, strat)
            if reason:
                reasons.append(reason)
                print(reason, file=sys.stderr)

        if reasons:
            halt_reason = reasons[0] + (
                f"; {len(reasons) - 1} more strategy drift(s) logged"
                if len(reasons) > 1 else ""
            )
            halt_flag.set_halt(reason=halt_reason, source="regime_monitor")
            return 1
    finally:
        session.close()

    print(f"regime_monitor OK: {today.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
