"""Daily regime monitor — writes per-strategy daily snapshots to ``regime_daily``
and halts via the shared halt flag when the rolling 7d-vs-30d signal drifts
>2pp from baseline.

Closes premortem failure mode 08 (TJK pool dynamics drift) **structurally** —
this canary creates the audit trail and the alerting plumbing. The current
``implied_takeout`` formula is a known proxy with limited validity (see Notes).

Notes
-----
* ``mean_pool_proxy_tl`` is a synthesizing heuristic — there is no real pool
  size in the picks ledger. We approximate it as ``stake * 1000`` averaged
  per winning pick. The shape, not the absolute level, is what matters.
* ``implied_takeout = 1 - (mean_prob * mean_payout) / mean_payout`` reduces
  algebraically to ``1 - mean_prob`` because the ``mean_payout`` factors
  cancel. So the metric we record under that name actually measures the
  *average model unconfidence on winners* for the strategy. It will drift
  when the model recalibrates around winners, NOT when TJK changes the
  takeout rate. To detect actual pool/takeout drift we'd need the raw pool
  size from TJK (not exposed) or a stake-weighted return ratio over ALL
  graded picks (hit + miss). Tracking this metric is still useful as a
  model-calibration canary, but the column name and docstring intent are
  aspirational, not literal. See memory ``project_regime_monitor_metric_caveat``.
* The drift threshold (>2pp 7d-vs-30d) is a calibration knob.
* Filter is ``hit=True`` so the sample is sparse (≈3 winners/strategy/day);
  day-over-day noise on small n means the 2pp threshold may fire on noise.
* ``model_prob_pct`` is stored as percentage 0-100 in the picks ledger, hence
  the divide-by-100 to get a fraction.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from statistics import mean

from sqlalchemy import func

from ganyan.db.models import Pick, RegimeDaily
from ganyan.db.session import get_session
from ganyan.predictor import halt_flag


STRATEGIES = ("uclu_box6", "sirali_ikili_top1")
TAKEOUT_DRIFT_PP_THRESHOLD = 2.0


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
    probs = [float(p.model_prob_pct or 0) / 100 for p in today_picks if p.model_prob_pct]
    n = len(today_picks)
    mean_payout = mean(payouts)
    pool_proxy = sum(float(p.stake_tl or 0) * 1000 for p in today_picks) / max(n, 1)

    if probs:
        expected_return = mean(probs) * mean_payout
        implied_takeout = max(0.0, 1.0 - (expected_return / max(mean_payout, 1)))
    else:
        implied_takeout = None

    realized_vs_expected = None
    if probs and mean_payout:
        realized_vs_expected = mean_payout / max(mean(probs) * mean_payout, 1e-6)

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
            # Idempotent on (snapshot_date, strategy) unique constraint:
            # delete any existing snapshot for today+strategy, then insert fresh.
            session.query(RegimeDaily).filter(
                RegimeDaily.snapshot_date == today,
                RegimeDaily.strategy == strat,
            ).delete()
            session.add(row)
        session.commit()

        for strat in STRATEGIES:
            reason = _check_takeout_drift(session, today, strat)
            if reason:
                halt_flag.set_halt(reason=reason, source="regime_monitor")
                print(reason, file=sys.stderr)
                return 1
    finally:
        session.close()

    print(f"regime_monitor OK: {today.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
