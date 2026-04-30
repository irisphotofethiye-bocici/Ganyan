"""Top-1-probability trip-wire.

Compares today's average top-1 model probability across all races on the
card against a rolling 90-day baseline.

Asymmetric policy (since 2026-04-30): the failure mode this guard exists
for is *under*-confidence — feature pipelines (AGF, last_six, KGS) that
haven't published, dropping every horse to a uniform 1/N softmax. That
shows up as today's avg top-1 *below* baseline.  *Over*-confidence
(today's avg above baseline) usually means MORE feature data is
available than baseline (e.g. a new feature shipped, or the re-predict
cron is squeezing late-money concentration into the avg).  Different
phenomenon — worth surfacing, but not worth halting advice over.

So:
- z < -sigma  → ``halt``: hide page body, demand bypass
- |z| > sigma → ``anomalous``: surface a soft yellow warning, render anyway
- otherwise   → ``ok``: small green OK line

Refactored out of the ``ganyan advice`` CLI so the web ``/advice``
route can call the same baseline math without duplicating the SQL.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry


def compute_trip_wire(
    session: Session,
    target_date: date,
    *,
    lookback_days: int = 90,
    min_baseline_days: int = 30,
) -> dict | None:
    """Return today vs ``lookback_days`` baseline top-1 stats, or None.

    Returns None when:
    - ``target_date`` has no predictions yet
    - fewer than ``min_baseline_days`` days of historical data
    - baseline std is degenerate (<1e-9)

    Otherwise returns a dict with ``today_avg``, ``baseline_mean``,
    ``baseline_std``, ``z_score``, ``n_baseline_days``.
    """
    since = target_date - timedelta(days=lookback_days + 1)
    per_race_top1 = (
        select(
            Race.date.label("d"),
            Race.id.label("rid"),
            func.max(RaceEntry.predicted_probability).label("top1"),
        )
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .where(Race.date >= since, Race.date <= target_date)
        .where(RaceEntry.predicted_probability.is_not(None))
        .group_by(Race.id, Race.date)
        .subquery()
    )
    rows = session.execute(
        select(
            per_race_top1.c.d,
            func.avg(per_race_top1.c.top1).label("daily_avg"),
        )
        .group_by(per_race_top1.c.d)
        .order_by(per_race_top1.c.d)
    ).all()
    daily = {row.d: float(row.daily_avg) for row in rows}
    today_avg = daily.pop(target_date, None)
    if today_avg is None:
        return None
    if len(daily) < min_baseline_days:
        return None
    arr = np.fromiter(daily.values(), dtype="float64")
    baseline_mean = float(arr.mean())
    baseline_std = float(arr.std())
    if baseline_std < 1e-9:
        return None
    return {
        "today_avg": today_avg,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "z_score": (today_avg - baseline_mean) / baseline_std,
        "n_baseline_days": len(daily),
    }


def is_halt(trip_info: dict | None, sigma: float = 2.0) -> bool:
    """Hard-halt when today is *under*-confident vs baseline.

    Asymmetric: only z < -sigma triggers halt.  Over-confident days
    (z > +sigma) are surfaced via :func:`is_anomalous` as a soft warning.
    """
    return trip_info is not None and trip_info["z_score"] < -sigma


def is_anomalous(trip_info: dict | None, sigma: float = 2.0) -> bool:
    """True when today's z-score crosses ±sigma in either direction.

    Includes halt cases — callers branch on :func:`is_halt` first.
    """
    return trip_info is not None and abs(trip_info["z_score"]) > sigma
