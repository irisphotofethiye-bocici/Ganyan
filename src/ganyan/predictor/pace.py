"""Pace-preference feature: characterise each horse by the kind of
race-pace they finish well in.

Hard data constraint: TJK exposes only the **winner's** and the
**runner-up's** last-800m sectional time per race (`pace_l800_leader_s`
on the Race row).  Per-horse fractionals aren't published, so classic
Beyer-style pace figures aren't reachable on this data.

What we *can* compute:
- For each (track, distance_bucket), the historical mean and stdev of
  ``pace_l800_leader_s``.  This is the local "par" for race-finishing
  pace.
- A signed z-score of every observed l800 against its (track, bucket)
  par.  Positive z = slower-than-usual finish (race grinded, pace
  attrition); negative z = faster-than-usual finish (closers' kick).
- For each horse, look at races where they finished **top-3** and
  average those races' pace-z scores → "pace preference" score per
  horse.  Negative = horse does well in fast-finish races (closer
  type); positive = horse does well in slow-finish races (front-runner
  / wire-to-wire type).

This is a *horse-level* feature.  Race-level pace prediction
(predicting the upcoming race's expected pace from the lineup
composition) is a separate, larger project — file under future work
if this MVP gives lift.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry
from ganyan.predictor.bayes.data import distance_bucket_for


PaceHistoryEntry = Tuple[date, float]   # (race_date, pace_z_score for that race)


def compute_pace_baseline(
    session: Session,
    to_date: date | None = None,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """Per (track_id, distance_bucket): (mean_l800, stdev_l800).

    Aggregates all resulted races with a recorded leader's last-800m.
    Returns sigma=0 entries with stdev=1.0 to avoid division blow-up.
    """
    q = (
        select(Race.track_id, Race.distance_meters, Race.pace_l800_leader_s)
        .where(Race.pace_l800_leader_s.is_not(None))
        .where(Race.distance_meters.is_not(None))
    )
    if to_date is not None:
        q = q.where(Race.date <= to_date)

    bucketed: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for tid, dist_m, l800 in session.execute(q):
        if l800 is None or l800 <= 0:
            continue
        b = distance_bucket_for(dist_m)
        bucketed[(tid, b)].append(float(l800))

    out: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for k, vals in bucketed.items():
        if len(vals) < 5:
            continue
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        sd = var ** 0.5 if var > 1e-9 else 1.0
        out[k] = (m, sd)
    return out


def build_horse_pace_history(
    session: Session,
    baseline: Dict[Tuple[int, int], Tuple[float, float]],
    to_date: date | None = None,
) -> Dict[int, List[PaceHistoryEntry]]:
    """For each horse, sorted (race_date, pace_z_for_that_race) where
    the horse finished top-3.

    Why top-3 only: when a horse finishes deep we cannot tell whether
    the recorded l800 (the *leader's* last-800m) characterised the
    horse's race-shape at all — they may have been dropped early and
    the front-of-field pace is irrelevant to them.  Limiting to top-3
    keeps the signal interpretable: "horse showed they handle this
    kind of finish."
    """
    q = (
        select(
            RaceEntry.horse_id, Race.date, Race.track_id,
            Race.distance_meters, Race.pace_l800_leader_s,
            RaceEntry.finish_position,
        )
        .join(Race, Race.id == RaceEntry.race_id)
        .where(RaceEntry.finish_position.in_((1, 2, 3)))
        .where(Race.pace_l800_leader_s.is_not(None))
        .where(Race.distance_meters.is_not(None))
    )
    if to_date is not None:
        q = q.where(Race.date <= to_date)

    history: Dict[int, List[PaceHistoryEntry]] = defaultdict(list)
    for hid, race_date, tid, dist_m, l800, _fp in session.execute(q):
        if l800 is None or l800 <= 0 or not dist_m:
            continue
        b = distance_bucket_for(dist_m)
        baseline_pair = baseline.get((tid, b))
        if baseline_pair is None:
            continue
        m, sd = baseline_pair
        z = (float(l800) - m) / sd
        history[hid].append((race_date, z))

    for hid in history:
        history[hid].sort(key=lambda t: t[0])
    return history


def horse_pace_score(
    history: Dict[int, List[PaceHistoryEntry]],
    horse_id: int,
    as_of_date: date,
    n_recent: int = 5,
) -> float | None:
    """Mean pace-z-score over the last ``n_recent`` top-3 finishes
    strictly before ``as_of_date``.  None when no qualifying history.

    Negative result = horse historically finishes top-3 in fast-pace
    races (closer type).  Positive = horse finishes top-3 in slow-pace
    races (front-runner / wire-to-wire type).
    """
    runs = history.get(horse_id)
    if not runs:
        return None
    prior = [z for d, z in runs if d < as_of_date]
    if not prior:
        return None
    take = prior[-n_recent:]
    return sum(take) / len(take)
