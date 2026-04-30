"""Beyer-style speed figures: adjust horse finish times for daily track variant.

Idea (paraphrased from Andrew Beyer's classic methodology):
- Pist x mesafe için "par" zamanı = tarihsel ortalama kazananın saniyesi.
- Pist günlük "variant" = o gün o pistteki kazanan ortalama saniye − par.
- Düzeltilmiş zaman = ham finish_time − variant.  (Yavaş gün = pozitif
  variant, atın zamanından düşülür → atı normalize ederiz.)

We expose two things:
- ``compute_track_variants(session)``  → dict[(track_id, dist_bucket, date)] → variant
- ``build_horse_speed_history(session, variants)`` → dict[horse_id] → sorted list of
  ``(race_date, sec_per_meter_adjusted)``.  Caller takes a recency-weighted
  mean of the entries with date < target → per-horse "speed score" in
  seconds-per-meter (lower = faster).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry
from ganyan.predictor.bayes.data import distance_bucket_for
from ganyan.scraper.parser import parse_eid_to_seconds


VariantKey = Tuple[int, int, date]   # (track_id, distance_bucket, race_date)


def compute_track_variants(
    session: Session,
    from_date: date | None = None,
    to_date: date | None = None,
) -> Dict[VariantKey, float]:
    """Compute daily track variants per (track, distance_bucket, date).

    variant = mean(winning_time_seconds for races that day at this track
              and distance bucket)
              − global_mean(winning_time_seconds for this track+bucket).

    Positive variant = slower-than-usual day; negative = faster.
    """
    q = (
        select(
            Race.track_id, Race.distance_meters, Race.date,
            RaceEntry.finish_time,
        )
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .where(RaceEntry.finish_position == 1)
        .where(RaceEntry.finish_time.is_not(None))
        .where(Race.distance_meters.is_not(None))
    )
    if from_date is not None:
        q = q.where(Race.date >= from_date)
    if to_date is not None:
        q = q.where(Race.date <= to_date)

    # daily_times[(track, bucket, day)] = list of winning seconds
    daily_times: Dict[VariantKey, List[float]] = defaultdict(list)
    # global_times[(track, bucket)] = list across all days
    global_times: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    for track_id, distance_m, race_date, ftime in session.execute(q):
        secs = parse_eid_to_seconds(ftime)
        if secs is None or secs <= 0 or distance_m is None:
            continue
        bucket = distance_bucket_for(distance_m)
        daily_times[(track_id, bucket, race_date)].append(secs)
        global_times[(track_id, bucket)].append(secs)

    par_by_tb = {
        k: sum(v) / len(v) for k, v in global_times.items() if v
    }

    variants: Dict[VariantKey, float] = {}
    for (track_id, bucket, race_date), times in daily_times.items():
        par = par_by_tb.get((track_id, bucket))
        if par is None or not times:
            continue
        variants[(track_id, bucket, race_date)] = (
            sum(times) / len(times) - par
        )
    return variants


def build_horse_speed_history(
    session: Session,
    variants: Dict[VariantKey, float],
    from_date: date | None = None,
    to_date: date | None = None,
) -> Dict[int, List[Tuple[date, float]]]:
    """Per horse, produce sorted (race_date, adjusted_sec_per_meter) list.

    sec_per_meter is the standardised "speed" (lower = faster) — adjusted
    finish_time divided by race distance, comparable across distances.
    """
    q = (
        select(
            RaceEntry.horse_id, Race.date, Race.track_id,
            Race.distance_meters, RaceEntry.finish_time,
        )
        .join(Race, Race.id == RaceEntry.race_id)
        .where(RaceEntry.finish_time.is_not(None))
        .where(Race.distance_meters.is_not(None))
    )
    if from_date is not None:
        q = q.where(Race.date >= from_date)
    if to_date is not None:
        q = q.where(Race.date <= to_date)

    history: Dict[int, List[Tuple[date, float]]] = defaultdict(list)
    for horse_id, race_date, track_id, distance_m, ftime in session.execute(q):
        secs = parse_eid_to_seconds(ftime)
        if secs is None or secs <= 0 or not distance_m:
            continue
        bucket = distance_bucket_for(distance_m)
        variant = variants.get((track_id, bucket, race_date), 0.0)
        adjusted = secs - variant
        spm = adjusted / float(distance_m)  # seconds per meter
        history[horse_id].append((race_date, spm))

    for hid in history:
        history[hid].sort(key=lambda t: t[0])
    return history


def horse_speed_score(
    history: Dict[int, List[Tuple[date, float]]],
    horse_id: int,
    as_of_date: date,
    n_recent: int = 3,
) -> float | None:
    """Mean adjusted sec/meter over the last ``n_recent`` races strictly
    before ``as_of_date``.  None when the horse has no prior runs.
    Lower = faster.
    """
    runs = history.get(horse_id)
    if not runs:
        return None
    prior = [spm for d, spm in runs if d < as_of_date]
    if not prior:
        return None
    take = prior[-n_recent:]
    return sum(take) / len(take)
