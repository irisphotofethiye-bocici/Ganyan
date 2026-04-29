"""Build training/holdout frames for the Bayesian Plackett-Luce model.

Encode horse, jockey, sire, and (track × distance-bucket) as dense
integer indices so PyMC's coords machinery can use them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import Race


DISTANCE_BUCKETS: Tuple[int, ...] = (1300, 1600, 1900, 2400)


def distance_bucket_for(meters: int) -> int:
    for i, boundary in enumerate(DISTANCE_BUCKETS):
        if meters < boundary:
            return i
    return len(DISTANCE_BUCKETS)


@dataclass
class TrainingFrame:
    orderings: Dict[int, List[int]] = field(default_factory=dict)
    horse_index: Dict[int, int] = field(default_factory=dict)
    jockey_index: Dict[str, int] = field(default_factory=dict)
    sire_index: Dict[str, int] = field(default_factory=dict)
    track_dist_index: Dict[Tuple[int, int], int] = field(default_factory=dict)
    jockey_of_horse_in_race: List[int] = field(default_factory=list)
    sire_of_horse_in_race: List[int] = field(default_factory=list)
    track_dist_of_race: Dict[int, int] = field(default_factory=dict)
    agf_of_horse_in_race: List[float] = field(default_factory=list)


def _intern(idx: Dict, key) -> int:
    if key not in idx:
        idx[key] = len(idx)
    return idx[key]


def build_training_frame(
    session: Session,
    from_date: date,
    to_date: date,
    min_field_size: int = 3,
) -> TrainingFrame:
    frame = TrainingFrame()
    _intern(frame.sire_index, "")

    races = session.execute(
        select(Race).where(
            Race.date >= from_date,
            Race.date <= to_date,
        ).order_by(Race.date, Race.race_number)
    ).scalars().all()

    for r in races:
        finishers = [
            e for e in r.entries
            if e.finish_position is not None and e.jockey is not None
        ]
        if len(finishers) < min_field_size:
            continue
        finishers.sort(key=lambda e: e.finish_position)
        horse_ids: List[int] = []
        jockey_ids: List[int] = []
        sire_ids: List[int] = []
        agfs: List[float] = []
        for e in finishers:
            horse_ids.append(_intern(frame.horse_index, e.horse_id))
            jockey_ids.append(_intern(frame.jockey_index, e.jockey))
            sire_name = (e.horse.sire or "") if e.horse else ""
            sire_ids.append(_intern(frame.sire_index, sire_name))
            agfs.append(float(e.agf) if e.agf is not None else 0.0)
        track_dist = (r.track_id, distance_bucket_for(r.distance_meters or 0))
        frame.track_dist_of_race[r.id] = _intern(frame.track_dist_index, track_dist)
        frame.orderings[r.id] = horse_ids
        frame.jockey_of_horse_in_race.extend(jockey_ids)
        frame.sire_of_horse_in_race.extend(sire_ids)
        frame.agf_of_horse_in_race.extend(agfs)

    return frame
