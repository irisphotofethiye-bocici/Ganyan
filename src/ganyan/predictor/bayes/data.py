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


def matrices_for_pymc(frame: TrainingFrame):
    """Build padded matrices for vectorized PL loglik.

    Returns a dict with:
      horse_idx_mat:   (R, K_max) int  — horse indices, 0 padded
      jockey_idx_mat:  (R, K_max) int  — jockey indices, 0 padded
      track_dist_arr:  (R,)       int
      agf_z_mat:       (R, K_max) float — within-race z-score, 0 padded
      valid_mask:      (R, K_max) bool
      horse_to_sire:   (n_horses,) int
    """
    import numpy as np

    race_ids = list(frame.orderings.keys())
    R = len(race_ids)
    K_max = max(len(frame.orderings[rid]) for rid in race_ids) if R else 0
    n_horses = len(frame.horse_index)

    horse_idx_mat = np.zeros((R, K_max), dtype="int64")
    jockey_idx_mat = np.zeros((R, K_max), dtype="int64")
    valid_mask = np.zeros((R, K_max), dtype="bool")
    agf_z_mat = np.zeros((R, K_max), dtype="float64")
    track_dist_arr = np.zeros(R, dtype="int64")

    has_jockey = len(frame.jockey_of_horse_in_race) > 0
    has_agf = len(frame.agf_of_horse_in_race) > 0
    has_track_dist = len(frame.track_dist_of_race) > 0

    flat_idx = 0
    for r, rid in enumerate(race_ids):
        order = frame.orderings[rid]
        n = len(order)
        horse_idx_mat[r, :n] = order
        if has_jockey:
            jockey_idx_mat[r, :n] = frame.jockey_of_horse_in_race[
                flat_idx : flat_idx + n
            ]
        if has_track_dist:
            track_dist_arr[r] = frame.track_dist_of_race[rid]
        if has_agf:
            agf_slice = np.asarray(
                frame.agf_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            )
            if agf_slice.std() > 1e-9:
                agf_z = (agf_slice - agf_slice.mean()) / agf_slice.std()
            else:
                agf_z = np.zeros_like(agf_slice)
            agf_z_mat[r, :n] = agf_z
        valid_mask[r, :n] = True
        flat_idx += n

    horse_to_sire = np.zeros(n_horses, dtype="int64")
    seen: set[int] = set()
    flat_horses = [h for order in frame.orderings.values() for h in order]
    for h, s in zip(flat_horses, frame.sire_of_horse_in_race):
        if h not in seen:
            horse_to_sire[h] = s
            seen.add(h)

    return {
        "horse_idx_mat": horse_idx_mat,
        "jockey_idx_mat": jockey_idx_mat,
        "track_dist_arr": track_dist_arr,
        "agf_z_mat": agf_z_mat,
        "valid_mask": valid_mask,
        "horse_to_sire": horse_to_sire,
    }


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
