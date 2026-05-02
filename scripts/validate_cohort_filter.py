"""One-off validation: does the cohort filter actually beat the takeout floor?

Splits graded picks over the 2025-01-01 → 2026-01-30 window into
  (A) kept cohort   — races advice would have surfaced
  (B) filtered-out cohort — races advice would have skipped (Maiden /Dişi,
      field>=13, Şanlıurfa/Bursa)

and compares ROI per strategy. If kept-cohort ROI does not beat
filtered-out cohort ROI by >= 5pp, the +5,870 TL claim from
``project_cohort_filter`` memory is small-sample noise; operator should
flip the default to OFF.

Mirrors the exact ``_cohort_skip_reason`` logic at
  src/ganyan/web/routes.py:1315
  src/ganyan/cli/main.py:2009
(both sites checked 2026-05-02; logic is identical).

Closes premortem failure mode 05.

Output: ``logs/cohort_filter_validation_<DATE>.json`` (committed as
audit trail) plus stdout dump.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from sqlalchemy.orm import joinedload

from ganyan.db import get_session
from ganyan.db.models import Pick, Race


COHORT_SKIP_TRACKS = {"Şanlıurfa", "Bursa"}
COHORT_FIELD_THRESHOLD = 13


def is_in_filter_cohort(race: Race) -> bool:
    """Mirror src/ganyan/web/routes.py:1315 / src/ganyan/cli/main.py:2009.

    Returns True if the race would be filtered out by the cohort filter
    (i.e. advice would skip it).
    """
    rt = race.race_type or ""
    if "Maiden /Dişi" in rt or "Maiden Dişi" in rt:
        return True
    if len(race.entries) >= COHORT_FIELD_THRESHOLD:
        return True
    track_name = race.track.name if race.track else ""
    if track_name in COHORT_SKIP_TRACKS:
        return True
    return False


def _strategy_roi(picks: list[Pick]) -> tuple[float | None, int]:
    if not picks:
        return None, 0
    n = len(picks)
    sum_stake = sum(float(p.stake_tl or 0) for p in picks)
    sum_payout = sum(float(p.payout_tl or 0) for p in picks)
    if sum_stake <= 0:
        return None, n
    return (sum_payout - sum_stake) / sum_stake, n


def _verdict(kept_roi: float | None, filt_roi: float | None) -> str:
    if kept_roi is None or filt_roi is None:
        return "INSUFFICIENT_DATA"
    gap_pp = (kept_roi - filt_roi) * 100
    if gap_pp >= 5:
        return "FILTER_HELPS — keep default ON"
    if gap_pp >= 0:
        return "INCONCLUSIVE — gap < 5pp; consider flipping OFF"
    return "FILTER_HURTS — flip default OFF"


def main() -> int:
    session = get_session()
    all_picks = (
        session.query(Pick)
        .join(Race, Pick.race_id == Race.id)
        .filter(Race.date >= date(2025, 1, 1))
        .filter(Race.date <= date(2026, 1, 30))
        .filter(Pick.graded.is_(True))
        .all()
    )

    # Build {race_id: Race} map with eager-loaded track + entries so
    # is_in_filter_cohort doesn't N+1.
    race_ids = {p.race_id for p in all_picks}
    races_by_id = {
        r.id: r
        for r in (
            session.query(Race)
            .options(joinedload(Race.track), joinedload(Race.entries))
            .filter(Race.id.in_(race_ids))
            .all()
        )
    }

    results: dict[str, dict] = {}
    for strat in ("uclu_box6", "sirali_ikili_top1"):
        strat_picks = [p for p in all_picks if p.strategy == strat]
        kept = [
            p for p in strat_picks
            if not is_in_filter_cohort(races_by_id[p.race_id])
        ]
        filtered = [
            p for p in strat_picks
            if is_in_filter_cohort(races_by_id[p.race_id])
        ]

        kept_roi, kept_n = _strategy_roi(kept)
        filt_roi, filt_n = _strategy_roi(filtered)

        gap_pp = (
            (kept_roi - filt_roi) * 100
            if (kept_roi is not None and filt_roi is not None)
            else None
        )

        results[strat] = {
            "kept_n": kept_n,
            "kept_roi": kept_roi,
            "filtered_n": filt_n,
            "filtered_roi": filt_roi,
            "gap_pp": gap_pp,
            "verdict": _verdict(kept_roi, filt_roi),
        }

    session.close()

    out_path = (
        Path("logs")
        / f"cohort_filter_validation_{date.today().isoformat()}.json"
    )
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\n→ {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
