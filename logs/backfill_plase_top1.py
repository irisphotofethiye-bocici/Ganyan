"""One-shot: insert + grade plase_top1 picks for races that already have
the other four strategies in the picks ledger.

Run once after the plase_top1 strategy was added to STRATEGIES (2026-05-06).
Idempotent — generate_picks_for_race only adds missing strategies and
grade_all_pending only touches ungraded rows.

    uv run python logs/backfill_plase_top1.py
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import distinct

from ganyan.db import get_session
from ganyan.db.models import Pick
from ganyan.predictor.picks import generate_picks_for_race, grade_all_pending


def main(since: date | None = None) -> None:
    session = get_session()
    try:
        q = session.query(distinct(Pick.race_id))
        if since is not None:
            q = q.filter(Pick.generated_at >= since)
        race_ids = [rid for (rid,) in q.all()]
        print(f"Visiting {len(race_ids)} races…")

        added = 0
        for i, race_id in enumerate(race_ids, 1):
            picks = generate_picks_for_race(session, race_id, refresh=False)
            added += len(picks)
            if i % 200 == 0:
                session.commit()
                print(f"  {i}/{len(race_ids)} races, {added} picks added")
        session.commit()
        print(f"Total added: {added} picks")

        graded = grade_all_pending(session)
        session.commit()
        print(f"Graded: {graded} picks")
    finally:
        session.close()


if __name__ == "__main__":
    main()
