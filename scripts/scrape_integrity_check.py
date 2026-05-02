"""Daily scrape-integrity canary — fingerprint stability of agf and
related categorical fields against a 7-day rolling baseline.

Closes premortem failure mode 04. The four prior scraper bugs in memory
(gate mismap, pre-11:30 NULL, results-parse, picks staleness) all wrote
*plausible* values rather than missing ones — schema validation passes,
NULL counts look normal, but the model degrades. This canary catches
distribution shifts that signal silent column-misalignment.

Sets the halt flag on any of:
- today's agf mean drifts >2pp from 7-day rolling mean per track
- last_six string format match rate (must match ^[0-9KDÇ-]+$) drops below 95%
"""

from __future__ import annotations

import re
import sys
from datetime import date, timedelta

from sqlalchemy import func

from ganyan.db.models import Race, RaceEntry
from ganyan.db.session import get_session
from ganyan.predictor import halt_flag


LAST_SIX_RE = re.compile(r"^[0-9KDÇ-]+$")
AGF_DRIFT_PP_THRESHOLD = 2.0
LAST_SIX_FORMAT_THRESHOLD = 0.95


def _today_agf_per_track(session, today):
    rows = (
        session.query(Race.track_id, func.avg(RaceEntry.agf))
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(Race.date == today, RaceEntry.agf.isnot(None))
        .group_by(Race.track_id)
        .all()
    )
    return {tid: float(m or 0) for tid, m in rows}


def _baseline_agf_per_track(session, today, lookback=7):
    cutoff = today - timedelta(days=lookback)
    rows = (
        session.query(Race.track_id, func.avg(RaceEntry.agf))
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(Race.date >= cutoff, Race.date < today, RaceEntry.agf.isnot(None))
        .group_by(Race.track_id)
        .all()
    )
    return {tid: float(m or 0) for tid, m in rows}


def _last_six_format_rate(session, today):
    entries = (
        session.query(RaceEntry.last_six)
        .join(Race)
        .filter(Race.date == today, RaceEntry.last_six.isnot(None))
        .all()
    )
    if not entries:
        return 1.0
    matches = sum(1 for (s,) in entries if LAST_SIX_RE.match(str(s).strip()))
    return matches / len(entries)


def main() -> int:
    today = date.today()
    with get_session() as session:
        today_agf = _today_agf_per_track(session, today)
        baseline_agf = _baseline_agf_per_track(session, today)

        for track_id, today_mean in today_agf.items():
            base = baseline_agf.get(track_id)
            if base is None:
                continue
            drift = abs(today_mean - base)
            if drift > AGF_DRIFT_PP_THRESHOLD:
                reason = (
                    f"scrape_integrity: track {track_id} AGF mean drifted "
                    f"{drift:.2f}pp ({today_mean:.2f} vs 7d baseline {base:.2f})"
                )
                halt_flag.set_halt(reason=reason, source="scrape_integrity_check")
                print(reason, file=sys.stderr)
                return 1

        format_rate = _last_six_format_rate(session, today)
        if format_rate < LAST_SIX_FORMAT_THRESHOLD:
            reason = (
                f"scrape_integrity: last_six format match rate {format_rate:.1%} "
                f"< threshold {LAST_SIX_FORMAT_THRESHOLD:.0%}"
            )
            halt_flag.set_halt(reason=reason, source="scrape_integrity_check")
            print(reason, file=sys.stderr)
            return 1

    print(f"scrape_integrity OK: {today.isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
