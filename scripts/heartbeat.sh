#!/usr/bin/env bash
# Daily 12:00 heartbeat — sets /tmp/ganyan-halt.flag on liveness/integrity failures.
#
# Checks:
#   1. /ops/health responds 200
#   2. today's race_entries have non-null agf (RaceEntry.agf, not agf_pct)
#   3. today's predictions have stddev > 0.01 per race (no uniform 1/N fallback)
#
# Halt-flag write preserves first-writer-wins: if a flag is already present,
# the existing reason (root cause) is preserved and we only log the new symptom.
# Still exits 1 on any failure so launchd marks the job non-zero.
#
# Exits 0 on success (no halt set), 1 on any failure.
set -u

HALT_FLAG_PATH="${GANYAN_HALT_FLAG_PATH:-/tmp/ganyan-halt.flag}"
HEALTH_URL="${GANYAN_HEALTH_URL:-http://localhost:5003/ops/health}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

set_halt() {
  local reason="$1"
  if [ ! -f "$HALT_FLAG_PATH" ]; then
    printf '{"reason": "%s", "source": "heartbeat.sh", "timestamp": "%s"}\n' \
      "$reason" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$HALT_FLAG_PATH"
    echo "HALT SET: $reason" >&2
  else
    echo "HALT ALREADY SET (preserving existing); detected: $reason" >&2
  fi
  exit 1
}

# Check 1: /ops/health
if ! curl -sf -o /dev/null "$HEALTH_URL"; then
  set_halt "heartbeat: /ops/health unreachable or non-200"
fi

# Check 2: today has non-null agf rows
NULL_COUNT=$(uv run python -c "
from datetime import date
from ganyan.db.session import get_session
from ganyan.db.models import RaceEntry, Race
with get_session() as s:
    q = s.query(RaceEntry).join(Race).filter(Race.date == date.today(), RaceEntry.agf.is_(None))
    print(q.count())
" 2>&1)

if [ -z "$NULL_COUNT" ] || ! [[ "$NULL_COUNT" =~ ^[0-9]+$ ]]; then
  set_halt "heartbeat: cannot query race_entries for today"
fi

if [ "$NULL_COUNT" -gt 0 ]; then
  TOTAL=$(uv run python -c "
from datetime import date
from ganyan.db.session import get_session
from ganyan.db.models import RaceEntry, Race
with get_session() as s:
    print(s.query(RaceEntry).join(Race).filter(Race.date == date.today()).count())
")
  if [ "$NULL_COUNT" = "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
    set_halt "heartbeat: 100% of today's race_entries have NULL agf ($TOTAL/$TOTAL)"
  fi
fi

# Check 3: per-race prediction uniformity guard
UNIFORM_RESULT=$(uv run python -c "
from datetime import date
from ganyan.db.session import get_session
from ganyan.db.models import Race
from ganyan.predictor.uniformity_guard import is_uniform
with get_session() as s:
    races = s.query(Race).filter(Race.date == date.today()).all()
    bad = []
    for r in races:
        probs = [float(e.predicted_probability or 0) for e in r.entries]
        if probs and is_uniform(probs):
            bad.append(r.id)
    print(','.join(map(str, bad)) if bad else 'OK')
")

if [ "$UNIFORM_RESULT" != "OK" ] && [ -n "$UNIFORM_RESULT" ]; then
  set_halt "heartbeat: uniform predictions detected on race(s) $UNIFORM_RESULT"
fi

echo "heartbeat OK: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
