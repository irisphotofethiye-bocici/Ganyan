"""Rolling kept-race ROI halt — fires when realized PnL on a strategy crosses
a negative threshold over the last N days with sufficient sample size.

Closes failure mode 02 from the 2026-05-02 premortem: the asymmetric trip wire
in trip_wire.py reads only model-prob z-score, never realized PnL. The May 2026
OOS retest already proved kept-race ROI is -20% to -30% on uclu_box6 and
sirali_ikili_top1; this canary surfaces that empirical reality back into the
advice gate.

Mirrors the query pattern of kelly.strategy_edge_stats (see kelly.py:79-141).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from ganyan.db.models import Pick


DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_MIN_N = 40
DEFAULT_HALT_THRESHOLD = -0.10  # halt when realized ROI <= -10%


def compute(
    session: Session,
    strategy: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_n: int = DEFAULT_MIN_N,
    halt_threshold: float = DEFAULT_HALT_THRESHOLD,
    now: Optional[datetime] = None,
) -> Optional[str]:
    """Return halt reason string if the strategy should be halted, else None."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    rows = (
        session.query(Pick)
        .filter(
            Pick.strategy == strategy,
            Pick.graded.is_(True),
            Pick.generated_at >= cutoff,
        )
        .all()
    )

    n = len(rows)
    if n < min_n:
        return None

    sum_stake = sum(float(p.stake_tl or 0) for p in rows)
    sum_payout = sum(float(p.payout_tl or 0) for p in rows)
    if sum_stake <= 0:
        return None

    roi = (sum_payout - sum_stake) / sum_stake
    if roi <= halt_threshold:
        return (
            f"{strategy}: rolling {lookback_days}d ROI {roi:+.1%} on n={n} "
            f"<= halt threshold {halt_threshold:+.0%}"
        )
    return None


def check_all_strategies(
    session: Session,
    strategies: Iterable[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_n: int = DEFAULT_MIN_N,
    halt_threshold: float = DEFAULT_HALT_THRESHOLD,
    now: Optional[datetime] = None,
) -> dict[str, Optional[str]]:
    """Run compute() for each strategy. Returns dict {strategy: reason_or_None}."""
    return {
        s: compute(session, s, lookback_days, min_n, halt_threshold, now)
        for s in strategies
    }
