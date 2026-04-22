"""Kelly-fraction stake sizing for the betting strategies.

For a bet with win probability ``p`` and decimal-odds payout multiplier
``b = payout/stake``, the Kelly-optimal fraction of bankroll to wager is

    K = (b * p - (1 - p)) / b

This maximises expected log-growth.  In practice full Kelly is too
volatile (it drawdowns hard on hit-rate variance), so real bettors use
a fraction of Kelly — typically 1/4 Kelly.

Per-race, ``p`` is the model's win probability for the specific pick
(e.g. ``Pick.model_prob_pct / 100``) and ``b`` is the historical
average winning-ticket payout multiplier for the strategy.  We can't
forecast this race's pool exactly, so we use the 3-month rolling
average as a stable point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import Pick, Race


@dataclass
class StrategyEdgeStats:
    """Historical edge summary a Kelly calc needs per strategy."""

    strategy: str
    n_graded: int
    n_hits: int
    hit_rate: float                    # observed win fraction
    avg_winning_payout_tl: float       # mean payout on winning tickets
    avg_stake_tl: float                # stake per ticket (constant per strategy)
    avg_b: float                       # (avg_winning_payout / avg_stake) - 1
    avg_model_prob: float              # mean model_prob_pct/100 across graded picks
    calibration_factor: float          # hit_rate / avg_model_prob (=1 if well-calibrated)

    def calibrate(self, raw_model_prob: float, max_multiplier: float = 3.0) -> float:
        """Scale a raw model prob so the mean matches empirical hit rate.

        Preserves the *ranking* of the model's confidence across races
        (a 0.5% raw stays below a 2% raw) while aligning the overall
        level with what actually happens, so Kelly gets a usable p.

        Capped at ``max_multiplier * hit_rate`` to stop an outlier raw
        prob from producing a full-bankroll Kelly bet.  Default 3×:
        a pick gets rated at most 3× as likely to win as the strategy's
        historical average.  For uclu_top1 (hit rate 4.2%) that caps at
        12.6% calibrated prob; for uclu_box6 (15.3%) at 45.9%.  Below
        the cap, calibration is linear in raw prob so Kelly still
        varies with confidence.
        """
        if self.avg_model_prob <= 0 or self.hit_rate <= 0:
            return 0.0
        scaled = raw_model_prob * self.calibration_factor
        cap = max_multiplier * self.hit_rate
        return max(0.0, min(scaled, cap, 1.0))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_graded": self.n_graded,
            "n_hits": self.n_hits,
            "hit_rate": round(self.hit_rate, 4),
            "avg_winning_payout_tl": round(self.avg_winning_payout_tl, 2),
            "avg_stake_tl": round(self.avg_stake_tl, 2),
            "avg_b": round(self.avg_b, 4),
            "avg_model_prob": round(self.avg_model_prob, 4),
            "calibration_factor": round(self.calibration_factor, 3),
        }


def strategy_edge_stats(
    session: Session,
    strategies: Iterable[str] = ("uclu_top1", "uclu_box6", "sirali_ikili_top1"),
    lookback_days: int = 90,
) -> dict[str, StrategyEdgeStats]:
    """Compute average winning payout + hit rate per strategy.

    Only graded, hit picks contribute to ``avg_winning_payout_tl``; the
    ``hit_rate`` divides by the full graded sample so it reflects real
    win fraction.
    """
    since = _date.today() - timedelta(days=lookback_days)
    out: dict[str, StrategyEdgeStats] = {}
    for strat in strategies:
        rows = (
            session.query(Pick)
            .join(Race, Race.id == Pick.race_id)
            .filter(
                Pick.strategy == strat,
                Pick.graded == True,  # noqa: E712
                Race.date >= since,
            )
            .all()
        )
        n = len(rows)
        if n == 0:
            continue
        stakes = [float(p.stake_tl) for p in rows]
        avg_stake = sum(stakes) / n
        wins = [p for p in rows if p.hit]
        n_hits = len(wins)
        win_payouts = [float(p.payout_tl or 0) for p in wins]
        avg_win_pay = sum(win_payouts) / n_hits if n_hits else 0.0
        hit_rate = n_hits / n
        b = (avg_win_pay / avg_stake - 1.0) if avg_stake > 0 else 0.0
        model_probs = [
            float(p.model_prob_pct) / 100.0
            for p in rows if p.model_prob_pct is not None
        ]
        avg_model_prob = sum(model_probs) / len(model_probs) if model_probs else 0.0
        calibration = (hit_rate / avg_model_prob) if avg_model_prob > 0 else 1.0
        out[strat] = StrategyEdgeStats(
            strategy=strat,
            n_graded=n,
            n_hits=n_hits,
            hit_rate=hit_rate,
            avg_winning_payout_tl=avg_win_pay,
            avg_stake_tl=avg_stake,
            avg_b=b,
            avg_model_prob=avg_model_prob,
            calibration_factor=calibration,
        )
    return out


def kelly_fraction(
    win_prob: float,
    b: float,
    kelly_multiplier: float = 0.25,
) -> float:
    """Fractional-Kelly fraction of bankroll to wager.

    Returns a value in ``[0, 1]``.  Zero means "no edge, skip".  Full
    Kelly (``kelly_multiplier=1.0``) is mathematically optimal for
    log-growth but is brutal in practice — variance of horse-racing
    exotic pools will drawdown a full-Kelly bettor ~50%+ routinely.
    Default ``0.25`` (quarter-Kelly) is the standard compromise.
    """
    if win_prob <= 0 or b <= 0:
        return 0.0
    q = 1.0 - win_prob
    raw = (b * win_prob - q) / b
    if raw <= 0:
        return 0.0
    scaled = raw * kelly_multiplier
    return min(max(scaled, 0.0), 1.0)


def suggested_stake_tl(
    win_prob: float,
    b: float,
    bankroll_tl: float,
    base_stake_tl: float,
    kelly_multiplier: float = 0.25,
) -> float:
    """Convert Kelly fraction into a concrete suggested stake.

    Clamped to the interval ``[0, base_stake_tl]``: we never recommend
    staking *more* than the flat-per-ticket baseline — Kelly is a cap,
    not a stake-up signal.  The idea is to let Kelly shrink (or skip)
    low-edge bets while keeping the standard ticket size on strong ones.
    """
    k = kelly_fraction(win_prob, b, kelly_multiplier=kelly_multiplier)
    if k <= 0:
        return 0.0
    raw = k * bankroll_tl
    return min(raw, base_stake_tl)
