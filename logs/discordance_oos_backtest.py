"""Out-of-sample discordance backtest on the post-backfill 2025+ history.

Live model trained on 2026-01-31 onwards (per models/lightgbm_ranker.meta.json),
so races BEFORE 2026-01-31 are properly out-of-sample for this model.

For each full-field resulted race in [2025-01-01, 2026-01-30]:
  - Run EnsemblePredictor (the production-path predictor)
  - Identify model_top1 (highest mean_probability)
  - Identify agf_top1 (highest agf among entries)
  - Identify actual winner (finish_position==1)
  - Bucket: concordant (model_top1 == agf_top1) vs discordant
  - Compute Ganyan strategy ROI per bucket assuming 1 TL stake on model_top1

Reports aggregate ROI + bootstrap 95% CI per bucket.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text
from ganyan.db.session import get_session
from ganyan.predictor.ml.ensemble import EnsemblePredictor

CUTOFF = date(2026, 1, 31)
FROM_DATE = date(2025, 1, 1)
STAKE_TL = 1.0
PROGRESS_EVERY = 200

MIN_WINDOW_DAYS = 365
MIN_RACE_COUNT = 1500


def assert_min_window(from_date, cutoff, n_races: int) -> None:
    """Enforce the V2-aftermath bar: ≥365 days AND ≥1500 races.

    Raises AssertionError if a one-off branch tries to shortcut the OOS window.
    Memory: feedback_oos_test_required (2026-05-02).
    """
    days = (cutoff - from_date).days
    assert days >= MIN_WINDOW_DAYS, (
        f"OOS window too short: {days} days < {MIN_WINDOW_DAYS} days. "
        "The V2 features failure (in-sample +3.6pp, OOS +0.07pp on 1500 races) "
        "settled the bar — partial windows replicate the same overfitting."
    )
    assert n_races >= MIN_RACE_COUNT, (
        f"OOS race count too low: {n_races} < {MIN_RACE_COUNT} races. "
        "Bar set by V2 retraction."
    )


@dataclass
class Bucket:
    name: str
    races: int = 0
    hits: int = 0
    stake_total: float = 0.0
    payout_total: float = 0.0
    per_race_pnl: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.races if self.races else 0.0

    @property
    def roi(self) -> float:
        return (self.payout_total - self.stake_total) / self.stake_total if self.stake_total else 0.0

    def bootstrap_roi_ci(self, n_boot: int = 2000) -> tuple[float, float]:
        if not self.per_race_pnl:
            return (0.0, 0.0)
        rng = random.Random(42)
        n = len(self.per_race_pnl)
        rois = []
        stake_per_race = STAKE_TL  # constant
        for _ in range(n_boot):
            sample = [self.per_race_pnl[rng.randrange(n)] for _ in range(n)]
            rois.append(sum(sample) / (n * stake_per_race))
        rois.sort()
        return (rois[int(0.025 * n_boot)], rois[int(0.975 * n_boot)])


def main() -> None:
    started = time.time()
    print(f"=== discordance OOS backtest start {time.strftime('%H:%M:%S')} ===")
    print(f"window: {FROM_DATE} to {CUTOFF - timedelta(days=1)}")
    s = get_session()

    # Pull all full-field OOS resulted races with payout
    races = s.execute(text(f"""
        SELECT r.id, r.date, r.ganyan_payout_tl
        FROM races r
        JOIN race_entries re ON re.race_id=r.id
        WHERE r.date >= '{FROM_DATE}'
          AND r.date < '{CUTOFF}'
          AND r.status='resulted'
          AND r.ganyan_payout_tl IS NOT NULL
        GROUP BY r.id, r.date, r.ganyan_payout_tl
        HAVING COUNT(re.id) >= 4
        ORDER BY r.date, r.id
    """)).fetchall()
    print(f"OOS full-field resulted races with payout: {len(races)}")
    assert_min_window(FROM_DATE, CUTOFF, len(races))

    p = EnsemblePredictor(s)
    concord = Bucket("CONCORDANT")
    discord = Bucket("DISCORDANT")
    skipped_no_agf = 0
    skipped_no_finish = 0

    for i, (race_id, race_date, payout_tl) in enumerate(races, start=1):
        # Per-race entries: model probs (via predict) + agf + finish_position
        try:
            preds = p.predict(race_id)
        except Exception as e:
            print(f"  race {race_id}: predict failed: {e}")
            continue
        if not preds:
            continue

        # Map hid -> mean_probability
        model_probs = {pred.horse_id: pred.mean_probability for pred in preds}
        if not model_probs:
            continue
        model_top1_hid = max(model_probs, key=model_probs.get)

        # Pull entries for AGF + finish_position
        entries = s.execute(text(f"""
            SELECT horse_id, agf, finish_position FROM race_entries
            WHERE race_id={race_id} AND horse_id IS NOT NULL
        """)).fetchall()

        # AGF top1
        agf_entries = [(e[0], float(e[1])) for e in entries if e[1] is not None]
        if not agf_entries:
            skipped_no_agf += 1
            continue
        agf_top1_hid = max(agf_entries, key=lambda x: x[1])[0]

        # Actual winner
        winners = [e[0] for e in entries if e[2] == 1]
        if not winners:
            skipped_no_finish += 1
            continue
        actual_winner_hid = winners[0]

        # Bucket
        bucket = concord if model_top1_hid == agf_top1_hid else discord
        bucket.races += 1
        bucket.stake_total += STAKE_TL
        if model_top1_hid == actual_winner_hid:
            bucket.hits += 1
            payout = STAKE_TL * float(payout_tl)
            bucket.payout_total += payout
            bucket.per_race_pnl.append(payout - STAKE_TL)
        else:
            bucket.per_race_pnl.append(-STAKE_TL)

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            eta = (len(races) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:>5}/{len(races)}] {elapsed/60:.1f}min elapsed, {eta:.1f}min ETA — concord={concord.races} discord={discord.races}")

    elapsed = time.time() - started
    print(f"\n=== done in {elapsed/60:.1f} min ===")
    print(f"skipped (no AGF data): {skipped_no_agf}")
    print(f"skipped (no finisher): {skipped_no_finish}")
    print()

    for b in (concord, discord):
        if b.races == 0:
            continue
        ci_lo, ci_hi = b.bootstrap_roi_ci()
        print(f"{b.name}: n={b.races} ({100*b.races/(concord.races+discord.races):.1f}%)")
        print(f"  hit_rate: {100*b.hit_rate:.2f}% ({b.hits}/{b.races})")
        print(f"  ROI: {100*b.roi:+.2f}%  CI95 [{100*ci_lo:+.2f}%, {100*ci_hi:+.2f}%]")
        print(f"  PnL: stake={b.stake_total:.0f} TL, payout={b.payout_total:.0f} TL, net={b.payout_total-b.stake_total:+.0f} TL")
        print()

    # Save raw data for follow-up
    with open("logs/discordance_oos_results.json", "w") as f:
        json.dump({
            "window": [str(FROM_DATE), str(CUTOFF - timedelta(days=1))],
            "model_trained_from": "2026-01-31",
            "concordant": {
                "n": concord.races, "hits": concord.hits,
                "hit_rate": concord.hit_rate, "roi": concord.roi,
            },
            "discordant": {
                "n": discord.races, "hits": discord.hits,
                "hit_rate": discord.hit_rate, "roi": discord.roi,
            },
        }, f, indent=2)
    print("results saved: logs/discordance_oos_results.json")


if __name__ == "__main__":
    main()
