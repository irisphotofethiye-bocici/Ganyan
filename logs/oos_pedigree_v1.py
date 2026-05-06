"""Head-to-head OOS comparison: live model vs pedigree_v1.

Both models trained on different windows. The shared OOS window is
[2025-01-01, 2025-05-04] — before the pedigree_v1 training start.

Bar to ship pedigree_v1 to live: top-1 hit rate ≥ +1pp on this window
(per feedback_oos_test_required.md). Window must be ≥365 days AND ≥1500
races (per V2 retraction; enforced by assert_min_window).
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select
from ganyan.db.session import get_session
from ganyan.db.models import Race, RaceEntry, RaceStatus
from ganyan.predictor.ml import MLPredictor, load_latest_model

# OOS window: from start-of-2025 up to the day before pedigree_v1's training cutoff.
# pedigree_v1 trains FROM_DATE = today - 90 days; so OOS ends today - 91 days.
# Window will be ~400 days, comfortably above the 365-day bar.
FROM_DATE = date(2025, 1, 1)
TO_DATE = date.today() - timedelta(days=91)
PROGRESS_EVERY = 200

MIN_WINDOW_DAYS = 365
MIN_RACE_COUNT = 1500


def assert_min_window(from_date, to_date, n_races: int) -> None:
    """Same bar enforced by discordance_oos_backtest."""
    days = (to_date - from_date).days + 1
    assert days >= MIN_WINDOW_DAYS, (
        f"OOS window too short: {days} days < {MIN_WINDOW_DAYS}"
    )
    assert n_races >= MIN_RACE_COUNT, (
        f"OOS race count too low: {n_races} < {MIN_RACE_COUNT}"
    )


def evaluate(predictor: MLPredictor, race_ids: list[int], session) -> dict:
    """Return {top1_hits, top3_hits, n} for the predictor across these races.

    Dead-heat winners (multiple horses with finish_position=1) count as a
    hit when the model's top-1 is ANY of the tied winners.
    """
    top1_hits = top3_hits = scored = 0
    for i, rid in enumerate(race_ids, 1):
        try:
            preds = predictor.predict(rid)
        except Exception:
            continue
        if not preds:
            continue
        winner_hids = {row[0] for row in session.execute(
            select(RaceEntry.horse_id).where(
                RaceEntry.race_id == rid,
                RaceEntry.finish_position == 1,
            )
        ).all()}
        if not winner_hids:
            continue
        scored += 1
        top3_hids = {p.horse_id for p in preds[:3]}
        if preds[0].horse_id in winner_hids:
            top1_hits += 1
        if winner_hids & top3_hids:
            top3_hits += 1
        if i % PROGRESS_EVERY == 0:
            print(f"    [{i}/{len(race_ids)}] top1={top1_hits}/{scored}={top1_hits/max(scored,1)*100:.1f}%", flush=True)
    return {"n": scored, "top1_hits": top1_hits, "top3_hits": top3_hits}


def main() -> None:
    print(f"=== oos_pedigree_v1 start {time.strftime('%H:%M:%S')} ===")
    print(f"window: {FROM_DATE} to {TO_DATE}")

    s = get_session()
    # Filter to full-field resulted races (≥4 entries) — matches the bar
    # used in discordance_oos_backtest.py to exclude degenerate fields.
    from sqlalchemy import func, text
    rows = s.execute(text("""
        SELECT r.id
        FROM races r
        JOIN race_entries re ON re.race_id = r.id
        WHERE r.date >= :from_date
          AND r.date <= :to_date
          AND r.status = 'resulted'
        GROUP BY r.id, r.date
        HAVING COUNT(re.id) >= 4
        ORDER BY r.date, r.id
    """), {"from_date": FROM_DATE, "to_date": TO_DATE}).fetchall()
    races = [r[0] for r in rows]
    print(f"OOS resulted races in window: {len(races)}")
    assert_min_window(FROM_DATE, TO_DATE, len(races))

    # Load both models
    print("\nLoading baseline (live) model...")
    baseline_model = load_latest_model(model_dir=Path("models"), model_name="lightgbm_ranker")
    baseline_predictor = MLPredictor(s, model=baseline_model)
    print(f"  baseline feature_columns: {len(baseline_model.feature_columns)} columns")

    print("Loading pedigree_v1 model...")
    pedigree_model = load_latest_model(
        model_dir=Path("models/pedigree_v1"),
        model_name="lightgbm_ranker_pedigree",
    )
    pedigree_predictor = MLPredictor(s, model=pedigree_model)
    print(f"  pedigree_v1 feature_columns: {len(pedigree_model.feature_columns)} columns")

    print("\n=== Evaluating BASELINE ===")
    t0 = time.time()
    baseline_stats = evaluate(baseline_predictor, races, s)
    t_baseline = time.time() - t0
    print(f"  done in {t_baseline/60:.1f}min")

    print("\n=== Evaluating PEDIGREE_V1 ===")
    t0 = time.time()
    pedigree_stats = evaluate(pedigree_predictor, races, s)
    t_pedigree = time.time() - t0
    print(f"  done in {t_pedigree/60:.1f}min")

    # Report
    print("\n=== Comparison on OOS window ===")
    for label, st in (("BASELINE", baseline_stats), ("PEDIGREE_V1", pedigree_stats)):
        if st["n"] == 0:
            print(f"  {label}: no scored races")
            continue
        t1 = st["top1_hits"] / st["n"] * 100
        t3 = st["top3_hits"] / st["n"] * 100
        print(f"  {label:<13} n={st['n']:>5}  top1={t1:>5.2f}%  top3={t3:>5.2f}%")

    if baseline_stats["n"] > 0 and pedigree_stats["n"] > 0:
        # Use the smaller n for comparison (race set may differ slightly if predict skipped any)
        b1 = baseline_stats["top1_hits"] / baseline_stats["n"]
        p1 = pedigree_stats["top1_hits"] / pedigree_stats["n"]
        b3 = baseline_stats["top3_hits"] / baseline_stats["n"]
        p3 = pedigree_stats["top3_hits"] / pedigree_stats["n"]
        print(f"\n  top-1 lift: {(p1 - b1) * 100:+.2f}pp")
        print(f"  top-3 lift: {(p3 - b3) * 100:+.2f}pp")
        print(f"\n  Ship bar: top-1 lift ≥ +1pp ⇒ {'PASS' if (p1 - b1) >= 0.01 else 'FAIL'}")

    out = Path("logs/oos_pedigree_v1_results.json")
    out.write_text(json.dumps({
        "window": [str(FROM_DATE), str(TO_DATE)],
        "n_races_in_window": len(races),
        "baseline": baseline_stats,
        "pedigree_v1": pedigree_stats,
    }, indent=2))
    print(f"\nresults saved: {out}")


if __name__ == "__main__":
    main()
