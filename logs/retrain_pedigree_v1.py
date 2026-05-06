"""Retrain LGBM ranker with the new pedigree + jockey-track features.

Adds 3 features over the live 39-feature model:
  - dam_win_rate
  - dam_surface_rate
  - jockey_track_win_rate

Saves to models/pedigree_v1/ so the live model is untouched. The OOS
backtest at logs/oos_pedigree_v1.py compares this against the live
model on the 2025+ window. Per feedback_oos_test_required.md: ship
only if OOS top-1 lift ≥ +1pp on ≥1500 races / ≥365 days.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

from ganyan.db.session import get_session
from ganyan.predictor.ml.trainer import train_ranker

OUT_DIR = Path("models/pedigree_v1")
OUT_NAME = "lightgbm_ranker_pedigree"
# 90-day window matches the live-model baseline (trained from 2026-01-31, ~95
# days back) and leaves the 2025-01-01 → ~2026-02-04 OOS window free for
# discordance-style evaluation. Training is ~5-10 min on a 90-day window.
FROM_DATE = date.today() - timedelta(days=90)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"=== retrain_pedigree_v1 (from {FROM_DATE}) start {time.strftime('%H:%M:%S')} ===")
    s = get_session()
    res = train_ranker(
        s,
        from_date=FROM_DATE,
        holdout_fraction=0.2,
        num_boost_round=500,
        early_stopping_rounds=100,
        model_dir=OUT_DIR,
        model_name=OUT_NAME,
        objective="rank",
    )
    elapsed = time.time() - started
    print(f"=== done in {elapsed:.0f}s ({elapsed/60:.1f}min) ===")
    print(f"Train races: {res.train_races} / Test races: {res.test_races}")
    print(f"Holdout metrics: {res.metrics}")
    print(f"Top-15 feature importances:")
    for name, imp in sorted(res.feature_importance.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name:<30} {imp:>10.2f}")

    # Persist a slim summary for status checks
    out = OUT_DIR / "retrain_summary.json"
    out.write_text(json.dumps({
        "from_date": FROM_DATE.isoformat(),
        "elapsed_seconds": elapsed,
        "metrics": res.metrics,
        "feature_importance": res.feature_importance,
    }, indent=2))
    print(f"Wrote summary to {out}")


if __name__ == "__main__":
    main()
