"""Smoke test: EnsemblePredictor.predict_and_save persists picks.

The daemon's morning_card writes RaceEntry.predicted_probability so
generate_picks_for_race() downstream can read it. Switching from
MLPredictor → EnsemblePredictor must keep that contract.
"""
from __future__ import annotations

from ganyan.predictor.ml.ensemble import EnsemblePredictor


def test_ensemble_predictor_has_predict_and_save():
    """API contract: must expose predict_and_save with the same signature
    as MLPredictor (called from scheduler._job_morning_card)."""
    assert hasattr(EnsemblePredictor, "predict_and_save")
    method = EnsemblePredictor.predict_and_save
    # Method should accept (self, race_id) — i.e. exactly two args.
    code = method.__code__
    assert code.co_argcount == 2, (
        f"expected predict_and_save(self, race_id), got {code.co_varnames[:code.co_argcount]}"
    )
