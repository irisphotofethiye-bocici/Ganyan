"""Tests for the Bayesian predictor inference."""
from __future__ import annotations

import numpy as np
import arviz as az

from ganyan.predictor.bayes.predictor import (
    BayesPrediction, predict_from_posterior,
)
from ganyan.predictor.bayes.data import TrainingFrame


def _build_idata(
    theta_means: np.ndarray,        # shape (n_horses,)
    sire_means: np.ndarray,         # shape (n_sires,)
    n_jockeys: int = 1,
    n_track_dist: int = 1,
    n_draws: int = 200,
):
    rng = np.random.default_rng(0)
    n_horses = len(theta_means)
    n_sires = len(sire_means)
    posterior = {
        "theta": np.tile(theta_means, (1, n_draws, 1))
                 + rng.normal(0, 0.01, (1, n_draws, n_horses)),
        "alpha_jockey": np.zeros((1, n_draws, n_jockeys)),
        "beta_sire": np.tile(sire_means, (1, n_draws, 1)),
        "gamma_track_dist": np.zeros((1, n_draws, n_track_dist)),
        "delta_agf": np.zeros((1, n_draws)),
        "sigma_theta": 0.1 * np.ones((1, n_draws)),
    }
    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    dims = {
        "theta": ["horse"],
        "alpha_jockey": ["jockey"],
        "beta_sire": ["sire"],
        "gamma_track_dist": ["track_dist"],
    }
    return az.from_dict(posterior=posterior, coords=coords, dims=dims)


def test_predict_from_posterior_orders_by_skill():
    frame = TrainingFrame()
    frame.horse_index = {101: 0, 102: 1, 103: 2}
    frame.jockey_index = {"jA": 0}
    frame.sire_index = {"": 0}
    frame.track_dist_index = {(3, 0): 0}
    idata = _build_idata(
        theta_means=np.array([2.0, 1.0, 0.0]),
        sire_means=np.array([0.0]),
    )

    race = {
        "horse_ids": [103, 102, 101],
        "jockeys": ["jA", "jA", "jA"],
        "sires": ["", "", ""],
        "track_id": 3,
        "distance_meters": 1200,
        "agfs": [10.0, 10.0, 10.0],
    }
    preds = predict_from_posterior(idata, frame, race)
    assert isinstance(preds, list)
    assert all(isinstance(p, BayesPrediction) for p in preds)
    by_horse = {p.horse_id: p for p in preds}
    assert by_horse[101].mean_prob > by_horse[102].mean_prob > by_horse[103].mean_prob
    assert abs(sum(p.mean_prob for p in preds) - 1.0) < 0.01
    for p in preds:
        assert p.lo_5 <= p.mean_prob <= p.hi_95


def test_predict_handles_cold_start_horse():
    frame = TrainingFrame()
    frame.horse_index = {101: 0}
    frame.jockey_index = {"jA": 0}
    frame.sire_index = {"": 0, "famousSire": 1}
    frame.track_dist_index = {(3, 0): 0}
    idata = _build_idata(
        theta_means=np.array([1.0]),
        sire_means=np.array([0.0, 0.5]),
    )

    race = {
        "horse_ids": [101, 999],
        "jockeys": ["jA", "jA"],
        "sires": ["", "famousSire"],
        "track_id": 3,
        "distance_meters": 1200,
        "agfs": [10.0, 10.0],
    }
    preds = predict_from_posterior(idata, frame, race)
    assert len(preds) == 2
    assert abs(sum(p.mean_prob for p in preds) - 1.0) < 0.01
