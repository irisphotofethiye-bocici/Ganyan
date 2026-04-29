"""Tests for the PyMC Plackett-Luce models."""
from __future__ import annotations

import numpy as np
import pytest

from ganyan.predictor.bayes.data import TrainingFrame
from ganyan.predictor.bayes.model import build_simple_pl_model, fit_advi


@pytest.fixture
def synthetic_frame() -> TrainingFrame:
    rng = np.random.default_rng(0)
    n_horses = 6
    n_races = 200
    true_skill = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    frame = TrainingFrame()
    for h in range(n_horses):
        frame.horse_index[h] = h
    for race_id in range(n_races):
        field = list(rng.choice(n_horses, size=4, replace=False))
        order: list[int] = []
        remaining = list(field)
        while remaining:
            skills = np.array([true_skill[h] for h in remaining])
            probs = np.exp(skills - skills.max())
            probs /= probs.sum()
            pick = rng.choice(remaining, p=probs)
            order.append(int(pick))
            remaining.remove(pick)
        frame.orderings[race_id] = order
    return frame


def test_simple_pl_recovers_top_horse(synthetic_frame):
    model = build_simple_pl_model(synthetic_frame)
    idata = fit_advi(model, n_iter=15_000, seed=0)
    posterior_skill_mean = idata.posterior["theta"].mean(("chain", "draw")).values
    assert int(np.argmax(posterior_skill_mean)) == 0
    assert posterior_skill_mean[0] > posterior_skill_mean[1]
    assert posterior_skill_mean[1] > posterior_skill_mean[3]
