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


def _decorate_with_hierarchy(frame: TrainingFrame) -> TrainingFrame:
    flat = [h for order in frame.orderings.values() for h in order]
    frame.jockey_index = {f"j{i}": i for i in range(3)}
    frame.track_dist_index = {(0, 0): 0, (0, 1): 1}
    frame.sire_index = {"": 0, "sireA": 1, "sireB": 2}
    frame.jockey_of_horse_in_race = [i % 3 for i in range(len(flat))]
    frame.sire_of_horse_in_race = [(h % 3) for h in flat]
    for rid in frame.orderings:
        frame.track_dist_of_race[rid] = rid % 2
    return frame


def test_hierarchical_model_compiles(synthetic_frame):
    """Smoke test — hierarchical model compiles and ADVI runs."""
    frame = _decorate_with_hierarchy(synthetic_frame)
    from ganyan.predictor.bayes.model import build_hierarchical_pl_model

    model = build_hierarchical_pl_model(frame)
    idata = fit_advi(model, n_iter=3_000, seed=0)
    assert "theta" in idata.posterior
    assert "alpha_jockey" in idata.posterior
    assert "gamma_track_dist" in idata.posterior
    assert "beta_sire" in idata.posterior


def test_hierarchical_with_agf_includes_delta(synthetic_frame):
    frame = _decorate_with_hierarchy(synthetic_frame)
    flat = [h for order in frame.orderings.values() for h in order]
    rng = np.random.default_rng(1)
    frame.agf_of_horse_in_race = list(rng.uniform(5, 50, size=len(flat)))

    from ganyan.predictor.bayes.model import build_hierarchical_pl_model_with_agf

    model = build_hierarchical_pl_model_with_agf(frame)
    idata = fit_advi(model, n_iter=3_000, seed=0)
    assert "delta_agf" in idata.posterior
    delta_mean = float(idata.posterior["delta_agf"].mean())
    assert -2.0 < delta_mean < 2.0
