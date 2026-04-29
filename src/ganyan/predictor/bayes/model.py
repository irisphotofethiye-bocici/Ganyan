"""PyMC Plackett-Luce models — simple and hierarchical."""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ganyan.predictor.bayes.data import TrainingFrame


def _plackett_luce_loglik(theta, orderings):
    total = pt.zeros((), dtype="float64")
    for order in orderings:
        order_arr = pt.constant(order, dtype="int64")
        chosen = theta[order_arr]
        rev = chosen[::-1]
        max_rev = rev.max()
        log_cumsum = pt.cumsum(pt.exp(rev - max_rev), axis=0)
        log_remaining_rev = pt.log(log_cumsum) + max_rev
        log_remaining = log_remaining_rev[::-1]
        total = total + pt.sum(chosen - log_remaining)
    return total


def build_simple_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    orderings = list(frame.orderings.values())
    coords = {"horse": list(range(n_horses))}
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="horse")
        pm.Potential("plackett_luce", _plackett_luce_loglik(theta, orderings))
    return model


def fit_advi(model: pm.Model, n_iter: int = 30_000, seed: int = 0):
    with model:
        approx = pm.fit(n_iter, method="advi", random_seed=seed, progressbar=False)
        idata = approx.sample(draws=2_000, random_seed=seed)
    return idata


def _plackett_luce_loglik_with_offsets(
    theta, alpha, gamma,
    orderings, jockey_per_entry, track_dist_per_race,
):
    total = pt.zeros((), dtype="float64")
    flat_idx = 0
    for race_idx, order in enumerate(orderings):
        order_arr = pt.constant(order, dtype="int64")
        n = len(order)
        jockey_slice = jockey_per_entry[flat_idx : flat_idx + n]
        flat_idx += n
        jockey_arr = pt.constant(jockey_slice, dtype="int64")
        td_idx = track_dist_per_race[race_idx]
        score = theta[order_arr] + alpha[jockey_arr] + gamma[td_idx]
        rev = score[::-1]
        max_rev = rev.max()
        log_cumsum = pt.cumsum(pt.exp(rev - max_rev), axis=0)
        log_remaining_rev = pt.log(log_cumsum) + max_rev
        log_remaining = log_remaining_rev[::-1]
        total = total + pt.sum(score - log_remaining)
    return total


def _build_horse_to_sire(frame: TrainingFrame, n_horses: int) -> np.ndarray:
    horse_to_sire = np.zeros(n_horses, dtype="int64")
    seen: set[int] = set()
    flat_horses = [h for order in frame.orderings.values() for h in order]
    for h, s in zip(flat_horses, frame.sire_of_horse_in_race):
        if h not in seen:
            horse_to_sire[h] = s
            seen.add(h)
    return horse_to_sire


def build_hierarchical_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    orderings = list(frame.orderings.values())
    track_dist_per_race = [frame.track_dist_of_race[rid] for rid in frame.orderings]

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    horse_to_sire = _build_horse_to_sire(frame, n_horses)

    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[horse_to_sire]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        gamma_track_dist = pm.Normal(
            "gamma_track_dist", 0.0, sigma_gamma, dims="track_dist",
        )
        pm.Potential(
            "plackett_luce_hier",
            _plackett_luce_loglik_with_offsets(
                theta, alpha_jockey, gamma_track_dist,
                orderings, frame.jockey_of_horse_in_race, track_dist_per_race,
            ),
        )
    return model
