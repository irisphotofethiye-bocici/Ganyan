"""PyMC Plackett-Luce models — simple and hierarchical (vectorized)."""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ganyan.predictor.bayes.data import TrainingFrame, matrices_for_pymc


def _vectorized_pl_loglik(score_matrix, valid_mask):
    """Plackett-Luce loglik on a padded (R, K_max) score matrix.

    For ordering positions k = 0..K-1 in race r:
        loglik_r = Σ_k (score[r,k] − logsumexp_{j≥k} score[r,j])

    Padding is masked out: pad positions get safe_score = −1e9 so they
    contribute 0 to the tail sums and to the per-position loglik.
    """
    safe_score = pt.where(valid_mask, score_matrix, -1e9)
    score_max = safe_score.max(axis=1, keepdims=True)
    shifted = safe_score - score_max
    exp_shifted = pt.exp(shifted)
    rev = exp_shifted[:, ::-1]
    cum_rev = pt.cumsum(rev, axis=1)
    cum = cum_rev[:, ::-1]
    log_tail_sum = pt.log(cum) + score_max
    pos_loglik = safe_score - log_tail_sum
    pos_loglik_valid = pt.where(valid_mask, pos_loglik, 0.0)
    return pos_loglik_valid.sum()


def build_simple_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    mats = matrices_for_pymc(frame)
    horse_idx_mat = mats["horse_idx_mat"]
    valid_mask = mats["valid_mask"]
    coords = {"horse": list(range(n_horses))}
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="horse")
        score = theta[horse_idx_mat]
        pm.Potential("plackett_luce", _vectorized_pl_loglik(score, valid_mask))
    return model


def build_hierarchical_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    mats = matrices_for_pymc(frame)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[mats["horse_to_sire"]]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        gamma_track_dist = pm.Normal(
            "gamma_track_dist", 0.0, sigma_gamma, dims="track_dist",
        )

        score = (
            theta[mats["horse_idx_mat"]]
            + alpha_jockey[mats["jockey_idx_mat"]]
            + gamma_track_dist[mats["track_dist_arr"]][:, None]
        )
        pm.Potential(
            "plackett_luce_hier",
            _vectorized_pl_loglik(score, mats["valid_mask"]),
        )
    return model


def build_hierarchical_pl_model_with_agf(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    mats = matrices_for_pymc(frame)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[mats["horse_to_sire"]]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        gamma_track_dist = pm.Normal(
            "gamma_track_dist", 0.0, sigma_gamma, dims="track_dist",
        )
        delta_agf = pm.Normal("delta_agf", 0.0, 1.0)

        score = (
            theta[mats["horse_idx_mat"]]
            + alpha_jockey[mats["jockey_idx_mat"]]
            + gamma_track_dist[mats["track_dist_arr"]][:, None]
            + delta_agf * mats["agf_z_mat"]
        )
        pm.Potential(
            "plackett_luce_agf",
            _vectorized_pl_loglik(score, mats["valid_mask"]),
        )
    return model


def fit_advi(model: pm.Model, n_iter: int = 30_000, seed: int = 0):
    with model:
        approx = pm.fit(n_iter, method="advi", random_seed=seed, progressbar=False)
        idata = approx.sample(draws=2_000, random_seed=seed)
    return idata
