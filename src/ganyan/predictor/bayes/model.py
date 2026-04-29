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
