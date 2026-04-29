"""Calibration metrics for race predictors.

Each race is represented as a list of (predicted_prob, is_winner) tuples.
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


Race = List[Tuple[float, bool]]


def top1_hit_rate(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    hits = 0
    for r in races:
        top = max(r, key=lambda t: t[0])
        if top[1]:
            hits += 1
    return hits / len(races)


def brier_top1(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    total = 0.0
    for r in races:
        top_prob, top_won = max(r, key=lambda t: t[0])
        total += (top_prob - (1.0 if top_won else 0.0)) ** 2
    return total / len(races)


def log_likelihood_of_winner(races: Iterable[Race]) -> float:
    races = list(races)
    if not races:
        return 0.0
    total = 0.0
    n = 0
    for r in races:
        for prob, won in r:
            if won:
                total += float(np.log(max(prob, 1e-12)))
                n += 1
                break
    return total / max(n, 1)
