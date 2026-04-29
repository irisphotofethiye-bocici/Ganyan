"""Tests for calibration metrics."""
from __future__ import annotations

import numpy as np
import pytest

from ganyan.predictor.bayes.calibration import (
    brier_top1, log_likelihood_of_winner, top1_hit_rate,
)


def test_top1_hit_rate_basic():
    races = [
        [(0.6, True), (0.3, False), (0.1, False)],
        [(0.5, False), (0.4, True), (0.1, False)],
        [(0.7, True), (0.2, False), (0.1, False)],
    ]
    assert top1_hit_rate(races) == pytest.approx(2 / 3)


def test_brier_top1_perfect_predictor():
    races = [
        [(1.0, True), (0.0, False)],
        [(1.0, True), (0.0, False)],
    ]
    assert brier_top1(races) == 0.0


def test_log_likelihood_uniform():
    races = [[(0.25, True), (0.25, False), (0.25, False), (0.25, False)]] * 5
    assert log_likelihood_of_winner(races) == pytest.approx(np.log(0.25))
