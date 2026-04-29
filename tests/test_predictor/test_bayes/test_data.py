"""Tests for the Bayesian training-frame builder."""
from __future__ import annotations

import inspect

from ganyan.predictor.bayes.data import (
    DISTANCE_BUCKETS, distance_bucket_for,
)


def test_distance_bucket_assigns_into_5_buckets():
    assert distance_bucket_for(1100) == 0
    assert distance_bucket_for(1300) == 1
    assert distance_bucket_for(1600) == 2
    assert distance_bucket_for(1900) == 3
    assert distance_bucket_for(2400) == 4
    assert len(DISTANCE_BUCKETS) == 4


def test_data_module_does_not_read_finish_time():
    from ganyan.predictor.bayes import data
    src = inspect.getsource(data)
    assert "finish_time" not in src
    assert "finish_position" in src
