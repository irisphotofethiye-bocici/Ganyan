"""Detect degenerate-uniform prediction fields (AGF-NULL / 1-N softmax fallback).

When AGF features are NULL at predict time (e.g., pre-11:30 scrape window,
Postgres down, scraper schema break), every horse lands on the same LGBM leaf,
softmax produces ~1/N probabilities, and the gate-ordered fallback ranks horses
by program number. The picks look plausible but carry zero information.

This guard catches that pathology by checking whether the per-race probability
distribution has near-zero standard deviation.
"""

from __future__ import annotations

from typing import Optional, Sequence

DEFAULT_STDDEV_THRESHOLD = 0.01


def is_uniform(probabilities: Sequence[float], stddev_threshold: float = DEFAULT_STDDEV_THRESHOLD) -> bool:
    """Return True if the probability vector is degenerately uniform.

    Empty or single-element vectors are treated as uniform (no information
    content). Threshold defaults to 0.01 — a real race field on this stack
    has stddev typically 0.05-0.15 across the field.
    """
    n = len(probabilities)
    if n < 2:
        return True
    mean = sum(probabilities) / n
    var = sum((p - mean) ** 2 for p in probabilities) / n
    stddev = var ** 0.5
    return stddev < stddev_threshold


def check_race_field(
    race_id: int,
    probabilities: Sequence[float],
    stddev_threshold: float = DEFAULT_STDDEV_THRESHOLD,
) -> Optional[str]:
    """If the race field is uniform, return a halt reason string. Else None."""
    if is_uniform(probabilities, stddev_threshold):
        return f"race {race_id}: uniform predictions detected (stddev < {stddev_threshold})"
    return None
