"""Smoke tests for trip-wire baseline computation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Horse, Race, RaceEntry, RaceStatus, Track
from ganyan.predictor.trip_wire import compute_trip_wire


@pytest.fixture
def session():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    s = Session(eng)
    yield s
    s.close()


def _seed(s: Session, target: date, daily_top1_pcts: list[tuple[date, float]]):
    """Seed N days of races, each with a single entry whose predicted_probability
    is the desired daily top-1 percent. compute_trip_wire takes max() per race
    so single-entry races make the test arithmetic crisp."""
    track = Track(name="TestTrack", city="Test")
    horse = Horse(name="TestHorse")
    s.add_all([track, horse])
    s.flush()
    for i, (d, top1) in enumerate(daily_top1_pcts):
        race = Race(
            track_id=track.id, date=d, race_number=1,
            status=RaceStatus.scheduled,
        )
        s.add(race)
        s.flush()
        s.add(RaceEntry(
            race_id=race.id, gate_number=1, horse_id=horse.id,
            predicted_probability=top1,
        ))
    s.commit()


def test_returns_none_when_no_predictions_today(session):
    today = date(2026, 4, 30)
    # 60 days of baseline but no row for today
    days = [(today - timedelta(days=d), 12.0) for d in range(1, 61)]
    _seed(session, today, days)
    assert compute_trip_wire(session, today) is None


def test_returns_none_with_thin_baseline(session):
    today = date(2026, 4, 30)
    # Only 5 days (today + 4 prior) — well under min_baseline_days=30
    days = [(today - timedelta(days=d), 12.0) for d in range(0, 5)]
    _seed(session, today, days)
    assert compute_trip_wire(session, today) is None


def test_z_score_positive_when_today_above_baseline(session):
    today = date(2026, 4, 30)
    # 60 days of baseline ~10%, today at 30% should fire positive z
    days = [(today - timedelta(days=d), 10.0) for d in range(1, 61)]
    days.insert(0, (today, 30.0))
    _seed(session, today, days)
    info = compute_trip_wire(session, today)
    # Constant baseline → std=0 → returns None per degenerate-std guard
    assert info is None


def test_z_score_calc_with_real_variance(session):
    today = date(2026, 4, 30)
    # 60 days where baseline varies 8-12%, today at 25%
    days = [
        (today - timedelta(days=d), 10.0 + (d % 5 - 2) * 1.0)
        for d in range(1, 61)
    ]
    days.insert(0, (today, 25.0))
    _seed(session, today, days)
    info = compute_trip_wire(session, today)
    assert info is not None
    assert info["today_avg"] == pytest.approx(25.0)
    assert 9.0 < info["baseline_mean"] < 11.0
    assert info["baseline_std"] > 0
    assert info["z_score"] > 5.0  # Way above baseline
    assert info["n_baseline_days"] == 60
