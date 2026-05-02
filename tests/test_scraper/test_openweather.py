"""Smoke tests for OpenWeatherSource plugin."""
from datetime import date, datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ganyan.config import Settings
from ganyan.db.models import Base, Race, RaceStatus, Track
from ganyan.scraper.external.openweather import (
    OpenWeatherSource,
    _bucket_from_owm,
    _payload_from_owm,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _seed_race(s, *, race_date: date, track_name: str = "Bursa"):
    track = Track(name=track_name, city=track_name)
    s.add(track)
    s.flush()
    race = Race(
        track_id=track.id, date=race_date, race_number=1,
        distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
    )
    s.add(race)
    s.flush()
    return race


_OWM_FIXTURE = {
    "weather": [{"main": "Clouds", "description": "scattered clouds"}],
    "main": {"temp": 18.4, "humidity": 62, "pressure": 1014},
    "wind": {"speed": 3.6},   # m/s → 13 kph
    "clouds": {"all": 40},    # <50 → bucket 1 (parçalı)
}


def test_no_api_key_returns_empty(session, monkeypatch):
    """Plugin must skip silently — not raise — when key is unset."""
    monkeypatch.setattr(
        "ganyan.scraper.external.openweather.get_settings",
        lambda: Settings(openweather_api_key=None),
    )
    _seed_race(session, race_date=date(2026, 5, 2))
    rows = OpenWeatherSource().fetch_for_date(session, date(2026, 5, 2))
    assert rows == []


def test_only_fetches_cities_with_races(session, monkeypatch):
    """Cities without races on target_date must not trigger HTTP calls."""
    monkeypatch.setattr(
        "ganyan.scraper.external.openweather.get_settings",
        lambda: Settings(openweather_api_key="dummy-key"),
    )
    _seed_race(session, race_date=date(2026, 5, 2), track_name="Bursa")

    calls = []
    def fake_fetch(api_key, lat, lon):
        calls.append((lat, lon))
        return _OWM_FIXTURE

    with patch(
        "ganyan.scraper.external.openweather._fetch_one", side_effect=fake_fetch,
    ):
        rows = OpenWeatherSource().fetch_for_date(session, date(2026, 5, 2))

    assert len(calls) == 1                # only Bursa
    assert len(rows) == 1
    assert rows[0].source_name == "openweather"
    assert rows[0].signal_type == "track_conditions"


def test_payload_schema_matches_tjk(session, monkeypatch):
    """Emitted payload must carry every key compute_track_conditions reads."""
    monkeypatch.setattr(
        "ganyan.scraper.external.openweather.get_settings",
        lambda: Settings(openweather_api_key="dummy-key"),
    )
    _seed_race(session, race_date=date(2026, 5, 2), track_name="İstanbul")

    with patch(
        "ganyan.scraper.external.openweather._fetch_one",
        return_value=_OWM_FIXTURE,
    ):
        rows = OpenWeatherSource().fetch_for_date(session, date(2026, 5, 2))

    payload = rows[0].payload
    assert payload["track_city"] == "İstanbul"
    assert payload["reading_date"] == "2026-05-02"
    assert payload["temperature_c"] == 18
    assert payload["humidity_pct"] == 62
    assert payload["pressure_mb"] == 1014
    assert payload["sky_bucket"] == 1        # 40% clouds → parçalı
    assert payload["wind_kph"] == 13         # 3.6 m/s → 13 kph
    # Keys compute_track_conditions consumes (features.py:702-744).
    for key in (
        "temperature_c", "humidity_pct", "pressure_mb",
        "sky_bucket", "wind_kph",
    ):
        assert key in payload


def test_sky_bucket_clouds_split():
    """Clouds → 1 (parçalı) below 50%, 2 (bulutlu) 50-89%, 4 (kapalı) ≥90%."""
    assert _bucket_from_owm("Clouds", 40) == 1
    assert _bucket_from_owm("Clouds", 70) == 2
    assert _bucket_from_owm("Clouds", 95) == 4
    assert _bucket_from_owm("Clear", None) == 0
    assert _bucket_from_owm("Rain", None) == 3
    assert _bucket_from_owm("Snow", None) == 4
    assert _bucket_from_owm(None, None) is None


def test_payload_handles_missing_fields():
    """OWM occasionally omits wind/clouds — payload must stay well-formed."""
    sparse = {
        "weather": [{"main": "Clear", "description": "clear sky"}],
        "main": {"temp": 22.0, "humidity": 50, "pressure": 1018},
    }
    payload = _payload_from_owm(
        sparse, track_city="Adana", target_date=date(2026, 5, 2),
    )
    assert payload["sky_bucket"] == 0
    assert payload["wind_kph"] is None
    assert payload["temperature_c"] == 22


def test_unknown_city_skipped(session, monkeypatch):
    """A track city not in _TRACK_GEO must be logged and skipped, not crash."""
    monkeypatch.setattr(
        "ganyan.scraper.external.openweather.get_settings",
        lambda: Settings(openweather_api_key="dummy-key"),
    )
    _seed_race(session, race_date=date(2026, 5, 2), track_name="MadeUpCity")

    with patch(
        "ganyan.scraper.external.openweather._fetch_one",
        return_value=_OWM_FIXTURE,
    ) as fetch_mock:
        rows = OpenWeatherSource().fetch_for_date(session, date(2026, 5, 2))

    fetch_mock.assert_not_called()
    assert rows == []
