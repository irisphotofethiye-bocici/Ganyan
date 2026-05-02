"""OpenWeatherMap track-conditions plugin (live companion to TJK).

TJK's PistBilgileri endpoint publishes with a ~3-week lag, so the
``track_conditions`` feature is always NaN at predict time.  This plugin
fills the live gap: for each TJK city racing on ``target_date`` it pulls
the current OpenWeatherMap reading and emits ``track_conditions`` rows
with the same payload schema the TJK plugin uses, so
``compute_track_conditions`` reads them transparently.

Free-tier endpoint, ~1M calls/month, ~7 cities × few calls/day = trivial
quota.  Coordinates are hardcoded per TJK hipodrom (more reliable than
city-name lookup, which trips on Turkish diacritics).
"""

from __future__ import annotations

import logging
from datetime import date as date_type, datetime
from typing import ClassVar

import httpx
from sqlalchemy.orm import Session

from ganyan.config import get_settings
from ganyan.db.models import Race, Track

from .base import ExternalSignalRow, ExternalSource


logger = logging.getLogger(__name__)

_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
_TIMEOUT = httpx.Timeout(10.0, connect=4.0)


# Hipodrom coordinates per TJK city.  Lat/lon avoids OWM's flaky
# Turkish-character handling.  Add a city here when TJK opens a new
# venue (e.g. Kocaeli back on the calendar).
_TRACK_GEO: dict[str, tuple[float, float]] = {
    "Adana": (37.0017, 35.3289),
    "Ankara": (39.9334, 32.8597),
    "Antalya": (36.9100, 30.7000),
    "Bursa": (40.2100, 29.0400),
    "Diyarbakır": (37.9144, 40.2306),
    "Elazığ": (38.6810, 39.2264),
    "İstanbul": (41.0050, 28.8470),    # Veliefendi
    "İzmir": (38.3793, 27.1428),       # Şirinyer
    "Kocaeli": (40.7654, 29.9408),
    "Şanlıurfa": (37.1671, 38.7955),
}


# OWM ``weather[0].main`` → ordinal aligned with TJK plugin _SKY_BUCKETS
# (0 açık, 1 parçalı bulutlu, 2 bulutlu, 3 yağmurlu, 4 kapalı).
_OWM_SKY_BUCKET: dict[str, int] = {
    "Clear": 0,
    "Clouds": 2,        # refined by clouds.all below
    "Mist": 2,
    "Haze": 2,
    "Fog": 2,
    "Smoke": 2,
    "Dust": 2,
    "Sand": 2,
    "Ash": 2,
    "Squall": 3,
    "Tornado": 3,
    "Drizzle": 3,
    "Rain": 3,
    "Thunderstorm": 3,
    "Snow": 4,
}


def _bucket_from_owm(
    weather_main: str | None, clouds_pct: int | None,
) -> int | None:
    """Map OWM weather + cloud-coverage % to the TJK 0–4 ordinal."""
    if not weather_main:
        return None
    base = _OWM_SKY_BUCKET.get(weather_main)
    if base is None:
        return None
    # "Clouds" splits into parçalı (1) / bulutlu (2) / kapalı (4) by %.
    if weather_main == "Clouds" and clouds_pct is not None:
        if clouds_pct < 50:
            return 1
        if clouds_pct >= 90:
            return 4
        return 2
    return base


def _fetch_one(api_key: str, lat: float, lon: float) -> dict | None:
    """Single OWM call.  Returns parsed JSON, or ``None`` on failure."""
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPError as exc:
        logger.warning("openweather fetch failed (%.4f,%.4f): %s", lat, lon, exc)
        return None


def _payload_from_owm(
    data: dict, *, track_city: str, target_date: date_type,
) -> dict:
    """Translate OWM JSON into the same payload shape the TJK plugin emits."""
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    clouds_pct = (data.get("clouds") or {}).get("all")
    weather_arr = data.get("weather") or [{}]
    weather_main = (weather_arr[0] or {}).get("main")
    weather_desc = (weather_arr[0] or {}).get("description")

    wind_ms = wind.get("speed")
    wind_kph = round(wind_ms * 3.6) if wind_ms is not None else None

    temp = main.get("temp")
    temp_c = int(round(temp)) if temp is not None else None

    return {
        "track_city": track_city,
        "track_name": track_city,
        "reading_date": target_date.isoformat(),
        "reading_time": datetime.now().strftime("%H:%M"),
        "temperature_c": temp_c,
        "humidity_pct": main.get("humidity"),
        "pressure_mb": main.get("pressure"),
        "sky_text": weather_desc,
        "sky_bucket": _bucket_from_owm(weather_main, clouds_pct),
        "wind_kph": wind_kph,
        "wind_text": None,
        "for_date": target_date.isoformat(),
    }


class OpenWeatherSource(ExternalSource):
    """Live track-side weather via OpenWeatherMap."""

    source_name: ClassVar[str] = "openweather"
    signal_types: ClassVar[tuple[str, ...]] = ("track_conditions",)

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        api_key = (get_settings().openweather_api_key or "").strip()
        if not api_key:
            logger.warning(
                "openweather: OPENWEATHER_API_KEY not set, skipping",
            )
            return []

        # Only fetch cities that actually race on target_date.
        cities = [
            row[0]
            for row in (
                session.query(Track.name)
                .join(Race, Race.track_id == Track.id)
                .filter(Race.date == target_date)
                .distinct()
                .all()
            )
        ]
        captured_at = datetime.now()
        out: list[ExternalSignalRow] = []
        for city in cities:
            geo = _TRACK_GEO.get(city)
            if geo is None:
                logger.warning(
                    "openweather: no geo for city %r — skipping", city,
                )
                continue
            data = _fetch_one(api_key, *geo)
            if data is None:
                continue
            payload = _payload_from_owm(
                data, track_city=city, target_date=target_date,
            )
            out.append(ExternalSignalRow(
                source_name=self.source_name,
                signal_type="track_conditions",
                payload=payload,
                captured_at=captured_at,
            ))
        logger.info(
            "openweather: %d readings for %s", len(out), target_date,
        )
        return out
