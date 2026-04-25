"""Bind raw external signals to concrete TJK race / race_entry rows.

Plugins (e.g. ``yarisrehberi``) capture signals in a source-native
format — a tipster's altılı parlay carries ``(sub_race_index 1-6,
horse_number)`` but no TJK race_id.  This module performs the
resolution pass after ingestion: it figures out which 6 TJK races the
altılı bundle covered on that date, joins ``horse_number`` to
``race_entries.gate_number`` to get a concrete row, and back-fills the
``race_id`` / ``race_entry_id`` columns on the original signal.

Heuristic for the altılı-bundle discovery (TJK doesn't publish it
through a clean endpoint):

1. **Title-tagged tickets**: ``"Ankara 2 Altılı"`` → restrict to the
   Ankara card on that date.  ``"İstanbul Altılı"`` → İstanbul.  We
   do prefix matching against known track names.
2. **Untagged tickets**: applied to every track that has ≥6 races on
   the date (so a single ticket may produce multiple resolutions, one
   per track).  Cheap rows; the model can deduplicate via
   ``COUNT(DISTINCT ticket_timestamp)`` at feature time.
3. **Sub-race → TJK race_number**: altılı is by convention the *last
   6* races of the day on the chosen track.  Sub-race 1 maps to
   ``race[-6]`` (chronologically), 6 to ``race[-1]``.

Same-tipster duplicates are not deduplicated at the signal level —
each pick is one row — but the feature aggregator groups by
``ticket_timestamp`` so each ticket counts once per horse.
"""

from __future__ import annotations

import logging
import re
from datetime import date as date_type, datetime, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import (
    ExternalSignal, Race, RaceEntry, Track,
)


logger = logging.getLogger(__name__)


# A ticket whose authored timestamp is older than this is treated as
# stale SEO/cached content and skipped during resolution — yarisrehberi
# notoriously serves Feb 2026 tickets on a current-day fetch.  Tighten
# to 1 day if/when fresh picks become available.
_TICKET_FRESHNESS_DAYS = 3
_DDMMYYYY_RE = re.compile(r"^\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})")


# Token map: ticket-title prefix → canonical TJK track name.  Kept
# explicit so a typo in the source doesn't silently bind to a wrong
# track (e.g. "İzmir" must match "İzmir", not get truncated to "İz").
_TRACK_TOKENS: list[tuple[str, str]] = [
    ("İstanbul", "İstanbul"),
    ("Istanbul", "İstanbul"),
    ("Ankara", "Ankara"),
    ("İzmir", "İzmir"),
    ("Izmir", "İzmir"),
    ("Bursa", "Bursa"),
    ("Adana", "Adana"),
    ("Diyarbakır", "Diyarbakır"),
    ("Diyarbakir", "Diyarbakır"),
    ("Şanlıurfa", "Şanlıurfa"),
    ("Sanliurfa", "Şanlıurfa"),
    ("Elazığ", "Elazığ"),
    ("Elazig", "Elazığ"),
    ("Kocaeli", "Kocaeli"),
    ("Antalya", "Antalya"),
]


def _track_from_title(title: str | None) -> str | None:
    if not title:
        return None
    for token, canonical in _TRACK_TOKENS:
        if token.lower() in title.lower():
            return canonical
    return None


def _ticket_age_days(ticket_ts: str | None, target_date: date_type) -> int | None:
    """Days between the ticket's authored timestamp and ``target_date``.

    Returns ``None`` when the timestamp can't be parsed.  Used to
    suppress stale SEO content from binding to current races.
    """
    if not ticket_ts:
        return None
    m = _DDMMYYYY_RE.match(ticket_ts)
    if not m:
        return None
    dd, mm, yyyy, _hh, _mi = m.groups()
    try:
        ticket_date = date_type(int(yyyy), int(mm), int(dd))
    except ValueError:
        return None
    return abs((target_date - ticket_date).days)


def _race_lookup(
    session: Session, target_date: date_type, track_name: str | None,
) -> dict[str, list[Race]]:
    """Return ``{track_name: [races sorted by race_number]}`` for ``target_date``.

    When ``track_name`` is provided, only that track's races are
    returned (still as a single-key dict for caller uniformity).
    """
    q = (
        session.query(Race)
        .join(Track, Track.id == Race.track_id)
        .filter(Race.date == target_date)
    )
    if track_name is not None:
        q = q.filter(Track.name == track_name)
    out: dict[str, list[Race]] = {}
    for race in q.order_by(Race.race_number.asc()).all():
        out.setdefault(race.track.name, []).append(race)
    return out


def _entry_by_gate(session: Session, race_id: int, gate: int) -> RaceEntry | None:
    """Look up the entry whose gate_number matches ``gate`` in the given race.

    Multiple entries can in theory share a gate (rare data anomaly).
    Returns the first match or ``None`` when nothing binds.
    """
    return (
        session.query(RaceEntry)
        .filter(
            RaceEntry.race_id == race_id,
            RaceEntry.gate_number == gate,
        )
        .order_by(RaceEntry.id.asc())
        .first()
    )


def resolve_unbound_signals(
    session: Session,
    target_date: date_type,
    source_name: str | None = None,
) -> int:
    """Resolve unbound external_signals rows for ``target_date``.

    Iterates rows where ``race_entry_id IS NULL`` whose ``payload``
    carries the un-resolved bundle info.  Sets ``race_id`` and
    ``race_entry_id`` when the heuristic produces a unique match.

    Returns the count of rows updated.  Caller must commit.

    ``source_name`` filters to a specific plugin (useful when only one
    plugin's rows need re-binding); ``None`` resolves across all.
    """
    q = (
        session.query(ExternalSignal)
        .filter(ExternalSignal.race_entry_id.is_(None))
    )
    if source_name is not None:
        q = q.filter(ExternalSignal.source_name == source_name)

    candidates = list(q.all())
    if not candidates:
        return 0

    # Filter to rows whose payload says the pick is for ``target_date``.
    matched = [
        r for r in candidates
        if r.payload and r.payload.get("for_date") == target_date.isoformat()
    ]
    if not matched:
        return 0

    # Pre-load all races for that date by track once, reused across
    # rows.  The per-row gate→entry lookup hits the indexed
    # (race_id, gate_number) columns and is cheap.
    by_track_full = _race_lookup(session, target_date, None)
    if not by_track_full:
        return 0

    n_updated = 0
    n_stale_skipped = 0
    for row in matched:
        payload = row.payload or {}
        sub = int(payload.get("sub_race_index") or 0)
        horse_no = payload.get("horse_number")
        if sub < 1 or sub > 6 or horse_no is None:
            continue

        # Stale-ticket guard: when the source publishes Feb tickets on
        # an April fetch, don't bind them to today's program.  Skip
        # without resolving so they sit unbound (and harmless) in the
        # table.  Only enforce when we *can* parse the timestamp;
        # missing-timestamp tickets get the benefit of the doubt.
        age = _ticket_age_days(payload.get("ticket_timestamp"), target_date)
        if age is not None and age > _TICKET_FRESHNESS_DAYS:
            n_stale_skipped += 1
            continue

        title_track = _track_from_title(payload.get("ticket_title"))
        if title_track and title_track in by_track_full:
            tracks = {title_track: by_track_full[title_track]}
        else:
            tracks = by_track_full

        # Pick first track whose card has ≥6 races and a gate match.
        # When the ticket is untagged we resolve to the first match,
        # not all matches — multiple-track explosion would inflate
        # tipster counts spuriously.
        bound = False
        for track_name, races in tracks.items():
            if len(races) < 6:
                continue
            tjk_race = races[-6 + (sub - 1)]
            entry = _entry_by_gate(session, tjk_race.id, int(horse_no))
            if entry is not None:
                row.race_id = tjk_race.id
                row.race_entry_id = entry.id
                bound = True
                break
        if bound:
            n_updated += 1

    session.flush()
    logger.info(
        "external resolver: %d/%d signal(s) bound for %s "
        "(%d stale skipped)",
        n_updated, len(matched), target_date, n_stale_skipped,
    )
    return n_updated


def fetch_and_resolve(
    session: Session, target_date: date_type, sources: Iterable[str] | None = None,
) -> dict[str, int]:
    """Run every registered source, persist, then resolve.

    Convenience for the scheduler.  ``sources`` filters which plugins
    to fire (default: all registered).  Returns ``{source: n_persisted}``;
    resolution count is logged but not returned per-source since
    resolution is cross-source by date.
    """
    from . import REGISTRY
    from .base import persist_signals

    results: dict[str, int] = {}
    plugins = sources or list(REGISTRY.keys())
    for name in plugins:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("external: unknown source %s, skipping", name)
            continue
        plugin = cls()
        try:
            rows = plugin.fetch_for_date(session, target_date)
            n = persist_signals(session, rows)
            results[name] = n
        except Exception:  # noqa: BLE001
            logger.exception("external: %s fetch failed", name)
            results[name] = 0
    resolve_unbound_signals(session, target_date)
    return results
