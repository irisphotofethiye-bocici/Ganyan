"""External-data plugin framework.

Each third-party source (yarisrehberi.com, altılıganyan.com, …) is a
subclass of :class:`ExternalSource` registered in :data:`REGISTRY`.
Rows are persisted into the ``external_signals`` table; downstream
features query that table.

Adding a new source: implement ``fetch_for_date`` returning a list of
:class:`ExternalSignalRow`, then register the class via
:func:`register_source`.  The CLI command ``ganyan scrape external``
discovers registered sources by name.
"""

from __future__ import annotations

from .base import ExternalSignalRow, ExternalSource, register_source, REGISTRY
from .tjk_discipline import TjkDisciplineSource
from .tjk_track_conditions import (
    TjkStewardReportsSource, TjkTrackConditionsSource,
)
from .tjk_workouts import TjkWorkoutSource
from .yarisrehberi import YarisRehberiTipsterSource


# Auto-register the bundled sources on import.  New sources should
# call register_source() at module import time the same way.
register_source(YarisRehberiTipsterSource)
register_source(TjkDisciplineSource)
register_source(TjkWorkoutSource)
register_source(TjkTrackConditionsSource)
register_source(TjkStewardReportsSource)

__all__ = [
    "ExternalSignalRow",
    "ExternalSource",
    "register_source",
    "REGISTRY",
]
