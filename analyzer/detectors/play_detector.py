"""Play detector (2E): a hand-reading sequence -> confirmed play events.

Thin wrapper that runs a full :class:`~analyzer.models.HandReading` sequence
through a fresh :class:`~analyzer.tracking.battle_state.BattleState`.
"""

from __future__ import annotations

from collections.abc import Iterable

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import HandReading, PlayEvent
from analyzer.tracking.battle_state import BattleState


class PlayDetector:
    """Detects cards played across a sequence of hand readings."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def detect(self, readings: Iterable[HandReading]) -> list[PlayEvent]:
        """Return the play events inferred from ``readings`` (in order)."""
        state = BattleState(self._settings)
        events: list[PlayEvent] = []
        for reading in readings:
            events.extend(state.update(reading))
        return events
