"""Crown detector (2G) -- SCAFFOLD (needs real footage to build).

Planned: read the crown counts for both sides from the ``crown_self`` /
``crown_opponent`` ROIs (template match on the crown/number glyphs) to detect
tower-destruction events. Footage-dependent; NOT wired into ``analyze``.
"""

from __future__ import annotations

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile


class CrownDetector:
    """Reads crown counts. Not implemented (2G)."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def read(self, frame, profile: CalibrationProfile) -> tuple[int, int]:
        """Return ``(self_crowns, opponent_crowns)``. Not implemented (2G)."""
        raise NotImplementedError(
            "CrownDetector (2G) requires real gameplay footage to calibrate."
        )
