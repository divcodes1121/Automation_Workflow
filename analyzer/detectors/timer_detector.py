"""Timer detector (2G) -- SCAFFOLD (needs real footage to build).

Planned: crop the ``timer`` ROI and read the match clock via digit template
matching (Clash Royale uses one font, so template digits beat general OCR) and
detect the overtime state. Footage-dependent; NOT wired into ``analyze``.
"""

from __future__ import annotations

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile


class TimerDetector:
    """Reads the match timer. Not implemented (2G)."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def read(self, frame, profile: CalibrationProfile) -> str:
        """Return the timer text (e.g. '1:32'). Not implemented (2G)."""
        raise NotImplementedError(
            "TimerDetector (2G) requires real gameplay footage to calibrate."
        )
