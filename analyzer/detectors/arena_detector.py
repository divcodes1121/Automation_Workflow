"""Arena detector (2F) -- SCAFFOLD (needs real footage to build).

Planned: once a card leaves the hand (2E), look inside the ``arena`` ROI for the
newly-appeared sprite and map its position to a lane (left / right / bridge /
back) and side (self / opponent). This is footage-dependent -- in-arena sprites
are animated, scaled, and rotated, unlike the static card art -- so it is only
scaffolded here and is NOT wired into :meth:`AnalyzerWorkflow.analyze`.
"""

from __future__ import annotations

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile


class ArenaDetector:
    """Locates where a played card appeared in the arena. Not implemented."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def locate(self, frame, profile: CalibrationProfile, card: str) -> str:
        """Return the lane a played ``card`` appeared in. Not implemented (2F)."""
        raise NotImplementedError(
            "ArenaDetector (2F) requires real gameplay footage to calibrate."
        )
