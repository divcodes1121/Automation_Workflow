"""Elixir detector (2G) -- SCAFFOLD (needs real footage to build).

Planned: crop the ``elixir_bar`` ROI and count filled pips (colour-threshold the
purple fill along the bar) to estimate current elixir. Footage-dependent; NOT
wired into ``analyze``.
"""

from __future__ import annotations

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile


class ElixirDetector:
    """Estimates current elixir. Not implemented (2G)."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def read(self, frame, profile: CalibrationProfile) -> int:
        """Return the estimated elixir count (0-10). Not implemented (2G)."""
        raise NotImplementedError(
            "ElixirDetector (2G) requires real gameplay footage to calibrate."
        )
