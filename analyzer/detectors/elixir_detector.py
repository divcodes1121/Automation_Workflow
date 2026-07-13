"""Elixir detector (2G): estimate current elixir (0-10) from the elixir bar.

The bar is a row of 10 magenta pips that fill left-to-right. Rather than counting
discrete pips (their separators are thin and anti-aliased), we threshold the
bar's magenta fill and measure how far along the bar the fill reaches: a column
counts as "filled" when enough of its height is magenta, and the filled fraction
x10 is the elixir. Validated on game_01 (sensible 0-10 across the match; both
bars read 0 at the victory frame).

Reads both bars via the ``elixir_self`` / ``elixir_opp`` ROIs. Returns None for a
bar whose ROI is missing from the profile.
"""

from __future__ import annotations

import cv2
import numpy as np

from analyzer.calibration.roi import crop
from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile

_SELF_ROI = "elixir_self"
_OPP_ROI = "elixir_opp"
# Magenta (elixir) in HSV: hue ~140-175, well-saturated, bright.
_H_LO, _H_HI, _S_MIN, _V_MIN = 140, 175, 90, 120


class ElixirDetector:
    """Estimates current elixir for both players (2G)."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def read(self, frame: np.ndarray, profile: CalibrationProfile) -> int | None:
        """Read the player's elixir (0-10), or None if the ROI is absent."""
        return self._read_bar(frame, profile, _SELF_ROI)

    def read_both(
        self, frame: np.ndarray, profile: CalibrationProfile
    ) -> tuple[int | None, int | None]:
        """Return ``(player_elixir, opponent_elixir)``; either may be None."""
        return (
            self._read_bar(frame, profile, _SELF_ROI),
            self._read_bar(frame, profile, _OPP_ROI),
        )

    def _read_bar(
        self, frame: np.ndarray, profile: CalibrationProfile, roi_key: str
    ) -> int | None:
        roi = profile.rois.get(roi_key)
        if roi is None:
            return None
        bar = crop(frame, roi)
        if bar.size == 0:
            return None
        # Skip the leftmost cap glyph, then threshold magenta fill per column.
        w = bar.shape[1]
        bar = bar[:, int(w * self._settings.elixir_left_trim):]
        if bar.size == 0:
            return None
        hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        magenta = (h >= _H_LO) & (h <= _H_HI) & (s > _S_MIN) & (v > _V_MIN)
        filled = (magenta.mean(axis=0) > self._settings.elixir_column_fill).mean()
        return int(round(float(filled) * 10))
