"""Timer detector (2G): read the match clock (M:SS) + phase from the timer ROI.

Clash Royale renders the clock in one bold font, so **digit template matching**
beats general OCR. The reader is deterministic and was validated frame-by-frame
against a countdown ground-truth model on game_01 (100% in regulation, ~96% in
mid-overtime; the red-on-red overtime endgame is confidence-flagged and cleaned
by the timeline's monotonic filter -- see :mod:`analyzer.tracking.match_state`).

Two rendering regimes need two binarizations (auto-selected by phase):

* **regulation** -- white (or red, final seconds) digits on a DARK badge. The
  ``max`` channel isolates any bright digit from the dark background.
* **overtime** -- white digits on a RED badge. ``max`` fails (red bg is bright),
  so the ``min`` channel isolates the fully-white digits from the red field.

Phase itself comes from the redness of the ROI **border** (background), which is
red only in true overtime -- not fooled by red digits in the regulation endgame.

Digit templates live in ``analyzer/assets/timer_digits/<d>_<regime>.png`` (two
exemplars per digit, one per regime); they are matched by binary-mask agreement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from analyzer.calibration.roi import crop
from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile, MatchPhase

logger = logging.getLogger(__name__)

# Normalized glyph size (must match how the committed templates were built).
_GLYPH_H, _GLYPH_W = 40, 28
_TIMER_ROI = "timer"


@dataclass(frozen=True)
class TimerReading:
    """A single-frame timer read (raw; no cross-frame smoothing)."""

    text: str | None  # "M:SS", or None when unreadable
    seconds: int | None  # parsed total seconds remaining, or None
    phase: MatchPhase
    confidence: float  # min per-digit mask agreement (0..1); 0 when unreadable


class TimerDetector:
    """Reads the match timer + phase from a frame (2G)."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()
        self._templates = self._load_templates()

    def _load_templates(self) -> dict[int, list[np.ndarray]]:
        """Load {digit: [boolean masks]} from the committed template PNGs."""
        tdir = self._settings.timer_templates_dir
        templates: dict[int, list[np.ndarray]] = {}
        for path in sorted(tdir.glob("*.png")):
            try:
                digit = int(path.name.split("_")[0])
            except ValueError:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.setdefault(digit, []).append(img > 127)
        if len(templates) < 10:
            logger.warning(
                "Timer digit templates incomplete (%d/10) in %s; timer reads will fail",
                len(templates), tdir,
            )
        return templates

    # -- per-frame read ------------------------------------------------------ #

    def read(self, frame: np.ndarray, profile: CalibrationProfile) -> TimerReading:
        """Read the timer ROI of one frame."""
        roi = profile.rois.get(_TIMER_ROI)
        if roi is None or not self._templates:
            return TimerReading(None, None, MatchPhase.UNKNOWN, 0.0)
        return self.read_region(crop(frame, roi))

    def read_region(self, region: np.ndarray) -> TimerReading:
        """Read an already-cropped timer ROI.

        Split out of :meth:`read` so callers that crop the ROI themselves -- the
        battle splitter crops it with FFmpeg, which is far cheaper than decoding
        whole 1320x2868 frames -- reuse this logic instead of duplicating it.
        """
        if not self._templates or region.size == 0:
            return TimerReading(None, None, MatchPhase.UNKNOWN, 0.0)

        overtime = self._is_overtime(region)
        phase = MatchPhase.OVERTIME if overtime else MatchPhase.REGULATION
        glyphs = self._segment(region, overtime)
        if len(glyphs) != 3:
            return TimerReading(None, None, phase, 0.0)

        digits, confs = [], []
        for _, gmask in glyphs:
            d, score = self._match_digit(gmask)
            digits.append(d)
            confs.append(score)
        minute, tens, ones = digits
        confidence = min(confs)
        # Format sanity + confidence gate: seconds tens is 0-5; low agreement is
        # an unreliable (usually overtime-endgame) read -> drop the value.
        if tens > 5 or confidence < self._settings.timer_min_confidence:
            return TimerReading(None, None, phase, confidence)
        seconds = minute * 60 + tens * 10 + ones
        return TimerReading(f"{minute}:{tens}{ones}", seconds, phase, confidence)

    # -- internals ----------------------------------------------------------- #

    @staticmethod
    def _is_overtime(region: np.ndarray) -> bool:
        """Overtime iff the ROI border (background) is red."""
        h, w = region.shape[:2]
        mask = np.ones((h, w), bool)
        mask[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)] = False
        return float(region[mask][:, 2].mean()) > 120.0  # mean border R (BGR)

    @staticmethod
    def _segment(region: np.ndarray, overtime: bool) -> list[tuple[int, np.ndarray]]:
        """Return up to 3 digit glyph masks, left-to-right (colon dropped)."""
        gray = region.min(axis=2) if overtime else region.max(axis=2)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = region.shape[:2]
        cand: list[tuple[int, int, np.ndarray]] = []
        for i in range(1, n):
            gx, gy, gw, gh, area = stats[i]
            # Tall + wide enough rejects the colon (short) and the bright edge
            # sliver / badge border (thin) that otherwise pollute the glyph set.
            if gh > 0.45 * h and gw > 0.05 * w and area > 0.01 * h * w:
                cand.append((area, gx, labels[gy:gy + gh, gx:gx + gw] == i))
        # The three digits are the largest solid glyphs.
        cand.sort(key=lambda t: t[0], reverse=True)
        glyphs = [(gx, gm) for _, gx, gm in cand[:3]]
        glyphs.sort(key=lambda t: t[0])
        return glyphs

    def _match_digit(self, glyph_mask: np.ndarray) -> tuple[int, float]:
        """Best digit for a glyph = max mask-agreement over all exemplars."""
        probe = cv2.resize(
            glyph_mask.astype(np.uint8) * 255, (_GLYPH_W, _GLYPH_H),
            interpolation=cv2.INTER_NEAREST,
        ) > 127
        best_digit, best_score = 0, -1.0
        for digit, exemplars in self._templates.items():
            score = max(float((probe == tpl).mean()) for tpl in exemplars)
            if score > best_score:
                best_digit, best_score = digit, score
        return best_digit, best_score
