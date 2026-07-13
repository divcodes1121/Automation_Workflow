"""Hand detector (2D): identify the four cards in hand on a single frame.

For each ``hand_slot_i`` ROI from the active calibration profile, the slot is
cropped and matched against the cached card templates (2A). A slot whose best
score clears ``hand_match_threshold`` is reported as that card; otherwise it is
left unmatched (empty / cycling / off-layout).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.calibration.roi import crop
from analyzer.models import CalibrationProfile, HandReading, HandSlotReading
from analyzer.template_index import TemplateIndex

logger = logging.getLogger(__name__)


class HandDetector:
    """Reads the four-card hand from a frame via template matching."""

    HAND_SLOTS = ("hand_slot_1", "hand_slot_2", "hand_slot_3", "hand_slot_4")
    OPP_HAND_SLOTS = ("opp_hand_slot_1", "opp_hand_slot_2", "opp_hand_slot_3", "opp_hand_slot_4")

    _FP_SIZE = (12, 12)  # downscale size for the per-slot change fingerprint

    def __init__(
        self,
        settings: AnalyzerSettings | None = None,
        template_index: TemplateIndex | None = None,
    ) -> None:
        self._settings = settings or get_analyzer_settings()
        self._index = template_index or TemplateIndex.load(self._settings)
        # Event-driven skip state: per-slot (fingerprint, cached identity).
        self._cache: dict[str, tuple[np.ndarray, tuple]] = {}
        self.slots_computed = 0
        self.slots_skipped = 0

    def detect(
        self,
        frame: np.ndarray,
        profile: CalibrationProfile,
        *,
        slot_keys: tuple[str, ...] = HAND_SLOTS,
        source_frame: int = 0,
        timestamp_seconds: float = 0.0,
    ) -> HandReading:
        """Detect the hand in one frame (BGR or grayscale).

        ``slot_keys`` selects which ROI set to read (player or opponent).
        """
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        skip = self._settings.hand_skip_unchanged
        threshold = self._settings.hand_change_threshold

        slots: list[HandSlotReading] = []
        for index, key in enumerate(slot_keys, start=1):
            roi = profile.rois.get(key)
            slot_gray = crop(gray, roi) if roi is not None else np.empty((0, 0), np.uint8)
            if roi is None or slot_gray.size == 0:
                slots.append(
                    HandSlotReading(slot=index, card=None, variant=None, score=0.0, matched=False)
                )
                continue

            # Event-driven skip: reuse the cached identity when this slot's pixels
            # are (near-)unchanged from the previous frame -- ORB is the expensive
            # part, and most consecutive frames have identical hands.
            fingerprint = cv2.resize(slot_gray, self._FP_SIZE)
            cached = self._cache.get(key)
            if skip and cached is not None and _mad(fingerprint, cached[0]) <= threshold:
                slug, variant, score, matched = cached[1]
                self.slots_skipped += 1
            else:
                slug, variant, score, matched = self._index.identify(
                    np.ascontiguousarray(slot_gray)
                )
                score = round(score, 4)
                self._cache[key] = (fingerprint, (slug, variant, score, matched))
                self.slots_computed += 1

            slots.append(
                HandSlotReading(
                    slot=index,
                    card=slug if matched else None,
                    variant=variant if matched else None,
                    score=score,
                    matched=matched,
                )
            )
        return HandReading(
            source_frame=source_frame, timestamp_seconds=timestamp_seconds, slots=slots
        )


def _mad(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two same-shape uint8 arrays (0-255)."""
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))
