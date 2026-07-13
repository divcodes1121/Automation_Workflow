"""ROI geometry: crop a region from a frame and draw a profile overlay.

Pure cv2/numpy helpers shared by the hand detector (crops) and the ``calibrate``
command (overlay preview).
"""

from __future__ import annotations

import cv2
import numpy as np

from analyzer.models import ROI, CalibrationProfile


def crop(frame: np.ndarray, roi: ROI) -> np.ndarray:
    """Return the sub-image of ``frame`` covered by ``roi`` (clamped to bounds)."""
    height, width = frame.shape[:2]
    x, y, w, h = roi.to_pixels(width, height)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)
    return frame[y1:y2, x1:x2]


def draw_rois(frame: np.ndarray, profile: CalibrationProfile) -> np.ndarray:
    """Return a copy of ``frame`` with every profile ROI drawn + labeled."""
    out = frame.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    height, width = out.shape[:2]
    for name, roi in profile.rois.items():
        x, y, w, h = roi.to_pixels(width, height)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out, name, (x + 3, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out
