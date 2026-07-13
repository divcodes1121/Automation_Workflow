"""Load device-specific calibration profiles.

Profiles live as JSON files under ``analyzer/calibration/profiles/`` and are
selected by name (``--profile`` / ``ANALYZER_PROFILE``). Calibrating a device
means editing one JSON (or re-tracing it via the ``calibrate`` preview) -- never
touching code. No cv2 needed here (pure model I/O).
"""

from __future__ import annotations

import logging

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import CalibrationProfile

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Raised when a calibration profile is missing or invalid."""


def load_profile(
    name: str | None = None, settings: AnalyzerSettings | None = None
) -> CalibrationProfile:
    """Load a calibration profile by name (defaults to ``settings.active_profile``)."""
    settings = settings or get_analyzer_settings()
    name = name or settings.active_profile
    path = settings.calibration_profiles_dir / f"{name}.json"
    if not path.is_file():
        available = ", ".join(available_profiles(settings)) or "none"
        raise CalibrationError(
            f"Unknown calibration profile {name!r} (looked for {path}). Available: {available}."
        )
    try:
        profile = CalibrationProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"Invalid calibration profile {name!r}: {exc}") from exc
    if profile.is_placeholder:
        logger.warning(
            "Profile %r is a placeholder -- ROIs are uncalibrated; tune against a real frame.",
            name,
        )
    return profile


def available_profiles(settings: AnalyzerSettings | None = None) -> list[str]:
    """List the profile names available on disk."""
    settings = settings or get_analyzer_settings()
    directory = settings.calibration_profiles_dir
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
