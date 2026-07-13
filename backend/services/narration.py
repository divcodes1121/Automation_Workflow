"""Prepare timeline narration for (future) text-to-speech (Feature 4).

:class:`NarrationService` turns a :class:`~backend.models.Timeline` into a
provider-neutral :class:`~backend.models.NarrationPackage`: for each segment it
keeps the original text, produces a cleaned/TTS-ready variant, copies the
estimated duration and stable ``segment_uuid``, and leaves delivery/audio slots
at inert defaults for a later voice provider to fill.

This is a **pure, deterministic transformation** — given the same ``Timeline`` it
always yields the same package (only ``generated_at`` varies). No TTS, no MP3, no
FFmpeg, no Whisper, no AI, and no provider-specific logic. Persistence is a
separate, optional concern (:meth:`save`), mirroring the other services. Emits
diagnostics via :mod:`logging`; no n8n dependency.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import NarrationPackage, PreparedNarrationSegment, Timeline

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
# Collapse runs of repeated ! or ? (e.g. "!!!" -> "!"). Dots are left alone so
# ellipses survive.
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1+")
# Insert a missing space after sentence punctuation glued to the next sentence
# ("deck.It" -> "deck. It"). Restricted to an uppercase follower to avoid
# touching decimals ("2.6") or lowercase abbreviations.
_SENTENCE_SPACE_RE = re.compile(r"([.!?])([A-Z])")

# Map unicode punctuation to ASCII so downstream TTS/consoles handle it cleanly.
_UNICODE_PUNCT = {
    "‘": "'", "’": "'",  # single quotes
    "“": '"', "”": '"',  # double quotes
    "–": "-", "—": "-",  # en/em dash
    "…": "...",               # ellipsis
}
_UNICODE_TRANSLATION = str.maketrans(
    {ord(k): v for k, v in _UNICODE_PUNCT.items()}
)

# Generic, provider- and game-independent abbreviation expansions. Deliberately
# excludes game terms (CR/Evo/MK stay untouched); game-specific normalisation, if
# ever needed, belongs in a separate stage. Applied in this fixed order.
_DEFAULT_ABBREVIATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bvs\b\.?", re.IGNORECASE), "versus"),
    (re.compile(r"\betc\b\.?", re.IGNORECASE), "et cetera"),
    (re.compile(r"\be\.g\b\.?", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\b\.?", re.IGNORECASE), "that is"),
    (re.compile(r"\bw/"), "with "),
    (re.compile(r"\s*&\s*"), " and "),
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class NarrationError(Exception):
    """Base class for all errors raised by :class:`NarrationService`."""


class NarrationPreparationError(NarrationError):
    """Raised when a timeline has no segments to prepare narration from."""


class NarrationService:
    """Prepares a :class:`Timeline` into a :class:`NarrationPackage`.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), used for the default save location.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def prepare(self, timeline: Timeline) -> NarrationPackage:
        """Transform ``timeline`` into a typed :class:`NarrationPackage`.

        Pure and deterministic — writes nothing.

        Raises
        ------
        NarrationPreparationError
            If the timeline contains no segments.
        """
        if not timeline.segments:
            raise NarrationPreparationError(
                f"Timeline {timeline.title!r} has no segments to prepare."
            )

        segments: list[PreparedNarrationSegment] = []
        for segment in timeline.segments:
            voice_text = segment.narration.voice
            segments.append(
                PreparedNarrationSegment(
                    segment_uuid=segment.segment_uuid,
                    voice_text=voice_text,
                    cleaned_text=self._clean_text(voice_text) or voice_text,
                    estimated_duration_seconds=segment.timing.estimated_duration_seconds,
                )
            )

        package = NarrationPackage(
            project_id=timeline.project_id,
            title=timeline.title,
            generated_at=datetime.now(timezone.utc),
            segments=segments,
        )
        logger.info(
            "Prepared narration for %r: %d segment(s), ~%.1fs estimated",
            package.title,
            package.segment_count,
            package.total_estimated_duration_seconds,
        )
        return package

    def save(self, package: NarrationPackage, destination: Path | None = None) -> Path:
        """Persist ``package`` as JSON and return the written path.

        Decoupled from :meth:`prepare`. ``destination`` is the seam for future
        per-project workspaces; when omitted, output goes to
        ``settings.output_dir/<slug>.narration.json``.
        """
        if destination is None:
            slug = _slugify(package.project_id or package.title)
            destination = self._settings.output_dir / f"{slug}.narration.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(package.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved narration for %r to %s", package.title, destination)
        return destination

    # -- Internal helpers (pure) ----------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Normalise narration text into a TTS-ready form.

        Collapses whitespace, maps unicode punctuation to ASCII, expands generic
        abbreviations, tidies repeated/again-glued punctuation. The original text
        is never mutated by this method (it operates on a copy).
        """
        cleaned = text.translate(_UNICODE_TRANSLATION)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        cleaned = self._expand_abbreviations(cleaned)
        cleaned = _REPEAT_PUNCT_RE.sub(r"\1", cleaned)
        cleaned = _SENTENCE_SPACE_RE.sub(r"\1 \2", cleaned)
        return _WHITESPACE_RE.sub(" ", cleaned).strip()

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        """Expand generic abbreviations in a fixed, deterministic order."""
        for pattern, replacement in _DEFAULT_ABBREVIATIONS:
            text = pattern.sub(replacement, text)
        return text


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "narration"
