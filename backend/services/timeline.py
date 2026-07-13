"""Build the master :class:`~backend.models.Timeline` from a project (Feature 3).

:class:`TimelineBuilderService` is a **pure transformation**: it takes a validated
:class:`~backend.models.Project`, derives ordered narration segments (via a
pluggable :class:`~backend.services.script_splitter.ScriptSplitter`), assigns
stable identifiers, and estimates cumulative timings from a configurable speaking
rate. No FFmpeg, no AI, and no filesystem side effects in :meth:`build`.

Persistence is a separate, optional concern (:meth:`save`), mirroring
:class:`~backend.services.metadata.MetadataService`. The service emits diagnostics
through :mod:`logging` and has no n8n dependency.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    NarrationSegment,
    Project,
    Timeline,
    TimelineSegment,
    TimelineTiming,
)
from backend.services.script_splitter import DefaultScriptSplitter, ScriptSplitter

logger = logging.getLogger(__name__)

# Minimum estimated duration for any segment, so very short lines still occupy a
# visible slice of the timeline. Seconds.
_MIN_SEGMENT_SECONDS = 1.0

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class TimelineError(Exception):
    """Base class for all errors raised by :class:`TimelineBuilderService`."""


class TimelineBuildError(TimelineError):
    """Raised when a project yields no usable narration to build a timeline."""


class TimelineBuilderService:
    """Builds a :class:`Timeline` from a :class:`Project`.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`). Supplies ``narration_wpm`` and the
        default save location.
    splitter:
        Optional narration-splitting strategy (defaults to
        :class:`~backend.services.script_splitter.DefaultScriptSplitter`).
        Injecting an alternative — e.g. a future AI splitter — changes how
        narration is segmented without altering timeline construction.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        splitter: ScriptSplitter | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._splitter: ScriptSplitter = splitter or DefaultScriptSplitter()

    # -- Public API -----------------------------------------------------------

    def build(self, project: Project) -> Timeline:
        """Transform ``project`` into a typed :class:`Timeline`. Writes nothing.

        Raises
        ------
        TimelineBuildError
            If no narration segments could be derived from the project.
        """
        narration_segments = self._splitter.split(project)
        if not narration_segments:
            raise TimelineBuildError(
                f"Project {project.title!r} produced no narration segments."
            )

        wpm = self._settings.narration_wpm
        segments: list[TimelineSegment] = []
        cursor = 0.0
        for index, narration in enumerate(narration_segments, start=1):
            duration = self._estimate_duration(narration, wpm)
            start, end = cursor, cursor + duration
            segments.append(
                TimelineSegment(
                    id=index,
                    timing=TimelineTiming(
                        estimated_start_seconds=round(start, 3),
                        estimated_end_seconds=round(end, 3),
                    ),
                    narration=narration,
                )
            )
            cursor = end

        timeline = Timeline(
            title=project.title,
            project_id=project.project_id,
            segments=segments,
            total_duration_seconds=round(cursor, 3),
            words_per_minute=wpm,
            is_estimated=True,
            generated_at=datetime.now(timezone.utc),
        )
        logger.info(
            "Built timeline for %r: %d segment(s), ~%.1fs estimated (%.0f wpm)",
            project.title,
            timeline.segment_count,
            timeline.total_duration_seconds,
            wpm,
        )
        return timeline

    def save(self, timeline: Timeline, destination: Path | None = None) -> Path:
        """Persist ``timeline`` as JSON and return the written path.

        Decoupled from :meth:`build` so callers that only need the object in
        memory can skip it. ``destination`` is the seam for future per-project
        workspaces (``projects/<name>/timeline.json``); when omitted, output goes
        to ``settings.output_dir/<slug>.timeline.json``.
        """
        if destination is None:
            slug = _slugify(timeline.project_id or timeline.title)
            destination = self._settings.output_dir / f"{slug}.timeline.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(timeline.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved timeline for %r to %s", timeline.title, destination)
        return destination

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _estimate_duration(narration: NarrationSegment, wpm: float) -> float:
        """Estimate a segment's spoken duration in seconds from its word count."""
        word_count = len(narration.voice.split())
        estimated = word_count / wpm * 60.0
        return max(estimated, _MIN_SEGMENT_SECONDS)


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "timeline"
