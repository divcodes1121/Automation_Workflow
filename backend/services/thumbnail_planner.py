"""Thumbnail planning — decide what the thumbnail shows (Feature 9A).

:class:`ThumbnailPlanner` reuses artifacts the pipeline already produced (the
:class:`~backend.models.ExecutionTimeline` + the base rendered video) to choose a
background frame, title and highlight, and emits a self-contained
:class:`~backend.models.ThumbnailPlan`. It does **no image work** — the renderer
(Feature 9B) executes the plan into ``thumbnail.png``.

Frame selection is pure and deterministic: highest-``importance`` segment →
first segment with a ``narration.visual`` → a fixed fraction of the video.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    ExecutionTimeline,
    FrameSelectionReason,
    FrameSource,
    SegmentImportance,
    ThumbnailCropMode,
    ThumbnailPlan,
)

logger = logging.getLogger(__name__)

# Rank importance so the "best" segment sorts highest; unset ranks lowest.
_IMPORTANCE_RANK = {
    SegmentImportance.HIGH: 3,
    SegmentImportance.MEDIUM: 2,
    SegmentImportance.LOW: 1,
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ThumbnailError(Exception):
    """Base class for all errors raised by :class:`ThumbnailPlanner`."""


class ThumbnailPlanError(ThumbnailError):
    """Raised when a thumbnail plan cannot be built."""


class ThumbnailPlanner:
    """Plans a video thumbnail from the execution timeline + base video.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying thumbnail style defaults.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def plan(
        self,
        execution: ExecutionTimeline,
        video: Path,
        thumbnail_prompt: str | None = None,
    ) -> ThumbnailPlan:
        """Build a :class:`ThumbnailPlan` for ``execution`` using ``video``.

        Raises
        ------
        ThumbnailPlanError
            If the timeline has no segments or the video can't be probed.
        """
        if not execution.segments:
            raise ThumbnailPlanError(
                f"Execution timeline {execution.title!r} has no segments."
            )

        metadata = self._probe(video)
        timestamp, highlight, reason, uuid = self._select(
            execution, metadata.duration_seconds, thumbnail_prompt
        )
        settings = self._settings
        badge = settings.thumbnail_badge_text.strip() or None

        plan = ThumbnailPlan(
            project_id=execution.project_id,
            title=execution.title,
            generated_at=datetime.now(timezone.utc),
            source_video=video,
            frame_source=FrameSource.VIDEO,
            video_duration_seconds=metadata.duration_seconds,
            video_width=metadata.width,
            video_height=metadata.height,
            video_fps=metadata.fps,
            target_frame_timestamp_seconds=round(timestamp, 3),
            selection_reason=reason,
            source_segment_uuid=uuid,
            target_width=settings.thumbnail_width,
            target_height=settings.thumbnail_height,
            crop_mode=ThumbnailCropMode.COVER,
            title_text=execution.title,
            highlight_text=highlight,
            badge_text=badge,
            blur_background=settings.thumbnail_blur_background,
            glow=settings.thumbnail_glow,
            safe_area_margin=settings.thumbnail_safe_area_margin,
        )
        logger.info(
            "Planned thumbnail for %r: frame @ %.2fs (%s), highlight=%r",
            plan.title,
            plan.target_frame_timestamp_seconds,
            plan.selection_reason.value,
            plan.highlight_text,
        )
        return plan

    def save(self, plan: ThumbnailPlan, destination: Path | None = None) -> Path:
        """Persist ``plan`` as JSON and return the written path."""
        if destination is None:
            slug = _slugify(plan.project_id or plan.title)
            destination = self._settings.edited_dir / f"{slug}.thumbnail_plan.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved thumbnail plan for %r to %s", plan.title, destination)
        return destination

    # -- Selection (pure) -----------------------------------------------------

    def _select(
        self,
        execution: ExecutionTimeline,
        duration: float,
        thumbnail_prompt: str | None,
    ) -> tuple[float, str | None, FrameSelectionReason, str | None]:
        """Choose ``(timestamp, highlight, reason, source_segment_uuid)``. Pure.

        Priority: highest importance -> first with a visual -> fixed fraction.
        Highlight priority (verbatim): segment visual -> thumbnail_prompt -> None.
        """
        segments = execution.segments

        best = max(
            segments,
            key=lambda s: _IMPORTANCE_RANK.get(s.narration.importance, 0),
        )
        if _IMPORTANCE_RANK.get(best.narration.importance, 0) > 0:
            reason = FrameSelectionReason.IMPORTANCE
            chosen = best
        else:
            chosen = next(
                (s for s in segments if s.narration.visual), None
            )
            reason = FrameSelectionReason.VISUAL if chosen else FrameSelectionReason.FALLBACK

        if chosen is not None:
            midpoint = (
                chosen.timing.actual_start_seconds + chosen.timing.actual_end_seconds
            ) / 2.0
            uuid = chosen.segment_uuid
            highlight = chosen.narration.visual or _clean_prompt(thumbnail_prompt)
        else:
            midpoint = self._settings.thumbnail_fallback_position * duration
            uuid = None
            highlight = _clean_prompt(thumbnail_prompt)

        timestamp = min(max(midpoint, 0.0), duration)
        return timestamp, highlight, reason, uuid

    # -- Internal -------------------------------------------------------------

    def _probe(self, video: Path):
        """Probe ``video`` for duration/resolution/fps (base render)."""
        path = Path(video)
        if not path.is_file():
            raise ThumbnailPlanError(f"Base video not found: {path}")

        from backend.services.metadata import MetadataError, MetadataService

        try:
            return MetadataService(self._settings).analyze(path)
        except MetadataError as exc:
            raise ThumbnailPlanError(f"Could not probe base video {path}: {exc}") from exc


def _clean_prompt(prompt: str | None) -> str | None:
    """Return the thumbnail prompt trimmed, or ``None`` (no keyword extraction)."""
    if prompt is None:
        return None
    text = prompt.strip()
    return text or None


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "thumbnail"
