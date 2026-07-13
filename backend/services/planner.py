"""Gameplay planning — assign footage ranges to narration segments (Feature 7A).

:class:`GameplayPlanner` consumes an :class:`~backend.models.ExecutionTimeline` plus
gameplay footage and produces an :class:`~backend.models.EditPlan`: one contiguous
gameplay range per narration segment, placed at the segment's measured audio slot.

This is **pure planning — no FFmpeg rendering.** It reads gameplay *metadata*
(durations via the F2 ffprobe path, preferring cached ``metadata.json``) but never
encodes video. The renderer (Feature 7B) turns the resulting plan into ``video.mp4``.

Assignment is deliberately dumb and deterministic: clips are consumed sequentially
in sorted order, one range per segment, with **no reuse**. If the footage cannot
cover the narration, :class:`EditPlanError` is raised listing every problem.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    EditPlan,
    EditPlanSegment,
    EditPlanSource,
    ExecutionTimeline,
    GameplayMetadata,
)

logger = logging.getLogger(__name__)

# Video containers scanned when a directory of footage is given.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
# Tolerance when checking a range fits within a clip (float rounding).
_FIT_EPSILON_SECONDS = 0.01

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class PlanningError(Exception):
    """Base class for all errors raised by :class:`GameplayPlanner`."""


class EditPlanError(PlanningError):
    """Raised when footage cannot be planned onto the narration (aggregated)."""


class GameplayPlanner:
    """Plans which gameplay to show under each narration segment.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def plan(
        self,
        execution: ExecutionTimeline,
        gameplay_sources: Path | list[Path] | None = None,
    ) -> EditPlan:
        """Build an :class:`EditPlan` for ``execution`` from gameplay footage.

        Parameters
        ----------
        execution:
            The synchronized timeline to plan against.
        gameplay_sources:
            A footage file, a directory to scan, or an explicit list of files.
            Defaults to scanning ``settings.gameplay_raw_dir``.

        Raises
        ------
        EditPlanError
            If no footage is found or it cannot cover the narration.
        """
        files = self._resolve_files(gameplay_sources)
        if not files:
            raise EditPlanError("No gameplay footage found to plan from.")

        sources = [self._source_for(path) for path in files]
        segments = self._assign(execution, sources)

        plan = EditPlan(
            project_id=execution.project_id,
            title=execution.title,
            generated_at=datetime.now(timezone.utc),
            source_videos=sources,
            segments=segments,
        )
        logger.info(
            "Planned %d segment(s) for %r across %d source video(s): %.1fs total",
            plan.segment_count,
            plan.title,
            len(sources),
            plan.total_duration_seconds,
        )
        return plan

    def save(self, plan: EditPlan, destination: Path | None = None) -> Path:
        """Persist ``plan`` as JSON and return the written path."""
        if destination is None:
            slug = _slugify(plan.project_id or plan.title)
            destination = self._settings.output_dir / f"{slug}.edit_plan.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved edit plan for %r to %s", plan.title, destination)
        return destination

    # -- Assignment (pure) ----------------------------------------------------

    @staticmethod
    def _assign(
        execution: ExecutionTimeline, sources: list[EditPlanSource]
    ) -> list[EditPlanSegment]:
        """Assign one contiguous gameplay range per segment. Pure; no I/O.

        Clips are consumed sequentially (no reuse). Raises :class:`EditPlanError`
        aggregating every shortfall.
        """
        total_footage = sum(s.duration_seconds for s in sources)
        total_needed = sum(seg.actual_duration_seconds for seg in execution.segments)
        longest_clip = max((s.duration_seconds for s in sources), default=0.0)

        problems: list[str] = []
        if total_needed > total_footage + _FIT_EPSILON_SECONDS:
            problems.append(
                f"not enough footage: need {total_needed:.1f}s but only "
                f"{total_footage:.1f}s available"
            )
        for seg in execution.segments:
            if seg.actual_duration_seconds > longest_clip + _FIT_EPSILON_SECONDS:
                problems.append(
                    f"segment {seg.index} needs {seg.actual_duration_seconds:.1f}s but the "
                    f"longest clip is only {longest_clip:.1f}s"
                )
        if problems:
            raise EditPlanError(
                "Cannot build edit plan:\n- " + "\n- ".join(problems)
            )

        planned: list[EditPlanSegment] = []
        clip_index = 0
        cursor = 0.0  # position within the current clip
        for seg in execution.segments:
            duration = seg.actual_duration_seconds
            # Advance to a clip that can fit this segment from the current cursor.
            while (
                clip_index < len(sources)
                and sources[clip_index].duration_seconds - cursor
                < duration - _FIT_EPSILON_SECONDS
            ):
                clip_index += 1
                cursor = 0.0
            if clip_index >= len(sources):
                # Guarded by the aggregate checks above, but stay explicit.
                raise EditPlanError(
                    f"ran out of footage assigning segment {seg.index}"
                )

            source = sources[clip_index]
            source_start = round(cursor, 3)
            source_end = round(cursor + duration, 3)
            cursor = source_end

            planned.append(
                EditPlanSegment(
                    segment_uuid=seg.segment_uuid,
                    index=seg.index,
                    source_file=source.file,
                    source_start_seconds=source_start,
                    source_end_seconds=source_end,
                    target_start_seconds=seg.timing.actual_start_seconds,
                    target_end_seconds=seg.timing.actual_end_seconds,
                    duration_seconds=round(duration, 3),
                    audio_file=seg.audio_file,
                    narration_visual=seg.narration.visual,
                )
            )
        return planned

    # -- Footage resolution ---------------------------------------------------

    def _resolve_files(
        self, gameplay_sources: Path | list[Path] | None
    ) -> list[Path]:
        """Resolve the footage argument into a sorted list of video files."""
        if gameplay_sources is None:
            return _scan_videos(self._settings.gameplay_raw_dir)
        if isinstance(gameplay_sources, list):
            return sorted(Path(p) for p in gameplay_sources)
        source = Path(gameplay_sources)
        if source.is_dir():
            return _scan_videos(source)
        if source.is_file():
            return [source]
        raise EditPlanError(f"Gameplay source not found: {source}")

    def _source_for(self, path: Path) -> EditPlanSource:
        """Build an :class:`EditPlanSource`, preferring cached F2 metadata."""
        metadata = self._load_metadata(path)
        return EditPlanSource(
            file=path,
            duration_seconds=metadata.duration_seconds,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
        )

    def _load_metadata(self, path: Path) -> GameplayMetadata:
        """Load cached ``metadata.json`` for ``path`` if present, else analyse it."""
        cached = self._settings.gameplay_metadata_dir / f"{path.stem}.json"
        if cached.is_file():
            try:
                return GameplayMetadata.model_validate_json(
                    cached.read_text(encoding="utf-8")
                )
            except ValueError:
                logger.warning("Ignoring unreadable cached metadata %s", cached)

        # Imported lazily so planning has no hard dependency until footage is probed.
        from backend.services.metadata import MetadataError, MetadataService

        try:
            return MetadataService(self._settings).analyze(path)
        except MetadataError as exc:
            raise EditPlanError(f"Could not read gameplay metadata for {path}: {exc}") from exc


def _scan_videos(directory: Path) -> list[Path]:
    """Return sorted video files directly inside ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    )


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "edit_plan"
