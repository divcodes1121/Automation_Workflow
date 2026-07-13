"""Timeline synchronization — merge estimates + audio into an execution timeline (Feature 6).

:class:`TimelineSynchronizer` merges three upstream artifacts, keyed by
``segment_uuid``:

* the estimated :class:`~backend.models.Timeline` (content + WPM estimates),
* the :class:`~backend.models.NarrationPackage` (cleaned text, delivery hints),
* the :class:`~backend.models.GeneratedNarration` (real audio + word timings),

into a single validated :class:`~backend.models.ExecutionTimeline`. Estimated
timings are replaced with **measured actual timings** computed by laying the
per-segment audio end-to-end. The merge is validated up front so downstream
stages (editor, subtitles, …) never inherit a broken timeline.

It performs read-only filesystem access (to confirm each audio file exists) but
writes nothing in :meth:`synchronize`; persistence is the separate :meth:`save`.
Emits diagnostics via :mod:`logging`; no n8n dependency.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    ExecutionSegment,
    ExecutionTimeline,
    GeneratedNarration,
    NarrationPackage,
    Timeline,
    TimelineTiming,
)

logger = logging.getLogger(__name__)

# Tolerance for a word timing running slightly past the measured audio length.
_TIMING_EPSILON_SECONDS = 0.05

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SynchronizationError(Exception):
    """Base class for all errors raised by :class:`TimelineSynchronizer`."""


class TimelineSynchronizationError(SynchronizationError):
    """Raised when the three artifacts cannot be merged into a valid timeline.

    Aggregates every detected problem so all issues are visible at once.
    """


class TimelineSynchronizer:
    """Merges timeline + narration + audio into a validated ``ExecutionTimeline``.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), used for the default save dir.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def synchronize(
        self,
        timeline: Timeline,
        narration_package: NarrationPackage,
        generated: GeneratedNarration,
    ) -> ExecutionTimeline:
        """Merge the three artifacts into a validated :class:`ExecutionTimeline`.

        Raises
        ------
        TimelineSynchronizationError
            If validation fails (missing/duplicate uuids, missing audio, bad
            durations or word timings, inconsistent sample rate, no provider).
        """
        prepared = {seg.segment_uuid: seg for seg in narration_package.segments}
        produced = {seg.segment_uuid: seg for seg in generated.segments}

        self._validate(timeline, narration_package, generated, prepared, produced)

        segments: list[ExecutionSegment] = []
        cursor = 0.0
        for timeline_segment in timeline.segments:
            uuid = timeline_segment.segment_uuid
            prepared_segment = prepared[uuid]
            produced_segment = produced[uuid]

            duration = produced_segment.duration_seconds
            start, end = cursor, cursor + duration
            cursor = end

            timing = TimelineTiming(
                estimated_start_seconds=timeline_segment.timing.estimated_start_seconds,
                estimated_end_seconds=timeline_segment.timing.estimated_end_seconds,
                actual_start_seconds=round(start, 3),
                actual_end_seconds=round(end, 3),
            )
            segments.append(
                ExecutionSegment(
                    segment_uuid=uuid,
                    index=timeline_segment.id,
                    narration=timeline_segment.narration,
                    cleaned_text=prepared_segment.cleaned_text,
                    speech_rate=prepared_segment.speech_rate,
                    emotion=prepared_segment.emotion,
                    timing=timing,
                    audio_offset_seconds=round(start, 3),
                    actual_duration_seconds=round(duration, 3),
                    audio_file=produced_segment.audio_file,
                    sample_rate=produced_segment.sample_rate,
                    word_timings=list(produced_segment.word_timings),
                    provider=generated.provider,
                )
            )

        execution = ExecutionTimeline(
            project_id=timeline.project_id,
            title=timeline.title,
            generated_at=datetime.now(timezone.utc),
            provider=generated.provider,
            sample_rate=generated.sample_rate,
            segments=segments,
            is_synchronized=True,
        )
        logger.info(
            "Synchronized timeline for %r: %d segment(s), %.1fs actual",
            execution.title,
            execution.segment_count,
            execution.total_actual_duration_seconds,
        )
        return execution

    def save(
        self, execution: ExecutionTimeline, destination: Path | None = None
    ) -> Path:
        """Persist ``execution`` as JSON and return the written path."""
        if destination is None:
            slug = _slugify(execution.project_id or execution.title)
            destination = self._settings.output_dir / f"{slug}.execution_timeline.json"

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(execution.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved execution timeline for %r to %s", execution.title, destination)
        return destination

    # -- Validation -----------------------------------------------------------

    def _validate(
        self,
        timeline: Timeline,
        narration_package: NarrationPackage,
        generated: GeneratedNarration,
        prepared: dict[str, object],
        produced: dict[str, object],
    ) -> None:
        """Collect every problem and raise once if any are found."""
        problems: list[str] = []

        timeline_uuids = [s.segment_uuid for s in timeline.segments]
        if len(timeline_uuids) != len(set(timeline_uuids)):
            problems.append("timeline contains duplicate segment_uuids")

        for timeline_segment in timeline.segments:
            uuid = timeline_segment.segment_uuid
            label = f"segment {timeline_segment.id} ({uuid[:8]})"

            if uuid not in prepared:
                problems.append(f"{label}: no matching prepared narration")
            if uuid not in produced:
                problems.append(f"{label}: no matching synthesised audio")
                continue  # nothing else to check without audio

            produced_segment = produced[uuid]

            audio_file = Path(produced_segment.audio_file)
            if not audio_file.is_file():
                problems.append(f"{label}: audio file missing ({audio_file})")

            if produced_segment.duration_seconds <= 0:
                problems.append(f"{label}: non-positive audio duration")

            self._check_word_timings(produced_segment, label, problems)

        # Every produced/prepared segment should map back to a timeline segment.
        extra_audio = set(produced) - set(timeline_uuids)
        if extra_audio:
            problems.append(
                f"audio for unknown segment_uuid(s): {sorted(u[:8] for u in extra_audio)}"
            )

        sample_rates = {s.sample_rate for s in generated.segments}
        if len(sample_rates) > 1:
            problems.append(f"inconsistent sample rates across segments: {sample_rates}")

        if not (generated.provider or "").strip():
            problems.append("generated narration has no provider")

        if problems:
            raise TimelineSynchronizationError(
                "Cannot synchronize timeline:\n- " + "\n- ".join(problems)
            )

    @staticmethod
    def _check_word_timings(produced_segment, label: str, problems: list[str]) -> None:
        """Validate word timings are sorted and within the segment's audio."""
        limit = produced_segment.duration_seconds + _TIMING_EPSILON_SECONDS
        previous_start = 0.0
        for word in produced_segment.word_timings:
            if word.start_seconds > word.end_seconds:
                problems.append(f"{label}: word timing start > end ({word.text!r})")
            if word.start_seconds < previous_start - _TIMING_EPSILON_SECONDS:
                problems.append(f"{label}: word timings out of order ({word.text!r})")
            if word.start_seconds < 0 or word.end_seconds > limit:
                problems.append(
                    f"{label}: word timing outside audio ({word.text!r})"
                )
            previous_start = word.start_seconds


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "execution_timeline"
