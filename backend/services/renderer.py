"""Video rendering — execute an EditPlan into a playable MP4 (Feature 7B).

:class:`VideoRenderer` is a **compiler, not an editor**: it makes no creative decisions.
It validates an :class:`~backend.models.EditPlan`, compiles it to a single
:class:`~backend.services.filtergraph.FFmpegCommand` (via the pure
:class:`~backend.services.filtergraph.FiltergraphBuilder`), runs FFmpeg once, and probes
the output for a :class:`~backend.models.RenderResult`.

The generated command and filter graph are saved to sidecar ``.txt`` files for debugging,
and :meth:`prepare` (validate + compile, no FFmpeg) backs a ``--dry-run`` path. Subtitles
and the reserved ``effects`` field are intentionally ignored here.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import EditPlan, RenderResult
from backend.services.filtergraph import FFmpegCommand, FiltergraphBuilder

logger = logging.getLogger(__name__)

# Tolerance when checking a trim range fits inside a clip.
_FIT_EPSILON_SECONDS = 0.05

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class RenderingError(Exception):
    """Base class for all errors raised by :class:`VideoRenderer`."""


class RenderError(RenderingError):
    """Raised when a plan cannot be rendered (validation or FFmpeg failure)."""


class VideoRenderer:
    """Renders an :class:`EditPlan` into a single MP4.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying ffmpeg path, x264
        quality knobs and the default output directory.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def prepare(
        self, plan: EditPlan, output: Path | None = None
    ) -> tuple[FFmpegCommand, Path]:
        """Validate ``plan`` and compile it to an :class:`FFmpegCommand`.

        No FFmpeg is run and nothing is written. Backs both :meth:`render` and
        the CLI ``--dry-run`` path.

        Raises
        ------
        RenderError
            If validation fails.
        """
        self._validate(plan)
        destination = output or (
            self._settings.edited_dir / f"{_slugify(plan.project_id or plan.title)}.mp4"
        )
        fps = plan.source_videos[0].fps
        command = FiltergraphBuilder.build(
            plan,
            destination,
            ffmpeg_path=self._settings.ffmpeg_path,
            fps=fps,
            crf=self._settings.render_crf,
            preset=self._settings.render_preset,
        )
        return command, destination

    def save_command(self, command: FFmpegCommand) -> tuple[Path, Path]:
        """Write the command and filter graph to sidecar ``.txt`` files.

        Returns ``(command_path, filtergraph_path)``.
        """
        stem = command.output.with_suffix("")
        command_path = stem.with_name(f"{stem.name}.render_command.txt")
        filtergraph_path = stem.with_name(f"{stem.name}.render_filtergraph.txt")
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(command.command_text() + "\n", encoding="utf-8")
        filtergraph_path.write_text(command.filtergraph_text() + "\n", encoding="utf-8")
        return command_path, filtergraph_path

    def render(self, plan: EditPlan, output: Path | None = None) -> RenderResult:
        """Render ``plan`` into an MP4 and return a :class:`RenderResult`.

        Raises
        ------
        RenderError
            If validation fails or FFmpeg exits non-zero.
        """
        command, destination = self.prepare(plan, output)
        self.save_command(command)

        logger.info("Rendering %r -> %s", plan.title, destination)
        started = time.perf_counter()
        completed = subprocess.run(command.argv, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip()[-2000:]
            raise RenderError(
                f"FFmpeg failed (exit {completed.returncode}) rendering {plan.title!r}.\n{tail}"
            )

        result = self._probe_output(
            destination,
            segment_count=plan.segment_count,
            input_clip_count=len(plan.source_videos),
            elapsed_seconds=round(elapsed, 3),
        )
        logger.info(
            "Rendered %r: %.1fs video in %.1fs (%.2gx realtime)",
            plan.title,
            result.duration_seconds,
            result.elapsed_seconds,
            result.render_speed if result.render_speed is not None else 0.0,
        )
        return result

    # -- Internals ------------------------------------------------------------

    def _validate(self, plan: EditPlan) -> None:
        """Fail early with an aggregated :class:`RenderError`."""
        problems: list[str] = []
        if not plan.segments:
            raise RenderError("Edit plan has no segments to render.")

        durations = {str(s.file): s.duration_seconds for s in plan.source_videos}
        resolutions = {(s.width, s.height) for s in plan.source_videos}
        fps_values = {round(s.fps, 3) for s in plan.source_videos}

        if len(resolutions) > 1:
            problems.append(
                f"incompatible source resolutions {resolutions}; normalization not enabled"
            )
        if len(fps_values) > 1:
            problems.append(
                f"incompatible source frame rates {fps_values}; normalization not enabled"
            )

        for segment in plan.segments:
            label = f"segment {segment.index}"
            source = Path(segment.source_file)
            audio = Path(segment.audio_file)
            if not source.is_file():
                problems.append(f"{label}: source clip missing ({source})")
            if not audio.is_file():
                problems.append(f"{label}: narration audio missing ({audio})")
            if segment.source_start_seconds < 0 or (
                segment.source_end_seconds <= segment.source_start_seconds
            ):
                problems.append(f"{label}: invalid trim range")
            clip_duration = durations.get(str(segment.source_file))
            if clip_duration is not None and (
                segment.source_end_seconds > clip_duration + _FIT_EPSILON_SECONDS
            ):
                problems.append(
                    f"{label}: trim end {segment.source_end_seconds:.2f}s exceeds clip "
                    f"duration {clip_duration:.2f}s"
                )

        if problems:
            raise RenderError("Cannot render edit plan:\n- " + "\n- ".join(problems))

    def _probe_output(
        self,
        output: Path,
        *,
        segment_count: int,
        input_clip_count: int,
        elapsed_seconds: float,
    ) -> RenderResult:
        """Probe the rendered file for the summary."""
        from backend.services.metadata import MetadataError, MetadataService

        try:
            metadata = MetadataService(self._settings).analyze(output)
        except MetadataError as exc:
            raise RenderError(f"Rendered file could not be probed: {exc}") from exc

        return RenderResult(
            output_file=output,
            duration_seconds=metadata.duration_seconds,
            video_codec=metadata.video_codec,
            audio_codec="aac",
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            segment_count=segment_count,
            input_clip_count=input_clip_count,
            elapsed_seconds=elapsed_seconds,
        )


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "video"
