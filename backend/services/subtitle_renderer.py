"""Subtitle burning — render captions onto a video (Feature 8B).

:class:`SubtitleRenderer` does exactly one thing: ``video + captions -> video with
burned-in subtitles``, via a single FFmpeg run using the ``ass`` (libass) filter,
copying audio untouched. It makes no timing/wording decisions.

It is **stateless**: an ``.ass`` input is used as-is; a ``.srt`` input is converted
to a *temporary* ASS inside the renderer and discarded afterwards — the source
caption files are never modified. The style comes from config (see
:func:`~backend.services.subtitles.build_ass_document`).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import RenderResult
from backend.services.subtitles import build_ass_document, parse_srt

logger = logging.getLogger(__name__)


class SubtitleRenderingError(Exception):
    """Base class for all errors raised by :class:`SubtitleRenderer`."""


class SubtitleBurnError(SubtitleRenderingError):
    """Raised when subtitles cannot be burned (bad inputs or FFmpeg failure)."""


class SubtitleRenderer:
    """Burns a subtitle file onto a video.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying the FFmpeg path, x264
        quality knobs and the default output directory.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def prepare(
        self,
        video: Path,
        subtitles: Path,
        output: Path | None,
        ass_dir: Path,
    ) -> tuple[list[str], Path, Path, int]:
        """Validate, resolve an ASS into ``ass_dir``, and build the FFmpeg argv.

        No FFmpeg is run. Returns ``(argv, output_path, ass_path, cue_count)``.

        Raises
        ------
        SubtitleBurnError
            If an input is missing or an SRT yields no cues.
        """
        problems: list[str] = []
        if not video.is_file():
            problems.append(f"video not found: {video}")
        if not subtitles.is_file():
            problems.append(f"subtitles not found: {subtitles}")
        if problems:
            raise SubtitleBurnError("Cannot burn subtitles:\n- " + "\n- ".join(problems))

        # Resolve to absolute: FFmpeg runs with cwd = ass_dir (for the bare ASS
        # filename), so relative video/output paths would break.
        out = (output or (self._settings.edited_dir / f"{video.stem}.subtitled.mp4")).resolve()
        ass_dir.mkdir(parents=True, exist_ok=True)

        if subtitles.suffix.lower() == ".ass":
            ass_path = ass_dir / subtitles.name
            if ass_path.resolve() != subtitles.resolve():
                shutil.copyfile(subtitles, ass_path)
            cue_count = sum(
                1 for line in ass_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("Dialogue:")
            )
        else:
            cues = parse_srt(subtitles)
            if not cues:
                raise SubtitleBurnError(f"No cues found in {subtitles}")
            ass_path = ass_dir / f"{subtitles.stem}.ass"
            ass_path.write_text(
                build_ass_document(cues, self._settings), encoding="utf-8"
            )
            cue_count = len(cues)

        argv = self._build_argv(video, ass_path, out)
        return argv, out, ass_path, cue_count

    def save_command(self, argv: list[str], destination: Path) -> Path:
        """Write the FFmpeg command to a debug text file (for ``--dry-run``)."""
        import shlex

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            " ".join(shlex.quote(a) for a in argv) + "\n", encoding="utf-8"
        )
        return destination

    def burn(
        self, video: Path, subtitles: Path, output: Path | None = None
    ) -> RenderResult:
        """Burn ``subtitles`` onto ``video`` and return a :class:`RenderResult`.

        Stateless: any SRT->ASS conversion happens in a temp dir that is removed
        afterwards.

        Raises
        ------
        SubtitleBurnError
            On bad inputs or a non-zero FFmpeg exit.
        """
        with tempfile.TemporaryDirectory(prefix="burn_") as tmp:
            argv, out, ass_path, cue_count = self.prepare(
                video, subtitles, output, ass_dir=Path(tmp)
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Burning subtitles onto %s -> %s", video.name, out)
            started = time.perf_counter()
            completed = subprocess.run(
                argv, cwd=str(ass_path.parent), capture_output=True, text=True
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                tail = (completed.stderr or "").strip()[-2000:]
                raise SubtitleBurnError(
                    f"FFmpeg failed (exit {completed.returncode}) burning subtitles.\n{tail}"
                )

        result = self._probe_output(out, cue_count=cue_count, elapsed_seconds=round(elapsed, 3))
        logger.info(
            "Burned %d cue(s) onto %r in %.1fs", cue_count, out.name, result.elapsed_seconds
        )
        return result

    # -- Internals ------------------------------------------------------------

    def _build_argv(self, video: Path, ass_path: Path, output: Path) -> list[str]:
        """Build the single-pass FFmpeg command (ass filter, copy audio)."""
        return [
            str(self._settings.ffmpeg_path),
            "-y",
            "-i", str(video.resolve()),
            # Bare filename; FFmpeg is run with cwd = ass_path.parent to sidestep
            # Windows drive-colon escaping in the filtergraph.
            "-vf", f"ass={ass_path.name}",
            "-c:v", "libx264",
            "-preset", self._settings.render_preset,
            "-crf", str(self._settings.render_crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output),
        ]

    def _probe_output(
        self, output: Path, *, cue_count: int, elapsed_seconds: float
    ) -> RenderResult:
        """Probe the burned file for the summary."""
        from backend.services.metadata import MetadataError, MetadataService

        try:
            metadata = MetadataService(self._settings).analyze(output)
        except MetadataError as exc:
            raise SubtitleBurnError(f"Burned file could not be probed: {exc}") from exc

        return RenderResult(
            output_file=output,
            duration_seconds=metadata.duration_seconds,
            video_codec=metadata.video_codec,
            audio_codec="copy",
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            segment_count=cue_count,
            input_clip_count=1,
            elapsed_seconds=elapsed_seconds,
        )
