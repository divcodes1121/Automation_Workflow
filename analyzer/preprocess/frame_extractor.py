"""Frame extraction for the Gameplay Analyzer (slice 2B).

:class:`FrameExtractor` samples a recording into a folder of frames + a
self-describing manifest, so every later detector reads pre-extracted frames
instead of each re-opening the video. It is pure FFmpeg orchestration -- no
OpenCV, no templates, no detection.

Frames are named by their **source-frame index** with an ``f`` prefix
(``f000000.png``), so a filename maps straight back to a timestamp
(``source_frame / source_fps``) and leaves the bare stem free for future
sidecars. The manifest records the source hash, all extraction settings, and the
exact ffmpeg version + command, so a cache can be validated before reuse and a
run can be reproduced. FFmpeg/ffprobe are invoked via subprocess at the paths
configured in ``.env`` (this machine keeps them off PATH).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from analyzer.config import FRAMES_VERSION, AnalyzerSettings, get_analyzer_settings
from analyzer.models import ExtractedFrame, FramesManifest

logger = logging.getLogger(__name__)

# Accepted format spellings -> file extension.
_FORMAT_EXT = {"png": "png", "jpg": "jpg", "jpeg": "jpg"}


class FrameExtractionError(Exception):
    """Base class for frame-extraction failures."""


class VideoNotFoundError(FrameExtractionError):
    """Raised when the input video does not exist."""


class FrameProbeError(FrameExtractionError):
    """Raised when ffprobe cannot read the video's stream info."""


class FrameExtractor:
    """Samples a recording into cached frames + a manifest.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~analyzer.config.get_analyzer_settings`).
    """

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings: AnalyzerSettings = settings or get_analyzer_settings()

    # -- Public API -----------------------------------------------------------

    def extract(
        self,
        video: str | Path,
        *,
        sample_fps: float | None = None,
        image_format: str | None = None,
        dry_run: bool = False,
    ) -> FramesManifest:
        """Extract sampled frames from ``video`` and return a manifest.

        Raises
        ------
        VideoNotFoundError, FrameProbeError, FrameExtractionError
        """
        video = Path(video)
        if not video.is_file():
            raise VideoNotFoundError(str(video))

        fmt = self._normalize_format(image_format or self._settings.frame_image_format)
        source_fps, duration, width, height = self._probe(video)
        requested = float(sample_fps or self._settings.frame_sample_fps)
        effective_fps = min(requested, source_fps)

        video_sha = self._sha256(video)
        video_mtime = datetime.fromtimestamp(video.stat().st_mtime, tz=timezone.utc)
        out_dir = self._settings.frames_root() / video.stem
        manifest_path = out_dir / "frames_manifest.json"
        keep = self._settings.frame_keep_existing

        # Validated cache reuse: only skip when the recording AND the extraction
        # parameters match what is already on disk.
        if keep and manifest_path.is_file():
            reused = self._try_reuse(manifest_path, video_sha, effective_fps, fmt)
            if reused is not None:
                logger.info("Reusing valid frame cache for %s (%d frames)", video.name, reused.frame_count)
                return reused

        # The ffmpeg invocation (stored form points at the final frames dir; the
        # real run writes to a temp dir and is then renamed by source frame).
        base_argv = self._ffmpeg_base_argv(video, effective_fps, fmt)
        display_command = " ".join([*base_argv, str(out_dir / f"%06d.{fmt}")])

        if dry_run:
            expected = int(math.floor(duration * effective_fps))
            return self._manifest(
                video, video_sha, video_mtime, source_fps, duration, effective_fps,
                fmt, width, height, frames=[], frame_count=expected,
                ffmpeg_command=display_command, frames_dir=out_dir,
            )

        frames = self._run_and_collect(base_argv, out_dir, fmt, source_fps, effective_fps)
        return self._manifest(
            video, video_sha, video_mtime, source_fps, duration, effective_fps,
            fmt, width, height, frames=frames, frame_count=len(frames),
            ffmpeg_command=display_command, frames_dir=out_dir,
        )

    def save(self, manifest: FramesManifest, destination: Path | None = None) -> Path:
        """Write ``frames_manifest.json`` into the frames dir. Returns its path."""
        dest = destination or Path(manifest.frames_dir) / "frames_manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved frames manifest (%d frames) to %s", manifest.frame_count, dest)
        return dest

    # -- Extraction internals -------------------------------------------------

    def _ffmpeg_base_argv(self, video: Path, effective_fps: float, fmt: str) -> list[str]:
        """The ffmpeg argv up to (but not including) the output pattern."""
        argv = [
            self._settings.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video.resolve()),
            "-vf",
            f"fps={effective_fps:g}",
        ]
        if fmt == "png":
            argv += ["-compression_level", str(self._settings.frame_png_compression)]
        else:  # jpg
            argv += ["-q:v", str(self._jpeg_qscale(self._settings.frame_jpeg_quality))]
        return argv

    def _run_and_collect(
        self,
        base_argv: list[str],
        out_dir: Path,
        fmt: str,
        source_fps: float,
        effective_fps: float,
    ) -> list[ExtractedFrame]:
        """Run ffmpeg into a temp dir, then rename outputs by source-frame index."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            argv = [*base_argv, str(tmp / f"%06d.{fmt}")]
            try:
                proc = subprocess.run(argv, capture_output=True, text=True)
            except FileNotFoundError as exc:
                raise FrameExtractionError(f"ffmpeg not found at {self._settings.ffmpeg_path!r}") from exc
            if proc.returncode != 0:
                raise FrameExtractionError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")

            outputs = sorted(tmp.glob(f"*.{fmt}"))
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            frames: list[ExtractedFrame] = []
            for index, src in enumerate(outputs):
                source_frame = round(index * source_fps / effective_fps)
                name = f"f{source_frame:06d}.{fmt}"
                dest = out_dir / name
                shutil.move(str(src), str(dest))
                frames.append(
                    ExtractedFrame(
                        index=index,
                        filename=name,
                        source_frame=source_frame,
                        timestamp_seconds=round(source_frame / source_fps, 4),
                        sha256=self._sha256(dest),
                    )
                )
        return frames

    # -- Manifest / reuse -----------------------------------------------------

    def _manifest(
        self,
        video: Path,
        video_sha: str,
        video_mtime: datetime,
        source_fps: float,
        duration: float,
        effective_fps: float,
        fmt: str,
        width: int,
        height: int,
        *,
        frames: list[ExtractedFrame],
        frame_count: int,
        ffmpeg_command: str,
        frames_dir: Path,
    ) -> FramesManifest:
        """Assemble a :class:`FramesManifest` (with timestamp stats)."""
        timestamps = [f.timestamp_seconds for f in frames]
        first_ts = timestamps[0] if timestamps else 0.0
        last_ts = timestamps[-1] if timestamps else 0.0
        if len(timestamps) > 1:
            average_spacing = round((last_ts - first_ts) / (len(timestamps) - 1), 4)
        else:
            average_spacing = 0.0

        return FramesManifest(
            extraction_version=FRAMES_VERSION,
            video=video.name,
            video_path=video.resolve(),
            video_sha256=video_sha,
            video_modified_at=video_mtime,
            source_fps=source_fps,
            duration_seconds=duration,
            sample_fps=effective_fps,
            image_format=fmt,
            png_compression=self._settings.frame_png_compression,
            jpeg_quality=self._settings.frame_jpeg_quality,
            keep_existing=self._settings.frame_keep_existing,
            ffmpeg_version=self._ffmpeg_version(),
            ffmpeg_command=ffmpeg_command,
            width=width,
            height=height,
            frame_count=frame_count,
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            average_spacing=average_spacing,
            frames_dir=frames_dir,
            frames=frames,
            extracted_at=datetime.now(timezone.utc),
        )

    def _try_reuse(
        self, manifest_path: Path, video_sha: str, effective_fps: float, fmt: str
    ) -> FramesManifest | None:
        """Return the cached manifest iff it matches the requested run."""
        try:
            existing = FramesManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        matches = (
            existing.video_sha256 == video_sha
            and existing.extraction_version == FRAMES_VERSION
            and existing.image_format == fmt
            and math.isclose(existing.sample_fps, effective_fps, rel_tol=1e-6, abs_tol=1e-6)
        )
        return existing if matches else None

    # -- ffprobe / helpers ----------------------------------------------------

    def _probe(self, video: Path) -> tuple[float, float, int, int]:
        """Return ``(source_fps, duration, width, height)`` via ffprobe."""
        argv = [
            self._settings.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(video),
        ]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise FrameProbeError(f"ffprobe not found at {self._settings.ffprobe_path!r}") from exc
        if proc.returncode != 0:
            raise FrameProbeError(proc.stderr.strip() or "ffprobe failed")
        try:
            data = json.loads(proc.stdout)
            stream = data["streams"][0]
            source_fps = self._parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))
            width = int(stream["width"])
            height = int(stream["height"])
            duration = float(
                stream.get("duration") or data.get("format", {}).get("duration") or 0.0
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise FrameProbeError(f"could not parse ffprobe output: {exc}") from exc
        return source_fps, duration, width, height

    def _ffmpeg_version(self) -> str:
        """First line of ``ffmpeg -version`` (or 'unknown' if unavailable)."""
        try:
            proc = subprocess.run(
                [self._settings.ffmpeg_path, "-version"], capture_output=True, text=True
            )
        except FileNotFoundError:
            return "unknown"
        if proc.returncode != 0 or not proc.stdout:
            return "unknown"
        return proc.stdout.splitlines()[0].strip()

    @staticmethod
    def _normalize_format(value: str) -> str:
        """Normalize a format spelling to a file extension ('png'|'jpg')."""
        ext = _FORMAT_EXT.get(value.strip().lower())
        if ext is None:
            raise FrameExtractionError(
                f"Unsupported image format {value!r} (use png or jpg)."
            )
        return ext

    @staticmethod
    def _parse_fps(rate: str | None) -> float:
        """Parse an ffprobe frame-rate string like '30000/1001' -> float."""
        if not rate:
            raise FrameProbeError("missing frame rate")
        try:
            if "/" in rate:
                num, den = rate.split("/", 1)
                value = float(num) / float(den)
            else:
                value = float(rate)
        except (ValueError, ZeroDivisionError) as exc:
            raise FrameProbeError(f"invalid frame rate {rate!r}") from exc
        if value <= 0:
            raise FrameProbeError(f"non-positive frame rate {rate!r}")
        return round(value, 6)

    @staticmethod
    def _jpeg_qscale(quality: int) -> int:
        """Map JPEG quality (1..100) to ffmpeg mjpeg -q:v (2=best..31=worst)."""
        qscale = round(2 + (100 - quality) * (31 - 2) / 99)
        return max(2, min(31, qscale))

    @staticmethod
    def _sha256(path: Path) -> str:
        """SHA-256 of a file's bytes."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
