"""Gameplay video metadata extraction (Feature 2 — Gameplay Analyzer).

:class:`MetadataService` inspects a gameplay ``.mp4`` with ``ffprobe`` (via the
``ffmpeg-python`` wrapper) and returns a strongly typed
:class:`~backend.models.GameplayMetadata` object. Persistence is a separate,
optional concern: :meth:`MetadataService.analyze` never writes to disk, while
:meth:`MetadataService.save` handles JSON output.

The service is stateless apart from injected configuration, favours composition,
and isolates every filesystem/subprocess touch (``_probe``, ``_hash_file``) from
the pure parsing step (``_build_metadata``) so the parsing is unit-testable with
a captured ffprobe fixture and no binary present. It emits diagnostics through
:mod:`logging` and never uses ``print``. It has no n8n dependency.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ffmpeg

from backend.config import Settings, get_settings
from backend.models import GameplayMetadata

logger = logging.getLogger(__name__)

# Read the source file in 1 MiB chunks when hashing to bound memory use.
_HASH_CHUNK_SIZE = 1024 * 1024


class MetadataError(Exception):
    """Base class for all errors raised by :class:`MetadataService`."""


class VideoNotFoundError(MetadataError):
    """Raised when the video path does not exist or is not a regular file."""


class FFprobeNotAvailableError(MetadataError):
    """Raised when the ``ffprobe`` binary cannot be located or executed."""


class MetadataProbeError(MetadataError):
    """Raised when ffprobe runs but fails (corrupted/unsupported input)."""


class MetadataParseError(MetadataError):
    """Raised when ffprobe output lacks the fields we require (e.g. no video stream)."""


class MetadataService:
    """Extracts technical metadata from gameplay videos using ffprobe.

    Parameters
    ----------
    settings:
        Optional configuration override. Defaults to the shared
        :func:`~backend.config.get_settings` instance. Injecting a custom
        :class:`Settings` (e.g. a temp ``metadata_dir``) makes the service easy
        to test in isolation.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def analyze(self, video_path: str | Path) -> GameplayMetadata:
        """Analyse ``video_path`` and return its metadata. Writes nothing.

        Parameters
        ----------
        video_path:
            Path to the gameplay video to inspect.

        Raises
        ------
        VideoNotFoundError
            If the path is missing or not a file.
        FFprobeNotAvailableError
            If the ffprobe binary cannot be found/executed.
        MetadataProbeError
            If ffprobe fails on the input (corrupted/unsupported).
        MetadataParseError
            If ffprobe output is missing required fields.
        """
        path = Path(video_path)
        if not path.is_file():
            raise VideoNotFoundError(f"Video file not found: {path}")

        logger.info("Analyzing gameplay video %s", path)
        probe_data = self._probe(path)
        video_hash = self._hash_file(path)
        metadata = self._build_metadata(
            probe_data=probe_data,
            file_path=path,
            file_size=path.stat().st_size,
            video_hash=video_hash,
        )
        logger.info(
            "Analyzed %s: %s, %.2fs, %.3g fps, %s",
            path.name,
            metadata.resolution,
            metadata.duration_seconds,
            metadata.fps,
            metadata.video_codec,
        )
        return metadata

    def save(self, metadata: GameplayMetadata) -> Path:
        """Persist ``metadata`` as JSON and return the written path.

        This is intentionally decoupled from :meth:`analyze`: callers that only
        need the object in memory can skip it. Output goes to
        ``settings.gameplay_metadata_dir/<source-stem>.json``.
        """
        destination_dir = self._settings.gameplay_metadata_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{metadata.source_file.stem}.json"
        destination.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved metadata for %s to %s", metadata.source_file.name, destination)
        return destination

    # -- Internal helpers (isolated I/O + pure parsing) -----------------------

    def _probe(self, path: Path) -> dict[str, Any]:
        """Run ffprobe on ``path`` and return the parsed JSON as a dict."""
        try:
            return ffmpeg.probe(str(path), cmd=self._settings.ffprobe_path)
        except FileNotFoundError as exc:
            raise FFprobeNotAvailableError(
                f"ffprobe executable not found (configured as "
                f"{self._settings.ffprobe_path!r}). Install FFmpeg or set "
                f"FFPROBE_PATH in your .env."
            ) from exc
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise MetadataProbeError(
                f"ffprobe failed for {path}. The file may be corrupted or "
                f"unsupported.\n{stderr.strip()}"
            ) from exc

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Return the SHA-256 hex digest of the file at ``path`` (streamed)."""
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _build_metadata(
        probe_data: dict[str, Any],
        file_path: Path,
        file_size: int,
        video_hash: str,
    ) -> GameplayMetadata:
        """Build a :class:`GameplayMetadata` from raw ffprobe output.

        Pure function: no I/O. Selects the first video stream and reads
        container-level ``format`` data. Raises :class:`MetadataParseError` when
        required information is absent.
        """
        streams = probe_data.get("streams", [])
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        if video_stream is None:
            raise MetadataParseError(
                f"No video stream found in {file_path}; not a playable video?"
            )

        fmt = probe_data.get("format", {})

        width = video_stream.get("width")
        height = video_stream.get("height")
        if width is None or height is None:
            raise MetadataParseError(
                f"Video stream in {file_path} is missing width/height."
            )

        duration = _to_float(fmt.get("duration") or video_stream.get("duration")) or 0.0
        bitrate = _to_int(fmt.get("bit_rate") or video_stream.get("bit_rate"))

        return GameplayMetadata(
            source_file=file_path,
            file_size_bytes=file_size,
            video_hash=video_hash,
            container_format=fmt.get("format_name", "unknown"),
            duration_seconds=duration,
            video_codec=video_stream.get("codec_name", "unknown"),
            width=int(width),
            height=int(height),
            fps=_parse_frame_rate(video_stream.get("r_frame_rate")),
            bitrate_bps=bitrate,
            creation_date=_parse_creation_time(fmt.get("tags", {})),
            analyzed_at=datetime.now(timezone.utc),
        )


def _parse_frame_rate(rate: str | None) -> float:
    """Parse an ffprobe frame-rate string like ``"30000/1001"`` into fps."""
    if not rate:
        return 0.0
    try:
        numerator, _, denominator = rate.partition("/")
        den = float(denominator) if denominator else 1.0
        if den == 0:
            return 0.0
        return float(numerator) / den
    except (ValueError, TypeError):
        return 0.0


def _parse_creation_time(tags: dict[str, Any]) -> datetime | None:
    """Extract a ``creation_time`` tag as a datetime, if present and parseable."""
    raw = tags.get("creation_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    """Best-effort float conversion; returns ``None`` on failure."""
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _to_int(value: Any) -> int | None:
    """Best-effort int conversion; returns ``None`` on failure."""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None
