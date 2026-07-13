"""Subtitle generation — turn an ExecutionTimeline into captions.srt (Feature 8A).

:class:`SubtitleGenerator` chunks each segment's per-word timings into short,
readable caption cues with **absolute** timing (``audio_offset + word``) and writes
a standard ``.srt``. Pure text generation — **no FFmpeg**; burning subtitles onto
the video is Feature 8B, so wording/timing/style can change without re-rendering.

Readability rules: greedy fill by max chars/line-duration, preferring to break after
sentence-ending punctuation; merge a tiny trailing cue into the previous one; then a
post-pass enforces a minimum duration and a maximum reading speed (chars/second) by
extending cue ends (never past the next cue's start). Segments with no word timings
fall back to a single ``cleaned_text`` cue.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    ExecutionSegment,
    ExecutionTimeline,
    SubtitleCue,
    SubtitleTrack,
)

logger = logging.getLogger(__name__)

# Characters that must not be preceded by a space when joining tokens.
_NO_SPACE_BEFORE = set(".,!?;:%)]}\"")
# Fraction of a limit past which a sentence-ending token triggers a flush.
_SENTENCE_BREAK_RATIO = 0.6

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SubtitleError(Exception):
    """Base class for all errors raised by :class:`SubtitleGenerator`."""


class SubtitleGenerationError(SubtitleError):
    """Raised when a timeline yields no subtitles (e.g. no segments)."""


@dataclass
class _RawCue:
    """Mutable working cue before indexing/serialisation."""

    start: float
    end: float
    text: str
    word_count: int


class SubtitleGenerator:
    """Generates an SRT :class:`SubtitleTrack` from an :class:`ExecutionTimeline`.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying the chunking limits and
        the default save directory.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._max_chars = self._settings.subtitle_max_chars
        self._max_duration = self._settings.subtitle_max_line_duration
        self._min_duration = self._settings.subtitle_min_duration
        self._max_cps = self._settings.subtitle_max_cps

    # -- Public API -----------------------------------------------------------

    def generate(self, execution: ExecutionTimeline) -> SubtitleTrack:
        """Produce a :class:`SubtitleTrack` from ``execution``. Pure; no I/O.

        Raises
        ------
        SubtitleGenerationError
            If the timeline has no segments to caption.
        """
        if not execution.segments:
            raise SubtitleGenerationError(
                f"Execution timeline {execution.title!r} has no segments."
            )

        raw: list[_RawCue] = []
        for segment in execution.segments:
            raw.extend(self._chunk_segment(segment))
        if not raw:
            raise SubtitleGenerationError(
                f"No captions could be generated for {execution.title!r}."
            )

        self._enforce_durations(raw)

        cues = [
            SubtitleCue(
                index=i,
                start_seconds=round(cue.start, 3),
                end_seconds=round(cue.end, 3),
                text=cue.text,
            )
            for i, cue in enumerate(raw, start=1)
        ]
        track = SubtitleTrack(
            project_id=execution.project_id,
            title=execution.title,
            generated_at=datetime.now(timezone.utc),
            cues=cues,
        )
        logger.info(
            "Generated %d subtitle cue(s) for %r (%.1fs)",
            track.cue_count,
            track.title,
            track.total_duration_seconds,
        )
        return track

    def to_srt(self, track: SubtitleTrack) -> str:
        """Serialise ``track`` to SRT text (plain export format)."""
        return self._to_srt(track)

    def to_ass(self, track: SubtitleTrack) -> str:
        """Serialise ``track`` to ASS text (styled working format, from config)."""
        return build_ass_document(track.cues, self._settings)

    def save(self, track: SubtitleTrack, destination: Path | None = None) -> Path:
        """Write both the styled ``.ass`` (working) and ``.srt`` (export) files.

        Returns the ``.ass`` path — the canonical working artifact the burner reads.
        A ``destination`` of any extension only sets the base name/location.
        """
        if destination is None:
            base = self._settings.edited_dir / _slugify(track.project_id or track.title)
        else:
            base = Path(destination).with_suffix("")
        ass_path = base.with_suffix(".ass")
        srt_path = base.with_suffix(".srt")
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_text(self.to_ass(track), encoding="utf-8")
        srt_path.write_text(self.to_srt(track), encoding="utf-8")
        logger.info("Saved subtitles for %r to %s and %s", track.title, ass_path, srt_path)
        return ass_path

    # -- Chunking (pure) ------------------------------------------------------

    def _chunk_segment(self, segment: ExecutionSegment) -> list[_RawCue]:
        """Split one segment's word timings into readable cues."""
        offset = segment.audio_offset_seconds
        segment_end = offset + segment.actual_duration_seconds
        words = segment.word_timings

        if not words:
            text = " ".join(segment.cleaned_text.split())
            return [_RawCue(offset, segment_end, text, len(text.split()))]

        cues: list[_RawCue] = []
        group: list = []
        group_start: float | None = None

        for word in words:
            token = (word.text or "").strip()
            if not token:
                continue
            candidate = self._smart_join([*(w.text for w in group), token])
            cand_start = group_start if group_start is not None else offset + word.start_seconds
            cand_duration = (offset + word.end_seconds) - cand_start

            if group and (
                len(candidate) > self._max_chars or cand_duration > self._max_duration
            ):
                cues.append(self._make_cue(group, group_start, offset, segment_end))
                group, group_start = [], None

            if not group:
                group_start = offset + word.start_seconds
            group.append(word)

            # Prefer breaking after sentence-ending punctuation once near a limit.
            line = self._smart_join([w.text for w in group])
            line_duration = (offset + word.end_seconds) - group_start
            if token[-1] in ".!?" and (
                len(line) >= _SENTENCE_BREAK_RATIO * self._max_chars
                or line_duration >= _SENTENCE_BREAK_RATIO * self._max_duration
            ):
                cues.append(self._make_cue(group, group_start, offset, segment_end))
                group, group_start = [], None

        if group:
            cues.append(self._make_cue(group, group_start, offset, segment_end))

        return self._merge_tiny_trailing(cues)

    def _make_cue(
        self, words: list, start: float, offset: float, segment_end: float
    ) -> _RawCue:
        """Build a raw cue from grouped words, clamping the end to the audio."""
        text = self._smart_join([w.text for w in words])
        end = min(offset + words[-1].end_seconds, segment_end)
        if end < start:
            end = start
        word_count = sum(
            1 for w in words if any(ch.isalnum() for ch in (w.text or ""))
        )
        return _RawCue(start=start, end=end, text=text, word_count=word_count)

    def _merge_tiny_trailing(self, cues: list[_RawCue]) -> list[_RawCue]:
        """Merge a final <=2-word cue into the previous one when limits allow."""
        if len(cues) < 2:
            return cues
        last, prev = cues[-1], cues[-2]
        if last.word_count > 2:
            return cues
        merged_text = self._join_texts(prev.text, last.text)
        merged_duration = last.end - prev.start
        cps = len(merged_text) / merged_duration if merged_duration > 0 else float("inf")
        if (
            len(merged_text) <= self._max_chars
            and merged_duration <= self._max_duration
            and cps <= self._max_cps
        ):
            cues[-2] = _RawCue(
                start=prev.start,
                end=last.end,
                text=merged_text,
                word_count=prev.word_count + last.word_count,
            )
            cues.pop()
        return cues

    def _enforce_durations(self, cues: list[_RawCue]) -> None:
        """Extend too-short / too-fast cues in place, never past the next start."""
        for i, cue in enumerate(cues):
            required = max(self._min_duration, len(cue.text) / self._max_cps)
            if cue.end - cue.start >= required:
                continue
            desired_end = cue.start + required
            if i + 1 < len(cues):
                desired_end = min(desired_end, cues[i + 1].start)
            if desired_end > cue.end:
                cue.end = desired_end

    # -- Formatting (pure) ----------------------------------------------------

    @staticmethod
    def _smart_join(tokens) -> str:
        """Join token texts with spaces, none before trailing punctuation."""
        out = ""
        for raw in tokens:
            token = (raw or "").strip()
            if not token:
                continue
            if out and token[0] not in _NO_SPACE_BEFORE:
                out += " "
            out += token
        return out

    @staticmethod
    def _join_texts(first: str, second: str) -> str:
        """Join two cue texts with the same spacing rule."""
        if second and second[0] not in _NO_SPACE_BEFORE:
            return f"{first} {second}"
        return f"{first}{second}"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as an SRT timestamp ``HH:MM:SS,mmm``."""
        total_ms = int(round(max(seconds, 0.0) * 1000))
        hours, total_ms = divmod(total_ms, 3_600_000)
        minutes, total_ms = divmod(total_ms, 60_000)
        secs, millis = divmod(total_ms, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _to_srt(self, track: SubtitleTrack) -> str:
        """Serialise a track to SRT text."""
        blocks = [
            f"{cue.index}\n"
            f"{self._format_timestamp(cue.start_seconds)} --> "
            f"{self._format_timestamp(cue.end_seconds)}\n"
            f"{cue.text}"
            for cue in track.cues
        ]
        return "\n\n".join(blocks) + "\n"


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "captions"


# ------------------------------------------------------------------------- #
# ASS (working format) + SRT parsing — shared with the subtitle renderer (8B)
# ------------------------------------------------------------------------- #
# Reference canvas the ASS style is authored against; libass scales to the
# real video, so styling is resolution-independent.
_ASS_PLAY_RES = (1920, 1080)

_NAMED_COLOURS = {
    "white": "&H00FFFFFF",
    "black": "&H00000000",
    "yellow": "&H0000FFFF",
    "red": "&H000000FF",
    "green": "&H0000FF00",
    "blue": "&H00FF0000",
    "cyan": "&H00FFFF00",
    "magenta": "&H00FF00FF",
}

_SRT_TS_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


def _ass_colour(value: str) -> str:
    """Convert a colour (name / ``#RRGGBB`` / ``&H…``) to ASS ``&HAABBGGRR``."""
    text = value.strip()
    if text[:2].lower() == "&h":
        return "&H" + text[2:].upper()
    if text.startswith("#") and len(text) == 7:
        r, g, b = text[1:3], text[3:5], text[5:7]
        return f"&H00{b}{g}{r}".upper()
    return _NAMED_COLOURS.get(text.lower(), "&H00FFFFFF")


def _ass_time(seconds: float) -> str:
    """Format seconds as an ASS timestamp ``H:MM:SS.cc`` (centiseconds)."""
    centis = int(round(max(seconds, 0.0) * 100))
    hours, centis = divmod(centis, 360_000)
    minutes, centis = divmod(centis, 6_000)
    secs, centis = divmod(centis, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(text: str) -> str:
    """Escape cue text for an ASS Dialogue line (newlines -> ``\\N``)."""
    return text.replace("\n", "\\N")


def build_ass_document(cues: list[SubtitleCue], settings) -> str:
    """Build a styled ASS subtitle document from cues and the config style."""
    primary = _ass_colour(settings.subtitle_primary_colour)
    outline_colour = _ass_colour(settings.subtitle_outline_colour)
    back = _ass_colour("black")
    style = (
        f"Style: Default,{settings.subtitle_font},{settings.subtitle_font_size},"
        f"{primary},&H000000FF,{outline_colour},{back},"
        f"0,0,0,0,100,100,0,0,1,"
        f"{settings.subtitle_outline},{settings.subtitle_shadow},2,20,20,"
        f"{settings.subtitle_margin_v},1"
    )
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {_ASS_PLAY_RES[0]}",
        f"PlayResY: {_ASS_PLAY_RES[1]}",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        style,
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for cue in cues:
        lines.append(
            f"Dialogue: 0,{_ass_time(cue.start_seconds)},{_ass_time(cue.end_seconds)},"
            f"Default,,0,0,0,,{_ass_escape(cue.text)}"
        )
    return "\n".join(lines) + "\n"


def _parse_srt_timestamp(value: str) -> float:
    """Parse an SRT timestamp ``HH:MM:SS,mmm`` into seconds."""
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_srt(path: Path) -> list[SubtitleCue]:
    """Parse an SRT file into :class:`SubtitleCue`\\ s (for the burner's fallback)."""
    content = Path(path).read_text(encoding="utf-8").strip()
    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        if "-->" in lines[0]:
            time_line, text_lines = lines[0], lines[1:]
        else:
            time_line, text_lines = lines[1], lines[2:]
        match = _SRT_TS_RE.search(time_line)
        if not match:
            continue
        text = " ".join(text_lines).strip() or " "
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_seconds=_parse_srt_timestamp(match.group(1)),
                end_seconds=_parse_srt_timestamp(match.group(2)),
                text=text,
            )
        )
    return cues
