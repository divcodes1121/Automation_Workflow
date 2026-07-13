"""Gaming-highlights editor (Phase 3.1 + 3.2): event-synced cuts, original audio.

The pivot from "AI documentary" to "gaming highlight editor". Instead of narrating
a whole match over a linear slice of footage, this cuts the ORIGINAL recording
into short windows placed around the *real* event timestamps the analyzer already
produces (`gameplay_analysis.json`), and keeps the ORIGINAL Clash Royale audio —
no TTS. The analyzer's events drive the editing.

This first slice is deliberately minimal (the "minimal proof" scope): pick ~5-6
marquee moments (Rockets, the Double-Elixir switch, Overtime), cut a tight window
around each real timestamp, and concatenate with the game audio intact. Captions
(3.4) and effects (3.3) layer on top later. It is fully ADDITIVE — the frozen
narrated pipeline is untouched — and stays decoupled from the analyzer package
(reads the analysis JSON as a dict).
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.models import HighlightClip, HighlightPlan

logger = logging.getLogger(__name__)

# Window around each event. Pre-roll is generous because play-detection confirms
# a card ~1-2s AFTER the real placement, so the actual action sits *before* the
# recorded timestamp; post-roll covers the payoff (e.g. a Rocket landing).
_PRE_ROLL_S = 2.5
_POST_ROLL_S = 4.0
_MERGE_GAP_S = 0.75  # windows closer than this are fused into one continuous beat
_MAX_CLIPS = 6       # "minimal proof" cap
_CRF = 20


class HighlightError(ValueError):
    """Raised when a highlight edit cannot be built or rendered."""


class HighlightEditor:
    """Selects marquee events and renders a gameplay-only highlight reel."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -- Planning (pure) ----------------------------------------------------- #

    def build(self, analysis: dict[str, Any], video: Path) -> HighlightPlan:
        """Select marquee moments and their event-synced windows (no I/O)."""
        if not isinstance(analysis, dict) or "events" not in analysis:
            raise HighlightError(
                "not a gameplay analysis (missing 'events'); pass a "
                "gameplay_analysis.json produced by the analyzer"
            )
        duration = float(analysis.get("duration_seconds") or 0.0)
        selected = _select_marquee(analysis)
        windows = _windows(selected, duration)

        clips = [
            HighlightClip(
                index=i,
                event_timestamp_seconds=round(w["ts"], 3),
                card=w["card"],
                phase=w["phase"],
                label=w["label"],
                source_start_seconds=round(w["start"], 3),
                source_end_seconds=round(w["end"], 3),
                duration_seconds=round(w["end"] - w["start"], 3),
            )
            for i, w in enumerate(windows)
        ]
        return HighlightPlan(
            generated_at=datetime.now(timezone.utc),
            source_analysis=str(analysis.get("video", video.name)),
            video=str(video),
            clip_count=len(clips),
            total_duration_seconds=round(sum(c.duration_seconds for c in clips), 3),
            clips=clips,
        )

    def save(self, plan: HighlightPlan, destination: Path | None = None) -> Path:
        """Persist the highlight plan as JSON."""
        dest = destination or self._settings.edited_dir / f"{Path(plan.video).stem}.highlight_plan.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved highlight plan (%d clips) to %s", plan.clip_count, dest)
        return dest

    # -- Render (FFmpeg) ----------------------------------------------------- #

    def render(self, plan: HighlightPlan, video: Path, output: Path | None = None) -> Path:
        """Cut the planned windows from ``video`` and concat, keeping game audio."""
        if not plan.clips:
            raise HighlightError("highlight plan has no clips to render")
        if not video.is_file():
            raise HighlightError(f"gameplay video not found: {video}")

        dest = output or self._settings.edited_dir / f"{video.stem}.highlight.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        argv = self._ffmpeg_argv(plan, video, dest)

        logger.info("Rendering highlight reel (%d clips) -> %s", plan.clip_count, dest)
        started = time.perf_counter()
        completed = subprocess.run(argv, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip()[-2000:]
            raise HighlightError(f"FFmpeg failed (exit {completed.returncode}).\n{tail}")
        logger.info(
            "Rendered highlight reel: %.1fs in %.1fs", plan.total_duration_seconds, elapsed
        )
        return dest

    def _ffmpeg_argv(self, plan: HighlightPlan, video: Path, dest: Path) -> list[str]:
        """One filter_complex trim+concat over a single source; keeps V+A."""
        parts: list[str] = []
        labels: list[str] = []
        for i, clip in enumerate(plan.clips):
            s, e = clip.source_start_seconds, clip.source_end_seconds
            parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS[v{i}]")
            parts.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[v{i}][a{i}]")
        n = len(plan.clips)
        parts.append("".join(labels) + f"concat=n={n}:v=1:a=1[outv][outa]")
        filtergraph = ";".join(parts)
        return [
            self._settings.ffmpeg_path, "-y", "-i", str(video),
            "-filter_complex", filtergraph,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-crf", str(_CRF), "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "192k",
            str(dest),
        ]


# --------------------------------------------------------------------------- #
# Selection (pure)
# --------------------------------------------------------------------------- #
def _state_for(analysis: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    states = analysis.get("match_states") or []
    ref = event.get("match_state_ref")
    if isinstance(ref, int) and 0 <= ref < len(states):
        return states[ref]
    return {}


def _select_marquee(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Pick the marquee moments (deterministic, no ML): Rockets + the switch to
    Double Elixir + the start of Overtime. Deduped, sorted by time, capped."""
    events = analysis.get("events", [])
    picks: dict[int, dict[str, Any]] = {}  # event index -> record (dedup)
    seen_double = seen_overtime = False

    for idx, event in enumerate(events):
        state = _state_for(analysis, event)
        phase = state.get("phase")
        mult = state.get("elixir_multiplier")
        card = event.get("card")
        ts = float(event.get("timestamp_seconds") or 0.0)

        label = None
        if phase == "overtime" and not seen_overtime:
            label, seen_overtime = "OVERTIME", True
        elif card == "rocket":
            label = "ROCKET"
        elif phase == "regulation" and mult == 2 and not seen_double:
            label, seen_double = "DOUBLE ELIXIR", True

        if label is not None:
            picks[idx] = {"ts": ts, "card": card, "phase": phase, "label": label}

    ordered = sorted(picks.values(), key=lambda r: r["ts"])
    return ordered[:_MAX_CLIPS]


def _windows(selected: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """Turn events into clamped, merged [start, end] windows around each timestamp."""
    windows: list[dict[str, Any]] = []
    for rec in selected:
        start = max(0.0, rec["ts"] - _PRE_ROLL_S)
        end = rec["ts"] + _POST_ROLL_S
        if duration > 0:
            end = min(end, duration)
        if end <= start:
            continue
        if windows and start <= windows[-1]["end"] + _MERGE_GAP_S:
            # Overlapping/adjacent -> fuse into one continuous beat (keep 1st label).
            windows[-1]["end"] = max(windows[-1]["end"], end)
            continue
        windows.append({**rec, "start": start, "end": end})
    return windows
