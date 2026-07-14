"""Gaming-highlights editor (Phase 3.1 + 3.2): story-arc cuts, original audio.

The pivot from "AI documentary" to "gaming highlight editor". The analyzer's real
event timestamps (`gameplay_analysis.json`) drive the editing; the ORIGINAL Clash
Royale audio is kept — no TTS anywhere in this path.

The reel is a STORY, not a chronological summary, built around the deck's
signature win condition (default: rocket; parameterized so Hog/Miner/Graveyard
archetypes generalize later):

    HOOK   — open immediately on the first signature play (tight window)
    BEAT   — each middle play: ~1-2s before placement, launch, impact, aftermath
    FLASH  — a sub-2s phase splash (Double Elixir / Overtime) inserted only when
             the phase change falls BETWEEN plays (never its own long clip)
    HERO   — the final signature play gets the longest window (the payoff)
    VICTORY— the end screen closes the reel (recordings end on the banner)

Everything stays chronological (no reordering — honest editing), hard cuts only
(fast transitions/effects are the next slice; clip `role`s exist so that slice
can style each beat differently). Fully additive: the narrated pipeline is
untouched, and this module reads the analysis JSON as a dict (never imports the
analyzer package).
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.models import HighlightClip, HighlightPlan, HighlightRole

logger = logging.getLogger(__name__)

# Play-detection confirms a card ~1-2s AFTER the real placement, so the actual
# action sits BEFORE the recorded timestamp; windows lead with that in mind.
_HOOK_PRE_S, _HOOK_POST_S = 2.0, 3.5     # open right on the action
_BEAT_PRE_S, _BEAT_POST_S = 2.5, 3.0     # placement -> launch -> impact -> 1s after
_HERO_PRE_S, _HERO_POST_S = 2.5, 5.0     # the payoff breathes a little longer
_FLASH_PRE_S, _FLASH_POST_S = 0.5, 1.2   # phase splash: blink-and-it's-gone
_VICTORY_TAIL_S = 3.0                    # recordings end on the Victory banner
_MERGE_GAP_S = 0.25                      # near-adjacent windows fuse into one beat
_MAX_REEL_S = 45.0                       # Shorts budget
_CRF = 20

# Viewer-facing card names for labels ("rocket" -> "ROCKET").
def _card_label(slug: str) -> str:
    return slug.replace("-", " ").upper()


class HighlightError(ValueError):
    """Raised when a highlight edit cannot be built or rendered."""


class HighlightEditor:
    """Builds and renders a story-arc highlight reel around a signature card."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -- Planning (pure) ----------------------------------------------------- #

    def build(
        self, analysis: dict[str, Any], video: Path, *, signature_card: str = "rocket"
    ) -> HighlightPlan:
        """Select the story beats and their event-synced windows (no I/O)."""
        if not isinstance(analysis, dict) or "events" not in analysis:
            raise HighlightError(
                "not a gameplay analysis (missing 'events'); pass a "
                "gameplay_analysis.json produced by the analyzer"
            )
        duration = float(analysis.get("duration_seconds") or 0.0)
        plays = [
            float(e.get("timestamp_seconds") or 0.0)
            for e in analysis.get("events", [])
            if e.get("card") == signature_card
        ]
        if not plays:
            raise HighlightError(
                f"no '{signature_card}' plays found in this match; pick the deck's "
                "signature card with --card"
            )
        plays.sort()

        windows = _story_windows(plays, _phase_changes(analysis), duration, signature_card)
        clips = [
            HighlightClip(
                index=i,
                role=w["role"],
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

    def render(
        self,
        plan: HighlightPlan,
        video: Path,
        output: Path | None = None,
        *,
        effects: bool = True,
    ) -> Path:
        """Cut the planned windows from ``video`` and concat, keeping game audio.

        With ``effects`` (default), each clip additionally gets its role's edit
        recipe (zoom/shake/flash/callout) from the Effects Engine.
        """
        if not plan.clips:
            raise HighlightError("highlight plan has no clips to render")
        if not video.is_file():
            raise HighlightError(f"gameplay video not found: {video}")

        dest = output or self._settings.edited_dir / f"{video.stem}.highlight.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Absolute video/dest because ffmpeg runs with cwd = project root: the
        # callout's fontfile must be a RELATIVE, colon-free path (this ffmpeg's
        # filter parser splits on the drive ':' even inside quotes - the same
        # Windows gotcha the 8B subtitle burn hit).
        argv = self._ffmpeg_argv(plan, video.resolve(), dest.resolve(), effects=effects)

        logger.info("Rendering highlight reel (%d clips) -> %s", plan.clip_count, dest)
        started = time.perf_counter()
        completed = subprocess.run(
            argv, capture_output=True, text=True, cwd=str(self._settings.project_root)
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip()[-2000:]
            raise HighlightError(f"FFmpeg failed (exit {completed.returncode}).\n{tail}")
        logger.info(
            "Rendered highlight reel: %.1fs in %.1fs", plan.total_duration_seconds, elapsed
        )
        return dest

    def _ffmpeg_argv(
        self, plan: HighlightPlan, video: Path, dest: Path, *, effects: bool = True
    ) -> list[str]:
        """One filter_complex trim+concat over a single source; keeps V+A."""
        chains: dict[int, str] = {}
        if effects:
            # Imported lazily; recipes are data in backend/recipes/.
            from backend.services.effects_engine import EffectsEngine, EffectsError

            try:
                engine = EffectsEngine()
                width, height = self._probe_dims(video)
                # Seed the variant cycling from the video so it's reproducible
                # yet differs across matches.
                seed = engine.stable_seed(Path(plan.video).stem)
                chains = engine.plan_chains(plan.clips, width, height, seed=seed)
            except EffectsError as exc:
                raise HighlightError(f"effects engine: {exc}") from exc

        parts: list[str] = []
        labels: list[str] = []
        for i, clip in enumerate(plan.clips):
            s, e = clip.source_start_seconds, clip.source_end_seconds
            fx = chains.get(i, "")
            parts.append(f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS{fx}[v{i}]")
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

    def _probe_dims(self, video: Path) -> tuple[int, int]:
        """Source frame dimensions via ffprobe (needed by the effects crop)."""
        import json as _json

        completed = subprocess.run(
            [
                self._settings.ffprobe_path, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v", str(video),
            ],
            capture_output=True, text=True,
        )
        try:
            stream = _json.loads(completed.stdout)["streams"][0]
            return int(stream["width"]), int(stream["height"])
        except (ValueError, KeyError, IndexError) as exc:
            raise HighlightError(f"could not probe video dimensions: {video}") from exc


# --------------------------------------------------------------------------- #
# Story construction (pure)
# --------------------------------------------------------------------------- #
# A phase/multiplier change only counts when SUSTAINED this many consecutive
# timeline states (~seconds). Real phases last minutes; the match-intro screen
# can misread as "overtime" for a couple of unreadable frames (seen in game_01
# at t=2-3s, timer confidence 0.0) and must not become a flash.
_SUSTAIN_STATES = 5


def _phase_changes(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """First sustained Double-Elixir switch + Overtime start, from the 2G timeline."""
    states = analysis.get("match_states") or []

    def first_sustained(pred) -> float | None:
        run_start: float | None = None
        run_len = 0
        for state in states:
            if pred(state):
                if run_len == 0:
                    run_start = float(state.get("timestamp_seconds") or 0.0)
                run_len += 1
                if run_len >= _SUSTAIN_STATES:
                    return run_start
            else:
                run_len = 0
        return None

    changes: list[dict[str, Any]] = []
    double_ts = first_sustained(
        lambda s: s.get("phase") == "regulation" and s.get("elixir_multiplier") == 2
    )
    if double_ts is not None:
        changes.append({"ts": double_ts, "label": "DOUBLE ELIXIR", "phase": "regulation"})
    overtime_ts = first_sustained(lambda s: s.get("phase") == "overtime")
    if overtime_ts is not None:
        changes.append({"ts": overtime_ts, "label": "OVERTIME", "phase": "overtime"})
    return changes


def _story_windows(
    plays: list[float],
    phase_changes: list[dict[str, Any]],
    duration: float,
    card: str,
) -> list[dict[str, Any]]:
    """Arrange hook/beats/hero/victory + phase flashes into merged windows."""
    label = _card_label(card)
    raw: list[dict[str, Any]] = []

    # Hook = first play, tight. Hero = last play, longer. Middles = beats.
    for i, ts in enumerate(plays):
        if i == 0:
            role, pre, post, lab = HighlightRole.HOOK, _HOOK_PRE_S, _HOOK_POST_S, label
        elif i == len(plays) - 1:
            role, pre, post, lab = HighlightRole.HERO, _HERO_PRE_S, _HERO_POST_S, f"FINAL {label}"
        else:
            role, pre, post, lab = HighlightRole.BEAT, _BEAT_PRE_S, _BEAT_POST_S, label
        raw.append(
            {"ts": ts, "role": role, "card": card, "phase": None, "label": lab,
             "start": max(0.0, ts - pre), "end": min(duration, ts + post) if duration else ts + post}
        )

    # Merge overlapping/near-adjacent play windows (keeps the earlier clip's
    # label/role; a fused hook+beat still opens the reel correctly).
    raw.sort(key=lambda w: w["start"])
    merged: list[dict[str, Any]] = []
    for w in raw:
        if merged and w["start"] <= merged[-1]["end"] + _MERGE_GAP_S:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            # A hero folded into the previous window keeps the payoff role.
            if w["role"] == HighlightRole.HERO:
                merged[-1]["role"] = HighlightRole.HERO
                merged[-1]["label"] = w["label"]
            continue
        merged.append(w)

    # Phase flashes: only when the change lands BETWEEN reel windows (never
    # before the hook or inside a play window) — a blink insert, not a clip.
    first_start, last_end = merged[0]["start"], merged[-1]["end"]
    for change in phase_changes:
        ts = change["ts"]
        if not (first_start < ts < last_end):
            continue
        if any(w["start"] - _FLASH_PRE_S <= ts <= w["end"] + _FLASH_PRE_S for w in merged):
            continue
        merged.append(
            {"ts": ts, "role": HighlightRole.FLASH, "card": None,
             "phase": change["phase"], "label": change["label"],
             "start": max(0.0, ts - _FLASH_PRE_S),
             "end": min(duration, ts + _FLASH_POST_S) if duration else ts + _FLASH_POST_S}
        )

    # Victory screen closes the reel (recordings end on the banner). Skip it if
    # the hero window already reaches the end.
    if duration > _VICTORY_TAIL_S and last_end < duration - _MERGE_GAP_S:
        merged.append(
            {"ts": duration, "role": HighlightRole.VICTORY, "card": None,
             "phase": None, "label": "VICTORY",
             "start": duration - _VICTORY_TAIL_S, "end": duration}
        )

    merged.sort(key=lambda w: w["start"])

    # Shorts budget: if over, trim middle BEATs first (hook/hero/victory stay).
    def total() -> float:
        return sum(w["end"] - w["start"] for w in merged)

    while total() > _MAX_REEL_S:
        beats = [w for w in merged if w["role"] == HighlightRole.BEAT]
        if not beats:
            break
        merged.remove(beats[len(beats) // 2])  # drop from the middle outward
    return merged
