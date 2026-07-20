"""Battle splitter (Phase 3.3): find each match inside a long recording.

A single screen recording usually holds several matches back to back, separated
by lobby / battle-log / matchmaking screens. This module locates each match and
cuts it out, so a whole session can be dropped in a folder and processed by one
command.

**The signal is the match clock.** Clash Royale only renders the countdown
during a battle, so the 2G :class:`~analyzer.detectors.timer_detector.
TimerDetector` doubles as an "am I in a match?" oracle -- no new detector, and
no expensive ORB card matching. Only the timer ROI is decoded (cropped straight
out by FFmpeg), which is far cheaper than decoding whole 1320x2868 frames.

Turning that raw signal into boundaries takes three rules, each measured against
a real 14.8-minute, 3-match recording:

1. **Gap-merge.** In each match's final minute the clock reads only every *other*
   second, which would otherwise shatter one battle into ~15 fragments. Runs are
   merged across gaps up to ``split_max_gap_s``. There is real headroom: the
   largest gap *inside* a battle was 8s, the smallest gap *between* matches 35s.
2. **Minimum duration.** Accidental captures (a replay opened by mistake, a
   match abandoned at the start) show up as blocks of a few seconds. Anything
   under ``split_min_battle_s`` is discarded -- on the reference recording the
   real matches were 177-268s and the accidents 2s and 15s, so nothing sits near
   the threshold.
3. **Scene snapping.** The clock marks the *battle*, but a watchable clip starts
   on the loading screen and ends after the crowns banner. Both lie outside the
   readable window, so each boundary extends outward to the nearest hard cut
   found by a short, windowed FFmpeg scene-detect pass (a full-recording pass
   costs minutes; searching only around the ~6 boundaries costs seconds). With
   no cut found, a fixed pre/post-roll is used instead.

Cutting is stream-copy (no re-encode), so splitting a session is I/O bound
rather than CPU bound. Stream copy can only cut on keyframes, so a start snaps
to the keyframe *at or after* the boundary -- landing a frame inside the loading
screen rather than a frame of the previous match's lobby.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import (
    BoundarySource,
    CalibrationProfile,
    DetectedBattle,
    DiscardedBlock,
    MatchPhase,
    SplitPlan,
)

logger = logging.getLogger(__name__)

_TIMER_ROI = "timer"
# `metadata=print` writes a "pts_time:<float>" line then a "lavfi.scd.score=<f>"
# line for every frame scdet scores.
_PTS_RE = re.compile(r"pts_time:([0-9.]+)")
_SCORE_RE = re.compile(r"lavfi\.scd\.score=([0-9.]+)")


class BattleSplitError(RuntimeError):
    """Raised when a recording cannot be segmented into matches."""


class BattleSplitter:
    """Locates and extracts the individual matches in a long recording."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    # -- Planning ------------------------------------------------------------ #

    def plan(
        self,
        video: str | Path,
        *,
        profile: CalibrationProfile,
        sample_fps: float | None = None,
    ) -> SplitPlan:
        """Locate every match in ``video`` (no files written)."""
        video = Path(video)
        if not video.is_file():
            raise BattleSplitError(f"recording not found: {video}")
        s = self._settings
        fps = sample_fps or s.split_sample_fps

        width, height, duration = self._probe(video)
        roi = profile.rois.get(_TIMER_ROI)
        if roi is None:
            raise BattleSplitError(
                f"profile {profile.name!r} has no 'timer' ROI; the splitter needs it "
                "to tell battles from menus"
            )
        if profile.is_placeholder:
            logger.warning(
                "Profile %r is a placeholder (uncalibrated); split boundaries may be wrong",
                profile.name,
            )

        readings = self._sample_timer(video, roi, width, height, fps)
        blocks, discarded = self._blocks(readings, fps)

        warnings: list[str] = []
        if not blocks:
            warnings.append(
                "no matches found -- check the profile's timer ROI matches this "
                "recording's resolution/device"
            )

        battles = self._to_battles(video, blocks, readings, fps, duration, warnings)
        return SplitPlan(
            video=str(video),
            video_sha256=_sha256(video),
            duration_seconds=round(duration, 3),
            width=width,
            height=height,
            profile_name=profile.name,
            sample_fps=fps,
            battles=battles,
            discarded=discarded,
            warnings=warnings,
            generated_at=datetime.now(timezone.utc),
        )

    def save(self, plan: SplitPlan, destination: Path | None = None) -> Path:
        """Persist the split plan as JSON."""
        dest = destination or (
            self._settings.analysis_output_dir / f"{Path(plan.video).stem}.split_plan.json"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved split plan (%d matches) to %s", plan.battle_count, dest)
        return dest

    # -- Timer sampling ------------------------------------------------------ #

    def _sample_timer(
        self, video: Path, roi, width: int, height: int, fps: float
    ) -> list[tuple[float, object]]:
        """Sample the timer ROI at ``fps`` -> ``[(timestamp, TimerReading), ...]``.

        FFmpeg crops the ROI during decode, so only a ~200x90 image per sample
        reaches Python instead of a full frame.
        """
        import cv2

        from analyzer.detectors.timer_detector import TimerDetector, TimerReading

        unreadable = TimerReading(None, None, MatchPhase.UNKNOWN, 0.0)

        x, y, w, h = roi.to_pixels(width, height)
        if w <= 0 or h <= 0:
            raise BattleSplitError(f"timer ROI is empty at {width}x{height}")

        detector = TimerDetector(self._settings)
        readings: list[tuple[float, object]] = []
        with tempfile.TemporaryDirectory(prefix="cr_timer_") as tmp:
            tmpdir = Path(tmp)
            argv = [
                self._settings.ffmpeg_path, "-v", "error", "-i", str(video),
                "-vf", f"fps={fps:g},crop={w}:{h}:{x}:{y}",
                "-start_number", "0", str(tmpdir / "t%06d.png"),
            ]
            logger.info("Sampling timer ROI (%dx%d @ %g fps) from %s", w, h, fps, video.name)
            completed = subprocess.run(argv, capture_output=True, text=True)
            if completed.returncode != 0:
                tail = (completed.stderr or "").strip()[-2000:]
                raise BattleSplitError(f"FFmpeg timer sampling failed.\n{tail}")

            for path in sorted(tmpdir.glob("t*.png")):
                index = int(path.stem[1:])
                region = cv2.imread(str(path))
                reading = detector.read_region(region) if region is not None else unreadable
                readings.append((index / fps, reading))
        logger.info(
            "Timer sampled: %d frames, %d readable",
            len(readings), sum(1 for _, r in readings if r.text),
        )
        return readings

    # -- Block detection (pure) ---------------------------------------------- #

    def _blocks(
        self, readings: list[tuple[float, object]], fps: float
    ) -> tuple[list[tuple[float, float]], list[DiscardedBlock]]:
        """Group readable samples into candidate matches; reject the rest."""
        s = self._settings
        step = 1.0 / fps

        # Contiguous runs of *plausible* samples. A Clash Royale clock never
        # exceeds 3:00, so anything above that is a misread -- the crowns banner
        # and its confetti produce spurious 3-glyph matches (a "6:00" was read
        # over game_01's win banner) that would otherwise extend the match.
        runs: list[dict] = []
        current: dict | None = None
        for ts, reading in readings:
            valid = bool(reading.text) and reading.seconds is not None and (
                reading.seconds <= s.split_max_clock_seconds
            )
            if valid:
                if current is None:
                    current = {
                        "start": ts, "first_ts": ts, "first_sec": reading.seconds,
                        "first_phase": reading.phase,
                    }
                current.update(
                    end=ts + step, last_ts=ts,
                    last_sec=reading.seconds, last_phase=reading.phase,
                )
            elif current is not None:
                runs.append(current)
                current = None
        if current is not None:
            runs.append(current)

        # Fuse runs that are the SAME match seen through a gap. Proximity alone is
        # not enough: an accidentally-opened replay sat 9s before game_03's real
        # start and would have been absorbed. A real clock also has to have ticked
        # down by roughly the elapsed wall time across the gap, which the replay
        # (2:56 -> 2:57, i.e. time going backwards) fails.
        merged: list[dict] = []
        for run in runs:
            if merged and self._continues(merged[-1], run):
                merged[-1].update(
                    end=run["end"], last_ts=run["last_ts"],
                    last_sec=run["last_sec"], last_phase=run["last_phase"],
                )
            else:
                merged.append(run)

        # Keep only blocks long enough to be a real match.
        blocks, discarded = [], []
        for run in merged:
            begin, end = run["start"], run["end"]
            if end - begin >= s.split_min_battle_s:
                blocks.append((begin, end))
            else:
                discarded.append(
                    DiscardedBlock(
                        start_seconds=round(begin, 3),
                        end_seconds=round(end, 3),
                        duration_seconds=round(end - begin, 3),
                        reason=(
                            f"only {end - begin:.0f}s of readable clock "
                            f"(minimum {s.split_min_battle_s:.0f}s) -- "
                            "likely a replay or an abandoned match"
                        ),
                    )
                )
        return blocks, discarded

    def _continues(self, previous: dict, nxt: dict) -> bool:
        """Is ``nxt`` the same match as ``previous``, seen after a gap?"""
        s = self._settings
        if nxt["start"] - previous["end"] > s.split_max_gap_s:
            return False
        # Overtime restarts the clock, so an upward jump is legitimate exactly
        # when the phase changed; otherwise time must have advanced.
        if previous["last_phase"] != nxt["first_phase"]:
            return True
        elapsed = nxt["first_ts"] - previous["last_ts"]
        drop = previous["last_sec"] - nxt["first_sec"]
        return abs(drop - elapsed) <= s.split_clock_drift_s

    def _to_battles(
        self,
        video: Path,
        blocks: list[tuple[float, float]],
        readings: list[tuple[float, object]],
        fps: float,
        duration: float,
        warnings: list[str],
    ) -> list[DetectedBattle]:
        """Extend each block out to its loading screen / crowns banner."""
        battles: list[DetectedBattle] = []
        for i, (clock_start, clock_end) in enumerate(blocks):
            # Same validity rule as _blocks, so the reported clocks cannot show a
            # misread (an impossible "6:00" was appearing over the win banner).
            inside = [
                r for ts, r in readings
                if clock_start <= ts <= clock_end and r.text
                and r.seconds is not None and r.seconds <= self._settings.split_max_clock_seconds
            ]
            # Never run into the neighbouring match when searching for a cut.
            floor = battles[-1].end_seconds if battles else 0.0
            ceiling = blocks[i + 1][0] if i + 1 < len(blocks) else duration

            start, start_src = self._snap_start(video, clock_start, floor)
            end, end_src = self._snap_end(video, clock_end, ceiling, duration)

            battles.append(
                DetectedBattle(
                    index=i,
                    clock_start_seconds=round(clock_start, 3),
                    clock_end_seconds=round(clock_end, 3),
                    first_clock=inside[0].text if inside else None,
                    last_clock=inside[-1].text if inside else None,
                    start_seconds=round(start, 3),
                    end_seconds=round(end, 3),
                    duration_seconds=round(end - start, 3),
                    start_source=start_src,
                    end_source=end_src,
                    readable_samples=len(inside),
                    total_samples=int(round((clock_end - clock_start) * fps)),
                    reached_overtime=any(
                        r.phase == MatchPhase.OVERTIME for r in inside
                    ),
                    confidence=round(
                        min(1.0, len(inside) / max(1.0, (clock_end - clock_start) * fps)), 4
                    ),
                )
            )
        return battles

    # -- Boundary snapping --------------------------------------------------- #

    def _snap_start(
        self, video: Path, clock_start: float, floor: float
    ) -> tuple[float, BoundarySource]:
        """Walk back from the clock to the first frame of the loading screen.

        The pre-battle chain is ``lobby -> loading splash -> arena fade -> VS ->
        battle``. Every step is a cut, so "earliest cut" would drift back into
        whatever the lobby was doing. The *loading splash* is the strongest cut of
        the group by a wide margin -- it replaces the entire screen with full-bleed
        artwork -- so the boundary is the **highest-scoring** cut in the window.
        Measured on the reference recording, the splash scored 43.0/44.2/43.4
        against best rivals of 21.3/20.9/32.7.
        """
        s = self._settings
        lower = max(floor, clock_start - s.split_scene_search_s)
        if clock_start - lower < 0.5:
            return max(floor, clock_start), BoundarySource.CLAMPED

        cuts = [c for c in self._scene_cuts(video, lower, clock_start - 0.5)
                if c[1] >= s.split_scene_min_score]
        if cuts:
            return max(cuts, key=lambda c: c[1])[0], BoundarySource.SCENE_CUT
        return max(floor, clock_start - s.split_pre_roll_s), BoundarySource.FIXED_ROLL

    def _snap_end(
        self, video: Path, clock_end: float, ceiling: float, duration: float
    ) -> tuple[float, BoundarySource]:
        """Extend past the clock to include the crowns banner, then stop.

        After the clock stops: "Match Over!" -> confetti -> the crowned WINNER
        banner -> the app returns to the lobby through the loading splash. That
        splash is the boundary, and it is the **first** strong cut in the window --
        not the strongest, since the lobby that follows generates bigger ones
        (game_03: the wanted cut scored 23.2, a later lobby cut 31.0).

        The search starts slightly BEFORE the clock: an overtime clock can freeze
        on screen and keep reading through the result screen, so ``clock_end`` may
        already sit past the real boundary (game_02's froze at 0:09 and read ~0.3s
        beyond it). The lookback is kept short because gameplay itself produces no
        cut this strong -- measured peak during play was 2.3, versus 23+ for a
        screen change -- so it cannot land mid-battle.
        """
        s = self._settings
        upper = min(ceiling, duration, clock_end + s.split_scene_search_s)
        lower = max(clock_end - s.split_end_lookback_s, 0.0)
        if upper - lower < 0.5:
            return min(ceiling, duration), BoundarySource.CLAMPED

        cuts = [c for c in self._scene_cuts(video, lower, upper)
                if c[1] >= s.split_scene_min_score]
        if cuts:
            return min(cuts, key=lambda c: c[0])[0], BoundarySource.SCENE_CUT
        fallback = min(ceiling, duration, clock_end + s.split_post_roll_s)
        return fallback, BoundarySource.FIXED_ROLL

    def _scene_cuts(self, video: Path, start: float, end: float) -> list[tuple[float, float]]:
        """``[(timestamp, score), ...]`` for ``[start, end]`` (windowed, so fast).

        Uses ``scdet``, which scores *every* frame 0-100, rather than
        ``select=gt(scene,..)``, so the caller can rank cuts by strength instead of
        only seeing which passed a threshold.
        """
        if end <= start:
            return []
        argv = [
            self._settings.ffmpeg_path, "-v", "error",
            "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
            "-vf", "scdet=threshold=0,metadata=print:file=-",
            "-an", "-f", "null", "-",
        ]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            logger.warning(
                "Scene detection failed in %.1f-%.1fs; falling back to fixed roll", start, end
            )
            return []
        # metadata=print emits "pts_time:<t>" then "lavfi.scd.score=<s>" per frame.
        # `-ss` before `-i` rebases timestamps to 0, so add the window offset back.
        cuts: list[tuple[float, float]] = []
        pending: float | None = None
        for line in (completed.stdout or "").splitlines():
            match = _PTS_RE.search(line)
            if match:
                pending = float(match.group(1))
                continue
            match = _SCORE_RE.search(line)
            if match and pending is not None:
                cuts.append((start + pending, float(match.group(1))))
                pending = None
        return cuts

    # -- Cutting + merging --------------------------------------------------- #

    def split(self, plan: SplitPlan, output_dir: Path | None = None) -> list[Path]:
        """Cut each detected match out of the recording (stream copy)."""
        if not plan.battles:
            raise BattleSplitError("split plan contains no matches to cut")
        video = Path(plan.video)
        if not video.is_file():
            raise BattleSplitError(f"recording not found: {video}")

        dest_dir = output_dir or self._settings.split_output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = video.stem

        written: list[Path] = []
        for battle in plan.battles:
            dest = dest_dir / f"{stem}_game_{battle.index + 1:02d}.mp4"
            start = self._snap_to_keyframe(video, battle.start_seconds)
            argv = [
                self._settings.ffmpeg_path, "-y", "-v", "error",
                # Both -ss and -to are INPUT options here: -ss seeks fast and -to
                # stays an absolute source timestamp. `start` is the first keyframe
                # at or after the boundary, because stream copy cannot cut between
                # keyframes and rounding DOWN would prepend a second of the lobby.
                "-ss", f"{start:.3f}",
                "-to", f"{battle.end_seconds:.3f}",
                "-i", str(video),
                # No -avoid_negative_ts: "make_zero" KEEPS the pre-seek GOP and
                # rebases it to zero, so the clip opens on a second of the lobby
                # instead of the loading screen (measured: +1.3s of Battle Log on
                # every clip). Letting FFmpeg drop that pre-roll is what makes the
                # first frame the loading splash.
                "-map", "0", "-map_metadata", "0", "-c", "copy",
                str(dest),
            ]
            completed = subprocess.run(argv, capture_output=True, text=True)
            if completed.returncode != 0:
                tail = (completed.stderr or "").strip()[-2000:]
                raise BattleSplitError(f"FFmpeg failed cutting match {battle.index + 1}.\n{tail}")
            battle.output_path = str(dest)
            written.append(dest)
            logger.info(
                "Cut match %d (%.1fs) -> %s",
                battle.index + 1, battle.duration_seconds, dest.name,
            )
        return written

    def _snap_to_keyframe(self, video: Path, timestamp: float) -> float:
        """Keyframe nearest ``timestamp`` that is not meaningfully before it.

        Stream copy can only start on a keyframe, and they sit ~1s apart here, so
        the boundary has to move. Snapping strictly forward is wrong when the
        boundary lands just after a keyframe: game_02's loading screen began 0.166s
        after one, and jumping to the next (0.834s later) skipped most of the
        splash. Snapping strictly backward is worse -- it prepends the lobby, the
        very thing the split removes.

        So: take the closest keyframe, allowing it to precede the boundary by at
        most ``split_keyframe_backfill_s``. That bounds any leftover previous-screen
        frames to a fraction of a second while keeping the intended opening.
        """
        argv = [
            self._settings.ffprobe_path, "-v", "error", "-select_streams", "v:0",
            # Bounded window: reading the whole packet index of a multi-GB
            # recording would cost far more than the cut itself.
            "-read_intervals", f"{max(0.0, timestamp - 2.0):.3f}%{timestamp + 3.0:.3f}",
            "-show_packets", "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0", str(video),
        ]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            logger.warning("Could not probe keyframes near %.2fs; cutting as-is", timestamp)
            return timestamp

        floor_ts = timestamp - self._settings.split_keyframe_backfill_s
        best: float | None = None
        for line in (completed.stdout or "").splitlines():
            pts, _, flags = line.partition(",")
            if "K" not in flags:
                continue
            try:
                value = float(pts)
            except ValueError:
                continue
            if value < floor_ts:
                continue
            if best is None or abs(value - timestamp) < abs(best - timestamp):
                best = value
        if best is None:
            return timestamp
        # Nudge just past the keyframe: asking for its exact PTS can float-round
        # down onto the PREVIOUS one and bring a second of lobby back.
        return best + 0.010

    def merge(self, clips: list[Path], destination: Path) -> Path:
        """Concatenate the cut matches into one gameplay-only recording.

        Stream copy again: every clip came from the same source, so codec
        parameters already match and no re-encode is needed.
        """
        if not clips:
            raise BattleSplitError("nothing to merge")
        missing = [c for c in clips if not c.is_file()]
        if missing:
            raise BattleSplitError(f"missing clip(s) to merge: {missing}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="cr_concat_") as tmp:
            listing = Path(tmp) / "clips.txt"
            listing.write_text(
                "".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8"
            )
            argv = [
                self._settings.ffmpeg_path, "-y", "-v", "error",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(destination),
            ]
            completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0:
            tail = (completed.stderr or "").strip()[-2000:]
            raise BattleSplitError(f"FFmpeg failed merging {len(clips)} clips.\n{tail}")
        logger.info("Merged %d matches -> %s", len(clips), destination)
        return destination

    # -- ffprobe ------------------------------------------------------------- #

    def _probe(self, video: Path) -> tuple[int, int, float]:
        """Return ``(width, height, duration)`` via ffprobe."""
        argv = [
            self._settings.ffprobe_path, "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-show_format", "-print_format", "json", str(video),
        ]
        try:
            completed = subprocess.run(argv, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise BattleSplitError(
                f"ffprobe not found at {self._settings.ffprobe_path!r}"
            ) from exc
        if completed.returncode != 0:
            raise BattleSplitError(completed.stderr.strip() or "ffprobe failed")
        try:
            data = json.loads(completed.stdout)
            stream = data["streams"][0]
            return (
                int(stream["width"]),
                int(stream["height"]),
                float(stream.get("duration") or data["format"]["duration"]),
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise BattleSplitError(f"could not parse ffprobe output: {exc}") from exc


def _sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
