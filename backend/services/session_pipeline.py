"""Session pipeline (Phase 3.3): one recording in, split + merge + shorts out.

The "drop a video in a folder and run one command" path. A session recording
holding several matches becomes:

    gameplay/raw/<stem>_game_NN.mp4      one clip per match (loading -> crowns)
    edited/<stem>_merged.mp4             all matches back to back, no menus
    edited/<stem>_game_NN.short.mp4      one Shorts-length highlight per match

Stages run in order and fail fast, each recorded with its own timing:

    SPLIT -> MERGE -> per match: ANALYZE -> SHORT

**Why the analyzer runs as a subprocess.** The analyzer is deliberately isolated
-- it never imports the backend, and its only contract with the pipeline is the
JSON it writes (``split_plan.json``, ``gameplay_analysis.json``). Driving it
through its CLI keeps that boundary intact and mirrors how the Kokoro TTS
sidecar is invoked. Everything downstream of the JSON is an ordinary in-process
call to :class:`~backend.services.highlight_editor.HighlightEditor`.

Shorts are rendered with effects and memes OFF: raw event-synced cuts carrying
the original Clash Royale audio.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterator

from backend.config import Settings, get_settings
from backend.models import SessionMatch, SessionResult, UploadResult
from backend.services.highlight_editor import HIGHLIGHT_CARDS

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv"}
# Frame rate the per-match analysis samples at (see _analyze).
_ANALYZE_SAMPLE_FPS = 1.0


class SessionStage(StrEnum):
    """Stages of a session run (values double as timing/artifact keys)."""

    SPLIT = "split"
    MERGE = "merge"
    ANALYZE = "analyze"
    SHORT = "short"
    UPLOAD = "upload"


class SessionError(RuntimeError):
    """Raised when a session recording cannot be processed."""

    def __init__(self, stage: SessionStage, message: str) -> None:
        super().__init__(f"[{stage.value}] {message}")
        self.stage = stage


class SessionPipeline:
    """Turns one session recording into per-match clips, a merge, and shorts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -- Inbox --------------------------------------------------------------- #

    def pending(self, inbox: Path | None = None) -> list[Path]:
        """Recordings sitting in the inbox, oldest first."""
        folder = inbox or self._settings.incoming_dir
        if not folder.is_dir():
            return []
        return sorted(
            (p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES),
            key=lambda p: p.stat().st_mtime,
        )

    def archive(self, recording: Path) -> Path:
        """Move a processed recording out of the inbox.

        Without this a scheduled daily run would find yesterday's file still
        sitting there and publish the same session again. Collisions get a
        numeric suffix rather than overwriting the earlier capture.
        """
        destination = self._settings.gameplay_archive_dir / recording.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            n = 2
            while (alt := destination.with_stem(f"{destination.stem}_{n}")).exists():
                n += 1
            destination = alt
        recording.replace(destination)
        logger.info("Archived %s -> %s", recording.name, destination)
        return destination

    def cleanup(
        self,
        result: SessionResult,
        *,
        raw: Path | None = None,
        force: bool = False,
    ) -> tuple[list[Path], int]:
        """Delete a session's media once it is safely on YouTube.

        A session costs about 6.8 GB on disk (clips, the merged long-form, the
        shorts, and ~2.2 GB of analyzer frame cache), so a daily run fills a drive
        fast. Everything removed here is either already published or regenerable.

        **Gated on every video having uploaded.** Deleting footage after a failed
        upload would destroy the only copy of something that never got published,
        so unless ``force`` is set a short upload count aborts the whole cleanup.

        ``raw`` is the original recording. It is the one irreplaceable file --
        every other artifact is derived from it -- so the caller passes it in
        explicitly rather than it being swept up by default.

        JSON artifacts (split plan, analyses, upload results) are deliberately
        kept: they are kilobytes, and they are the only remaining record of what
        was published and where.
        """
        expected = len([m for m in result.matches if m.short_path]) + (
            1 if result.merged_path else 0
        )
        if not force and len(result.uploads) < expected:
            raise SessionError(
                SessionStage.UPLOAD,
                f"refusing to delete media: only {len(result.uploads)} of {expected} "
                "videos uploaded. The footage is the only copy of anything that "
                "did not publish. Re-run the upload, or pass force to override.",
            )

        targets: list[Path] = []
        for match in result.matches:
            targets.append(match.clip_path)
            if match.short_path:
                targets.append(match.short_path)
            # The analyzer's extracted-frame cache for this clip. Addressed by
            # path rather than through the analyzer (which the backend must not
            # import); it is a regenerable cache, not source data.
            targets.append(
                self._settings.project_root
                / "analyzer" / "cache" / "frames" / "v1" / match.clip_path.stem
            )
        if result.merged_path:
            targets.append(result.merged_path)
        if raw is not None:
            targets.append(raw)

        removed: list[Path] = []
        freed = 0
        for target in targets:
            try:
                if target.is_dir():
                    size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                    shutil.rmtree(target)
                elif target.is_file():
                    size = target.stat().st_size
                    target.unlink()
                else:
                    continue
            except OSError as exc:
                logger.warning("Could not delete %s: %s", target, exc)
                continue
            removed.append(target)
            freed += size
            logger.info("Deleted %s (%.0f MB)", target.name, size / 1024 / 1024)
        logger.info("Cleanup freed %.2f GB across %d items", freed / 1024**3, len(removed))
        return removed, freed

    # -- Run ----------------------------------------------------------------- #

    def run(
        self,
        recording: Path,
        *,
        profile: str | None = None,
        merge: bool = True,
        shorts: bool = True,
        signature_card: str | None = None,
        sample_fps: float | None = None,
        upload: bool = False,
        privacy: str | None = None,
        schedule: bool = True,
        on_stage: Callable[[SessionStage, str], None] | None = None,
    ) -> SessionResult:
        """Process one recording end to end.

        With ``upload`` the merged long-form video and every short are published
        to YouTube. It defaults to off, and the privacy default is ``private``,
        so publishing is always a deliberate act rather than a side effect of
        processing footage.
        """
        if not recording.is_file():
            raise SessionError(SessionStage.SPLIT, f"recording not found: {recording}")

        started = time.perf_counter()
        timings: dict[str, float] = {}
        completed: list[SessionStage] = []
        warnings: list[str] = []

        @contextmanager
        def stage(name: SessionStage, detail: str = "") -> Iterator[None]:
            if on_stage:
                on_stage(name, detail)
            begin = time.perf_counter()
            yield
            timings[name.value] = round(
                timings.get(name.value, 0.0) + time.perf_counter() - begin, 2
            )
            if name not in completed:
                completed.append(name)

        # 1. Split the recording into one clip per match.
        with stage(SessionStage.SPLIT, recording.name):
            plan_path, clips, plan = self._split(recording, profile, sample_fps)
        if not clips:
            raise SessionError(
                SessionStage.SPLIT,
                "no matches found in this recording -- check the calibration "
                "profile matches the capture device/resolution",
            )
        warnings.extend(plan.get("warnings") or [])

        matches = [
            SessionMatch(
                index=battle["index"],
                clip_path=Path(battle["output_path"]),
                start_seconds=battle["start_seconds"],
                end_seconds=battle["end_seconds"],
                duration_seconds=battle["duration_seconds"],
            )
            for battle in plan["battles"]
            if battle.get("output_path")
        ]

        merged_path: Path | None = None
        if merge:
            with stage(SessionStage.MERGE, f"{len(clips)} matches"):
                merged_path = self._merge(plan_path, recording)

        # 2. Per match: analyze, then cut a short from the real event timings.
        if shorts:
            for match in matches:
                label = f"match {match.index + 1}/{len(matches)}"
                with stage(SessionStage.ANALYZE, label):
                    try:
                        match.analysis_path = self._analyze(match.clip_path, profile)
                    except SessionError as exc:
                        match.note = str(exc)
                        warnings.append(f"{label}: analysis failed, no short")
                        continue
                with stage(SessionStage.SHORT, label):
                    self._short(match, signature_card, warnings, label)

        uploads: list[UploadResult] = []
        if upload:
            with stage(SessionStage.UPLOAD, f"{len(matches)} short(s) + long"):
                uploads = self._upload(matches, merged_path, privacy, warnings, schedule)

        result = SessionResult(
            recording=recording,
            split_plan_path=plan_path,
            merged_path=merged_path,
            matches=matches,
            uploads=uploads,
            completed_stages=[s.value for s in completed],
            stage_timings=timings,
            warnings=warnings,
            total_elapsed_seconds=round(time.perf_counter() - started, 2),
            created_at=datetime.now(timezone.utc),
        )
        return result

    def save(self, result: SessionResult, destination: Path | None = None) -> Path:
        """Persist the session result as JSON."""
        dest = destination or (
            self._settings.edited_dir / f"{result.recording.stem}.session_result.json"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved session result to %s", dest)
        return dest

    # -- Stages -------------------------------------------------------------- #

    def _split(
        self, recording: Path, profile: str | None, sample_fps: float | None
    ) -> tuple[Path, list[Path], dict[str, Any]]:
        """Run the analyzer's splitter; return (plan path, clips, plan dict)."""
        argv = [sys.executable, "-m", "analyzer.main", "split", str(recording)]
        if profile:
            argv += ["--profile", profile]
        if sample_fps is not None:
            argv += ["--sample-fps", str(sample_fps)]
        self._run_analyzer(argv, SessionStage.SPLIT)

        plan_path = self._analysis_dir() / f"{recording.stem}.split_plan.json"
        if not plan_path.is_file():
            raise SessionError(SessionStage.SPLIT, f"no split plan written at {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        clips = [
            Path(b["output_path"]) for b in plan.get("battles", []) if b.get("output_path")
        ]
        return plan_path, clips, plan

    def _merge(self, plan_path: Path, recording: Path) -> Path:
        """Concatenate the plan's clips into one gameplay-only video."""
        destination = self._merged_path(recording)
        self._run_analyzer(
            [sys.executable, "-m", "analyzer.main", "merge", str(plan_path),
             "--output", str(destination)],
            SessionStage.MERGE,
        )
        if not destination.is_file():
            raise SessionError(SessionStage.MERGE, f"merge produced no file at {destination}")
        return destination

    def _analyze(self, clip: Path, profile: str | None) -> Path:
        """Run the analyzer on one match clip; return its analysis JSON path."""
        argv = [
            sys.executable, "-m", "analyzer.main", "analyze", str(clip),
            # 1 fps is what the analyzer's own regression baseline was measured
            # at (~2.7 min per match). The config default of 5 fps would quintuple
            # both runtime and the frame cache for no gain here: shorts only need
            # play timestamps, which resolve fine at 1 fps.
            "--sample-fps", str(_ANALYZE_SAMPLE_FPS),
        ]
        if profile:
            argv += ["--profile", profile]
        self._run_analyzer(argv, SessionStage.ANALYZE)
        analysis_path = self._analysis_dir() / f"{clip.stem}.gameplay_analysis.json"
        if not analysis_path.is_file():
            raise SessionError(
                SessionStage.ANALYZE, f"no analysis written at {analysis_path}"
            )
        return analysis_path

    def _short(
        self,
        match: SessionMatch,
        signature_card: str | None,
        warnings: list[str],
        label: str,
    ) -> None:
        """Build + render one short from a match's analysis (no effects)."""
        from backend.services.highlight_editor import HighlightEditor, HighlightError

        analysis = json.loads(match.analysis_path.read_text(encoding="utf-8"))
        # Default: summarise the match across every win condition and spell that
        # was played, rather than repeating a single card. Those are the plays
        # that decide a game, so the reel reads as "what happened in this match".
        # An explicit --card pins the reel to one card instead.
        cards = None if signature_card else _played_highlight_cards(analysis)
        card = signature_card or _signature_card(analysis)
        if card is None:
            match.note = "no card plays detected; nothing to build a short around"
            warnings.append(f"{label}: {match.note}")
            return

        editor = HighlightEditor(self._settings)
        dest = self._settings.edited_dir / f"{match.clip_path.stem}.short.mp4"
        try:
            plan = editor.build(
                analysis, match.clip_path, signature_card=card, cards=cards
            )
            editor.save(plan)
            editor.render(plan, match.clip_path, dest, effects=False, memes=False)
        except HighlightError as exc:
            match.note = str(exc)
            warnings.append(f"{label}: could not build a short ({exc})")
            return
        match.short_path = dest
        match.signature_card = card
        match.short_duration_seconds = plan.total_duration_seconds

    def _upload(
        self,
        matches: list[SessionMatch],
        merged_path: Path | None,
        privacy: str | None,
        warnings: list[str],
        schedule: bool = True,
    ) -> list[UploadResult]:
        """Publish the merged long-form video and each short to YouTube.

        Uploads continue past a per-video failure so one rejected clip does not
        strand the rest of the day's batch; every failure lands in ``warnings``.
        """
        from backend.services.publish_metadata import (
            find_thumbnail,
            load_analysis,
            next_slot,
            parse_slots,
            session_metadata,
            short_metadata,
        )
        from backend.services.uploader import (
            UploadAuthError,
            UploadError,
            UploadRequestError,
            YouTubeUploader,
        )

        uploader = YouTubeUploader(self._settings)
        if not uploader.credentials_present():
            raise SessionError(
                SessionStage.UPLOAD,
                "no YouTube credentials: put your OAuth desktop client JSON at "
                f"{self._settings.youtube_client_secret_file}",
            )

        analyses = [load_analysis(m.analysis_path) for m in matches]
        # Each video gets its own publish slot. Releasing a day's batch at one
        # instant makes the videos compete with each other for the same session;
        # staggering them covers several browsing windows instead.
        s = self._settings
        long_slot = next_slot(s.publish_long_at) if schedule else None
        short_slots = parse_slots(s.publish_shorts_at) if schedule else []

        jobs: list[tuple[str, Path, Any, datetime | None]] = []
        if merged_path is not None:
            durations = [m.duration_seconds for m in matches]
            anchor = next((m.signature_card for m in matches if m.signature_card), None)
            jobs.append((
                "long",
                merged_path,
                session_metadata(analyses, durations, signature_card=anchor),
                long_slot,
            ))
        shorts_done = 0
        used_titles: set[str] = {j[2].title for j in jobs}
        for match, analysis in zip(matches, analyses):
            if match.short_path is None:
                continue
            slot = (
                next_slot(short_slots[shorts_done % len(short_slots)])
                if short_slots else None
            )
            shorts_done += 1
            meta_short = short_metadata(
                analysis, match.signature_card, index=match.index, avoid=used_titles
            )
            used_titles.add(meta_short.title)
            jobs.append((
                f"short {match.index + 1}",
                match.short_path,
                meta_short,
                slot,
            ))

        results: list[UploadResult] = []
        for label, video, meta, slot in jobs:
            try:
                thumbnail = find_thumbnail(video)
                if thumbnail:
                    logger.info("Using thumbnail %s for %s", thumbnail.name, label)
                request = uploader.prepare_video(
                    video,
                    title=meta.title,
                    description=meta.description,
                    tags=meta.tags,
                    thumbnail=thumbnail,
                    privacy=privacy,
                    publish_at=slot,
                )
                result = uploader.upload_request(request)
                uploader.save(result)
                results.append(result)
                if slot:
                    logger.info(
                        "Uploaded %s -> %s (goes public %s IST)",
                        label, result.url, slot.strftime("%Y-%m-%d %H:%M"),
                    )
                else:
                    logger.info("Uploaded %s -> %s", label, result.url)
            except UploadAuthError:
                raise  # Auth is fatal for the whole batch, not just this video.
            except (UploadRequestError, UploadError) as exc:
                warnings.append(f"{label}: upload failed ({exc})")
                logger.warning("Upload failed for %s: %s", label, exc)
        return results

    # -- Helpers ------------------------------------------------------------- #

    def _run_analyzer(self, argv: list[str], stage: SessionStage) -> None:
        """Invoke the analyzer CLI, surfacing its output on failure."""
        logger.info("Running: %s", " ".join(argv[2:]))
        completed = subprocess.run(
            argv, capture_output=True, text=True, cwd=str(self._settings.project_root)
        )
        if completed.returncode != 0:
            tail = ((completed.stderr or "") + (completed.stdout or "")).strip()[-2000:]
            raise SessionError(
                stage, f"analyzer exited {completed.returncode}.\n{tail}"
            )

    def _analysis_dir(self) -> Path:
        """Where the analyzer writes its JSON artifacts."""
        return self._settings.project_root / "gameplay" / "analysis"

    def _merged_path(self, recording: Path) -> Path:
        """Where the analyzer's --merge writes the combined video."""
        return self._settings.gameplay_raw_dir / f"{recording.stem}_merged.mp4"


def _played_highlight_cards(analysis: dict[str, Any]) -> set[str] | None:
    """Win conditions and spells the player actually played this match.

    The set a summary reel is cut from. ``None`` when the deck contains none of
    them, which sends the caller back to the single most-played card so a short
    still gets made.
    """
    played = {e["card"] for e in analysis.get("events") or [] if e.get("card")}
    return (played & HIGHLIGHT_CARDS) or None


def _signature_card(analysis: dict[str, Any]) -> str | None:
    """The card a match's story should be built around.

    Prefers a played card that makes a visible moment (see ``HIGHLIGHT_CARDS``),
    falling back to the most-played card when the deck has none -- so any deck
    yields a short without configuration. Within either group the pick is the
    most-played, ties breaking toward the earliest play, keeping it deterministic.
    """
    events = [e for e in analysis.get("events") or [] if e.get("card")]
    if not events:
        return None
    counts = Counter(e["card"] for e in events)
    preferred = {c: n for c, n in counts.items() if c in HIGHLIGHT_CARDS}
    pool = preferred or counts

    best = max(pool.values())
    leaders = {card for card, n in pool.items() if n == best}
    for event in events:  # events are chronological
        if event["card"] in leaders:
            return event["card"]
    return None
