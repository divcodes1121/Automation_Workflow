"""Orchestration seam for the Gameplay Analyzer.

:class:`AnalyzerWorkflow` is the single object callers (the CLI today, n8n later)
drive. Each method owns the business logic of one analyzer stage; keeping them
discrete is what lets an orchestrator sequence them without embedding logic.

Slice 2A exposes only :meth:`build_template_library`. The detector chain that
turns a recording into ``gameplay_analysis.json`` will attach here as an
``analyze(video)`` method in later slices; it is documented now (as a stub) so
the seam is visible, but intentionally unimplemented -- it needs real footage.
"""

from __future__ import annotations

import logging
from pathlib import Path

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import FramesManifest, GameplayAnalysis, HandReading, TemplateLibrary

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

logger = logging.getLogger(__name__)


class AnalyzerWorkflow:
    """Coordinates the analyzer stages for one recording.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~analyzer.config.get_analyzer_settings`).
    """

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings: AnalyzerSettings = settings or get_analyzer_settings()

    def build_template_library(self, dry_run: bool = False) -> TemplateLibrary:
        """Preprocess the bundled card art into a cached template library.

        Delegates to :class:`~analyzer.preprocess.build_templates.TemplateBuilder`.
        Builds and (unless ``dry_run``) persists the manifest.
        """
        # Imported lazily so importing the workflow does not pull in OpenCV.
        from analyzer.preprocess.build_templates import TemplateBuilder

        builder = TemplateBuilder(self._settings)
        library = builder.build(dry_run=dry_run)
        if not dry_run:
            builder.save(library)
        return library

    def extract_frames(
        self,
        video: str | Path,
        *,
        sample_fps: float | None = None,
        image_format: str | None = None,
        dry_run: bool = False,
    ) -> FramesManifest:
        """Sample a recording into cached frames + a manifest.

        Delegates to :class:`~analyzer.preprocess.frame_extractor.FrameExtractor`.
        Extracts and (unless ``dry_run``) persists the manifest.
        """
        # Imported lazily to keep the workflow import light.
        from analyzer.preprocess.frame_extractor import FrameExtractor

        extractor = FrameExtractor(self._settings)
        manifest = extractor.extract(
            video, sample_fps=sample_fps, image_format=image_format, dry_run=dry_run
        )
        if not dry_run:
            extractor.save(manifest)
        return manifest

    def calibrate(
        self,
        source: str | Path,
        *,
        profile_name: str | None = None,
        output: Path | None = None,
    ) -> Path:
        """Draw the active profile's ROIs over a frame -> a preview PNG (2C).

        ``source`` may be an image or a video (first frame is used). The preview
        is how a device profile is visually tuned against a real capture.
        """
        import cv2

        from analyzer.calibration.profiles import load_profile
        from analyzer.calibration.roi import draw_rois

        profile = load_profile(profile_name, self._settings)
        frame = self._load_frame(source)
        overlay = draw_rois(frame, profile)
        stem = Path(source).stem
        dest = output or self._settings.analysis_output_dir / f"{stem}.calibration_preview.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), overlay)
        logger.info("Wrote calibration preview (%s) to %s", profile.name, dest)
        return dest

    def detect_hand(
        self, frame_path: str | Path, *, profile_name: str | None = None
    ) -> HandReading:
        """Detect the four-card hand in a single frame image (2D)."""
        from analyzer.calibration.profiles import load_profile
        from analyzer.detectors.hand_detector import HandDetector

        profile = load_profile(profile_name, self._settings)
        frame = self._load_frame(frame_path)
        return HandDetector(self._settings).detect(frame, profile)

    def analyze(
        self,
        video: str | Path,
        *,
        profile_name: str | None = None,
        sample_fps: float | None = None,
    ) -> GameplayAnalysis:
        """Run the full analysis: video -> gameplay_analysis.json (2B-2I).

        Reads BOTH hands per frame (replays show the opponent's too),
        reconstructs both decks with confidence, detects the player's plays, and
        records analyzer metrics. Lane/context (2F/2G) stay null with a warning
        until those footage-dependent detectors exist. An optional, config-gated
        change-detection foundation (``hand_skip_unchanged``) reuses unchanged
        frames -- off by default (correctness first).
        """
        import time

        import cv2

        from analyzer.calibration.profiles import load_profile
        from analyzer.detectors.elixir_detector import ElixirDetector
        from analyzer.detectors.hand_detector import HandDetector
        from analyzer.detectors.play_detector import PlayDetector
        from analyzer.detectors.timer_detector import TimerDetector
        from analyzer.models import AnalyzerMetrics
        from analyzer.preprocess.frame_extractor import FrameExtractor
        from analyzer.tracking.deck_reconstructor import DeckReconstructor
        from analyzer.tracking.event_builder import EventBuilder
        from analyzer.tracking.match_state import RawSample, build_timeline

        s = self._settings
        profile = load_profile(profile_name, s)
        manifest = FrameExtractor(s).extract(video, sample_fps=sample_fps)
        frames_dir = Path(manifest.frames_dir)

        detector = HandDetector(s)
        player_recon = DeckReconstructor("player", s)
        opponent_recon = DeckReconstructor("opponent", s)
        player_readings: list[HandReading] = []

        # 2G match-state: sample the timer + elixir ~once per interval (they only
        # change ~1/sec, so this is far sparser than the hand read every frame).
        timer_detector = TimerDetector(s)
        elixir_detector = ElixirDetector(s)
        raw_samples: list[RawSample] = []
        last_state_ts: float | None = None

        conf_norm = 2 * s.orb_min_matches  # ORB score at which a slot reads ~100% confident
        m_frames = m_slots = m_ids = m_unknown = 0
        orb_sum = conf_sum = 0.0
        started = time.perf_counter()

        for frame in manifest.frames:
            image = cv2.imread(str(frames_dir / frame.filename))
            if image is None:
                logger.warning("Could not read frame %s; skipping", frame.filename)
                continue

            # Per-slot event-driven skip (unchanged slots reuse the cached card)
            # is handled inside HandDetector.detect, gated by hand_skip_unchanged.
            player = detector.detect(
                image, profile, slot_keys=HandDetector.HAND_SLOTS,
                source_frame=frame.source_frame, timestamp_seconds=frame.timestamp_seconds,
            )
            opponent = detector.detect(
                image, profile, slot_keys=HandDetector.OPP_HAND_SLOTS,
                source_frame=frame.source_frame, timestamp_seconds=frame.timestamp_seconds,
            )
            m_frames += 1
            player_readings.append(player)
            player_recon.update(player)
            opponent_recon.update(opponent)

            ts = frame.timestamp_seconds
            if last_state_ts is None or (ts - last_state_ts) >= s.match_state_interval_s:
                timer = timer_detector.read(image, profile)
                self_elixir, opp_elixir = elixir_detector.read_both(image, profile)
                raw_samples.append(
                    RawSample(
                        timestamp_seconds=ts,
                        source_frame=frame.source_frame,
                        phase=timer.phase,
                        time_text=timer.text,
                        time_seconds=timer.seconds,
                        timer_confidence=timer.confidence,
                        player_elixir=self_elixir,
                        opponent_elixir=opp_elixir,
                    )
                )
                last_state_ts = ts
            for reading in (player, opponent):
                for slot in reading.slots:
                    m_slots += 1
                    if slot.matched:
                        m_ids += 1
                        orb_sum += slot.score
                        conf_sum += min(1.0, slot.score / conf_norm)
                    else:
                        m_unknown += 1

        elapsed = time.perf_counter() - started
        metrics = AnalyzerMetrics(
            frames_processed=m_frames,
            frames_skipped=0,  # frame-level skip retired in favour of per-slot
            slots_analyzed=m_slots,
            slots_skipped=detector.slots_skipped,
            cards_identified=m_ids,
            unknown_slots=m_unknown,
            average_orb_matches=round(orb_sum / m_ids, 2) if m_ids else 0.0,
            average_confidence=round(conf_sum / m_ids, 4) if m_ids else 0.0,
            processing_seconds=round(elapsed, 2),
            fps_processed=round(m_frames / elapsed, 2) if elapsed > 0 else 0.0,
            matching_method=s.matching_method,
        )

        match_states = build_timeline(raw_samples)
        plays = PlayDetector(s).detect(player_readings)
        builder = EventBuilder(s)
        analysis = builder.build(
            plays,
            video=manifest.video,
            video_sha256=manifest.video_sha256,
            source_fps=manifest.source_fps,
            duration_seconds=manifest.duration_seconds,
            sample_fps=manifest.sample_fps,
            frame_count=manifest.frame_count,
            profile_name=profile.name,
            match_states=match_states,
            player_deck=player_recon.deck(),
            opponent_deck=opponent_recon.deck(),
            metrics=metrics,
        )
        builder.save(analysis)
        return analysis

    def _load_frame(self, source: str | Path):
        """Load a single frame from an image path, or the first frame of a video."""
        import cv2

        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(f"Frame source not found: {source}")
        if source.suffix.lower() in _IMAGE_SUFFIXES:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Could not read image: {source}")
            return image
        capture = cv2.VideoCapture(str(source))
        try:
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok or image is None:
            raise ValueError(f"Could not read first frame of video: {source}")
        return image
