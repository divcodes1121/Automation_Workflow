"""Workflow orchestration seam for the AI Creator Studio.

:class:`WorkflowManager` is the single object n8n drives to produce a video.
n8n owns the *sequencing* of steps and any retry/branching logic; each method
below owns the *business logic* of one pipeline stage. Keeping the stages as
discrete, independently callable methods is what lets n8n orchestrate without
embedding any logic of its own.

Every stage is currently a stub raising :class:`NotImplementedError`. The real
implementations will be delegated to focused modules under
:mod:`backend.modules` in later phases.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from backend.config import Settings, get_settings
from backend.models import (
    EditPlan,
    ExecutionTimeline,
    GameplayMetadata,
    GeneratedNarration,
    NarrationPackage,
    Project,
    RenderResult,
    RunResult,
    SubtitleTrack,
    ThumbnailPlan,
    ThumbnailResult,
    Timeline,
    UploadResult,
)

logger = logging.getLogger(__name__)


class RunStage(StrEnum):
    """Ordered identifiers for the stages :meth:`WorkflowManager.run` chains.

    The string values double as the keys in :attr:`RunResult.artifacts` /
    :attr:`RunResult.stage_timings` and as the labels the CLI reports.
    """

    TIMELINE = "timeline"
    NARRATION = "narration"
    SPEECH = "speech"
    SYNC = "synchronize"
    PLAN = "plan"
    RENDER = "render"
    SUBTITLE = "subtitle"
    BURN = "burn"
    THUMBNAIL_PLAN = "thumbnail-plan"
    THUMBNAIL = "thumbnail"
    UPLOAD = "upload"


class PipelineError(Exception):
    """A pipeline stage failed during :meth:`WorkflowManager.run`.

    Carries the :class:`RunStage` that failed so the caller can report *where*
    and map the underlying cause (chained via ``__cause__``) to an exit code.
    The workflow layer stays free of any CLI/``ExitCode`` knowledge.
    """

    def __init__(self, stage: RunStage, cause: Exception) -> None:
        self.stage = stage
        super().__init__(f"Pipeline failed at stage '{stage.value}': {cause}")


class WorkflowManager:
    """Coordinates the end-to-end video-production pipeline for one project.

    Parameters
    ----------
    settings:
        Optional configuration override. Defaults to the process-wide
        :func:`~backend.config.get_settings` singleton, which makes the manager
        trivially testable by injecting a custom :class:`Settings`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._project: Project | None = None

    @property
    def project(self) -> Project | None:
        """The currently loaded project, or ``None`` if none is loaded yet."""
        return self._project

    def run(
        self,
        project: Project,
        *,
        gameplay_sources: Path | list[Path] | None = None,
        upload: bool = False,
        privacy: str | None = None,
        result_path: Path | None = None,
        on_stage: Callable[[RunStage], None] | None = None,
    ) -> RunResult:
        """Run the whole production pipeline in one process.

        This is the single entry point n8n/CI drive: it calls the service
        classes **directly** (no CLI subprocesses), threading each stage's
        output into the next and recording every artifact path + elapsed time.

        The chain is timeline -> narration -> speech -> synchronize -> plan ->
        render -> subtitle -> burn -> thumbnail-plan -> thumbnail, then
        **upload only when** ``upload=True`` (otherwise the run stops after the
        thumbnail and returns cleanly -- the finished ``*.subtitled.mp4`` and
        ``*.thumbnail.png`` are on disk).

        Fail-fast: the first stage to raise stops the run and is re-raised as a
        :class:`PipelineError` naming the stage (original error chained via
        ``__cause__``). ``on_stage`` is invoked at the start of each stage so a
        caller can drive progress UI/notifications without touching this code.

        The resulting :class:`~backend.models.RunResult` is saved to
        ``edited/<slug>.run_result.json`` and returned.
        """
        # Services imported lazily to keep the orchestration module decoupled
        # and avoid import cycles (mirrors the per-stage methods below).
        from backend.services.narration import NarrationService
        from backend.services.planner import GameplayPlanner
        from backend.services.renderer import VideoRenderer
        from backend.services.speech import SpeechSynthesisService
        from backend.services.subtitle_renderer import SubtitleRenderer
        from backend.services.subtitles import SubtitleGenerator
        from backend.services.synchronizer import TimelineSynchronizer
        from backend.services.thumbnail_planner import ThumbnailPlanner
        from backend.services.thumbnail_renderer import ThumbnailRenderer
        from backend.services.timeline import TimelineBuilderService
        from backend.services.uploader import YouTubeUploader

        settings = self._settings
        self._project = project

        artifacts: dict[str, str] = {}
        stage_timings: dict[str, float] = {}
        completed: list[str] = []
        run_started = time.perf_counter()

        @contextmanager
        def stage(step: RunStage):
            if on_stage is not None:
                on_stage(step)
            started = time.perf_counter()
            try:
                yield
            except PipelineError:
                raise
            except Exception as exc:  # re-raised, tagged with the failing stage
                raise PipelineError(step, exc) from exc
            stage_timings[step.value] = round(time.perf_counter() - started, 3)
            completed.append(step.value)

        timeline_service = TimelineBuilderService(settings)
        narration_service = NarrationService(settings)
        speech_service = SpeechSynthesisService(settings)
        sync_service = TimelineSynchronizer(settings)
        planner = GameplayPlanner(settings)
        renderer = VideoRenderer(settings)
        subtitle_generator = SubtitleGenerator(settings)
        subtitle_renderer = SubtitleRenderer(settings)
        thumbnail_planner = ThumbnailPlanner(settings)
        thumbnail_renderer = ThumbnailRenderer(settings)
        uploader = YouTubeUploader(settings)

        with stage(RunStage.TIMELINE):
            timeline = timeline_service.build(project)
            artifacts[RunStage.TIMELINE.value] = str(timeline_service.save(timeline))

        with stage(RunStage.NARRATION):
            package = narration_service.prepare(timeline)
            artifacts[RunStage.NARRATION.value] = str(narration_service.save(package))

        with stage(RunStage.SPEECH):
            generated = speech_service.synthesize(package)
            artifacts[RunStage.SPEECH.value] = str(speech_service.save(generated))

        with stage(RunStage.SYNC):
            execution = sync_service.synchronize(timeline, package, generated)
            artifacts[RunStage.SYNC.value] = str(sync_service.save(execution))

        with stage(RunStage.PLAN):
            plan = planner.plan(execution, gameplay_sources=gameplay_sources)
            artifacts[RunStage.PLAN.value] = str(planner.save(plan))

        with stage(RunStage.RENDER):
            base_video = renderer.render(plan).output_file
            artifacts[RunStage.RENDER.value] = str(base_video)

        with stage(RunStage.SUBTITLE):
            track = subtitle_generator.generate(execution)
            # save() returns the .ass path -- the working file the burner reads.
            subtitle_path = subtitle_generator.save(track)
            artifacts[RunStage.SUBTITLE.value] = str(subtitle_path)

        with stage(RunStage.BURN):
            final_video = subtitle_renderer.burn(base_video, subtitle_path).output_file
            artifacts[RunStage.BURN.value] = str(final_video)

        with stage(RunStage.THUMBNAIL_PLAN):
            # Plan from the base render (no burned-in subtitles on the frame).
            thumb_plan = thumbnail_planner.plan(execution, base_video)
            artifacts[RunStage.THUMBNAIL_PLAN.value] = str(thumbnail_planner.save(thumb_plan))

        with stage(RunStage.THUMBNAIL):
            thumbnail_path = thumbnail_renderer.render(thumb_plan).output_file
            artifacts[RunStage.THUMBNAIL.value] = str(thumbnail_path)

        upload_result: UploadResult | None = None
        if upload:
            with stage(RunStage.UPLOAD):
                upload_result = uploader.upload(
                    project, final_video, thumbnail_path, privacy=privacy
                )
                artifacts[RunStage.UPLOAD.value] = str(uploader.save(upload_result))

        result = RunResult(
            project_id=project.project_id or project.title,
            title=project.title,
            completed_stages=completed,
            artifacts=artifacts,
            stage_timings=stage_timings,
            video_path=final_video,
            thumbnail_path=thumbnail_path,
            requested_upload=upload,
            uploaded=upload_result is not None,
            upload=upload_result,
            total_elapsed_seconds=round(time.perf_counter() - run_started, 3),
            created_at=datetime.now(timezone.utc),
        )
        self._save_run_result(project, result, destination=result_path)
        logger.info(
            "Run complete for %r: %d stages, %.1fs (uploaded=%s)",
            project.title,
            len(completed),
            result.total_elapsed_seconds,
            result.uploaded,
        )
        return result

    def _save_run_result(
        self, project: Project, result: RunResult, destination: Path | None = None
    ) -> Path:
        """Write ``edited/<slug>.run_result.json`` (the whole-run record)."""
        base = (project.project_id or project.title).strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "project"
        dest = destination or self._settings.edited_dir / f"{slug}.run_result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved run result for %r to %s", project.title, dest)
        return dest

    def load_project(self, path: str | Path) -> Project:
        """Load and validate the project that the rest of the pipeline acts on.

        Delegates to :func:`backend.script_loader.load_project` and caches the
        result on the manager for subsequent stages.
        """
        # Imported lazily to keep the module graph flat and avoid a hard import
        # cycle between orchestration and the I/O boundary.
        from backend.script_loader import load_project

        self._project = load_project(path)
        return self._project

    def analyze_gameplay(self, video_path: str | Path) -> GameplayMetadata:
        """Extract technical metadata from a gameplay video and persist it.

        Delegates to :class:`~backend.services.metadata.MetadataService`,
        keeping orchestration free of business logic. Analysis and persistence
        are separate calls at the service layer; here we do both.
        """
        # Imported lazily to keep the orchestration module decoupled from the
        # concrete service and avoid import cycles.
        from backend.services.metadata import MetadataService

        service = MetadataService(self._settings)
        metadata = service.analyze(video_path)
        service.save(metadata)
        return metadata

    def build_timeline(self, project: Project) -> Timeline:
        """Build the master timeline that every later stage consumes.

        Delegates to :class:`~backend.services.timeline.TimelineBuilderService`.
        Persistence is optional at the service layer; here we build and save.
        """
        from backend.services.timeline import TimelineBuilderService

        service = TimelineBuilderService(self._settings)
        timeline = service.build(project)
        service.save(timeline)
        return timeline

    def prepare_narration(self, timeline: Timeline) -> NarrationPackage:
        """Prepare a provider-neutral narration package from the timeline.

        Delegates to :class:`~backend.services.narration.NarrationService`.
        Persistence is optional at the service layer; here we prepare and save.
        """
        from backend.services.narration import NarrationService

        service = NarrationService(self._settings)
        package = service.prepare(timeline)
        service.save(package)
        return package

    def synthesize_speech(self, package: NarrationPackage) -> GeneratedNarration:
        """Synthesise per-segment narration audio via the speech provider.

        Delegates to :class:`~backend.services.speech.SpeechSynthesisService`,
        which runs Kokoro in an isolated Python 3.12 worker. Produces the audio
        and its manifest.
        """
        from backend.services.speech import SpeechSynthesisService

        service = SpeechSynthesisService(self._settings)
        generated = service.synthesize(package)
        service.save(generated)
        return generated

    def synchronize_timeline(
        self,
        timeline: Timeline,
        package: NarrationPackage,
        generated: GeneratedNarration,
    ) -> ExecutionTimeline:
        """Merge timeline + narration + audio into a validated execution timeline.

        Delegates to :class:`~backend.services.synchronizer.TimelineSynchronizer`.
        """
        from backend.services.synchronizer import TimelineSynchronizer

        service = TimelineSynchronizer(self._settings)
        execution = service.synchronize(timeline, package, generated)
        service.save(execution)
        return execution

    def plan_gameplay(
        self,
        execution: ExecutionTimeline,
        gameplay_sources: Path | list[Path] | None = None,
    ) -> EditPlan:
        """Plan which gameplay to show under each narration segment.

        Delegates to :class:`~backend.services.planner.GameplayPlanner`. Pure
        planning — produces an :class:`EditPlan`, no rendering.
        """
        from backend.services.planner import GameplayPlanner

        service = GameplayPlanner(self._settings)
        plan = service.plan(execution, gameplay_sources=gameplay_sources)
        service.save(plan)
        return plan

    def render_video(self, plan: EditPlan) -> RenderResult:
        """Render an edit plan into a playable MP4.

        Delegates to :class:`~backend.services.renderer.VideoRenderer` (executes
        the plan via one FFmpeg run; no creative decisions).
        """
        from backend.services.renderer import VideoRenderer

        return VideoRenderer(self._settings).render(plan)

    def generate_subtitles(self, execution: ExecutionTimeline) -> SubtitleTrack:
        """Generate an SRT subtitle track from the execution timeline.

        Delegates to :class:`~backend.services.subtitles.SubtitleGenerator`
        (word-timed captions from the timeline's own word timings; no Whisper).
        """
        from backend.services.subtitles import SubtitleGenerator

        service = SubtitleGenerator(self._settings)
        track = service.generate(execution)
        service.save(track)
        return track

    def burn_subtitles(self, video: Path, subtitles: Path) -> RenderResult:
        """Burn a subtitle file onto a video, producing the final MP4.

        Delegates to :class:`~backend.services.subtitle_renderer.SubtitleRenderer`.
        """
        from backend.services.subtitle_renderer import SubtitleRenderer

        return SubtitleRenderer(self._settings).burn(video, subtitles)

    def plan_thumbnail(self, execution: ExecutionTimeline, video: Path) -> ThumbnailPlan:
        """Plan a thumbnail from the execution timeline + base video.

        Delegates to :class:`~backend.services.thumbnail_planner.ThumbnailPlanner`
        (pure planning; the 9B renderer turns the plan into an image).
        """
        from backend.services.thumbnail_planner import ThumbnailPlanner

        service = ThumbnailPlanner(self._settings)
        plan = service.plan(execution, video)
        service.save(plan)
        return plan

    def edit_video(self) -> None:
        """Assemble the final long-form video from clips, voice and overlays.

        TODO(phase-3): Delegate to ``backend.modules.editor`` using FFmpeg
        (``settings.ffmpeg_path``). Intentionally unimplemented for now.
        """
        raise NotImplementedError("edit_video is not implemented yet")

    def generate_thumbnail(self, plan: ThumbnailPlan) -> ThumbnailResult:
        """Render a thumbnail image from a thumbnail plan.

        Delegates to :class:`~backend.services.thumbnail_renderer.ThumbnailRenderer`
        (executes the plan; no design decisions).
        """
        from backend.services.thumbnail_renderer import ThumbnailRenderer

        return ThumbnailRenderer(self._settings).render(plan)

    def upload_video(
        self, project: Project, video: Path, thumbnail: Path
    ) -> UploadResult:
        """Upload the finished video + thumbnail to YouTube.

        Delegates to :class:`~backend.services.uploader.YouTubeUploader`
        (upload-only; metadata comes from the project).
        """
        from backend.services.uploader import YouTubeUploader

        service = YouTubeUploader(self._settings)
        result = service.upload(project, video, thumbnail)
        service.save(result)
        return result

    def generate_shorts(self) -> None:
        """Render the vertical shorts described by ``project.shorts``.

        TODO(phase-4): Delegate to ``backend.modules.shorts`` — reframe to 9:16,
        apply hooks/captions, export into ``settings.gameplay_shorts_dir``.
        """
        raise NotImplementedError("generate_shorts is not implemented yet")
