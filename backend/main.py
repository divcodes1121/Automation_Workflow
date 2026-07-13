"""Command-line entry point for the AI Creator Studio.

Feature commands are exposed::

    python -m backend.main validate  path/to/project.json     # Feature 1
    python -m backend.main analyze   gameplay/raw/example.mp4  # Feature 2
    python -m backend.main timeline  path/to/project.json      # Feature 3
    python -m backend.main narration path/to/project.json      # Feature 4
    python -m backend.main voice     path/to/project.json      # Feature 5
    python -m backend.main synchronize path/to/project.json    # Feature 6
    python -m backend.main plan      path/to/project.json      # Feature 7A
    python -m backend.main render    path/to/project.json      # Feature 7B
    python -m backend.main subtitle  path/to/project.json      # Feature 8A
    python -m backend.main burn      path/to/project.json      # Feature 8B
    python -m backend.main thumbnail-plan path/to/project.json # Feature 9A
    python -m backend.main thumbnail path/to/project.json       # Feature 9B
    python -m backend.main upload    path/to/project.json       # Feature 10
    python -m backend.main run       path/to/project.json       # Feature 11

The CLI is intentionally *thin*: it parses arguments, calls a service, renders a
Rich summary, and maps failures to distinct process exit codes (read by n8n).
All domain logic lives elsewhere.

Exit codes
----------
0
    Success.
1
    Any other unexpected error.
2
    Project file not found.
3
    Invalid JSON.
4
    Project schema validation failed.
5
    Video file not found.
6
    ffprobe unavailable / probe failed / metadata unparseable.
7
    Timeline could not be built from the project.
8
    Narration package could not be prepared.
9
    Speech provider/runtime unavailable.
10
    Speech synthesis failed.
11
    Timeline synchronization/validation failed.
12
    Edit plan could not be built (insufficient/invalid footage).
13
    Video render failed (validation or FFmpeg error).
14
    Subtitle generation failed.
15
    Subtitle burn failed (validation or FFmpeg error).
16
    Thumbnail plan could not be built.
17
    Thumbnail render failed (validation or FFmpeg/Pillow error).
18
    Upload failed (invalid request or YouTube API error).
19
    Upload failed (missing/invalid OAuth credentials).
"""

from __future__ import annotations

import logging
import re
from enum import IntEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from backend.config import get_settings
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
from backend.script_loader import (
    ScriptFileNotFoundError,
    ScriptParseError,
    ScriptValidationError,
    load_project,
)
from backend.services.metadata import (
    FFprobeNotAvailableError,
    MetadataParseError,
    MetadataProbeError,
    MetadataService,
    VideoNotFoundError,
)
from backend.services.narration import (
    NarrationPreparationError,
    NarrationService,
)
from backend.services.kokoro_runner import (
    SpeechProviderUnavailableError,
    SpeechSynthesisError,
)
from backend.services.speech import SpeechSynthesisService
from backend.services.planner import EditPlanError, GameplayPlanner
from backend.services.renderer import RenderError, VideoRenderer
from backend.services.subtitle_renderer import SubtitleBurnError, SubtitleRenderer
from backend.services.subtitles import SubtitleGenerationError, SubtitleGenerator
from backend.services.thumbnail_planner import ThumbnailPlanError, ThumbnailPlanner
from backend.services.thumbnail_renderer import (
    ThumbnailRenderError,
    ThumbnailRenderer,
    layout_path,
)
from backend.services.uploader import (
    UploadAuthError,
    UploadError,
    UploadRequestError,
    YouTubeUploader,
)
from backend.services.synchronizer import (
    TimelineSynchronizationError,
    TimelineSynchronizer,
)
from backend.services.timeline import TimelineBuildError, TimelineBuilderService
from backend.utils.logging_config import configure_logging
from backend.workflow import PipelineError, RunStage, WorkflowManager

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="AI Creator Studio - Clash Royale YouTube pipeline.",
)


class ExitCode(IntEnum):
    """Process exit codes, consumed by n8n to branch on outcome."""

    SUCCESS = 0
    UNEXPECTED = 1
    FILE_NOT_FOUND = 2
    PARSE_ERROR = 3
    VALIDATION_ERROR = 4
    VIDEO_NOT_FOUND = 5
    PROBE_ERROR = 6
    TIMELINE_ERROR = 7
    NARRATION_ERROR = 8
    SPEECH_UNAVAILABLE = 9
    SPEECH_ERROR = 10
    SYNC_ERROR = 11
    PLAN_ERROR = 12
    RENDER_ERROR = 13
    SUBTITLE_ERROR = 14
    BURN_ERROR = 15
    THUMBNAIL_ERROR = 16
    THUMBNAIL_RENDER_ERROR = 17
    UPLOAD_ERROR = 18
    UPLOAD_AUTH_ERROR = 19


def _human_bytes(num: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _human_duration(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS``)."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _load_project_or_exit(project_file: Path) -> Project:
    """Load and validate a project file, mapping failures to CLI exit codes.

    Shared by the ``validate`` and ``timeline`` commands so both report project
    problems identically.
    """
    try:
        return load_project(project_file)
    except ScriptFileNotFoundError as exc:
        console.print(f"[bold red]File not found:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.FILE_NOT_FOUND)
    except ScriptParseError as exc:
        console.print(f"[bold red]Invalid JSON:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PARSE_ERROR)
    except ScriptValidationError as exc:
        console.print(f"[bold red]Validation failed:[/bold red] {exc}")
        if exc.__cause__ is not None:
            console.print(f"[dim]{exc.__cause__}[/dim]")
        raise typer.Exit(code=ExitCode.VALIDATION_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while loading project")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)


# --------------------------------------------------------------------------- #
# Feature 1: project validation
# --------------------------------------------------------------------------- #
def _render_project_summary(project: Project) -> None:
    """Render a Rich summary of the validated project to the console."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Title", project.title)
    table.add_row("Shorts", str(project.short_count))
    table.add_row("Tags", ", ".join(project.tags))
    table.add_row("Upload time", project.upload_time.isoformat())
    table.add_row("Category", project.category.value)
    table.add_row("Language", project.language)
    table.add_row("Status", project.status.value)

    console.print(
        Panel(
            table,
            title="[bold green]Project validated[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )


@app.command()
def validate(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to load and validate.",
    ),
) -> None:
    """Load a project JSON file, validate it, and print a Rich summary."""
    configure_logging()
    project = _load_project_or_exit(project_file)
    _render_project_summary(project)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 2: gameplay analysis
# --------------------------------------------------------------------------- #
def _render_metadata_summary(
    metadata: GameplayMetadata, saved_to: Path | None
) -> None:
    """Render a Rich summary of extracted gameplay metadata."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Source", metadata.source_file.name)
    table.add_row("Duration", _human_duration(metadata.duration_seconds))
    table.add_row("FPS", f"{metadata.fps:.3g}")
    table.add_row("Resolution", metadata.resolution)
    table.add_row("Codec", metadata.video_codec)
    bitrate = (
        f"{metadata.bitrate_bps / 1_000_000:.2f} Mbps"
        if metadata.bitrate_bps
        else "unknown"
    )
    table.add_row("Bitrate", bitrate)
    table.add_row("File size", _human_bytes(metadata.file_size_bytes))
    table.add_row("Video hash", metadata.video_hash[:16] + "...")
    table.add_row(
        "Created",
        metadata.creation_date.isoformat() if metadata.creation_date else "unknown",
    )
    if saved_to is not None:
        table.add_row("Saved to", str(saved_to))

    console.print(
        Panel(
            table,
            title="[bold green]Gameplay analyzed[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )


@app.command()
def analyze(
    video_file: Path = typer.Argument(
        ...,
        help="Path to the gameplay video (.mp4) to inspect.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Write the extracted metadata to a JSON file.",
    ),
) -> None:
    """Extract technical metadata from a gameplay video via ffprobe."""
    configure_logging()

    service = MetadataService()
    try:
        metadata = service.analyze(video_file)
        saved_to = service.save(metadata) if save else None
    except VideoNotFoundError as exc:
        console.print(f"[bold red]Video not found:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.VIDEO_NOT_FOUND)
    except FFprobeNotAvailableError as exc:
        console.print(f"[bold red]ffprobe unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PROBE_ERROR)
    except (MetadataProbeError, MetadataParseError) as exc:
        console.print(f"[bold red]Could not analyze video:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PROBE_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while analyzing gameplay")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_metadata_summary(metadata, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 3: timeline builder
# --------------------------------------------------------------------------- #
def _truncate(text: str, limit: int = 44) -> str:
    """Shorten ``text`` for table display, appending an ellipsis if clipped."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."


def _render_timeline_summary(timeline: Timeline, saved_to: Path | None) -> None:
    """Render a Rich summary of the built timeline to the console."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", timeline.title)
    header.add_row("Segments", str(timeline.segment_count))
    header.add_row("Est. duration", _human_duration(timeline.total_duration_seconds))
    header.add_row("Speaking rate", f"{timeline.words_per_minute:.0f} wpm")
    header.add_row("Timing", "estimated" if timeline.is_estimated else "measured")

    segments = Table(box=None, pad_edge=False, show_edge=False)
    segments.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    segments.add_column("Est. span", style="magenta", no_wrap=True)
    segments.add_column("Imp.", style="yellow", no_wrap=True)
    segments.add_column("Voice", style="white")
    segments.add_column("Visual", style="dim")
    for seg in timeline.segments:
        span = (
            f"{seg.timing.estimated_start_seconds:.1f}"
            f"-{seg.timing.estimated_end_seconds:.1f}s"
        )
        importance = seg.narration.importance.value if seg.narration.importance else "-"
        segments.add_row(
            str(seg.id),
            span,
            importance,
            _truncate(seg.voice),
            _truncate(seg.visual) if seg.visual else "-",
        )

    console.print(
        Panel(
            header,
            title="[bold green]Timeline built[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(segments)
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def timeline(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to build a timeline from.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Write the timeline to a JSON file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit path for the timeline JSON (default: output/<slug>.timeline.json).",
    ),
) -> None:
    """Validate a project, build its master timeline, and print a Rich summary."""
    configure_logging()
    project = _load_project_or_exit(project_file)

    service = TimelineBuilderService()
    try:
        built = service.build(project)
        saved_to = service.save(built, destination=output) if save else None
    except TimelineBuildError as exc:
        console.print(f"[bold red]Could not build timeline:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.TIMELINE_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while building timeline")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_timeline_summary(built, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 4: narration engine (preparation layer)
# --------------------------------------------------------------------------- #
def _render_narration_summary(
    package: NarrationPackage, saved_to: Path | None
) -> None:
    """Render a Rich summary of the prepared narration package."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", package.title)
    header.add_row("Segments", str(package.segment_count))
    header.add_row("Schema", package.schema_version)
    header.add_row(
        "Est. duration",
        _human_duration(package.total_estimated_duration_seconds),
    )

    segments = Table(box=None, pad_edge=False, show_edge=False)
    segments.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    segments.add_column("Dur", style="magenta", justify="right", no_wrap=True)
    segments.add_column("Emphasis", style="yellow", no_wrap=True)
    segments.add_column("Emotion", style="green", no_wrap=True)
    segments.add_column("Cleaned text", style="white")
    for index, seg in enumerate(package.segments, start=1):
        segments.add_row(
            str(index),
            f"{seg.estimated_duration_seconds:.1f}s",
            ", ".join(seg.emphasis_words) if seg.emphasis_words else "-",
            seg.emotion.value if seg.emotion else "-",
            _truncate(seg.cleaned_text),
        )

    console.print(
        Panel(
            header,
            title="[bold green]Narration prepared[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(segments)
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def narration(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to prepare narration for.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Write the narration package to a JSON file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit path for the narration JSON (default: output/<slug>.narration.json).",
    ),
) -> None:
    """Validate a project, build its timeline, and prepare a narration package."""
    configure_logging()
    project = _load_project_or_exit(project_file)

    try:
        timeline = TimelineBuilderService().build(project)
        package = NarrationService().prepare(timeline)
        saved_to = NarrationService().save(package, destination=output) if save else None
    except (TimelineBuildError, NarrationPreparationError) as exc:
        console.print(f"[bold red]Could not prepare narration:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.NARRATION_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while preparing narration")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_narration_summary(package, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 5: speech synthesis (Kokoro sidecar)
# --------------------------------------------------------------------------- #
def _render_speech_summary(generated: GeneratedNarration, manifest: Path | None) -> None:
    """Render a Rich summary of the synthesised narration."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", generated.title)
    header.add_row("Provider", f"{generated.provider} ({generated.voice})")
    header.add_row("Segments", str(generated.segment_count))
    header.add_row("Sample rate", f"{generated.sample_rate} Hz")
    header.add_row("Audio duration", _human_duration(generated.total_audio_seconds))
    header.add_row("Synthesis time", f"{generated.total_synthesis_seconds:.2f} s")
    rtf = generated.realtime_factor
    header.add_row("RTF", f"{rtf:.3g}" if rtf is not None else "-")

    segments = Table(box=None, pad_edge=False, show_edge=False)
    segments.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    segments.add_column("Audio", style="magenta", justify="right", no_wrap=True)
    segments.add_column("Words", style="yellow", justify="right", no_wrap=True)
    segments.add_column("File", style="white")
    for seg in generated.segments:
        segments.add_row(
            str(seg.index),
            f"{seg.duration_seconds:.1f}s",
            str(len(seg.word_timings)),
            seg.audio_file.name,
        )

    console.print(
        Panel(
            header,
            title="[bold green]Speech synthesised[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(segments)
    if generated.segments:
        console.print(f"[dim]Audio dir {generated.segments[0].audio_file.parent}[/dim]")
    if manifest is not None:
        console.print(f"[dim]Manifest {manifest}[/dim]")


@app.command()
def voice(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to synthesise narration for.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for the per-segment WAVs + manifest "
        "(default: output/narration/<slug>).",
    ),
) -> None:
    """Synthesise per-segment narration audio via the Kokoro sidecar (Feature 5)."""
    configure_logging()
    project = _load_project_or_exit(project_file)

    service = SpeechSynthesisService()
    try:
        timeline = TimelineBuilderService().build(project)
        package = NarrationService().prepare(timeline)
        generated = service.synthesize(package, output_dir=output_dir)
        manifest = service.save(generated)
    except SpeechProviderUnavailableError as exc:
        console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
    except (TimelineBuildError, NarrationPreparationError, SpeechSynthesisError) as exc:
        console.print(f"[bold red]Speech synthesis failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while synthesising speech")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_speech_summary(generated, manifest)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 6: timeline synchronization
# --------------------------------------------------------------------------- #
def _render_execution_summary(
    execution: ExecutionTimeline, saved_to: Path | None
) -> None:
    """Render a Rich summary of the synchronized execution timeline."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", execution.title)
    header.add_row("Provider", execution.provider)
    header.add_row("Segments", str(execution.segment_count))
    header.add_row("Sample rate", f"{execution.sample_rate} Hz")
    header.add_row(
        "Actual duration", _human_duration(execution.total_actual_duration_seconds)
    )
    header.add_row("Synchronized", "yes" if execution.is_synchronized else "no")

    segments = Table(box=None, pad_edge=False, show_edge=False)
    segments.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    segments.add_column("Est span", style="dim", no_wrap=True)
    segments.add_column("Actual span", style="magenta", no_wrap=True)
    segments.add_column("Words", style="yellow", justify="right", no_wrap=True)
    segments.add_column("Audio", style="white")
    for seg in execution.segments:
        est = (
            f"{seg.timing.estimated_start_seconds:.1f}-"
            f"{seg.timing.estimated_end_seconds:.1f}s"
        )
        act = (
            f"{seg.timing.actual_start_seconds:.1f}-"
            f"{seg.timing.actual_end_seconds:.1f}s"
        )
        segments.add_row(
            str(seg.index), est, act, str(len(seg.word_timings)), seg.audio_file.name
        )

    console.print(
        Panel(
            header,
            title="[bold green]Timeline synchronized[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(segments)
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def synchronize(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to synchronize.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Write the execution timeline to a JSON file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit path for the execution timeline JSON "
        "(default: output/<slug>.execution_timeline.json).",
    ),
) -> None:
    """Build the full pipeline and merge it into a validated ExecutionTimeline."""
    configure_logging()
    project = _load_project_or_exit(project_file)

    try:
        timeline = TimelineBuilderService().build(project)
        package = NarrationService().prepare(timeline)
        generated = SpeechSynthesisService().synthesize(package)
    except SpeechProviderUnavailableError as exc:
        console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
    except (TimelineBuildError, NarrationPreparationError, SpeechSynthesisError) as exc:
        console.print(f"[bold red]Pipeline failed before synchronization:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error before synchronization")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    synchronizer = TimelineSynchronizer()
    try:
        execution = synchronizer.synchronize(timeline, package, generated)
        saved_to = synchronizer.save(execution, destination=output) if save else None
    except TimelineSynchronizationError as exc:
        console.print(f"[bold red]Synchronization failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SYNC_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error during synchronization")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_execution_summary(execution, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 7A: gameplay planner (Video Composer, part 1)
# --------------------------------------------------------------------------- #
def _build_execution_timeline(project: Project) -> ExecutionTimeline:
    """Run the full pipeline (timeline -> narration -> speech -> sync) for ``project``.

    Raises the underlying service exceptions; the caller maps them to exit codes.
    """
    timeline = TimelineBuilderService().build(project)
    package = NarrationService().prepare(timeline)
    generated = SpeechSynthesisService().synthesize(package)
    return TimelineSynchronizer().synchronize(timeline, package, generated)


def _render_plan_summary(plan: EditPlan, saved_to: Path | None) -> None:
    """Render a Rich summary of the edit plan."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", plan.title)
    header.add_row("Segments", str(plan.segment_count))
    header.add_row("Source videos", str(len(plan.source_videos)))
    header.add_row("Total duration", _human_duration(plan.total_duration_seconds))

    segments = Table(box=None, pad_edge=False, show_edge=False)
    segments.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    segments.add_column("Target", style="magenta", no_wrap=True)
    segments.add_column("Source clip", style="white")
    segments.add_column("Src span", style="yellow", no_wrap=True)
    segments.add_column("Visual", style="dim")
    for seg in plan.segments:
        segments.add_row(
            str(seg.index),
            f"{seg.target_start_seconds:.1f}-{seg.target_end_seconds:.1f}s",
            seg.source_file.name,
            f"{seg.source_start_seconds:.1f}-{seg.source_end_seconds:.1f}s",
            _truncate(seg.narration_visual) if seg.narration_visual else "-",
        )

    console.print(
        Panel(
            header,
            title="[bold green]Edit plan built[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(segments)
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def plan(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to plan gameplay for.",
    ),
    gameplay: Path | None = typer.Option(
        None,
        "--gameplay",
        help="Gameplay file or directory (default: the gameplay/raw directory).",
    ),
    from_execution: Path | None = typer.Option(
        None,
        "--from-execution",
        help="Load a saved execution_timeline.json instead of running the pipeline.",
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Write the edit plan to a JSON file."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Explicit path for the edit plan JSON (default: output/<slug>.edit_plan.json).",
    ),
) -> None:
    """Plan which gameplay to show under each narration segment (Feature 7A)."""
    configure_logging()

    # Obtain the ExecutionTimeline: either loaded, or built via the full pipeline.
    if from_execution is not None:
        if not from_execution.is_file():
            console.print(f"[bold red]Execution timeline not found:[/bold red] {from_execution}")
            raise typer.Exit(code=ExitCode.FILE_NOT_FOUND)
        try:
            execution = ExecutionTimeline.model_validate_json(
                from_execution.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            console.print(f"[bold red]Invalid execution timeline:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.VALIDATION_ERROR)
    else:
        project = _load_project_or_exit(project_file)
        try:
            execution = _build_execution_timeline(project)
        except SpeechProviderUnavailableError as exc:
            console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
        except (
            TimelineBuildError,
            NarrationPreparationError,
            SpeechSynthesisError,
            TimelineSynchronizationError,
        ) as exc:
            console.print(f"[bold red]Pipeline failed before planning:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.SPEECH_ERROR)
        except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
            logger.exception("Unexpected error building execution timeline")
            console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.UNEXPECTED)

    planner = GameplayPlanner()
    try:
        edit_plan = planner.plan(execution, gameplay_sources=gameplay)
        saved_to = planner.save(edit_plan, destination=output) if save else None
    except EditPlanError as exc:
        console.print(f"[bold red]Could not build edit plan:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PLAN_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while planning gameplay")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_plan_summary(edit_plan, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 7B: video renderer (Video Composer, part 2)
# --------------------------------------------------------------------------- #
def _render_result_summary(result: RenderResult) -> None:
    """Render a Rich summary of a completed render."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Segments", str(result.segment_count))
    table.add_row("Input clips", str(result.input_clip_count))
    table.add_row("Output duration", _human_duration(result.duration_seconds))
    table.add_row("Video codec", result.video_codec)
    table.add_row("Audio codec", result.audio_codec)
    table.add_row("Resolution", result.resolution)
    table.add_row("FPS", f"{result.fps:.3g}")
    table.add_row("Render time", f"{result.elapsed_seconds:.1f} s")
    speed = result.render_speed
    table.add_row("Render speed", f"{speed:.2f}x realtime" if speed is not None else "-")

    console.print(
        Panel(
            table,
            title="[bold green]Video rendered[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(f"[dim]Output {result.output_file}[/dim]")


def _edit_plan_or_exit(
    project_file: Path, gameplay: Path | None, from_plan: Path | None
) -> EditPlan:
    """Obtain an EditPlan: load one, or build it via the full pipeline + planner."""
    if from_plan is not None:
        if not from_plan.is_file():
            console.print(f"[bold red]Edit plan not found:[/bold red] {from_plan}")
            raise typer.Exit(code=ExitCode.FILE_NOT_FOUND)
        try:
            return EditPlan.model_validate_json(from_plan.read_text(encoding="utf-8"))
        except ValueError as exc:
            console.print(f"[bold red]Invalid edit plan:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.VALIDATION_ERROR)

    project = _load_project_or_exit(project_file)
    try:
        execution = _build_execution_timeline(project)
        return GameplayPlanner().plan(execution, gameplay_sources=gameplay)
    except SpeechProviderUnavailableError as exc:
        console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
    except (
        TimelineBuildError,
        NarrationPreparationError,
        SpeechSynthesisError,
        TimelineSynchronizationError,
    ) as exc:
        console.print(f"[bold red]Pipeline failed before render:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_ERROR)
    except EditPlanError as exc:
        console.print(f"[bold red]Could not build edit plan:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PLAN_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error building edit plan")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)


@app.command()
def render(
    project_file: Path = typer.Argument(
        ...,
        help="Path to the project JSON file to render.",
    ),
    gameplay: Path | None = typer.Option(
        None, "--gameplay", help="Gameplay file or directory (default: gameplay/raw)."
    ),
    from_plan: Path | None = typer.Option(
        None, "--from-plan", help="Render a saved edit_plan.json instead of running the pipeline."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate + generate the FFmpeg command/filtergraph without rendering."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output video path (default: edited/<slug>.mp4)."
    ),
) -> None:
    """Compile the edit plan into a playable MP4 via one FFmpeg run (Feature 7B)."""
    configure_logging()
    edit_plan = _edit_plan_or_exit(project_file, gameplay, from_plan)

    renderer = VideoRenderer()
    if dry_run:
        try:
            command, destination = renderer.prepare(edit_plan, output=output)
            command_path, filtergraph_path = renderer.save_command(command)
        except RenderError as exc:
            console.print(f"[bold red]Validation failed:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.RENDER_ERROR)
        console.print("[green]OK[/green] Validation")
        console.print("[green]OK[/green] Filtergraph generated")
        console.print("[green]OK[/green] FFmpeg command generated")
        console.print(f"[dim]Command:    {command_path}[/dim]")
        console.print(f"[dim]Filtergraph:{filtergraph_path}[/dim]")
        console.print(f"Output would be: [bold]{destination}[/bold]  (no rendering performed)")
        raise typer.Exit(code=ExitCode.SUCCESS)

    try:
        result = renderer.render(edit_plan, output=output)
    except RenderError as exc:
        console.print(f"[bold red]Render failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.RENDER_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while rendering")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_result_summary(result)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 8A: subtitle generator (captions.srt)
# --------------------------------------------------------------------------- #
def _execution_or_exit(
    project_file: Path, from_execution: Path | None
) -> ExecutionTimeline:
    """Load an ExecutionTimeline, or build one via the full pipeline."""
    if from_execution is not None:
        if not from_execution.is_file():
            console.print(f"[bold red]Execution timeline not found:[/bold red] {from_execution}")
            raise typer.Exit(code=ExitCode.FILE_NOT_FOUND)
        try:
            return ExecutionTimeline.model_validate_json(
                from_execution.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            console.print(f"[bold red]Invalid execution timeline:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.VALIDATION_ERROR)

    project = _load_project_or_exit(project_file)
    try:
        return _build_execution_timeline(project)
    except SpeechProviderUnavailableError as exc:
        console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
    except (
        TimelineBuildError,
        NarrationPreparationError,
        SpeechSynthesisError,
        TimelineSynchronizationError,
    ) as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error building execution timeline")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)


def _render_subtitle_summary(track: SubtitleTrack, saved_to: Path | None) -> None:
    """Render a Rich summary of the generated subtitle track."""
    header = Table(show_header=False, box=None, pad_edge=False)
    header.add_column("Field", style="bold cyan", no_wrap=True)
    header.add_column("Value", style="white")
    header.add_row("Title", track.title)
    header.add_row("Cues", str(track.cue_count))
    header.add_row("Format", track.format)
    header.add_row("Total duration", _human_duration(track.total_duration_seconds))

    preview = Table(box=None, pad_edge=False, show_edge=False)
    preview.add_column("#", style="bold cyan", justify="right", no_wrap=True)
    preview.add_column("Span", style="magenta", no_wrap=True)
    preview.add_column("Text", style="white")
    for cue in track.cues[:5]:
        preview.add_row(
            str(cue.index),
            f"{cue.start_seconds:.1f}-{cue.end_seconds:.1f}s",
            _truncate(cue.text),
        )

    console.print(
        Panel(
            header,
            title="[bold green]Subtitles generated[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(preview)
    if track.cue_count > 5:
        console.print(f"[dim]... and {track.cue_count - 5} more cue(s)[/dim]")
    if saved_to is not None:
        # save() returns the .ass (working); the .srt export is its sibling.
        console.print(f"[dim]Saved {saved_to} (+ {saved_to.with_suffix('.srt').name})[/dim]")


@app.command()
def subtitle(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file to generate subtitles for."
    ),
    from_execution: Path | None = typer.Option(
        None, "--from-execution", help="Load a saved execution_timeline.json (skips the pipeline)."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Write the captions (.ass working + .srt export)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Base output path (default: edited/<slug>); writes .ass + .srt."
    ),
) -> None:
    """Generate word-timed captions (.ass working + .srt export) (Feature 8A)."""
    configure_logging()
    execution = _execution_or_exit(project_file, from_execution)

    generator = SubtitleGenerator()
    try:
        track = generator.generate(execution)
        saved_to = generator.save(track, destination=output) if save else None
    except SubtitleGenerationError as exc:
        console.print(f"[bold red]Could not generate subtitles:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SUBTITLE_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while generating subtitles")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_subtitle_summary(track, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 8B: subtitle burn (video + captions -> video_with_subtitles.mp4)
# --------------------------------------------------------------------------- #
def _burn_inputs(
    project_file: Path,
    gameplay: Path | None,
    from_video: Path | None,
    from_subtitles: Path | None,
) -> tuple[Path, Path]:
    """Resolve (video, subtitles): use provided files, else run the full pipeline."""
    if from_video is not None and from_subtitles is not None:
        return from_video, from_subtitles

    project = _load_project_or_exit(project_file)
    try:
        execution = _build_execution_timeline(project)
        plan = GameplayPlanner().plan(execution, gameplay_sources=gameplay)
        render_result = VideoRenderer().render(plan)
        track = SubtitleGenerator().generate(execution)
        ass_path = SubtitleGenerator().save(track)
    except SpeechProviderUnavailableError as exc:
        console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
    except (
        TimelineBuildError,
        NarrationPreparationError,
        SpeechSynthesisError,
        TimelineSynchronizationError,
    ) as exc:
        console.print(f"[bold red]Pipeline failed before burn:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SPEECH_ERROR)
    except EditPlanError as exc:
        console.print(f"[bold red]Could not build edit plan:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PLAN_ERROR)
    except RenderError as exc:
        console.print(f"[bold red]Render failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.RENDER_ERROR)
    except SubtitleGenerationError as exc:
        console.print(f"[bold red]Could not generate subtitles:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.SUBTITLE_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error building burn inputs")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    return render_result.output_file, ass_path


@app.command()
def burn(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file (used when inputs aren't provided)."
    ),
    from_video: Path | None = typer.Option(
        None, "--from-video", help="Existing base video to burn onto."
    ),
    from_subtitles: Path | None = typer.Option(
        None, "--from-subtitles", help="Existing .ass (or .srt) captions to burn."
    ),
    gameplay: Path | None = typer.Option(
        None, "--gameplay", help="Gameplay file/dir when running the full pipeline."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate + build the FFmpeg command without burning."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path (default: edited/<video>.subtitled.mp4)."
    ),
) -> None:
    """Burn captions onto the video -> video_with_subtitles.mp4 (Feature 8B)."""
    configure_logging()
    video, subtitles = _burn_inputs(project_file, gameplay, from_video, from_subtitles)

    renderer = SubtitleRenderer()
    if dry_run:
        try:
            argv, out, _ass, _cues = renderer.prepare(
                video, subtitles, output, ass_dir=get_settings().edited_dir
            )
        except SubtitleBurnError as exc:
            console.print(f"[bold red]Validation failed:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.BURN_ERROR)
        cmd_path = renderer.save_command(
            argv, out.with_name(f"{out.stem}.burn_command.txt")
        )
        console.print("[green]OK[/green] Validation")
        console.print("[green]OK[/green] ASS built")
        console.print("[green]OK[/green] FFmpeg command generated")
        console.print(f"[dim]Command: {cmd_path}[/dim]")
        console.print(f"Output would be: [bold]{out}[/bold]  (no burning performed)")
        raise typer.Exit(code=ExitCode.SUCCESS)

    try:
        result = renderer.burn(video, subtitles, output=output)
    except SubtitleBurnError as exc:
        console.print(f"[bold red]Burn failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.BURN_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while burning subtitles")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_result_summary(result)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 9A: thumbnail planner (thumbnail_plan.json)
# --------------------------------------------------------------------------- #
def _thumbnail_inputs(
    project_file: Path,
    gameplay: Path | None,
    from_execution: Path | None,
    from_video: Path | None,
) -> tuple[ExecutionTimeline, Path, str | None]:
    """Resolve (execution, base video, thumbnail_prompt) for the planner."""
    project: Project | None = None
    if from_execution is not None and from_video is not None:
        execution = _execution_or_exit(project_file, from_execution)
        video = from_video
    else:
        project = _load_project_or_exit(project_file)
        try:
            execution = _build_execution_timeline(project)
            edit_plan = GameplayPlanner().plan(execution, gameplay_sources=gameplay)
            video = VideoRenderer().render(edit_plan).output_file
        except SpeechProviderUnavailableError as exc:
            console.print(f"[bold red]Speech provider unavailable:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.SPEECH_UNAVAILABLE)
        except (
            TimelineBuildError,
            NarrationPreparationError,
            SpeechSynthesisError,
            TimelineSynchronizationError,
        ) as exc:
            console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.SPEECH_ERROR)
        except EditPlanError as exc:
            console.print(f"[bold red]Could not build edit plan:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.PLAN_ERROR)
        except RenderError as exc:
            console.print(f"[bold red]Render failed:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.RENDER_ERROR)
        except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
            logger.exception("Unexpected error building thumbnail inputs")
            console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.UNEXPECTED)

    # thumbnail_prompt only lives on the Project; load it if we don't have it.
    if project is None:
        project = _load_project_or_exit(project_file)
    return execution, video, project.thumbnail_prompt


def _render_thumbnail_plan_summary(plan: ThumbnailPlan, saved_to: Path | None) -> None:
    """Render a Rich summary of the thumbnail plan."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Title", plan.title_text)
    table.add_row(
        "Frame",
        f"{plan.target_frame_timestamp_seconds:.2f}s ({plan.selection_reason.value})",
    )
    table.add_row("Highlight", plan.highlight_text or "-")
    table.add_row("Badge", plan.badge_text or "-")
    table.add_row("Target size", f"{plan.target_width}x{plan.target_height}")
    table.add_row("Source video", plan.source_video.name)

    console.print(
        Panel(
            table,
            title="[bold green]Thumbnail planned[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command(name="thumbnail-plan")
def thumbnail_plan(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file to plan a thumbnail for."
    ),
    from_execution: Path | None = typer.Option(
        None, "--from-execution", help="Load a saved execution_timeline.json (skips pipeline)."
    ),
    from_video: Path | None = typer.Option(
        None, "--from-video", help="Existing base video to grab the frame from."
    ),
    gameplay: Path | None = typer.Option(
        None, "--gameplay", help="Gameplay file/dir when running the full pipeline."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Write the thumbnail plan to a JSON file."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Explicit path (default: edited/<slug>.thumbnail_plan.json)."
    ),
) -> None:
    """Plan a thumbnail (frame/title/highlight/style) -> thumbnail_plan.json (Feature 9A)."""
    configure_logging()
    execution, video, thumbnail_prompt = _thumbnail_inputs(
        project_file, gameplay, from_execution, from_video
    )

    planner = ThumbnailPlanner()
    try:
        plan = planner.plan(execution, video, thumbnail_prompt=thumbnail_prompt)
        saved_to = planner.save(plan, destination=output) if save else None
    except ThumbnailPlanError as exc:
        console.print(f"[bold red]Could not plan thumbnail:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.THUMBNAIL_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while planning thumbnail")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_thumbnail_plan_summary(plan, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 9B: thumbnail renderer (thumbnail.png)
# --------------------------------------------------------------------------- #
def _thumbnail_plan_or_exit(
    project_file: Path,
    gameplay: Path | None,
    from_execution: Path | None,
    from_video: Path | None,
    from_plan: Path | None,
) -> ThumbnailPlan:
    """Obtain a ThumbnailPlan: load one, or build it (planner, maybe full pipeline)."""
    if from_plan is not None:
        if not from_plan.is_file():
            console.print(f"[bold red]Thumbnail plan not found:[/bold red] {from_plan}")
            raise typer.Exit(code=ExitCode.FILE_NOT_FOUND)
        try:
            return ThumbnailPlan.model_validate_json(from_plan.read_text(encoding="utf-8"))
        except ValueError as exc:
            console.print(f"[bold red]Invalid thumbnail plan:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.VALIDATION_ERROR)

    execution, video, thumbnail_prompt = _thumbnail_inputs(
        project_file, gameplay, from_execution, from_video
    )
    try:
        return ThumbnailPlanner().plan(execution, video, thumbnail_prompt=thumbnail_prompt)
    except ThumbnailPlanError as exc:
        console.print(f"[bold red]Could not plan thumbnail:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.THUMBNAIL_ERROR)


def _render_thumbnail_summary(result: ThumbnailResult) -> None:
    """Render a Rich summary of a rendered thumbnail."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Size", f"{result.width}x{result.height}")
    table.add_row(
        "Frame",
        f"{result.actual_frame_timestamp_seconds:.2f}s "
        f"(requested {result.requested_frame_timestamp_seconds:.2f}s)",
    )
    table.add_row("Render time", f"{result.elapsed_seconds:.2f} s")
    console.print(
        Panel(
            table,
            title="[bold green]Thumbnail rendered[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    console.print(f"[dim]Output {result.output_file}[/dim]")


@app.command()
def thumbnail(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file to render a thumbnail for."
    ),
    from_plan: Path | None = typer.Option(
        None, "--from-plan", help="Render a saved thumbnail_plan.json (fast)."
    ),
    from_execution: Path | None = typer.Option(
        None, "--from-execution", help="Load a saved execution_timeline.json (skips pipeline)."
    ),
    from_video: Path | None = typer.Option(
        None, "--from-video", help="Existing base video to grab the frame from."
    ),
    gameplay: Path | None = typer.Option(
        None, "--gameplay", help="Gameplay file/dir when running the full pipeline."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate + build the layout without rendering the image."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output PNG path (default: edited/<slug>.thumbnail.png)."
    ),
) -> None:
    """Render a thumbnail.png from a thumbnail plan (Feature 9B)."""
    configure_logging()
    plan = _thumbnail_plan_or_exit(
        project_file, gameplay, from_execution, from_video, from_plan
    )

    renderer = ThumbnailRenderer()
    if dry_run:
        try:
            layout = renderer._layout(plan)  # noqa: SLF001 — internal layout for dry-run
            out = renderer.output_path(plan, output)
            layout_file = renderer.save_layout(layout, layout_path(out))
        except ThumbnailRenderError as exc:
            console.print(f"[bold red]Validation failed:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.THUMBNAIL_RENDER_ERROR)
        console.print("[green]OK[/green] Validation (plan + video + font)")
        console.print("[green]OK[/green] Layout built")
        console.print(f"[dim]Layout: {layout_file}[/dim]")
        console.print(f"Output would be: [bold]{out}[/bold]  (no rendering performed)")
        raise typer.Exit(code=ExitCode.SUCCESS)

    try:
        result = renderer.render(plan, output=output)
    except ThumbnailRenderError as exc:
        console.print(f"[bold red]Thumbnail render failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.THUMBNAIL_RENDER_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while rendering thumbnail")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_thumbnail_summary(result)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 10: YouTube upload (upload_result.json)
# --------------------------------------------------------------------------- #
def _project_slug(project: Project) -> str:
    """Filesystem slug from a project (matches the service slugs)."""
    base = (project.project_id or project.title).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", base).strip("_") or "project"


def _render_upload_summary(result: UploadResult, saved_to: Path | None) -> None:
    """Render a Rich summary of an upload."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Video ID", result.video_id)
    table.add_row("URL", result.url)
    table.add_row("Privacy", result.privacy)
    table.add_row("Status", result.status)
    table.add_row("Thumbnail", "uploaded" if result.thumbnail_uploaded else "skipped")
    table.add_row("Upload time", f"{result.elapsed_seconds:.1f} s")
    console.print(
        Panel(
            table,
            title="[bold green]Uploaded to YouTube[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    if saved_to is not None:
        console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def upload(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file (metadata source)."
    ),
    video: Path | None = typer.Option(
        None, "--video", help="Video to upload (default: edited/<slug>.subtitled.mp4)."
    ),
    thumbnail: Path | None = typer.Option(
        None, "--thumbnail", help="Thumbnail (default: edited/<slug>.thumbnail.png)."
    ),
    privacy: str | None = typer.Option(
        None, "--privacy", help="private | unlisted | public (default: config)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate + build the request without uploading."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output path (default: edited/<slug>.upload_result.json)."
    ),
) -> None:
    """Upload the video + thumbnail to YouTube -> upload_result.json (Feature 10)."""
    configure_logging()
    project = _load_project_or_exit(project_file)
    settings = get_settings()
    slug = _project_slug(project)
    video = video or settings.edited_dir / f"{slug}.subtitled.mp4"
    thumbnail = thumbnail or settings.edited_dir / f"{slug}.thumbnail.png"

    uploader = YouTubeUploader()
    if dry_run:
        try:
            request = uploader.prepare(project, video, thumbnail, privacy=privacy)
        except UploadRequestError as exc:
            console.print(f"[bold red]Invalid upload request:[/bold red] {exc}")
            raise typer.Exit(code=ExitCode.UPLOAD_ERROR)
        req_path = uploader.save_request(
            request, settings.edited_dir / f"{slug}.upload_request.json"
        )
        console.print(f"[green]OK[/green] Video exists ({video})")
        console.print(f"[green]OK[/green] Thumbnail exists ({thumbnail})")
        console.print(f"[green]OK[/green] Privacy = {request['privacy']}")
        console.print(
            f"[green]OK[/green] Category = {request['body']['snippet']['categoryId']}"
        )
        console.print(
            f"Credentials present = {'yes' if uploader.credentials_present() else 'no'}"
        )
        console.print(f"[dim]Title: {request['body']['snippet']['title']}[/dim]")
        console.print(f"[dim]Request: {req_path}[/dim]  (no upload performed)")
        raise typer.Exit(code=ExitCode.SUCCESS)

    try:
        result = uploader.upload(project, video, thumbnail, privacy=privacy)
        saved_to = uploader.save(result, destination=output)
    except UploadAuthError as exc:
        console.print(f"[bold red]Authentication failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UPLOAD_AUTH_ERROR)
    except (UploadRequestError, UploadError) as exc:
        console.print(f"[bold red]Upload failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UPLOAD_ERROR)
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error while uploading")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    _render_upload_summary(result, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


# --------------------------------------------------------------------------- #
# Feature 11: project runner (full pipeline in one process)
# --------------------------------------------------------------------------- #

# The stages `run` always executes (upload is appended only with --upload).
_RUN_STAGE_PLAN: tuple[RunStage, ...] = (
    RunStage.TIMELINE,
    RunStage.NARRATION,
    RunStage.SPEECH,
    RunStage.SYNC,
    RunStage.PLAN,
    RunStage.RENDER,
    RunStage.SUBTITLE,
    RunStage.BURN,
    RunStage.THUMBNAIL_PLAN,
    RunStage.THUMBNAIL,
)

_GAMEPLAY_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi")

# Maps each service exception to its stage's exit code. Order matters: the most
# specific type must precede its base (UploadAuthError before UploadError).
_EXIT_CODE_BY_EXCEPTION: tuple[tuple[type[BaseException], ExitCode], ...] = (
    (TimelineBuildError, ExitCode.TIMELINE_ERROR),
    (NarrationPreparationError, ExitCode.NARRATION_ERROR),
    (SpeechProviderUnavailableError, ExitCode.SPEECH_UNAVAILABLE),
    (SpeechSynthesisError, ExitCode.SPEECH_ERROR),
    (TimelineSynchronizationError, ExitCode.SYNC_ERROR),
    (EditPlanError, ExitCode.PLAN_ERROR),
    (RenderError, ExitCode.RENDER_ERROR),
    (SubtitleGenerationError, ExitCode.SUBTITLE_ERROR),
    (SubtitleBurnError, ExitCode.BURN_ERROR),
    (ThumbnailPlanError, ExitCode.THUMBNAIL_ERROR),
    (ThumbnailRenderError, ExitCode.THUMBNAIL_RENDER_ERROR),
    (UploadAuthError, ExitCode.UPLOAD_AUTH_ERROR),
    (UploadError, ExitCode.UPLOAD_ERROR),  # base — keep after UploadAuthError
)


def _exit_code_for(exc: BaseException) -> ExitCode:
    """Map a stage failure to the same exit code its standalone command uses."""
    for exc_type, code in _EXIT_CODE_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return code
    return ExitCode.UNEXPECTED


def _count_gameplay_sources(gameplay: Path | None) -> tuple[Path, int]:
    """Resolve the gameplay source (default: gameplay/raw) and count videos."""
    source = gameplay or get_settings().gameplay_raw_dir
    if source.is_file():
        return source, 1
    if source.is_dir():
        count = sum(
            1 for p in source.iterdir() if p.suffix.lower() in _GAMEPLAY_EXTENSIONS
        )
        return source, count
    return source, 0


def _render_run_summary(result: RunResult, saved_to: Path) -> None:
    """Render a Rich summary of a full pipeline run.

    Short fields go in the table; the long file paths are printed as dim lines
    below (a fixed-width table cell would wrap + truncate them with a glyph the
    Windows console cannot encode).
    """
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Stages", f"{len(result.completed_stages)} completed")
    if result.uploaded and result.upload is not None:
        table.add_row("Upload", f"{result.upload.url} ({result.upload.privacy})")
    elif result.requested_upload:
        table.add_row("Upload", "requested - failed")
    else:
        table.add_row("Upload", "skipped (no --upload)")
    table.add_row("Elapsed", _human_duration(result.total_elapsed_seconds))
    console.print(
        Panel(
            table,
            title="[bold green]Pipeline run complete[/bold green]",
            subtitle="[dim]AI Creator Studio[/dim]",
            border_style="green",
            expand=False,
        )
    )
    timings = "  ".join(
        f"{stage}={seconds:.1f}s" for stage, seconds in result.stage_timings.items()
    )
    console.print(f"[dim]{timings}[/dim]")
    console.print(f"[dim]Video     {result.video_path}[/dim]")
    console.print(f"[dim]Thumbnail {result.thumbnail_path}[/dim]")
    console.print(f"[dim]Saved to {saved_to}[/dim]")


@app.command()
def run(
    project_file: Path = typer.Argument(
        ..., help="Path to the project JSON file to run end-to-end."
    ),
    gameplay: Path | None = typer.Option(
        None,
        "--gameplay",
        help="Gameplay file or directory (default: the gameplay/raw directory).",
    ),
    upload: bool = typer.Option(
        False,
        "--upload",
        help="Also upload to YouTube after the thumbnail (default: stop after thumbnail).",
    ),
    privacy: str | None = typer.Option(
        None, "--privacy", help="private | unlisted | public for --upload (default: config)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report the planned stages + inputs without executing."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Path for run_result.json (default: edited/<slug>.run_result.json).",
    ),
) -> None:
    """Run the whole pipeline in one process -> run_result.json (Feature 11)."""
    configure_logging()
    project = _load_project_or_exit(project_file)

    if dry_run:
        source, count = _count_gameplay_sources(gameplay)
        stages = list(_RUN_STAGE_PLAN) + ([RunStage.UPLOAD] if upload else [])
        console.print(
            f"[bold]Would run {len(stages)} stages:[/bold] "
            + " -> ".join(s.value for s in stages)
        )
        console.print(
            f"[green]OK[/green] Gameplay source: {source} ({count} video file(s) found)"
        )
        if upload:
            creds = YouTubeUploader().credentials_present()
            console.print(
                f"Upload requested; credentials present = {'yes' if creds else 'no'}"
            )
        else:
            console.print("[dim]Upload not requested (stops after thumbnail).[/dim]")
        console.print("[dim](no work performed)[/dim]")
        raise typer.Exit(code=ExitCode.SUCCESS)

    def _announce(step: RunStage) -> None:
        console.print(f"[dim]-> {step.value}...[/dim]")

    try:
        result = WorkflowManager().run(
            project,
            gameplay_sources=gameplay,
            upload=upload,
            privacy=privacy,
            result_path=output,
            on_stage=_announce,
        )
    except PipelineError as exc:
        cause = exc.__cause__ or exc
        console.print(
            f"[bold red]Pipeline failed at stage '{exc.stage.value}':[/bold red] {cause}"
        )
        raise typer.Exit(code=_exit_code_for(cause))
    except Exception as exc:  # noqa: BLE001 — final safety net for the CLI.
        logger.exception("Unexpected error during run")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    saved_to = output or get_settings().edited_dir / f"{_project_slug(project)}.run_result.json"
    _render_run_summary(result, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


if __name__ == "__main__":
    app()
