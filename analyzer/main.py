"""Command-line entry point for the Gameplay Analyzer.

Commands::

    python -m analyzer.main build-templates            # Slice 2A

The CLI is intentionally *thin*: it parses arguments, calls the workflow,
renders a Rich summary, and maps failures to distinct process exit codes. All
business logic lives in :mod:`analyzer.preprocess` / :mod:`analyzer.workflow`.

The analyzer is a separate program from ``backend.main`` and has its own small
exit-code space.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyzer.config import get_analyzer_settings
from analyzer.models import FramesManifest, GameplayAnalysis, HandReading, TemplateLibrary

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Gameplay Analyzer - Clash Royale CV analysis (Phase 2).",
)


@app.callback()
def _main() -> None:
    """Gameplay Analyzer CLI.

    A no-op group callback so the app keeps explicit subcommand names (Typer
    otherwise collapses a single-command app to no subcommand). Future analyzer
    commands register alongside ``build-templates``.
    """


class ExitCode(IntEnum):
    """Process exit codes for the analyzer CLI."""

    SUCCESS = 0
    UNEXPECTED = 1
    ASSET_DIR_NOT_FOUND = 2
    BUILD_ERROR = 3
    VIDEO_NOT_FOUND = 4
    PROBE_ERROR = 5
    EXTRACTION_ERROR = 6
    CALIBRATION_ERROR = 7
    DETECT_ERROR = 8
    ANALYZE_ERROR = 9


def _render_template_summary(library: TemplateLibrary, saved_to: Path | None) -> None:
    """Render a Rich summary of a template-library build (ASCII-only)."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for variant, count in sorted(library.variant_counts.items()):
        table.add_row(variant.title(), str(count))
    table.add_row("Total", str(library.card_count))
    table.add_row("Average size", f"{library.average_width} x {library.average_height}")
    if library.largest_template and library.smallest_template:
        table.add_row("Largest", library.largest_template)
        table.add_row("Smallest", library.smallest_template)
    table.add_row("Skipped", str(len(library.skipped)))
    console.print(
        Panel(
            table,
            title="[bold green]Templates built[/bold green]",
            subtitle="[dim]Gameplay Analyzer[/dim]",
            border_style="green",
            expand=False,
        )
    )
    missing = [t.slug for t in library.templates if not t.has_base_match]
    if missing:
        console.print(f"[yellow]No base-card match:[/yellow] {', '.join(missing)}")
    for note in library.skipped:
        console.print(f"[dim]skipped {note}[/dim]")
    if saved_to is not None:
        console.print(f"[dim]Manifest {saved_to}[/dim]")
    else:
        console.print("[dim](dry run - nothing written)[/dim]")


def _render_frames_summary(manifest: FramesManifest, saved_to: Path | None) -> None:
    """Render a Rich summary of a frame-extraction run (ASCII-only)."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Video", manifest.video)
    table.add_row("Source", f"{manifest.source_fps:g} fps, {manifest.duration_seconds:.1f} s, "
                            f"{manifest.width}x{manifest.height}")
    table.add_row("Sample fps", f"{manifest.sample_fps:g}")
    table.add_row("Format", manifest.image_format)
    table.add_row("Frames", str(manifest.frame_count))
    if manifest.frames:
        table.add_row(
            "Span",
            f"{manifest.first_timestamp:.2f}-{manifest.last_timestamp:.2f} s "
            f"(~{manifest.average_spacing:.3f} s apart)",
        )
    console.print(
        Panel(
            table,
            title="[bold green]Frames extracted[/bold green]",
            subtitle="[dim]Gameplay Analyzer[/dim]",
            border_style="green",
            expand=False,
        )
    )
    if saved_to is not None:
        console.print(f"[dim]Frames dir {manifest.frames_dir}[/dim]")
        console.print(f"[dim]Manifest   {saved_to}[/dim]")
    else:
        console.print(f"[dim]Would run: {manifest.ffmpeg_command}[/dim]")
        console.print("[dim](dry run - nothing written)[/dim]")


@app.command(name="extract-frames")
def extract_frames(
    video: Path = typer.Argument(..., help="Path to the recording to sample."),
    sample_fps: float | None = typer.Option(
        None, "--sample-fps", help="Frames per second to sample (default: config, 5.0)."
    ),
    image_format: str | None = typer.Option(
        None, "--format", help="png | jpg (default: config, png)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Probe + report the plan without extracting."
    ),
    keep_existing: bool = typer.Option(
        False, "--keep-existing", help="Reuse a valid existing cache instead of re-extracting."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Manifest path (default: <frames_dir>/frames_manifest.json)."
    ),
) -> None:
    """Sample a recording into cached frames + a manifest (Slice 2B)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Imported lazily so `import analyzer.main` stays light.
    from analyzer.preprocess.frame_extractor import (
        FrameExtractionError,
        FrameExtractor,
        FrameProbeError,
        VideoNotFoundError,
    )

    settings = get_analyzer_settings()
    if keep_existing:
        # --keep-existing is a per-invocation override of the config default.
        settings = settings.model_copy(update={"frame_keep_existing": True})
    if not dry_run:
        settings.ensure_directories()

    extractor = FrameExtractor(settings)
    try:
        manifest = extractor.extract(
            video, sample_fps=sample_fps, image_format=image_format, dry_run=dry_run
        )
    except VideoNotFoundError as exc:
        console.print(f"[bold red]Video not found:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.VIDEO_NOT_FOUND)
    except FrameProbeError as exc:
        console.print(f"[bold red]Could not probe video:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PROBE_ERROR)
    except FrameExtractionError as exc:
        console.print(f"[bold red]Frame extraction failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.EXTRACTION_ERROR)
    except Exception as exc:  # noqa: BLE001 - final safety net for the CLI.
        logger.exception("Unexpected error while extracting frames")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    saved_to = None
    if not dry_run:
        # A reused cache is already saved; only re-save a freshly built manifest.
        saved_to = extractor.save(manifest, destination=output)
    _render_frames_summary(manifest, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


@app.command(name="build-templates")
def build_templates(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scan + report without writing the cache."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Manifest path (default: analyzer/cache/templates/v1/template_manifest.json).",
    ),
) -> None:
    """Preprocess bundled card art into a cached template library (Slice 2A)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Imported lazily so `import analyzer.main` works before OpenCV is installed.
    from analyzer.preprocess.build_templates import TemplateBuilder, TemplateBuildError

    settings = get_analyzer_settings()
    if not dry_run:
        settings.ensure_directories()

    builder = TemplateBuilder(settings)
    try:
        library = builder.build(dry_run=dry_run)
    except TemplateBuildError as exc:
        console.print(f"[bold red]Cannot build templates:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.ASSET_DIR_NOT_FOUND)
    except Exception as exc:  # noqa: BLE001 - final safety net for the CLI.
        logger.exception("Unexpected error while building templates")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    saved_to = None if dry_run else builder.save(library, destination=output)
    _render_template_summary(library, saved_to)
    raise typer.Exit(code=ExitCode.SUCCESS)


def _render_hand(reading: HandReading) -> None:
    """Print a hand reading as a small table (ASCII-only)."""
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Slot", style="bold cyan")
    table.add_column("Card", style="white")
    table.add_column("Variant", style="white")
    table.add_column("Score", style="white")
    table.add_column("Matched", style="white")
    for slot in reading.slots:
        table.add_row(
            str(slot.slot), slot.card or "-", slot.variant or "-",
            f"{slot.score:.3f}", "yes" if slot.matched else "no",
        )
    console.print(
        Panel(table, title="[bold green]Hand[/bold green]",
              subtitle="[dim]Gameplay Analyzer[/dim]", border_style="green", expand=False)
    )


@app.command(name="calibrate")
def calibrate(
    source: Path = typer.Argument(..., help="Image, or video (first frame is used)."),
    profile: str | None = typer.Option(None, "--profile", help="Calibration profile name."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Preview PNG path."),
) -> None:
    """Draw a profile's ROIs over a frame to visually tune it (Slice 2C)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from analyzer.calibration.profiles import CalibrationError
    from analyzer.workflow import AnalyzerWorkflow

    get_analyzer_settings().ensure_directories()
    try:
        preview = AnalyzerWorkflow().calibrate(source, profile_name=profile, output=output)
    except CalibrationError as exc:
        console.print(f"[bold red]Calibration error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.CALIBRATION_ERROR)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Could not read frame:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.DETECT_ERROR)
    console.print(f"[green]OK[/green] Calibration preview -> {preview}")
    raise typer.Exit(code=ExitCode.SUCCESS)


@app.command(name="detect-hand")
def detect_hand(
    frame: Path = typer.Argument(..., help="Frame image to read the hand from."),
    profile: str | None = typer.Option(None, "--profile", help="Calibration profile name."),
) -> None:
    """Identify the four hand cards in a single frame (Slice 2D)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from analyzer.calibration.profiles import CalibrationError
    from analyzer.template_index import TemplateIndexError
    from analyzer.workflow import AnalyzerWorkflow

    try:
        reading = AnalyzerWorkflow().detect_hand(frame, profile_name=profile)
    except CalibrationError as exc:
        console.print(f"[bold red]Calibration error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.CALIBRATION_ERROR)
    except (TemplateIndexError, FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Detection error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.DETECT_ERROR)
    _render_hand(reading)
    raise typer.Exit(code=ExitCode.SUCCESS)


@app.command(name="analyze")
def analyze(
    video: Path = typer.Argument(..., help="Recording to analyze into gameplay_analysis.json."),
    profile: str | None = typer.Option(None, "--profile", help="Calibration profile name."),
    sample_fps: float | None = typer.Option(None, "--sample-fps", help="Frames/sec to sample."),
) -> None:
    """Video -> gameplay_analysis.json via the full detector chain (2C-2E, 2H)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from analyzer.calibration.profiles import CalibrationError
    from analyzer.preprocess.frame_extractor import (
        FrameExtractionError,
        FrameProbeError,
        VideoNotFoundError,
    )
    from analyzer.template_index import TemplateIndexError
    from analyzer.workflow import AnalyzerWorkflow

    get_analyzer_settings().ensure_directories()
    try:
        analysis = AnalyzerWorkflow().analyze(video, profile_name=profile, sample_fps=sample_fps)
    except VideoNotFoundError as exc:
        console.print(f"[bold red]Video not found:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.VIDEO_NOT_FOUND)
    except FrameProbeError as exc:
        console.print(f"[bold red]Could not probe video:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.PROBE_ERROR)
    except CalibrationError as exc:
        console.print(f"[bold red]Calibration error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.CALIBRATION_ERROR)
    except (FrameExtractionError, TemplateIndexError) as exc:
        console.print(f"[bold red]Analyze failed:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.ANALYZE_ERROR)
    except Exception as exc:  # noqa: BLE001 - final safety net for the CLI.
        logger.exception("Unexpected error during analyze")
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_row("Video", analysis.video)
    table.add_row("Profile", analysis.profile_name)
    table.add_row("Frames", str(analysis.frame_count))
    table.add_row("Events", str(len(analysis.events)))
    console.print(
        Panel(table, title="[bold green]Gameplay analyzed[/bold green]",
              subtitle="[dim]Gameplay Analyzer[/dim]", border_style="green", expand=False)
    )
    from analyzer.report import format_report

    console.print(format_report(analysis))
    for warning in analysis.warnings:
        console.print(f"[yellow]note:[/yellow] {warning}")
    raise typer.Exit(code=ExitCode.SUCCESS)


@app.command(name="regression")
def regression(
    run: bool = typer.Option(
        True, "--run/--no-run",
        help="Re-analyze the videos (--run) or check existing analysis JSONs (--no-run).",
    ),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Write the baseline from the current results."
    ),
) -> None:
    """Verify the frozen deck engine hasn't regressed on the 3 validated matches."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from analyzer.regression import run_regression

    try:
        passed, report = run_regression(run=run, update=update_baseline)
    except FileNotFoundError as exc:
        console.print(f"[bold red]Regression setup error:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.VIDEO_NOT_FOUND)
    console.print(report)
    raise typer.Exit(code=ExitCode.SUCCESS if passed else ExitCode.BUILD_ERROR)


@app.command(name="completeness")
def completeness() -> None:
    """Report whether the template library covers the game's cards (2I+)."""
    from analyzer.completeness import check_completeness, format_completeness_report

    report = check_completeness(get_analyzer_settings())
    console.print(format_completeness_report(report))
    raise typer.Exit(code=ExitCode.SUCCESS if report.complete else ExitCode.BUILD_ERROR)


@app.command(name="report")
def report(
    analysis_json: Path = typer.Argument(..., help="A saved gameplay_analysis.json file."),
) -> None:
    """Print the ANALYZER REPORT for a saved analysis (Slice 2I)."""
    from analyzer.models import GameplayAnalysis
    from analyzer.report import format_report

    if not analysis_json.is_file():
        console.print(f"[bold red]Not found:[/bold red] {analysis_json}")
        raise typer.Exit(code=ExitCode.VIDEO_NOT_FOUND)
    try:
        analysis = GameplayAnalysis.model_validate_json(analysis_json.read_text(encoding="utf-8"))
    except ValueError as exc:
        console.print(f"[bold red]Invalid analysis JSON:[/bold red] {exc}")
        raise typer.Exit(code=ExitCode.UNEXPECTED)
    console.print(format_report(analysis))
    raise typer.Exit(code=ExitCode.SUCCESS)


if __name__ == "__main__":
    app()
