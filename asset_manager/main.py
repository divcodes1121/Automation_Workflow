"""Asset Manager CLI — sync, index, search, resolve, verify.

    python -m asset_manager.main sync                  # all keyless sources
    python -m asset_manager.main sync --source twemoji
    python -m asset_manager.main resolve emoji_fire
    python -m asset_manager.main search fire --category stickers
    python -m asset_manager.main verify
    python -m asset_manager.main stats
    python -m asset_manager.main sources

Output is ASCII-only (Windows cp1252 consoles).
"""

from __future__ import annotations

import logging
from enum import IntEnum

import typer

from asset_manager.manager import AssetManager

app = typer.Typer(add_completion=False, help="Auto-download and resolve editing assets.")


class ExitCode(IntEnum):
    OK = 0
    ERROR = 1
    NOT_FOUND = 2


def _log() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@app.callback()
def _main() -> None:
    """Asset Manager — keeps the editor free of hard-coded filenames."""


@app.command()
def sync(
    source: list[str] = typer.Option(
        None, "--source", "-s", help="Limit to these sources (default: all)."
    ),
    include_keyed: bool = typer.Option(
        False, "--include-keyed", help="Also run keyed sources (needs API keys in env)."
    ),
) -> None:
    """Download new assets and refresh the manifest."""
    _log()
    mgr = AssetManager()
    summary = mgr.sync(source or None, include_keyed=include_keyed)
    total = sum(summary.values())
    typer.echo(f"Synced {total} asset(s):")
    for name, n in summary.items():
        typer.echo(f"  {name:14} {n}")


@app.command("rebuild-manifest")
def rebuild_manifest() -> None:
    """Regenerate asset_manifest.json from the index."""
    _log()
    path = AssetManager().rebuild_manifest()
    typer.echo(f"Wrote {path}")


@app.command()
def resolve(
    name: str = typer.Argument(..., help="Symbolic name, e.g. emoji_fire / comic_boom."),
    pick_random: bool = typer.Option(False, "--random", help="Pick randomly among matches."),
) -> None:
    """Resolve a symbolic name to a real asset path."""
    _log()
    asset = AssetManager().resolve(name, pick_random=pick_random)
    if asset is None:
        typer.echo(f"No asset resolves '{name}'")
        raise typer.Exit(code=ExitCode.NOT_FOUND)
    typer.echo(asset.path)
    typer.echo(f"  id={asset.id} license={asset.license} tags={','.join(asset.tags)}")


@app.command()
def search(
    tags: list[str] = typer.Argument(..., help="Tags an asset must ALL have."),
    category: str = typer.Option(None, "--category", "-c", help="Restrict to a category prefix."),
    limit: int = typer.Option(25, "--limit", "-n"),
) -> None:
    """Find assets by tag(s)."""
    _log()
    results = AssetManager().search(list(tags), category=category, limit=limit)
    if not results:
        typer.echo("No matches")
        raise typer.Exit(code=ExitCode.NOT_FOUND)
    for a in results:
        typer.echo(f"{a.id:22} {a.category:18} {a.path}")


@app.command()
def random(
    category: str = typer.Argument(..., help="Category prefix, e.g. stickers or audio/bass."),
) -> None:
    """Return a random asset from a category."""
    _log()
    a = AssetManager().random(category)
    if a is None:
        typer.echo(f"No assets in category '{category}'")
        raise typer.Exit(code=ExitCode.NOT_FOUND)
    typer.echo(f"{a.id}  {a.path}")


@app.command()
def verify() -> None:
    """Drop missing/corrupt files; report duplicates."""
    _log()
    report = AssetManager().verify()
    for key in ("missing", "corrupt", "duplicates"):
        items = report.get(key, [])
        typer.echo(f"{key}: {len(items)}")
        for it in items:
            typer.echo(f"  {it}")


@app.command()
def stats() -> None:
    """Show library totals by category and source."""
    _log()
    s = AssetManager().stats()
    typer.echo(f"Total assets: {s['total']}")
    typer.echo("By source:")
    for k, v in s["by_source"].items():  # type: ignore[union-attr]
        typer.echo(f"  {k:14} {v}")
    typer.echo("By category:")
    for k, v in s["by_category"].items():  # type: ignore[union-attr]
        typer.echo(f"  {k:20} {v}")


@app.command()
def sources() -> None:
    """List registered sources and their key/licensing status."""
    _log()
    for name, src in AssetManager().sources.items():
        key = f"needs {src.key_env}" if src.requires_key else "keyless"
        typer.echo(f"  {name:14} [{key:22}] {src.license_note}")


if __name__ == "__main__":
    app()
