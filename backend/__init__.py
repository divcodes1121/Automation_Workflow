"""AI Creator Studio backend for the Clash Royale YouTube channel.

This package holds *all* business logic for the studio. It is designed to be
driven either from the command line (``python -m backend.main``) or by n8n,
which is responsible only for *orchestrating* workflows — never for computing
anything itself.

Sub-packages
------------
core
    Cross-cutting building blocks shared by feature modules.
modules
    Independent feature modules (gameplay analysis, subtitles, editing,
    thumbnails, uploads, shorts). Each module owns one concern.
utils
    Small, dependency-light helpers (logging configuration, etc.).

Top-level modules
-----------------
config
    Environment-driven configuration and secret handling.
models
    Strongly typed Pydantic domain models (the ``Project`` aggregate).
script_loader
    Loads and validates a project JSON file into a typed ``Project``.
workflow
    ``WorkflowManager`` — the orchestration seam n8n will drive.
main
    Typer CLI entry point.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
