"""Load and validate a project JSON file into a typed :class:`Project`.

This module is the studio's *input boundary*. It reads a JSON file from disk,
validates it against the domain schema, and returns a strongly typed
:class:`~backend.models.Project`. Every failure mode maps to a specific,
meaningful exception so callers — including n8n via process exit codes — can
react precisely.

The module never prints; it emits diagnostics through the standard
:mod:`logging` framework and raises on error.

Exception hierarchy
--------------------
``ScriptLoaderError``
    Base class for anything this module raises.
``ScriptFileNotFoundError``
    The path does not exist or is not a file.
``ScriptParseError``
    The file exists but is not valid JSON.
``ScriptValidationError``
    The JSON is well-formed but does not satisfy the :class:`Project` schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from backend.models import Project

logger = logging.getLogger(__name__)


class ScriptLoaderError(Exception):
    """Base class for all errors raised while loading a project script."""


class ScriptFileNotFoundError(ScriptLoaderError):
    """Raised when the project file does not exist or is not a regular file."""


class ScriptParseError(ScriptLoaderError):
    """Raised when the file content is not valid JSON."""


class ScriptValidationError(ScriptLoaderError):
    """Raised when JSON is valid but does not match the :class:`Project` schema.

    The original :class:`pydantic.ValidationError` is preserved as the cause
    (``__cause__``) so detailed field-level errors remain available.
    """


def load_project(path: str | Path) -> Project:
    """Load, parse and validate a project JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file describing the project.

    Returns
    -------
    Project
        A fully validated project aggregate.

    Raises
    ------
    ScriptFileNotFoundError
        If ``path`` does not point to an existing file.
    ScriptParseError
        If the file is not valid JSON.
    ScriptValidationError
        If the JSON does not satisfy the :class:`Project` schema.
    """
    file_path = Path(path)
    logger.debug("Loading project script from %s", file_path)

    if not file_path.is_file():
        raise ScriptFileNotFoundError(f"Project file not found: {file_path}")

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:  # Permission errors, unreadable file, etc.
        raise ScriptLoaderError(f"Could not read project file: {file_path}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScriptParseError(
            f"Invalid JSON in {file_path} at line {exc.lineno}, column {exc.colno}: "
            f"{exc.msg}"
        ) from exc

    try:
        project = Project.model_validate(data)
    except ValidationError as exc:
        raise ScriptValidationError(
            f"Project file {file_path} failed schema validation "
            f"({exc.error_count()} error(s)). See details above."
        ) from exc

    logger.info(
        "Loaded project '%s' with %d short(s)", project.title, project.short_count
    )
    return project
