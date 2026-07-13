"""Centralised logging configuration for the AI Creator Studio.

Feature modules obtain their logger with ``logging.getLogger(__name__)`` and
never configure handlers themselves. The application entry point calls
:func:`configure_logging` exactly once so all output is formatted consistently
— using Rich for readable, colourised console logs.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single Rich-based handler on the root logger.

    Idempotent: calling it more than once will not stack duplicate handlers.

    Parameters
    ----------
    level:
        Minimum level to emit (defaults to :data:`logging.INFO`).
    """
    root = logging.getLogger()
    if any(isinstance(handler, RichHandler) for handler in root.handlers):
        root.setLevel(level)
        return

    handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )
