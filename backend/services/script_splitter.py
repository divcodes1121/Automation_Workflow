"""Turning a :class:`~backend.models.Project` into ordered narration segments.

The splitter is the pluggable front of the timeline pipeline::

    Project -> ScriptSplitter -> list[NarrationSegment] -> TimelineBuilder -> Timeline

:class:`ScriptSplitter` is a ``Protocol`` (an in-process strategy, deliberately
*not* in :mod:`backend.interfaces`, which is reserved for external-tool swaps).
:class:`DefaultScriptSplitter` implements a deterministic fallback chain; a future
AI/semantic splitter can replace it without touching the timeline builder.

The module is pure: no I/O, no FFmpeg, no AI.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from backend.models import NarrationSegment, Project

# One or more blank lines separate paragraphs.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Split after sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class ScriptSplitter(Protocol):
    """Strategy that derives ordered narration segments from a project."""

    def split(self, project: Project) -> list[NarrationSegment]:
        """Return the project's narration as ordered :class:`NarrationSegment`\\ s."""
        ...


class DefaultScriptSplitter:
    """Deterministic splitter with a priority fallback chain.

    Resolution order:

    1. If ``project.segments`` is present and non-empty, use it **verbatim** —
       the author (or Claude) already decided the structure.
    2. Otherwise split ``long_script`` on blank-line paragraphs.
    3. If there are no paragraph breaks, split into sentences.
    4. If none of the above yields multiple parts, treat the whole script as a
       single segment.

    Paragraphs are preferred over sentences because they map to *ideas/scenes*,
    which keeps segments coarse enough for editing; sentence-level captioning can
    still happen later within a segment.
    """

    def split(self, project: Project) -> list[NarrationSegment]:
        """See :class:`ScriptSplitter`."""
        if project.segments:
            return list(project.segments)

        text = project.long_script.strip()
        parts = self._split_paragraphs(text)
        if len(parts) <= 1:
            parts = self._split_sentences(text)
        if not parts:
            parts = [text]

        return [NarrationSegment(voice=part) for part in parts]

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on blank lines, dropping empty fragments."""
        return [chunk.strip() for chunk in _PARAGRAPH_RE.split(text) if chunk.strip()]

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence boundaries, dropping empty fragments."""
        return [chunk.strip() for chunk in _SENTENCE_RE.split(text) if chunk.strip()]
