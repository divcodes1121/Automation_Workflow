"""Speech synthesis — turn a :class:`NarrationPackage` into per-segment audio (Feature 5).

:class:`SpeechSynthesisService` orchestrates synthesis but stays **provider-neutral**:
it delegates the actual audio generation to an injectable runner (default
:class:`~backend.services.kokoro_runner.KokoroRunner`). Swapping in a future Piper
or ElevenLabs runner requires no change here.

It produces **one WAV per narration segment** (no concatenation — that belongs to
the editor) plus a :class:`GeneratedNarration` manifest. Persistence of the JSON
manifests is a separate, optional :meth:`save` (the WAVs themselves are written by
the runner/worker during synthesis). Emits diagnostics via :mod:`logging`; no n8n
dependency.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from backend.config import Settings, get_settings
from backend.models import (
    GeneratedNarration,
    GeneratedNarrationSegment,
    NarrationPackage,
    WordTiming,
)
from backend.services.kokoro_runner import (
    KokoroRunner,
    SpeechSynthesisError,
)

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class SpeechRunner(Protocol):
    """A pluggable synthesis backend.

    Implemented today by :class:`~backend.services.kokoro_runner.KokoroRunner`;
    future providers (Piper, ElevenLabs) implement the same shape.
    """

    @property
    def provider_name(self) -> str: ...

    def is_available(self) -> bool: ...

    def synthesize(
        self, requests: list[dict[str, Any]], output_dir: Path
    ) -> dict[str, Any]: ...


class SpeechSynthesisService:
    """Synthesises narration audio via a pluggable runner.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`).
    runner:
        Optional synthesis backend (defaults to
        :class:`~backend.services.kokoro_runner.KokoroRunner`). Injecting a
        different runner is how alternative providers are swapped in.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        runner: SpeechRunner | None = None,
    ) -> None:
        self._settings: Settings = settings or get_settings()
        self._runner: SpeechRunner = runner or KokoroRunner(self._settings)

    # -- Public API -----------------------------------------------------------

    def synthesize(
        self, package: NarrationPackage, output_dir: Path | None = None
    ) -> GeneratedNarration:
        """Synthesise audio for every segment in ``package``.

        Writes one WAV per segment into ``output_dir`` (default
        ``settings.output_dir/narration/<slug>``) and returns the manifest.

        Raises
        ------
        SpeechSynthesisError
            If the package has no segments.
        SpeechProviderUnavailableError
            If the runner's runtime is missing.
        """
        if not package.segments:
            raise SpeechSynthesisError(
                f"Narration package {package.title!r} has no segments to synthesise."
            )

        target_dir = output_dir or (
            self._settings.output_dir
            / "narration"
            / _slugify(package.project_id or package.title)
        )

        requests = [
            {
                "index": index,
                "segment_uuid": segment.segment_uuid,
                "text": segment.cleaned_text,
            }
            for index, segment in enumerate(package.segments, start=1)
        ]

        response = self._runner.synthesize(requests, target_dir)
        segments = [
            GeneratedNarrationSegment(
                segment_uuid=item["segment_uuid"],
                index=item["index"],
                audio_file=Path(item["audio_file"]),
                duration_seconds=item["duration_seconds"],
                sample_rate=item["sample_rate"],
                synthesis_seconds=item["synthesis_seconds"],
                word_timings=[WordTiming(**word) for word in item.get("words", [])],
            )
            for item in response["segments"]
        ]

        generated = GeneratedNarration(
            project_id=package.project_id,
            title=package.title,
            generated_at=datetime.now(timezone.utc),
            provider=response["provider"],
            voice=response["voice"],
            sample_rate=response["sample_rate"],
            segments=segments,
            provider_data={
                "provider": response["provider"],
                "voice": response["voice"],
                "lang": response.get("lang"),
                "sample_rate": response["sample_rate"],
                "kokoro_version": response.get("kokoro_version"),
                "total_synthesis_seconds": round(
                    sum(s.synthesis_seconds for s in segments), 3
                ),
            },
        )
        logger.info(
            "Synthesised %d segment(s) for %r via %s: %.1fs audio, RTF %.3g",
            generated.segment_count,
            generated.title,
            generated.provider,
            generated.total_audio_seconds,
            generated.realtime_factor if generated.realtime_factor is not None else 0.0,
        )
        return generated

    def save(
        self, generated: GeneratedNarration, destination_dir: Path | None = None
    ) -> Path:
        """Write the manifest, timings and provider-data JSON files.

        Separate from :meth:`synthesize`. Defaults to the directory that already
        holds the generated WAVs. Returns the manifest path.
        """
        out_dir = destination_dir or _audio_dir(generated)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(generated.model_dump_json(indent=2), encoding="utf-8")

        timings = {
            "segments": [
                {
                    "index": seg.index,
                    "segment_uuid": seg.segment_uuid,
                    "words": [w.model_dump() for w in seg.word_timings],
                }
                for seg in generated.segments
            ]
        }
        (out_dir / "timings.json").write_text(
            json.dumps(timings, indent=2), encoding="utf-8"
        )
        (out_dir / "provider_data.json").write_text(
            json.dumps(generated.provider_data, indent=2), encoding="utf-8"
        )

        logger.info("Saved narration manifest for %r to %s", generated.title, manifest_path)
        return manifest_path


def _audio_dir(generated: GeneratedNarration) -> Path:
    """Directory holding the generated audio (parent of the first segment's WAV)."""
    if generated.segments:
        return generated.segments[0].audio_file.parent
    raise SpeechSynthesisError("GeneratedNarration has no segments to locate.")


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "narration"
