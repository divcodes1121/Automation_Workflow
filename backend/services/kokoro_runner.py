"""Runner that drives the Kokoro worker in its isolated Python 3.12 environment.

Kokoro is incompatible with the backend's Python 3.13, so it is treated as an
**external worker**, not a library: this runner never imports kokoro/torch. It
serialises a synthesis request, invokes ``kokoro_worker.py`` with the configured
3.12 interpreter as a subprocess, and parses the JSON response.

The runner is the swappable unit behind
:class:`~backend.services.speech.SpeechSynthesisService`; a future ``PiperRunner``
or ``ElevenLabsRunner`` would implement the same ``synthesize`` shape.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SpeechError(Exception):
    """Base class for all speech-synthesis errors."""


class SpeechProviderUnavailableError(SpeechError):
    """Raised when the TTS provider's runtime (interpreter/worker) is missing."""


class SpeechSynthesisError(SpeechError):
    """Raised when synthesis runs but fails (worker error / bad output)."""


class KokoroRunner:
    """Invokes the Kokoro worker in the configured Python 3.12 environment.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`). Supplies the interpreter/worker
        paths and the default voice/language.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    @property
    def provider_name(self) -> str:
        """Identifier for this provider."""
        return "kokoro"

    def is_available(self) -> bool:
        """Return whether the 3.12 interpreter and worker script both exist."""
        return (
            Path(self._settings.kokoro_python).is_file()
            and Path(self._settings.kokoro_worker).is_file()
        )

    def synthesize(
        self, requests: list[dict[str, Any]], output_dir: Path
    ) -> dict[str, Any]:
        """Synthesise ``requests`` into ``output_dir`` and return the worker response.

        Parameters
        ----------
        requests:
            ``[{"index": int, "segment_uuid": str, "text": str}, …]``.
        output_dir:
            Directory the worker writes the per-segment WAVs into.

        Raises
        ------
        SpeechProviderUnavailableError
            If the interpreter or worker script is missing.
        SpeechSynthesisError
            If the worker exits non-zero or emits unparseable output.
        """
        python = Path(self._settings.kokoro_python)
        worker = Path(self._settings.kokoro_worker)
        if not python.is_file():
            raise SpeechProviderUnavailableError(
                f"Kokoro interpreter not found at {python}. Set KOKORO_PYTHON to a "
                f"Python 3.12 env with kokoro installed (see experiments/kokoro/README.md)."
            )
        if not worker.is_file():
            raise SpeechProviderUnavailableError(
                f"Kokoro worker script not found at {worker}. Set KOKORO_WORKER."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        request = {
            "voice": self._settings.kokoro_voice,
            "lang": self._settings.kokoro_lang,
            "output_dir": str(output_dir),
            "segments": requests,
        }

        with tempfile.TemporaryDirectory(prefix="kokoro_") as tmp:
            request_path = Path(tmp) / "request.json"
            response_path = Path(tmp) / "response.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")

            logger.info(
                "Invoking Kokoro worker for %d segment(s) via %s",
                len(requests),
                python,
            )
            completed = subprocess.run(
                [str(python), str(worker), str(request_path), str(response_path)],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                tail = (completed.stderr or "").strip()[-2000:]
                raise SpeechSynthesisError(
                    f"Kokoro worker failed (exit {completed.returncode}).\n{tail}"
                )
            if not response_path.is_file():
                raise SpeechSynthesisError(
                    "Kokoro worker produced no response file."
                )
            try:
                return json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SpeechSynthesisError(
                    f"Kokoro worker returned invalid JSON: {exc}"
                ) from exc
