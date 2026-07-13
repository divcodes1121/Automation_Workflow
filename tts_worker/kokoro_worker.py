"""Kokoro speech-synthesis worker — runs in the isolated Python 3.12 env.

This script is **executed as a subprocess** by
:class:`backend.services.kokoro_runner.KokoroRunner`; it is **never imported** by
the Python 3.13 backend (which cannot import Kokoro/torch). It lives at the top
level, outside any package, precisely so nothing in ``backend`` can import it.

Contract (JSON files exchanged via argv):

    python kokoro_worker.py <request.json> <response.json>

Request::

    {
      "voice": "af_heart",
      "lang": "a",
      "output_dir": "…/output/narration/<slug>",
      "segments": [{"index": 1, "segment_uuid": "…", "text": "…"}, …]
    }

Response (written to ``response.json``)::

    {
      "provider": "kokoro", "voice": "…", "lang": "…",
      "sample_rate": 24000, "kokoro_version": "0.9.4",
      "segments": [{
        "index": 1, "segment_uuid": "…", "audio_file": "…/001.wav",
        "duration_seconds": 3.5, "sample_rate": 24000,
        "synthesis_seconds": 2.6,
        "words": [{"text": "This", "start_seconds": 0.275, "end_seconds": 0.475}, …]
      }, …]
    }

One WAV per segment is written to ``output_dir`` (``001.wav``, ``002.wav`` …).
Audio is **not** concatenated — that belongs to the editor stage.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Kokoro's fixed output sample rate.
SAMPLE_RATE = 24000


def _log(message: str) -> None:
    """Emit progress/diagnostics to stderr (stdout is reserved for cleanliness)."""
    print(f"[kokoro_worker] {message}", file=sys.stderr, flush=True)


def _to_mono(audio: Any):
    """Convert a Kokoro audio chunk (torch tensor / ndarray) to a 1-D float array."""
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().to("cpu").numpy()
    return np.asarray(audio).reshape(-1)


def _synthesize_segment(pipeline: Any, text: str, voice: str, wav_path: Path) -> dict:
    """Synthesise one segment to ``wav_path``; return its manifest entry data."""
    import numpy as np
    import soundfile as sf

    started = time.perf_counter()
    parts: list[Any] = []
    words: list[dict[str, Any]] = []
    offset = 0.0  # seconds of audio already emitted for this segment

    for chunk in pipeline(text, voice=voice):
        audio = getattr(chunk, "audio", None)
        if audio is None:
            continue
        samples = _to_mono(audio)
        for token in getattr(chunk, "tokens", None) or []:
            start_ts = getattr(token, "start_ts", None)
            end_ts = getattr(token, "end_ts", None)
            if start_ts is None or end_ts is None:
                continue
            words.append(
                {
                    "text": getattr(token, "text", None),
                    "start_seconds": round(start_ts + offset, 3),
                    "end_seconds": round(end_ts + offset, 3),
                }
            )
        offset += len(samples) / SAMPLE_RATE
        parts.append(samples)

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
    sf.write(wav_path, audio, SAMPLE_RATE)

    return {
        "audio_file": str(wav_path),
        "duration_seconds": round(len(audio) / SAMPLE_RATE, 3),
        "sample_rate": SAMPLE_RATE,
        "synthesis_seconds": round(time.perf_counter() - started, 3),
        "words": words,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        _log("usage: kokoro_worker.py <request.json> <response.json>")
        return 2

    request_path, response_path = Path(argv[1]), Path(argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    voice = request.get("voice", "af_heart")
    lang = request.get("lang", "a")
    output_dir = Path(request["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = request["segments"]

    import kokoro
    from kokoro import KPipeline

    _log(f"loading KPipeline(lang_code={lang!r}) …")
    pipeline = KPipeline(lang_code=lang)

    out_segments: list[dict[str, Any]] = []
    for segment in segments:
        index = int(segment["index"])
        wav_path = output_dir / f"{index:03d}.wav"
        data = _synthesize_segment(pipeline, segment["text"], voice, wav_path)
        data.update(index=index, segment_uuid=segment["segment_uuid"])
        out_segments.append(data)
        _log(
            f"segment {index:03d} -> {wav_path.name} "
            f"({data['duration_seconds']:.2f}s audio, "
            f"{data['synthesis_seconds']:.2f}s synth)"
        )

    response = {
        "provider": "kokoro",
        "voice": voice,
        "lang": lang,
        "sample_rate": SAMPLE_RATE,
        "kokoro_version": getattr(kokoro, "__version__", None),
        "segments": out_segments,
    }
    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    _log(f"done: {len(out_segments)} segment(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:  # noqa: BLE001 — surface any failure to the runner.
        _log(f"ERROR: {type(exc).__name__}: {exc}")
        raise
