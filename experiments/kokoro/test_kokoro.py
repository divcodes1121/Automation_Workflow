"""Kokoro TTS technical spike (Feature 5a) — standalone research script.

This is **not** production code. It imports nothing from ``backend`` and knows
nothing about ``Project``/``Timeline``/``NarrationPackage``. Its only job is to
validate how the Kokoro Python API behaves so we can design Feature 5.

It:

1. Detects the compute device (CPU / CUDA / MPS) and inference backend.
2. Synthesises a fixed test sentence with Kokoro.
3. Saves ``output.wav``.
4. Measures synthesis time, audio duration and the realtime factor (RTF).
5. Probes the API for available timing data (sentence / word / phoneme / callbacks).
6. Exports ``timings.json`` if any timing data is available.
7. Prints a concise report.

Run:  ``python experiments/kokoro/test_kokoro.py``
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

TEST_TEXT = "This is a test of the Clash Royale narration system."
HERE = Path(__file__).resolve().parent
OUTPUT_WAV = HERE / "output.wav"
TIMINGS_JSON = HERE / "timings.json"

# Kokoro's fixed output sample rate.
KOKORO_SAMPLE_RATE = 24000
# Voices to try, in order (v1.0 voice pack names).
CANDIDATE_VOICES = ("af_heart", "af_bella", "af_sky")
# American English G2P.
LANG_CODE = "a"


def detect_device() -> tuple[str, str, str]:
    """Return ``(device, device_name, backend)`` for the current machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0), f"torch/cuda {torch.version.cuda}"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "Apple Silicon (MPS)", "torch/mps"
    cpu_name = platform.processor() or platform.machine() or "unknown CPU"
    return "cpu", cpu_name, f"torch/cpu (threads={torch.get_num_threads()})"


def build_pipeline(device: str) -> Any:
    """Construct a KPipeline, passing ``device`` when the version supports it."""
    from kokoro import KPipeline

    try:
        return KPipeline(lang_code=LANG_CODE, device=device)
    except TypeError:
        # Older/newer signature without a device kwarg.
        return KPipeline(lang_code=LANG_CODE)


def synthesize(pipeline: Any) -> tuple[list[Any], float]:
    """Run synthesis, returning the list of result chunks and elapsed seconds."""
    last_error: Exception | None = None
    for voice in CANDIDATE_VOICES:
        try:
            start = time.perf_counter()
            results = list(pipeline(TEST_TEXT, voice=voice))
            elapsed = time.perf_counter() - start
            print(f"[info] synthesised with voice {voice!r}")
            return results, elapsed
        except Exception as exc:  # noqa: BLE001 — spike: try the next voice.
            last_error = exc
            print(f"[warn] voice {voice!r} failed: {exc}")
    raise RuntimeError(f"All candidate voices failed; last error: {last_error}")


def _to_numpy(audio: Any):
    """Convert a Kokoro audio chunk (torch tensor / ndarray) to a 1-D ndarray."""
    import numpy as np

    if hasattr(audio, "detach"):  # torch tensor
        audio = audio.detach().to("cpu").numpy()
    return np.asarray(audio).reshape(-1)


def extract_timings(results: list[Any]) -> dict[str, list[dict[str, Any]]]:
    """Probe result chunks for timing data.

    Returns a mapping of timing-type -> list of records. Kokoro exposes
    per-token (word-level) timestamps through ``result.tokens`` when alignment
    is available; each chunk also forms a natural sentence/segment boundary.
    """
    word_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        tokens = getattr(result, "tokens", None) or []
        seg_start: float | None = None
        seg_end: float | None = None
        for token in tokens:
            start_ts = getattr(token, "start_ts", None)
            end_ts = getattr(token, "end_ts", None)
            text = getattr(token, "text", None)
            if start_ts is None and end_ts is None:
                continue
            word_records.append(
                {
                    "segment": index,
                    "text": text,
                    "start": _round(start_ts),
                    "end": _round(end_ts),
                    "phonemes": getattr(token, "phonemes", None),
                }
            )
            if start_ts is not None:
                seg_start = start_ts if seg_start is None else min(seg_start, start_ts)
            if end_ts is not None:
                seg_end = end_ts if seg_end is None else max(seg_end, end_ts)

        graphemes = getattr(result, "graphemes", None)
        segment_records.append(
            {
                "segment": index,
                "text": graphemes,
                "start": _round(seg_start),
                "end": _round(seg_end),
            }
        )

    timings: dict[str, list[dict[str, Any]]] = {}
    if word_records:
        timings["word"] = word_records
    if any(r["start"] is not None for r in segment_records):
        timings["sentence"] = segment_records
    return timings


def _round(value: Any) -> Any:
    """Round floats to 3 dp; pass through ``None``."""
    return round(value, 3) if isinstance(value, (int, float)) else value


def main() -> int:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        print(f"[error] missing dependency: {exc}. See README.md for setup.")
        return 1

    try:
        device, device_name, backend = detect_device()
    except ImportError:
        print("[error] PyTorch is not installed. See README.md for setup.")
        return 1

    print(f"[info] device={device} ({device_name}) backend={backend}")
    print("[info] building pipeline (first run downloads the model)...")
    pipeline = build_pipeline(device)

    results, gen_seconds = synthesize(pipeline)

    audio = np.concatenate([_to_numpy(r.audio) for r in results if r.audio is not None])
    sample_rate = KOKORO_SAMPLE_RATE
    sf.write(OUTPUT_WAV, audio, sample_rate)
    audio_seconds = len(audio) / sample_rate
    rtf = gen_seconds / audio_seconds if audio_seconds else float("inf")

    timings = extract_timings(results)
    available_timing_types = sorted(timings.keys()) or ["none"]
    if timings:
        TIMINGS_JSON.write_text(json.dumps(timings, indent=2), encoding="utf-8")

    print("\n" + "=" * 52)
    print(" KOKORO TTS SPIKE REPORT")
    print("=" * 52)
    print(f" Device               : {device} ({device_name})")
    print(f" Backend              : {backend}")
    print(f" Generation time      : {gen_seconds:.3f} s")
    print(f" Audio duration       : {audio_seconds:.3f} s")
    print(f" Realtime factor (RTF): {rtf:.3f}  (<1 = faster than realtime)")
    print(f" Sample rate          : {sample_rate} Hz")
    print(f" Result chunks        : {len(results)}")
    print(f" Available timing types: {', '.join(available_timing_types)}")
    print(f" Output WAV           : {OUTPUT_WAV.name}")
    if timings:
        counts = ", ".join(f"{k}={len(v)}" for k, v in timings.items())
        print(f" Timings JSON         : {TIMINGS_JSON.name} ({counts})")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
