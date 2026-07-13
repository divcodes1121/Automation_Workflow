# Kokoro TTS — Technical Spike (Feature 5a)

**Status:** ✅ Validated. Kokoro **0.9.4** (current v1.0 line) synthesises speech with
**word-level timestamps** and runs **faster than realtime on CPU** — but **only on
Python 3.10–3.12, not the project's Python 3.13.**

Research spike only. `test_kokoro.py` imports nothing from `backend/`. It was run in an
isolated, uv-managed **Python 3.12** venv (`experiments/kokoro/.venv312`); the project
backend stays on 3.13, untouched.

---

## TL;DR

- **Works and sounds good.** Generated `output.wav` (24 kHz mono) for
  _"This is a test of the Clash Royale narration system."_
- **Word + sentence timestamps are available** (`timings.json`) — ideal for subtitle
  alignment and Feature 6 (audio-timeline enrichment).
- **Fast on CPU:** steady-state **RTF ≈ 0.72–0.76** (a 10-min narration ≈ ~7.5 min to
  synthesise on this 6-thread CPU; a GPU would be far faster).
- **Blocker:** Kokoro is **incompatible with Python 3.13** (project's pinned version).
  Integration must isolate it → see recommendation.

---

## Measured results (CPU)

Machine: AMD Ryzen (Family 23), 6 torch threads, no GPU. `hexgrad/Kokoro-82M`, voice
`af_heart`, lang `a` (American English).

| Metric | Cold (1st call) | Warm (cached) |
|--------|----------------:|--------------:|
| Generation time (3.5 s clip) | 4.441 s | **2.51–2.65 s** |
| **Realtime factor (RTF)** | 1.269 | **0.72–0.76** |
| Audio duration | 3.500 s | 3.500 s |
| Sample rate | 24 000 Hz | 24 000 Hz |

- The cold call includes a one-time `af_heart` **voice-pack download**; warm RTF is the
  number that matters for planning. RTF excludes the one-time **model load** (~a few s).
- Artifacts produced next to this README: **`output.wav`** (168 KB, PCM_16, mono, 3.5 s)
  and **`timings.json`**.

## Timing capabilities (what the API actually exposes)

| Type | Available? | Notes |
|------|:----------:|-------|
| **Word** timestamps | ✅ | `result.tokens[].start_ts` / `.end_ts` (seconds). 11 words captured. |
| **Sentence/segment** | ✅ | One `result` chunk per sentence; min/max of its token times. |
| **Phoneme** strings | ✅ (text only) | Each token carries IPA `phonemes` (e.g. `test → tˈɛst`) but **no per-phoneme timestamps**. |
| **Phoneme** timestamps | ❌ | Not exposed by `KPipeline`. |
| Streaming **callbacks** | ❌ | No progressive/streaming callback API; results are yielded per chunk after synthesis. |

Sample from `timings.json`:

```
This   0.275–0.475   is  0.475–0.588   a  0.588–0.688   test 0.688–1.238
... Clash 1.425–1.800  Royale 1.800–2.112  narration 2.112–2.575  system 2.575–3.250
sentence[0]: 0.275–3.400  "This is a test of the Clash Royale narration system."
```

---

## Environment, dependencies, install

- **Python:** 3.12.13 (provisioned via `uv python install 3.12`; standalone, isolated —
  does not touch system Python or the 3.13 backend).
- **Packages (in `.venv312`):** `kokoro==0.9.4`, `torch==2.13.0+cpu`, `numpy==2.5.1`,
  `soundfile`, `misaki` (G2P), `spacy` + **`en_core_web_sm==3.8.0`**.
- First run downloads `hexgrad/Kokoro-82M` (~330 MB) + the voice pack from Hugging Face.

**Install that worked:**
```bash
uv python install 3.12
uv venv --python 3.12 experiments/kokoro/.venv312
uv pip install --python experiments/kokoro/.venv312/Scripts/python.exe kokoro soundfile
# misaki's English G2P needs the spaCy model IN THE SAME venv:
uv pip install --python experiments/kokoro/.venv312/Scripts/python.exe \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
experiments/kokoro/.venv312/Scripts/python experiments/kokoro/test_kokoro.py
```

### Gotchas found
- **`en_core_web_sm` is a hard runtime dependency** of misaki's English G2P and is **not**
  pulled in automatically. When missing, misaki tries to auto-install it but resolved to
  the **wrong environment** (the active 3.13 `.venv`), so it must be installed explicitly
  into the 3.12 venv (as above).
- HF cache warns about symlinks on Windows (works, uses more disk) — harmless; silence with
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`. Also an unauthenticated-HF rate-limit warning.
- Console `cp1252` can't print IPA phonemes; read `timings.json` with UTF-8.

## Why Python 3.13 fails (the integration constraint)

- Kokoro **0.8.1–0.9.4** declare `Requires-Python >=3.10,<3.13` → pip refuses on 3.13.
- Kokoro **0.7.16** (newest uncapped) pins **`numpy==1.26.4`**, which has **no cp313
  wheel** → source build → fails (Meson `WinError 1392`, worsened by OneDrive).
- Net: no published Kokoro runs on 3.13 today. This machine has only 3.13 and 3.9, so a
  3.12 was provisioned with `uv`.

## Limitations

- CPU-only here; RTF ~0.73 is fine for **batch** narration, not low-latency streaming.
- No phoneme-level timing and no streaming callbacks (word-level is the finest timing).
- English requires the extra spaCy model; other languages need other misaki extras.
- `.venv312` is large (torch); keep it out of git.

---

## Recommendation for Feature 5

**Adopt Kokoro as an isolated sidecar behind the planned `VoiceProvider.generate(segment)`
seam.** Concretely:

1. Ship a dedicated **Python 3.12 environment** for TTS (uv-managed, exactly as here). The
   3.13 backend never imports Kokoro/torch.
2. Feature 5's `KokoroVoiceProvider` invokes that env via a **subprocess CLI** (or a tiny
   local HTTP service): input = a `PreparedNarrationSegment`'s `cleaned_text`; output = a
   WAV path (→ fills `output_audio`) + word timings + `provider_metadata`
   (`{"provider":"kokoro","voice":"af_heart","sample_rate":24000}`).
3. **Word timestamps flow straight into Feature 6** (audio-timeline enrichment): map each
   segment's word times to populate `TimelineTiming.actual_start/end`.
4. Bundle `en_core_web_sm` into that env's setup (it won't self-install correctly).
5. Because RTF < 1 on CPU, **no GPU is required** to start; a GPU is a pure speed upgrade.

**Fallback / alternative:** Piper (ONNX, no torch, **native Python 3.13**) remains the
zero-friction free default behind the same `VoiceProvider` interface if we'd rather avoid a
second Python runtime; Kokoro then stays the quality upgrade. Decide based on a Piper
voice-quality A/B — Kokoro's naturalness is the reason to keep the sidecar.

## Reproduce
```bash
experiments/kokoro/.venv312/Scripts/python experiments/kokoro/test_kokoro.py
```
Regenerates `output.wav` + `timings.json` and prints the report.
