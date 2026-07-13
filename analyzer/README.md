# Analyzer — Session Handoff / Status

Classical-CV engine that reconstructs decks + gameplay events from real Clash Royale
footage. Deliberately **independent** of `backend/` (the frozen F1–F11 pipeline); the
only coupling is the output file `gameplay_analysis.json`.

Read this first each session — it's the fast context. Deeper details live in
`memory/cr-workflow-roadmap.md`.

---

## Build discipline (do not violate)
- **Vertical slices, YAGNI, freeze after validating.** Confidence-flag over narrowing scope.
- **Never edit analyzer code while a background `analyze` runs** — lazy imports load a fresh
  model against a cached old one → `AttributeError` (e.g. `slots_skipped`). Let runs finish.
- **Verify before fabricating.** Before adding/altering a template, prove the in-game card does
  or doesn't match an existing PNG (ORB). The "Evolved Goblins" hack was rejected for skipping this.
- Windows: PowerShell primary; Rich/Panel output must be **ASCII-only** (cp1252). ffmpeg/ffprobe
  are **off PATH** — absolute paths come from `.env` (`FFMPEG_PATH`/`FFPROBE_PATH`).

---

## FROZEN — do not touch (v1.0-baseline, tagged + pushed)
Repo: github.com/divcodes1121/Automation_Workflow. Deck engine is **stable**.
- **Deck reconstruction**: top-8 confirmed cards by observation count; confidence
  `= min(1,count/strong)*min(1,avg_orb/quality_norm)`; `needs_review` flag.
  Generalizes across 3 games (player deck identical ×3; opponents distinct; 85–91%).
- **Template library COMPLETE**: 122 base + 41 pink evo + 14 gold "Heroes" (in-battle evo art,
  `CardVariant.HERO`) + 9 champions = **177 templates**. Furnace misdetect was really Evolved
  Goblins = `assets/Heroes/goblins.png`.
- **Frozen modules**: `TemplateBuilder`, `TemplateIndex`, `HandDetector`, `DeckReconstructor`,
  `completeness.py`. ORB matching, per-slot event-driven skip (MAD on 12×12 downscale).
- **Regression guard** (`regression.py` + `regression_baseline.json`): games 01/02/03, profile
  `iphone_16_pro_max`, 1 fps. Tolerances: conf −0.10, ident −0.05, events ±30%. Baseline:
  g1 8/8 93% / 8/8 95% / ident 91% / 62 ev · g2 8/8 97% / 8/8 100% / 93% / 60 ev · g3 8/8 91% /
  8/8 91% / 90% / 28 ev. **Run `python -m analyzer.main regression --run` before any commit.**

Gameplay videos live at `gameplay/raw/<game>.mp4` — **outside git** (gitignored, too large).

---

## DONE — 2G Match State (timer / elixir / phase)  [schema 1.2]
Every `GameEvent` now carries grounded match context. **Additive**: deck engine + event counts
unchanged, deck regression still **ALL PASS** (62/60/28 events). Validated end-to-end on game_01.

**Design — a match-state TIMELINE (not per-event duplication):**
- `GameplayAnalysis.match_states: list[MatchState]` — one snapshot ~every 1s (`match_state_interval_s`).
- Each `GameEvent.match_state_ref` = **index** of the nearest state (referenced, not embedded).
- `MatchState` = `{time_remaining "M:SS", time_remaining_seconds, phase, elixir_multiplier,
  player_elixir, opponent_elixir, player_crowns, opponent_crowns, timer_confidence}`.

**Timer reader** ([timer_detector.py](detectors/timer_detector.py)) — digit-template OCR, validated
frame-by-frame vs a countdown ground-truth model: **100% regulation, ~96% mid-overtime, 90.6%
whole-match**. Key techniques:
- Multi-exemplar templates in [assets/timer_digits/](assets/timer_digits/) — `<d>_reg.png` (dark bg)
  + `<d>_ot.png` (red bg), 20 PNGs. Rebuild via the scratch harvester (needs video + clock model).
- **Adaptive binarize**: regulation → `max`-channel (white/red digits on dark); overtime → `min`-channel
  (white digits on red). Segment = Otsu → 3 largest tall+wide components (drops colon/sliver/border).
- **Phase** = redness of the ROI **border** (background) — red only in true overtime; not fooled by
  red regulation-endgame digits.
- Per-digit **confidence gate** (`timer_min_confidence=0.5`) drops weak reads → time `None`, phase kept.
- The confidently-wrong red-on-red endgame reads are cleaned by the **monotonic filter** in
  [tracking/match_state.py](tracking/match_state.py): within a phase the clock only counts down;
  upward jumps / impossible drops → time `None`. Phase change resets the baseline.

**Elixir reader** ([elixir_detector.py](detectors/elixir_detector.py)) — HSV magenta fill-fraction of
the bar → 0–10, both players. Validated (10/10 early, 0/0 at victory).

**Phase / multiplier** — `phase` ∈ regulation|overtime (border-red). `elixir_multiplier` derived from
CR rules we trust: **2x** in regulation's final minute (≤1:00) + all overtime, else **1x**. **3x is
never asserted** without a signal (left None).

**Timer ROI** fixed in the profile: `x0.815 y0.188 w0.15 h0.032` (now committed).

Known lag: `play_stability_frames=2` debounce → confirmed `PlayEvent` trails actual placement ~1–2s.

**2G REMAINING (deferred):**
- **Crowns = tower destruction** — no HUD counter in normal play, so it's a tower-ROI change detector.
  `crown_detector.py` is still a scaffold; `player_crowns`/`opponent_crowns` stay `None`. Tower ROIs
  in the profile are **estimated**. Hardest piece — do last.
- **Match-state regression** — deck regression guards the frozen engine; a dedicated timer/elixir/phase
  accuracy regression (hand-labeled) is not yet added. The frame-by-frame spike is the current evidence.

---

## DEFERRED / BLOCKED
- **2F Arena Detector (spell localization)** — full plan approved 10/10 (spells-first) but the
  **frame-diff feasibility spike FAILED**: 5 heuristics couldn't isolate the rocket explosion
  from princess-tower projectile streaks / troop motion. Cast-timing (6fps hand-read) works;
  effect-isolation is the blocker. **No 2F code written.** `arena_detector.py` is a scaffold
  (raises NotImplementedError). Revisit with a better method or skip. Plan file:
  `C:\Users\singh\.claude\plans\here-s-what-i-want-binary-octopus.md`.
- Later: analyze a NEW recording · Claude commentary generation · feed results into F1–F11.

---

## Key models / conventions
- Pydantic v2, `_STRICT_CONFIG` = `ConfigDict(extra="forbid", str_strip_whitespace=True,
  validate_assignment=True)`; `StrEnum`; frozen `Settings` (`ConfigDict(frozen=True)`).
- `GameEvent` reserves `lane: str|None`, `context: dict|None`, `notes`.
  `GameplayAnalysis.schema_version = "1.1"`.
- Typer CLI (`analyzer/main.py`) with `@app.callback()`. Calibration = fractional ROIs
  (`ROI.to_pixels`), device JSON at `calibration/profiles/`.
- Commands: `analyze`, `report`, `regression [--run|--update-baseline]`, `calibrate`,
  `build-templates`, completeness check.

## Immediate next step
Build the **timer digit reader** (spike first, then module), then the other 2G modules, then
enrich events, then regression + commit. Don't edit analyzer code during a background run.
