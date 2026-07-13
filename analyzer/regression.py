"""Regression suite for the frozen deck engine.

Locks in the known-good analyzer behaviour on the three validated matches
(game_01/02/03) so that future work (Arena, troop tracking, ...) cannot silently
regress deck detection. A committed baseline records, per game, the reconstructed
decks + key quality numbers; the checker re-analyzes and verifies:

  * player deck still the exact 8 cards
  * opponent deck still the exact 8 cards
  * deck confidence not dropped beyond tolerance
  * slot identification rate not dropped beyond tolerance
  * play-event count within a relative band

The gameplay videos live outside git (too large), so this runs locally against
`gameplay/raw/<game>.mp4`. Baseline: `analyzer/regression_baseline.json`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import GameplayAnalysis

logger = logging.getLogger(__name__)

REGRESSION_GAMES = ("game_01", "game_02", "game_03")
REGRESSION_PROFILE = "iphone_16_pro_max"
REGRESSION_SAMPLE_FPS = 1.0

# Tolerances (fail if exceeded).
CONFIDENCE_DROP = 0.10   # deck confidence may fall at most this much
IDENTIFY_DROP = 0.05     # slot identify-rate may fall at most this much
EVENT_REL_BAND = 0.30    # play-event count within +/- this fraction

_BASELINE_PATH = Path(__file__).with_name("regression_baseline.json")


@dataclass
class GameSummary:
    """The regression-relevant slice of one game's analysis."""

    player_deck: list[str]
    opponent_deck: list[str]
    player_confidence: float
    opponent_confidence: float
    identify_rate: float
    event_count: int


def summarize(analysis: GameplayAnalysis) -> GameSummary:
    """Extract the regression summary from a full analysis."""
    pd = analysis.player_deck
    od = analysis.opponent_deck
    m = analysis.metrics
    ident = (m.cards_identified / m.slots_analyzed) if (m and m.slots_analyzed) else 0.0
    return GameSummary(
        player_deck=sorted(c.slug for c in pd.cards) if pd else [],
        opponent_deck=sorted(c.slug for c in od.cards) if od else [],
        player_confidence=round(pd.confidence, 3) if pd else 0.0,
        opponent_confidence=round(od.confidence, 3) if od else 0.0,
        identify_rate=round(ident, 3),
        event_count=len(analysis.events),
    )


def compare(current: GameSummary, baseline: GameSummary) -> list[str]:
    """Return a list of failure strings (empty == pass)."""
    fails: list[str] = []
    if set(current.player_deck) != set(baseline.player_deck):
        missing = set(baseline.player_deck) - set(current.player_deck)
        extra = set(current.player_deck) - set(baseline.player_deck)
        fails.append(f"player deck changed (missing {sorted(missing)}, new {sorted(extra)})")
    if set(current.opponent_deck) != set(baseline.opponent_deck):
        missing = set(baseline.opponent_deck) - set(current.opponent_deck)
        extra = set(current.opponent_deck) - set(baseline.opponent_deck)
        fails.append(f"opponent deck changed (missing {sorted(missing)}, new {sorted(extra)})")
    if current.player_confidence < baseline.player_confidence - CONFIDENCE_DROP:
        fails.append(
            f"player confidence dropped {baseline.player_confidence:.2f} -> {current.player_confidence:.2f}"
        )
    if current.opponent_confidence < baseline.opponent_confidence - CONFIDENCE_DROP:
        fails.append(
            f"opponent confidence dropped {baseline.opponent_confidence:.2f} -> {current.opponent_confidence:.2f}"
        )
    if current.identify_rate < baseline.identify_rate - IDENTIFY_DROP:
        fails.append(
            f"identify rate dropped {baseline.identify_rate:.2f} -> {current.identify_rate:.2f}"
        )
    lo = baseline.event_count * (1 - EVENT_REL_BAND)
    hi = baseline.event_count * (1 + EVENT_REL_BAND)
    if not (lo <= current.event_count <= hi):
        fails.append(
            f"event count {current.event_count} outside [{lo:.0f}, {hi:.0f}] (baseline {baseline.event_count})"
        )
    return fails


def _analysis_path(settings: AnalyzerSettings, game: str) -> Path:
    return settings.analysis_output_dir / f"{game}.gameplay_analysis.json"


def _current_summary(
    settings: AnalyzerSettings, game: str, *, run: bool
) -> GameSummary:
    """Get a game's current summary, optionally re-running the analyzer first."""
    if run:
        from analyzer.workflow import AnalyzerWorkflow

        video = settings.project_root / "gameplay" / "raw" / f"{game}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"regression video not found: {video}")
        analysis = AnalyzerWorkflow(settings).analyze(
            video, profile_name=REGRESSION_PROFILE, sample_fps=REGRESSION_SAMPLE_FPS
        )
        return summarize(analysis)
    path = _analysis_path(settings, game)
    if not path.is_file():
        raise FileNotFoundError(
            f"no analysis for {game} at {path}; run `analyzer analyze` first or pass --run"
        )
    return summarize(GameplayAnalysis.model_validate_json(path.read_text(encoding="utf-8")))


def load_baseline() -> dict[str, GameSummary]:
    if not _BASELINE_PATH.is_file():
        raise FileNotFoundError(
            f"no regression baseline at {_BASELINE_PATH}; create it with --update-baseline"
        )
    raw = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {game: GameSummary(**data) for game, data in raw["games"].items()}


def update_baseline(summaries: dict[str, GameSummary]) -> Path:
    payload = {
        "profile": REGRESSION_PROFILE,
        "sample_fps": REGRESSION_SAMPLE_FPS,
        "games": {game: asdict(s) for game, s in summaries.items()},
    }
    _BASELINE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return _BASELINE_PATH


def run_regression(
    *, run: bool, update: bool, settings: AnalyzerSettings | None = None
) -> tuple[bool, str]:
    """Run the regression. Returns ``(passed, report_text)``."""
    settings = settings or get_analyzer_settings()
    summaries = {g: _current_summary(settings, g, run=run) for g in REGRESSION_GAMES}

    if update:
        dest = update_baseline(summaries)
        return True, f"Baseline updated ({len(summaries)} games) -> {dest}"

    baseline = load_baseline()
    lines = ["=" * 52, "            DECK ENGINE REGRESSION", "=" * 52]
    all_pass = True
    for game in REGRESSION_GAMES:
        fails = compare(summaries[game], baseline[game]) if game in baseline else ["no baseline"]
        cur = summaries[game]
        status = "PASS" if not fails else "FAIL"
        all_pass = all_pass and not fails
        lines.append(
            f"[{status}] {game}: player {len(cur.player_deck)}/8 conf {cur.player_confidence:.0%}, "
            f"opp {len(cur.opponent_deck)}/8 conf {cur.opponent_confidence:.0%}, "
            f"ident {cur.identify_rate:.0%}, events {cur.event_count}"
        )
        for f in fails:
            lines.append(f"        - {f}")
    lines.append("=" * 52)
    lines.append("ALL PASS" if all_pass else "REGRESSIONS DETECTED")
    lines.append("=" * 52)
    return all_pass, "\n".join(lines)
