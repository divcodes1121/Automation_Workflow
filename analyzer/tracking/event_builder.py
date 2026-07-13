"""Event builder (2H): play events -> the gameplay_analysis.json artifact.

Assembles the analyzer's single output -- the handoff consumed by script
generation. Each play event becomes a :class:`~analyzer.models.GameEvent` with a
stable ``event_id`` and the departing card's match ``confidence``. ``lane`` and
``context`` stay ``null`` until the 2F/2G detectors exist; a top-level warning
records that.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import (
    AnalyzerMetrics,
    GameEvent,
    GameplayAnalysis,
    MatchState,
    PlayEvent,
    ReconstructedDeck,
)
from analyzer.tracking.match_state import nearest_index

logger = logging.getLogger(__name__)

_PENDING_WARNING = (
    "lane detection (2F) not yet implemented - lane is null pending real footage"
)


class EventBuilder:
    """Turns play events into a :class:`GameplayAnalysis` and persists it."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()

    def build(
        self,
        play_events: list[PlayEvent],
        *,
        video: str,
        video_sha256: str,
        source_fps: float,
        duration_seconds: float,
        sample_fps: float,
        frame_count: int,
        profile_name: str,
        match_states: list[MatchState] | None = None,
        player_deck: ReconstructedDeck | None = None,
        opponent_deck: ReconstructedDeck | None = None,
        metrics: AnalyzerMetrics | None = None,
        extra_warnings: list[str] | None = None,
    ) -> GameplayAnalysis:
        """Assemble a :class:`GameplayAnalysis` from confirmed play events + decks."""
        states = match_states or []
        events: list[GameEvent] = []
        for seq, play in enumerate(play_events):
            events.append(
                GameEvent(
                    event_id=f"play_{seq:06d}",
                    sequence_number=seq,
                    timestamp_seconds=play.timestamp_seconds,
                    source_frame=play.source_frame,
                    type="card_played",
                    card=play.card,
                    variant=play.variant,
                    slot=play.slot,
                    confidence=play.score,
                    # Reference the nearest match-state snapshot (2G) by index.
                    match_state_ref=nearest_index(states, play.timestamp_seconds),
                )
            )
        warnings = [_PENDING_WARNING, *(extra_warnings or [])]
        return GameplayAnalysis(
            video=video,
            video_sha256=video_sha256,
            source_fps=source_fps,
            duration_seconds=duration_seconds,
            sample_fps=sample_fps,
            frame_count=frame_count,
            profile_name=profile_name,
            events=events,
            match_states=states,
            warnings=warnings,
            generated_at=datetime.now(timezone.utc),
            player_deck=player_deck,
            opponent_deck=opponent_deck,
            metrics=metrics,
        )

    def save(self, analysis: GameplayAnalysis, destination: Path | None = None) -> Path:
        """Write ``<stem>.gameplay_analysis.json`` into the analysis output dir."""
        stem = Path(analysis.video).stem
        dest = destination or self._settings.analysis_output_dir / f"{stem}.gameplay_analysis.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved gameplay analysis (%d events) to %s", len(analysis.events), dest)
        return dest
