"""Battle state machine (2E): hand readings -> play events, flicker-resistant.

Template matching flickers frame-to-frame, so a naive "slot changed -> card
played" rule produces many false positives. :class:`BattleState` is
confidence/stability aware: it tracks a *confirmed* card per slot plus a pending
*candidate*, and only commits a change (emitting the departing card as a play)
once the candidate has been seen for ``play_stability_frames`` consecutive
readings. In Clash Royale a slot's card leaving the hand means it was played.
"""

from __future__ import annotations

from dataclasses import dataclass

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import HandReading, PlayEvent


@dataclass
class _SlotState:
    confirmed_card: str | None = None
    confirmed_variant: str | None = None
    confirmed_score: float = 0.0
    candidate_card: str | None = None
    candidate_variant: str | None = None
    candidate_count: int = 0


class BattleState:
    """Tracks the four hand slots over time and emits confirmed play events."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings = settings or get_analyzer_settings()
        self._stability = self._settings.play_stability_frames
        self._slots: dict[int, _SlotState] = {i: _SlotState() for i in range(1, 5)}

    def update(self, reading: HandReading) -> list[PlayEvent]:
        """Apply one hand reading; return any newly confirmed play events."""
        events: list[PlayEvent] = []
        for slot in reading.slots:
            state = self._slots.setdefault(slot.slot, _SlotState())
            matched = slot.card if slot.matched else None

            # Unmatched (empty / cycling / flicker): hold the confirmed card,
            # drop any pending candidate.
            if matched is None:
                state.candidate_card = None
                state.candidate_variant = None
                state.candidate_count = 0
                continue

            # First real sighting establishes the confirmed card (no play).
            if state.confirmed_card is None:
                state.confirmed_card = matched
                state.confirmed_variant = slot.variant
                state.confirmed_score = slot.score
                continue

            # Same card holds: refresh score, clear candidate.
            if matched == state.confirmed_card:
                state.confirmed_score = slot.score
                state.candidate_card = None
                state.candidate_variant = None
                state.candidate_count = 0
                continue

            # A different card: accumulate candidate confirmations.
            if matched == state.candidate_card:
                state.candidate_count += 1
            else:
                state.candidate_card = matched
                state.candidate_variant = slot.variant
                state.candidate_count = 1

            if state.candidate_count >= self._stability:
                # Confirmed change: the departing card was played.
                events.append(
                    PlayEvent(
                        source_frame=reading.source_frame,
                        timestamp_seconds=reading.timestamp_seconds,
                        card=state.confirmed_card,
                        variant=state.confirmed_variant,
                        slot=slot.slot,
                        score=round(state.confirmed_score, 4),
                    )
                )
                state.confirmed_card = state.candidate_card
                state.confirmed_variant = state.candidate_variant
                state.confirmed_score = slot.score
                state.candidate_card = None
                state.candidate_variant = None
                state.candidate_count = 0
        return events
