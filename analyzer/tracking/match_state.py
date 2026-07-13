"""Match-state timeline (2G): raw per-frame reads -> a clean, monotonic timeline.

The timer/elixir detectors read each sampled frame independently. This module
turns those raw reads into the :class:`~analyzer.models.MatchState` timeline that
events reference, applying the physics the detectors can't see on their own:

* **Monotonic countdown filter** -- within a phase the clock only counts *down*.
  A read that jumps *up* (or drops implausibly far) is a misread and its time is
  dropped to None (phase is kept). This is what cleans the confidently-wrong
  red-on-red overtime endgame reads that survive the per-digit confidence gate.
  Phase transitions (regulation 0:00 -> overtime 1:xx) legitimately reset the
  clock, so the filter resets its baseline whenever the phase changes.
* **Elixir multiplier** -- derived from the well-known CR rules we can stand
  behind: 2x in regulation's final minute and throughout overtime, else 1x.
  Triple elixir is never asserted without a dedicated signal (left None).

Pure data (no OpenCV); the detectors do the pixel work upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from analyzer.models import MatchPhase, MatchState

# Monotonic-filter tolerances (seconds), for ~1 Hz sampling.
_RISE_SLACK = 2.0   # allow tiny upward noise/jitter before calling a read bogus
_DROP_SLACK = 5.0   # extra room beyond the elapsed wall-time for a legit drop
_DOUBLE_ELIXIR_S = 60  # regulation switches to 2x with <= 1:00 remaining


@dataclass(frozen=True)
class RawSample:
    """One frame's raw match-state read, before cross-frame reconciliation."""

    timestamp_seconds: float
    source_frame: int
    phase: MatchPhase
    time_text: str | None
    time_seconds: int | None
    timer_confidence: float
    player_elixir: int | None
    opponent_elixir: int | None


def derive_multiplier(phase: MatchPhase, time_seconds: int | None) -> int | None:
    """Elixir generation multiplier from phase + time remaining (1x/2x, else None)."""
    if phase == MatchPhase.OVERTIME:
        return 2
    if phase == MatchPhase.REGULATION and time_seconds is not None:
        return 2 if time_seconds <= _DOUBLE_ELIXIR_S else 1
    return None


def build_timeline(samples: list[RawSample]) -> list[MatchState]:
    """Reconcile raw samples into the monotonic :class:`MatchState` timeline."""
    states: list[MatchState] = []
    last_phase: MatchPhase | None = None
    last_seconds: int | None = None
    last_ts: float | None = None

    for i, s in enumerate(samples):
        seconds, text = s.time_seconds, s.time_text

        # A real phase change resets the countdown baseline (clock jumps up).
        if s.phase in (MatchPhase.REGULATION, MatchPhase.OVERTIME) and s.phase != last_phase:
            last_seconds = None

        # Monotonic sanity within a phase: reject upward jumps and impossibly
        # large drops (the clock tracks wall-time ~1:1).
        if seconds is not None and last_seconds is not None:
            elapsed = (s.timestamp_seconds - last_ts) if last_ts is not None else 0.0
            if seconds > last_seconds + _RISE_SLACK or (
                last_seconds - seconds > elapsed + _DROP_SLACK
            ):
                seconds, text = None, None

        if seconds is not None:
            last_seconds = seconds
            last_ts = s.timestamp_seconds

        states.append(
            MatchState(
                index=i,
                timestamp_seconds=s.timestamp_seconds,
                source_frame=s.source_frame,
                time_remaining=text,
                time_remaining_seconds=seconds,
                phase=s.phase,
                elixir_multiplier=derive_multiplier(s.phase, seconds),
                player_elixir=s.player_elixir,
                opponent_elixir=s.opponent_elixir,
                timer_confidence=round(s.timer_confidence, 3),
            )
        )
        if s.phase in (MatchPhase.REGULATION, MatchPhase.OVERTIME):
            last_phase = s.phase

    return states


def nearest_index(states: list[MatchState], timestamp_seconds: float) -> int | None:
    """Index of the timeline state nearest a timestamp (None if empty)."""
    if not states:
        return None
    return min(
        range(len(states)),
        key=lambda i: abs(states[i].timestamp_seconds - timestamp_seconds),
    )
