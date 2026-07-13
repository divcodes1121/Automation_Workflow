"""Phase 3: turn a gameplay analysis into a Claude prompt for project.json.

This is the deterministic, network-free CORE of the "Claude Project Generator".
It reads the analyzer's ``gameplay_analysis.json`` (as a plain dict — the backend
stays decoupled from the ``analyzer`` package, coupled only by that JSON contract)
and distills it into a single, grounded prompt: role + the exact frozen
``Project`` schema + honest match facts + a compact machine-readable brief. A human
pastes that prompt into Claude and saves the reply as ``project.json``; the
existing ``validate`` / ``run`` commands take it from there.

The Claude call is intentionally NOT here — it's a swappable step (manual paste
now, an Anthropic-SDK adapter later behind the same seam, mirroring how
VoiceProvider/ScriptSplitter defer their backends). Building the prompt is what's
valuable and testable today.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import (
    GeneratedPrompt,
    SegmentImportance,
    VideoCategory,
)

logger = logging.getLogger(__name__)


class ProjectGenerationError(ValueError):
    """Raised when the analysis JSON cannot be turned into a prompt."""


# Channel defaults surfaced to Claude (it may override within the schema).
_DEFAULT_VOICE_STYLE = "energetic_male_en"
_DEFAULT_LANGUAGE = "en"


class ProjectGenerator:
    """Builds a Claude prompt (and its distilled record) from a match analysis."""

    def build(self, analysis: dict[str, Any]) -> GeneratedPrompt:
        """Distill an analysis dict into a :class:`GeneratedPrompt` (pure)."""
        if not isinstance(analysis, dict) or "events" not in analysis:
            raise ProjectGenerationError(
                "not a gameplay analysis (missing 'events'); pass a "
                "gameplay_analysis.json produced by the analyzer"
            )
        video = str(analysis.get("video", "gameplay.mp4"))
        player_deck = _deck_slugs(analysis.get("player_deck"))
        opponent_deck = _deck_slugs(analysis.get("opponent_deck"))
        plays = _play_by_play(analysis)

        prompt = _compose_prompt(analysis, player_deck, opponent_deck, plays)
        return GeneratedPrompt(
            generated_at=datetime.now(timezone.utc),
            source_analysis=video,
            video=video,
            player_deck=player_deck,
            opponent_deck=opponent_deck,
            play_count=len(plays),
            prompt=prompt,
        )

    def save(self, generated: GeneratedPrompt, destination: Path) -> Path:
        """Write the paste-ready prompt text to ``destination``."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(generated.prompt, encoding="utf-8")
        logger.info("Wrote Claude prompt (%d plays) to %s", generated.play_count, destination)
        return destination


# --------------------------------------------------------------------------- #
# Distillation
# --------------------------------------------------------------------------- #
def _deck_slugs(deck: Any) -> list[str]:
    """Card slugs from a reconstructed-deck dict (order preserved)."""
    if not isinstance(deck, dict):
        return []
    return [str(c.get("slug")) for c in deck.get("cards", []) if c.get("slug")]


def _deck_detail(deck: Any) -> list[dict[str, Any]]:
    """Per-card {card, confidence, uncertain} for the machine-readable brief."""
    if not isinstance(deck, dict):
        return []
    detail = []
    for c in deck.get("cards", []):
        if not c.get("slug"):
            continue
        detail.append(
            {
                "card": c["slug"],
                "confidence": round(float(c.get("confidence", 0.0)), 2),
                "uncertain": bool(c.get("needs_review", False)),
            }
        )
    return detail


def _clock(analysis: dict[str, Any], event: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    """Resolve an event's referenced match-state (clock/phase/elixir)."""
    states = analysis.get("match_states") or []
    ref = event.get("match_state_ref")
    if isinstance(ref, int) and 0 <= ref < len(states):
        return "state", states[ref]
    return "ts", None


def _play_by_play(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """One compact record per PLAYER card play, enriched with match state.

    Records stay in chronological order. ``clock`` is the real match clock when
    the timer was readable, else null (we never manufacture a clock from the
    video timestamp — a match clock can't read "4:33"). A play is ``uncertain``
    when its detection was low-confidence OR its card is not in the reconstructed
    deck (a transient misdetection, e.g. a one-off "valkyrie").
    """
    deck = set(_deck_slugs(analysis.get("player_deck")))
    plays: list[dict[str, Any]] = []
    for event in analysis.get("events", []):
        _, state = _clock(analysis, event)
        phase = None
        elixir_x = None
        p_elx = o_elx = None
        clock = None
        if state is not None:
            clock = state.get("time_remaining")  # None in the unreadable endgame
            phase = state.get("phase")
            elixir_x = state.get("elixir_multiplier")
            p_elx = state.get("player_elixir")
            o_elx = state.get("opponent_elixir")
        card = event.get("card")
        low_conf = (event.get("confidence") or 1.0) < 0.6
        plays.append(
            {
                "clock": clock,
                "phase": phase or "unknown",
                "elixir_x": elixir_x,
                "player_elixir": p_elx,
                "opponent_elixir": o_elx,
                "card": card,
                "uncertain": low_conf or (bool(deck) and card not in deck),
            }
        )
    return plays


# --------------------------------------------------------------------------- #
# Prompt composition
# --------------------------------------------------------------------------- #
def _compose_prompt(
    analysis: dict[str, Any],
    player_deck: list[str],
    opponent_deck: list[str],
    plays: list[dict[str, Any]],
) -> str:
    categories = ", ".join(c.value for c in VideoCategory)
    importances = ", ".join(i.value for i in SegmentImportance)
    duration = float(analysis.get("duration_seconds", 0.0))
    winner = analysis.get("winner")

    brief = {
        "video": analysis.get("video"),
        "duration_seconds": round(duration, 1),
        "winner": winner,
        "player_deck": _deck_detail(analysis.get("player_deck")),
        "opponent_deck": _deck_detail(analysis.get("opponent_deck")),
        "player_plays": plays,
    }
    brief_json = json.dumps(brief, indent=2, ensure_ascii=False)

    phases = sorted({p["phase"] for p in plays if p["phase"] != "unknown"})
    phase_note = ", ".join(phases) if phases else "unknown"

    return f"""\
You are an expert Clash Royale YouTube commentator and scriptwriter. You turn a
computer-vision analysis of a real ladder match into an engaging, accurate
long-form commentary video project.

# YOUR TASK
Write a `project.json` for ONE long-form YouTube video narrating the match
described under MATCH DATA. Return ONLY the JSON object — no markdown fences, no
prose before or after. It must pass strict schema validation (unknown keys are
rejected).

# OUTPUT SCHEMA (project.json)
A single JSON object with these fields:
- "title": string, 1-100 chars. Punchy, specific, YouTube-style.
- "description": string, 1-5000 chars. What the video covers.
- "tags": array of >=1 non-empty strings (search terms).
- "thumbnail_prompt": string. A vivid image-generation prompt for the thumbnail.
- "long_script": string. A one-paragraph summary of the full narration (the
  authoritative narration lives in "segments" below).
- "voice_style": string. Use "{_DEFAULT_VOICE_STYLE}".
- "upload_time": ISO-8601 datetime, e.g. "2026-07-20T17:00:00+00:00".
- "category": one of [{categories}].
- "language": string, e.g. "{_DEFAULT_LANGUAGE}".
- "shorts": array (may be []). Each: {{"title": str<=100, "script": str,
  "hook": str<=200, "duration_seconds": int 1-60}}. Add 1-2 if a moment stands
  out; otherwise use [].
- "segments": array of narration units driving the video timeline IN ORDER.
  Each: {{"id": int, "voice": str (what is said), "visual": str (the gameplay
  moment to show), "importance": one of [{importances}]}}.

# HOW TO WRITE THE COMMENTARY
- USE the "segments" array as the real script: one segment per beat of the match,
  in chronological order, each mapping narration ("voice") to a real moment
  ("visual"). 12-25 segments is a good long-form length.
- Be SPECIFIC and grounded in MATCH DATA: name the actual cards, cite the clock
  ("with 1:22 left"), the phase (single vs double elixir), and elixir counts.
  This specificity is the whole point — never write generic filler like
  "a card is played".
- Open with a hook segment (importance "high") and end with a wrap-up.
- Mark tower pushes / big plays / phase changes as "high" importance.

# HONESTY CONSTRAINTS (the analysis has known limits — do NOT fabricate)
- "player_plays" are the PLAYER's card plays only. The opponent's DECK is known,
  but individual opponent plays are NOT tracked — don't invent specific opponent
  plays; speak about their deck/threats in general terms.
- Card names are slugs (e.g. "skeleton-barrel"); use natural names in narration
  ("Skeleton Barrel").
- Cards/plays marked "uncertain": true are low-confidence detections — hedge or
  omit rather than state them as fact.
- Placement/lane/tower-targeting is NOT detected yet — don't claim exact lanes or
  which tower was hit.
- Play timestamps can lag the real placement by ~1-2s — keep timing language
  approximate ("around 1:20").
- "winner" may be null (unknown) — if so, don't declare a final result.

# MATCH DATA
Player deck: {", ".join(player_deck) or "(unknown)"}
Opponent deck: {", ".join(opponent_deck) or "(unknown)"}
Duration: {int(duration // 60)}:{int(duration % 60):02d}   Phases seen: {phase_note}
Player card plays: {len(plays)}

Full structured brief (authoritative):
```json
{brief_json}
```

Return ONLY the project.json object now.
"""
