"""Human-readable analyzer report (2I).

Turns a :class:`~analyzer.models.GameplayAnalysis` into a plain-text report that
surfaces both reconstructed decks, recognition metrics, and -- most usefully for
tuning -- a "Potential Errors" section that flags low-confidence deck cards and
transient event mismatches instead of silently accepting them. ASCII-only so it
prints on the Windows console.
"""

from __future__ import annotations

from analyzer.models import GameplayAnalysis, ReconstructedDeck

# Thresholds only for the human-readable *reason* text (the needs_review flag
# itself is confidence-based, set by the reconstructor).
_LOW_OBS = 10
_LOW_ORB = 30.0


def format_report(analysis: GameplayAnalysis) -> str:
    """Build the full ANALYZER REPORT string."""
    lines: list[str] = []
    lines.append("=" * 52)
    lines.append("               ANALYZER REPORT")
    lines.append("=" * 52)
    lines.append(f"Video   : {analysis.video}")
    lines.append(f"Profile : {analysis.profile_name}")

    _deck_section(lines, "Player deck", analysis.player_deck)
    _deck_section(lines, "Opponent deck", analysis.opponent_deck)

    m = analysis.metrics
    if m is not None:
        pct = 100 * m.cards_identified / m.slots_analyzed if m.slots_analyzed else 0.0
        lines.append("")
        skip_note = ""
        if m.slots_analyzed:
            skip_note = f" ({m.slots_skipped} skipped, {100*m.slots_skipped/m.slots_analyzed:.0f}%)"
        lines.append("Recognition")
        lines.append(f"  Frames processed : {m.frames_processed}")
        lines.append(f"  Slots analyzed   : {m.slots_analyzed}{skip_note}")
        lines.append(f"  Cards identified : {m.cards_identified} ({pct:.0f}%)")
        lines.append(f"  Unknown slots    : {m.unknown_slots}")
        lines.append(f"  Avg ORB / conf   : {m.average_orb_matches:.0f} / {m.average_confidence*100:.0f}%")
        lines.append(f"  Time / FPS       : {m.processing_seconds:.0f}s / {m.fps_processed:.2f} fps")

    problems = _potential_errors(analysis)
    lines.append("")
    lines.append("Potential Errors")
    if problems:
        for p in problems:
            lines.append(f"  - {p}")
    else:
        lines.append("  (none)")
    lines.append("=" * 52)
    return "\n".join(lines)


def _deck_section(lines: list[str], title: str, deck: ReconstructedDeck | None) -> None:
    lines.append("")
    if deck is None or not deck.cards:
        lines.append(f"{title}: (none reconstructed)")
        return
    state = "COMPLETE" if deck.complete else "PARTIAL"
    lines.append(
        f"{title} ({len(deck.cards)}/8 {state}, confidence {deck.confidence*100:.0f}%)"
    )
    for c in deck.cards:
        mark = "!!" if c.needs_review else "OK"
        name = c.slug if c.variant == "base" else f"{c.slug} [{c.variant}]"
        flag = "   <- NEEDS REVIEW" if c.needs_review else ""
        lines.append(
            f"  {mark} {name:<20}  obs={c.observation_count:<4} "
            f"ORB={c.average_match_score:>3.0f}  conf={c.confidence*100:>3.0f}%{flag}"
        )


def _potential_errors(analysis: GameplayAnalysis) -> list[str]:
    problems: list[str] = []
    deck_slugs: set[str] = set()
    for side, deck in (("player", analysis.player_deck), ("opponent", analysis.opponent_deck)):
        if deck is None:
            continue
        for c in deck.cards:
            deck_slugs.add(c.slug)
            if not c.needs_review:
                continue
            reasons = []
            if c.observation_count < _LOW_OBS:
                reasons.append(f"only {c.observation_count} observations")
            if c.average_match_score < _LOW_ORB:
                reasons.append(f"low avg ORB {c.average_match_score:.0f}")
            reason = "; ".join(reasons) or f"low confidence {c.confidence*100:.0f}%"
            problems.append(f"{c.slug} ({side}): {reason}")

    # Transient mismatches: cards that were "played" but never made a deck.
    played: dict[str, int] = {}
    for e in analysis.events:
        played[e.card] = played.get(e.card, 0) + 1
    for card, count in sorted(played.items(), key=lambda kv: -kv[1]):
        if card not in deck_slugs:
            times = "once" if count == 1 else f"{count} times"
            problems.append(f"{card}: played {times} but not in any reconstructed deck (transient)")
    return problems
