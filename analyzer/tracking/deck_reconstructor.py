"""Deck reconstruction (2I): rebuild a player's 8-card deck across a match.

Persistent battle understanding: rather than trusting any single frame, this
accumulates hand observations over the whole game. Each observed card keeps
confidence stats (first/last seen, observation count, average match score). The
reconstructed deck is the highest-confidence *confirmed* cards (seen at least
``deck_min_observations`` times, filtering one-off misdetections), capped at the
8 cards a Clash Royale deck always has. Because the deck is the top-8 by
observation count, a spurious card seen once or twice never displaces a real
card seen many times -- conflicts are resolved by confidence, not by replacement.
"""

from __future__ import annotations

from dataclasses import dataclass

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import DECK_SIZE, DeckCard, HandReading, ReconstructedDeck


@dataclass
class _CardStat:
    slug: str
    variant: str
    first_seen: float
    last_seen: float
    count: int
    score_sum: float


class DeckReconstructor:
    """Accumulates hand observations for one side into a :class:`ReconstructedDeck`."""

    def __init__(self, side: str, settings: AnalyzerSettings | None = None) -> None:
        self._side = side
        self._settings = settings or get_analyzer_settings()
        self._stats: dict[str, _CardStat] = {}

    def observe(self, slug: str, variant: str, timestamp: float, score: float) -> None:
        """Record a single sighting of ``slug`` at ``timestamp`` with ``score``."""
        stat = self._stats.get(slug)
        if stat is None:
            self._stats[slug] = _CardStat(slug, variant, timestamp, timestamp, 1, score)
            return
        stat.count += 1
        stat.score_sum += score
        stat.first_seen = min(stat.first_seen, timestamp)
        stat.last_seen = max(stat.last_seen, timestamp)

    def update(self, reading: HandReading) -> None:
        """Fold one hand reading's matched slots into the running deck."""
        for slot in reading.slots:
            if slot.matched and slot.card:
                self.observe(slot.card, slot.variant or "base", reading.timestamp_seconds, slot.score)

    def deck(self) -> ReconstructedDeck:
        """Build the current best-estimate deck (top-8 confirmed cards)."""
        settings = self._settings
        method = settings.matching_method
        min_obs = settings.deck_min_observations
        strong = settings.deck_strong_observations
        quality_norm = settings.deck_quality_norm
        review_below = settings.deck_review_confidence

        confirmed = [s for s in self._stats.values() if s.count >= min_obs]
        confirmed.sort(key=lambda s: (s.count, s.score_sum), reverse=True)
        top = confirmed[:DECK_SIZE]

        cards: list[DeckCard] = []
        for s in top:
            avg_score = s.score_sum / s.count
            # Confidence combines how OFTEN we saw it with how WELL it matched.
            confidence = min(1.0, s.count / strong) * min(1.0, avg_score / quality_norm)
            cards.append(
                DeckCard(
                    slug=s.slug,
                    variant=s.variant,
                    first_seen_time=round(s.first_seen, 2),
                    last_seen_time=round(s.last_seen, 2),
                    observation_count=s.count,
                    average_match_score=round(avg_score, 2),
                    matching_method=method,
                    confidence=round(confidence, 3),
                    needs_review=confidence < review_below,
                )
            )
        completion = min(DECK_SIZE, len(cards)) / DECK_SIZE * 100.0
        overall = sum(c.confidence for c in cards) / len(cards) if cards else 0.0
        return ReconstructedDeck(
            side=self._side,
            cards=cards,
            complete=len(cards) >= DECK_SIZE,
            completion_percent=round(completion, 1),
            confidence=round(overall, 3),
        )
