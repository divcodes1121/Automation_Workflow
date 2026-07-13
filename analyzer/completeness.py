"""Template completeness (2I+): does the library cover the cards in the game?

This is NOT a detector. It answers a different, crucial question: when the
analyzer reports a hand slot as "unknown", is that a *detector* failure or is
there simply **no template** for that card? The Evolved-Goblins case proved the
game ships art the library doesn't have; as CR adds champions/evolutions/cards,
the library must be kept in sync or every new card becomes a permanent false
"unknown".

The check compares the on-disk asset library against a curated expectation of
cards known to exist, and reports what is missing plus data-integrity issues
(e.g. an evolution whose slug does not match any base card). The expectation
lists are intentionally maintainable -- extend them as the game grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analyzer.config import AnalyzerSettings, get_analyzer_settings

# Champions (they live in assets/cards/). Update as new champions release.
EXPECTED_CHAMPIONS: frozenset[str] = frozenset(
    {
        "archer-queen", "golden-knight", "skeleton-king", "mighty-miner",
        "monk", "phoenix", "little-prince", "goblinstein", "boss-bandit",
    }
)

# Evolutions CONFIRMED to exist in-game but that may be absent from the asset
# set. Grow this as gaps are discovered (Evolved Goblins found 2026-07-12 via a
# misdetection: the in-game gold-helmet goblin matched weakly to 'furnace').
CONFIRMED_EVOLUTIONS: frozenset[str] = frozenset({"goblins"})


@dataclass
class CompletenessReport:
    base_count: int
    evolution_count: int
    hero_count: int = 0
    champions_present: list[str] = field(default_factory=list)
    champions_missing: list[str] = field(default_factory=list)
    missing_evolutions: list[str] = field(default_factory=list)
    evolutions_without_base: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not (
            self.champions_missing
            or self.missing_evolutions
            or self.evolutions_without_base
        )


def _slugs(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.png")} if directory.is_dir() else set()


def check_completeness(settings: AnalyzerSettings | None = None) -> CompletenessReport:
    """Compare the asset library against expected cards."""
    settings = settings or get_analyzer_settings()
    base = _slugs(settings.cards_dir)
    evolutions = _slugs(settings.evolutions_dir)
    heroes = _slugs(settings.heroes_dir)
    evolution_art = evolutions | heroes  # both evolution render styles

    return CompletenessReport(
        base_count=len(base),
        evolution_count=len(evolutions),
        hero_count=len(heroes),
        champions_present=sorted(EXPECTED_CHAMPIONS & base),
        champions_missing=sorted(EXPECTED_CHAMPIONS - base),
        missing_evolutions=sorted(CONFIRMED_EVOLUTIONS - evolution_art),
        # An evolution art whose slug has no matching base card is almost always
        # a typo (e.g. 'furnance' vs base 'furnace') or a stray file.
        evolutions_without_base=sorted(evolution_art - base),
    )


def format_completeness_report(report: CompletenessReport) -> str:
    """Render the completeness report as plain text (ASCII-only)."""
    lines: list[str] = []
    lines.append("=" * 52)
    lines.append("            TEMPLATE COMPLETENESS")
    lines.append("=" * 52)
    lines.append(f"Base cards      : {report.base_count}")
    lines.append(f"Evolutions      : {report.evolution_count} (pink) + {report.hero_count} (gold/in-battle)")
    lines.append(
        f"Champions       : {len(report.champions_present)}/"
        f"{len(report.champions_present) + len(report.champions_missing)} present"
    )
    lines.append("")
    lines.append("Missing (add these templates to stop false 'unknown's):")
    if report.champions_missing:
        for slug in report.champions_missing:
            lines.append(f"  X champion  {slug}")
    if report.missing_evolutions:
        for slug in report.missing_evolutions:
            lines.append(f"  X evolution {slug}  (e.g. Evolved {slug.replace('-', ' ').title()})")
    if not report.champions_missing and not report.missing_evolutions:
        lines.append("  (none known)")
    lines.append("")
    lines.append("Data issues:")
    if report.evolutions_without_base:
        for slug in report.evolutions_without_base:
            lines.append(f"  ! evolution '{slug}' has no matching base card (typo/stray?)")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(f"Library {'COMPLETE' if report.complete else 'has known GAPS'}.")
    lines.append("=" * 52)
    return "\n".join(lines)
