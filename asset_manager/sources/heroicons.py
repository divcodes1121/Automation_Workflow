"""Heroicons — MIT-licensed UI icons (arrows, markers), no API key.

Downloads a curated set of solid 24px SVGs straight from the tailwindlabs
GitHub repo (raw files; MIT allows redistribution). These back the editor's
UI-pointer needs: arrows, spotlights, markers, crosshair-ish targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

from asset_manager.sources.base import AssetSource, FetchResult, SourceError

logger = logging.getLogger(__name__)

_RAW = (
    "https://raw.githubusercontent.com/tailwindlabs/heroicons/master/"
    "optimized/24/solid/{name}.svg"
)

# icon name in repo -> extra tags
_ICONS: dict[str, list[str]] = {
    "arrow-up": ["arrow", "up", "pointer"],
    "arrow-down": ["arrow", "down", "pointer"],
    "arrow-left": ["arrow", "left", "pointer"],
    "arrow-right": ["arrow", "right", "pointer"],
    "arrow-trending-up": ["arrow", "trending", "up", "graph"],
    "arrow-trending-down": ["arrow", "trending", "down", "graph"],
    "cursor-arrow-rays": ["cursor", "click", "highlight"],
    "map-pin": ["pin", "marker", "location"],
    "star": ["star", "favorite", "rating"],
    "bolt": ["bolt", "lightning", "electric"],
    "fire": ["fire", "flame", "hot"],
    "trophy": ["trophy", "winner", "victory"],
    "eye": ["eye", "watch", "spotlight"],
    "magnifying-glass": ["magnify", "zoom", "search"],
    "exclamation-triangle": ["warning", "alert", "attention"],
    "question-mark-circle": ["question", "confused", "unknown"],
    "check-circle": ["check", "success", "correct"],
    "x-circle": ["cross", "fail", "wrong"],
    "hand-thumb-up": ["thumbsup", "like", "approve"],
    "hand-thumb-down": ["thumbsdown", "dislike", "fail"],
}


class HeroiconsSource(AssetSource):
    name = "heroicons"
    license_note = "Heroicons, MIT; SVGs from github.com/tailwindlabs/heroicons"
    requires_key = False

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        out_dir = dest_root / "icons"
        results: list[FetchResult] = []
        for icon, extra in _ICONS.items():
            slug = icon.replace("-", "_")
            dest = out_dir / f"heroicon_{slug}.svg"
            try:
                self._download(_RAW.format(name=icon), dest)
            except SourceError as exc:
                logger.warning("Icon %r failed: %s", icon, exc)
                continue
            results.append(
                FetchResult(
                    id=f"heroicons:{slug}",
                    name=f"icon {icon}",
                    path=dest,
                    category="icons",
                    license="MIT",
                    tags=["icon", "ui", *extra],
                    resolution="24x24",
                )
            )
        return results
