"""Twemoji — emoji graphics (CC-BY 4.0), no API key.

Downloads the curated emoji set the editor uses as stickers. Each emoji gets the
72x72 PNG (directly usable by ffmpeg overlay) plus its SVG master (crisp, for
future hi-res rasterization) as the preview.
"""

from __future__ import annotations

import logging
from pathlib import Path

from asset_manager.sources.base import AssetSource, FetchResult, SourceError

logger = logging.getLogger(__name__)

# jsDelivr mirror of jdecked/twemoji (fe0f variation selectors stripped).
_PNG = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/72x72/{code}.png"
_SVG = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@latest/assets/svg/{code}.svg"

# name -> (codepoint filename, extra tags). 'emoji' + the name are always tags.
_EMOJI: dict[str, tuple[str, list[str]]] = {
    "joy": ("1f602", ["laughing", "lol", "haha"]),
    "skull": ("1f480", ["dead", "bruh", "cringe"]),
    "crying": ("1f62d", ["sob", "sad", "tears"]),
    "fire": ("1f525", ["lit", "hot", "flames"]),
    "crown": ("1f451", ["king", "win", "royale"]),
    "flushed": ("1f633", ["shocked", "embarrassed"]),
    "mindblown": ("1f92f", ["shocked", "exploding", "wow"]),
    "crossbones": ("2620", ["skull", "death", "danger"]),
    "money": ("1f4b8", ["cash", "wings", "broke"]),
    "lightning": ("26a1", ["electric", "zap", "bolt"]),
    "rocket": ("1f680", ["launch", "space", "boost"]),
    "target": ("1f3af", ["bullseye", "hit", "aim"]),
    "boom": ("1f4a5", ["explosion", "impact", "collision"]),
    "clap": ("1f44f", ["applause", "clapping", "gg"]),
    "moai": ("1f5ff", ["stone", "gigachad", "stoneface"]),
}


class TwemojiSource(AssetSource):
    name = "twemoji"
    license_note = "Twemoji graphics, CC-BY 4.0 (attribution: Twitter/jdecked)"
    requires_key = False

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        out_dir = dest_root / "stickers" / "emojis"
        results: list[FetchResult] = []
        for name, (code, extra) in _EMOJI.items():
            png = out_dir / f"{name}.png"
            svg = out_dir / f"{name}.svg"
            try:
                self._download(_PNG.format(code=code), png)
            except SourceError as exc:
                logger.warning("Emoji %r failed: %s", name, exc)
                continue
            try:
                self._download(_SVG.format(code=code), svg)
            except Exception:  # noqa: BLE001 — SVG is optional (preview only)
                svg = None  # type: ignore[assignment]
            results.append(
                FetchResult(
                    id=f"twemoji:{name}",
                    name=f"emoji {name}",
                    path=png,
                    category="stickers/emojis",
                    license="CC-BY 4.0",
                    tags=["emoji", name, *extra],
                    resolution="72x72",
                    preview=svg,
                )
            )
        return results
