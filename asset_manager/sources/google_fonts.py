"""Google Fonts — caption/display typefaces (OFL / Apache), no API key.

Downloads a curated set of bold display fonts (the hyper-gaming caption look)
as raw TTFs from the github.com/google/fonts repo. Families live under a
license directory (``ofl/`` or ``apache/``) — recorded per asset; both licenses
permit bundling.
"""

from __future__ import annotations

import logging
from pathlib import Path

from asset_manager.sources.base import AssetSource, FetchResult, SourceError

logger = logging.getLogger(__name__)

_RAW = "https://raw.githubusercontent.com/google/fonts/main/{path}"

# family -> (repo path, license, extra tags). Verified against the repo.
_FONTS: dict[str, tuple[str, str, list[str]]] = {
    "bangers": ("ofl/bangers/Bangers-Regular.ttf", "OFL-1.1", ["comic", "shout", "impact"]),
    "luckiest_guy": ("apache/luckiestguy/LuckiestGuy-Regular.ttf", "Apache-2.0", ["comic", "bold", "fun"]),
    "anton": ("ofl/anton/Anton-Regular.ttf", "OFL-1.1", ["condensed", "bold", "headline"]),
    "bungee": ("ofl/bungee/Bungee-Regular.ttf", "OFL-1.1", ["blocky", "urban", "display"]),
    "russo_one": ("ofl/russoone/RussoOne-Regular.ttf", "OFL-1.1", ["esports", "tech", "bold"]),
    "archivo_black": ("ofl/archivoblack/ArchivoBlack-Regular.ttf", "OFL-1.1", ["heavy", "bold", "poster"]),
    "titan_one": ("ofl/titanone/TitanOne-Regular.ttf", "OFL-1.1", ["round", "bubbly", "cartoon"]),
    "permanent_marker": ("apache/permanentmarker/PermanentMarker-Regular.ttf", "Apache-2.0", ["marker", "handwritten"]),
    "press_start_2p": ("ofl/pressstart2p/PressStart2P-Regular.ttf", "OFL-1.1", ["pixel", "retro", "arcade"]),
}


class GoogleFontsSource(AssetSource):
    name = "google_fonts"
    license_note = "Google Fonts (OFL/Apache families); TTFs from github.com/google/fonts"
    requires_key = False

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        out_dir = dest_root / "fonts"
        results: list[FetchResult] = []
        for family, (path, license_, extra) in _FONTS.items():
            dest = out_dir / f"{family}.ttf"
            try:
                self._download(_RAW.format(path=path), dest)
            except SourceError as exc:
                logger.warning("Font %r failed: %s", family, exc)
                continue
            results.append(
                FetchResult(
                    id=f"google_fonts:{family}",
                    name=f"font {family.replace('_', ' ')}",
                    path=dest,
                    category="fonts",
                    license=license_,
                    tags=["font", "caption", family, *extra],
                )
            )
        if not results:
            raise SourceError("no fonts could be downloaded")
        return results
