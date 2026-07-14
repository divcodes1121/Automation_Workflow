"""Kenney — CC0 game assets (SFX, UI, particles), no API key.

Kenney offers free, CC0-licensed asset packs with a direct zip link embedded in
each asset page (donation optional). The adapter fetches the page, finds the
``kenney_<slug>.zip`` URL, downloads it once (cached under ``_cache/downloads``),
and extracts a curated selection into category folders.

Per-pack config keeps this data-driven: which packs, which file patterns, which
target category, and tag rules — adding a pack is one dict entry.
"""

from __future__ import annotations

import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from asset_manager.sources.base import _UA, AssetSource, FetchResult, SourceError

logger = logging.getLogger(__name__)

_PAGE = "https://kenney.nl/assets/{slug}"
_ZIP_RE = re.compile(r"href='(https://kenney\.nl/media/pages/assets/[^']+\.zip)'")


@dataclass(frozen=True)
class _Pack:
    slug: str                 # kenney.nl asset slug
    category: str             # target assets/ category
    include: str              # regex a zip member must match (against lowercase path)
    tags: tuple[str, ...]     # tags every extracted file gets
    limit: int = 60           # safety cap per pack


# The curated packs for the hyper-gaming style. Audio packs ship OGG (ffmpeg-friendly).
_PACKS: tuple[_Pack, ...] = (
    _Pack("impact-sounds", "audio/impacts", r"audio/.*\.ogg$", ("impact", "hit", "thud")),
    _Pack("interface-sounds", "audio/ui", r"audio/.*\.ogg$", ("ui", "click", "interface")),
    _Pack("digital-audio", "audio/ui", r"audio/.*\.ogg$", ("digital", "arcade", "retro")),
    _Pack("sci-fi-sounds", "audio/explosion", r"audio/.*(laser|explosion).*\.ogg$", ("scifi", "laser", "explosion")),
    _Pack("particle-pack", "particles", r"png.*transparent.*\.png$|.*transparent.*\.png$", ("particle", "vfx"), limit=90),
)


class KenneySource(AssetSource):
    name = "kenney"
    license_note = "Kenney game assets, CC0; per-pack zip downloads from kenney.nl"
    requires_key = False

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        results: list[FetchResult] = []
        cache = dest_root / "_cache" / "downloads"
        cache.mkdir(parents=True, exist_ok=True)
        for pack in _PACKS:
            try:
                results.extend(self._fetch_pack(pack, dest_root, cache))
            except SourceError as exc:
                logger.warning("Kenney pack %r failed: %s", pack.slug, exc)
        if not results:
            raise SourceError("no Kenney packs could be fetched")
        return results

    def _fetch_pack(self, pack: _Pack, dest_root: Path, cache: Path) -> list[FetchResult]:
        zip_path = cache / f"kenney_{pack.slug}.zip"
        if not zip_path.is_file():
            self._download(self._zip_url(pack.slug), zip_path, timeout=120)

        out_dir = dest_root / pack.category
        include = re.compile(pack.include)
        results: list[FetchResult] = []
        with zipfile.ZipFile(zip_path) as zf:
            names = [
                n for n in zf.namelist()
                if not n.endswith("/") and include.search(n.lower())
            ][: pack.limit]
            for member in names:
                stem = Path(member).stem.lower().replace(" ", "_")
                fname = f"kenney_{pack.slug}_{stem}{Path(member).suffix.lower()}"
                dest = out_dir / fname
                if not dest.is_file():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(member))
                # Tags: pack tags + words from the filename (footstep_concrete -> both).
                word_tags = [w for w in re.split(r"[^a-z0-9]+", stem) if len(w) > 2]
                results.append(
                    FetchResult(
                        id=f"kenney:{pack.slug}/{stem}",
                        name=f"{pack.slug} {stem}",
                        path=dest,
                        category=pack.category,
                        license="CC0",
                        tags=[*pack.tags, *word_tags],
                    )
                )
        if not results:
            raise SourceError(f"pack {pack.slug}: no members matched {pack.include!r}")
        return results

    @staticmethod
    def _zip_url(slug: str) -> str:
        """Scrape the pack page for its direct zip link (embedded, no auth)."""
        url = _PAGE.format(slug=slug)
        try:
            req = urllib.request.Request(url, headers=_UA)
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"page fetch failed: {url} ({exc})") from exc
        match = _ZIP_RE.search(html)
        if not match:
            raise SourceError(f"no zip link found on {url}")
        return match.group(1)
