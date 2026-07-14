"""GIPHY — reaction GIFs via the official API (key: ``GIPHY_API_KEY``).

Curated search queries map to sticker/GIF categories; for each query the top
results are downloaded as MP4 (the API's ``original_mp4`` rendition — small and
directly usable as an ffmpeg input, unlike heavy animated GIF decoding).

NOTE GIPHY content is licensed for use via the API with attribution; each asset
records ``license="GIPHY API terms"`` and keeps the GIPHY id in the asset id so
attribution/lookup stays possible.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from asset_manager.sources.base import _UA, AssetSource, FetchResult, SourceError

logger = logging.getLogger(__name__)

_SEARCH = "https://api.giphy.com/v1/gifs/search?api_key={key}&q={q}&limit={limit}&rating=pg-13"


@dataclass(frozen=True)
class _Query:
    q: str                  # search string
    slug: str               # filename/tag stem
    category: str           # target assets/ category
    tags: tuple[str, ...]
    limit: int = 4


# The meme-reaction set for the hyper-gaming style (skull/bruh/facepalm/etc).
_QUERIES: tuple[_Query, ...] = (
    _Query("bruh reaction", "bruh", "gifs/reactions", ("bruh", "reaction")),
    _Query("skull dead laughing", "skull", "gifs/reactions", ("skull", "dead", "laughing")),
    _Query("facepalm reaction", "facepalm", "gifs/reactions", ("facepalm", "fail")),
    _Query("shocked surprised reaction", "shocked", "gifs/reactions", ("shocked", "surprised")),
    _Query("no way reaction", "noway", "gifs/reactions", ("noway", "disbelief")),
    _Query("celebration lets go reaction", "letsgo", "gifs/reactions", ("celebration", "hype", "letsgo")),
    _Query("crying laughing meme", "lol", "gifs/memes", ("lol", "laughing", "meme")),
    _Query("gg well played gaming", "gg", "gifs/memes", ("gg", "gaming", "victory")),
)


class GiphySource(AssetSource):
    name = "giphy"
    license_note = "GIPHY API (respect API terms; attribution retained via asset id)"
    requires_key = True
    key_env = "GIPHY_API_KEY"

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        key = os.environ.get(self.key_env or "")
        if not key:
            raise SourceError(f"GIPHY needs {self.key_env} in the environment")
        results: list[FetchResult] = []
        for query in _QUERIES:
            try:
                results.extend(self._fetch_query(query, key, dest_root))
            except SourceError as exc:
                logger.warning("GIPHY query %r failed: %s", query.q, exc)
        if not results:
            raise SourceError("no GIPHY queries succeeded")
        return results

    def _fetch_query(self, query: _Query, key: str, dest_root: Path) -> list[FetchResult]:
        url = _SEARCH.format(key=key, q=urllib.parse.quote(query.q), limit=query.limit)
        try:
            req = urllib.request.Request(url, headers=_UA)
            payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"search failed: {query.q} ({exc})") from exc

        out_dir = dest_root / query.category
        results: list[FetchResult] = []
        for rank, gif in enumerate(payload.get("data", []), start=1):
            gid = gif.get("id", "")
            images = gif.get("images", {})
            # Prefer the mp4 rendition (ffmpeg-friendly); fall back to the gif.
            rendition = images.get("original_mp4") or {}
            src, suffix = rendition.get("mp4"), ".mp4"
            if not src:
                rendition = images.get("original") or {}
                src, suffix = rendition.get("url"), ".gif"
            if not gid or not src:
                continue
            dest = out_dir / f"giphy_{query.slug}_{rank:02d}{suffix}"
            try:
                self._download(src, dest, timeout=60)
            except SourceError as exc:
                logger.warning("GIPHY download failed: %s", exc)
                continue
            width = rendition.get("width")
            height = rendition.get("height")
            results.append(
                FetchResult(
                    id=f"giphy:{gid}",
                    name=(gif.get("title") or f"{query.slug} {rank}").strip(),
                    path=dest,
                    category=query.category,
                    license="GIPHY API terms",
                    tags=["giphy", "reaction", query.slug, *query.tags],
                    resolution=f"{width}x{height}" if width and height else None,
                )
            )
        if not results:
            raise SourceError(f"query {query.q!r}: nothing downloadable")
        return results
