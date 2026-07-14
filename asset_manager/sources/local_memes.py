"""Local user-managed packs — registers files the user drops into ``Memes/``.

Not a downloader: it indexes what's already on disk so the editor can resolve
those files symbolically like any other asset (the editor never references
filenames). ``Memes/`` is gitignored (third-party, licensing varies) — assets
register with ``license="user-managed"`` and keep their original location;
``category`` is metadata, not a storage path.

Tags come from filename words plus a curated alias map so the existing
recipe vocabulary (thud/hype/amazing/tension/comedy...) resolves cleanly.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from asset_manager.sources.base import AssetSource, FetchResult

logger = logging.getLogger(__name__)

_AUDIO_SUFFIXES = {".mp3", ".wav", ".ogg"}
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".gif"}

# filename-substring -> extra tags (the editing vocabulary).
_ALIASES: dict[str, list[str]] = {
    "thud": ["impact", "bass", "boom"],
    "yeah-boy": ["hype", "shout", "celebration"],
    "woooooaah": ["hype", "crowd", "wow"],
    "amazing": ["amazing", "reaction", "praise"],
    "wowow": ["amazing", "wow", "excited"],
    "waterphone": ["tension", "riser", "eerie"],
    "horn": ["comedy", "fail", "goofy"],
    "sus": ["comedy", "sus", "meme"],
    "what-meme": ["comedy", "confused", "what"],
    "wait-a-minute": ["comedy", "confused", "interrupt"],
    "fart": ["comedy", "fail", "gross"],
    "fah": ["comedy", "shout"],
    "i-got-this": ["comedy", "confident"],
    "memeclick": ["pop", "click", "ui"],
    "aa-with-reverb": ["comedy", "scream", "reverb"],
    "meow": ["comedy", "cat", "cute"],
    "disappointed": ["disappointed", "reaction", "subject"],
    "dancing": ["celebration", "dance", "party"],
}


class LocalMemesSource(AssetSource):
    name = "local"
    license_note = "User-managed Memes/ packs (third-party; licensing varies)"
    requires_key = False

    def fetch(self, dest_root: Path) -> list[FetchResult]:
        repo_root = dest_root.parent
        results: list[FetchResult] = []
        results += self._scan(repo_root / "Memes" / "SFX", "audio/meme", _AUDIO_SUFFIXES)
        results += self._scan(repo_root / "Memes" / "IMGS", "gifs/memes", _VIDEO_SUFFIXES)
        if not results:
            logger.info("No local Memes/ files found (folder is user-managed)")
        return results

    def _scan(self, folder: Path, category: str, suffixes: set[str]) -> list[FetchResult]:
        if not folder.is_dir():
            return []
        results: list[FetchResult] = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in suffixes or not path.is_file():
                continue
            stem = path.stem.lower()
            word_tags = [w for w in re.split(r"[^a-z]+", stem) if len(w) > 2]
            alias_tags = [t for key, tags in _ALIASES.items() if key in stem for t in tags]
            slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:60]
            results.append(
                FetchResult(
                    id=f"local:{slug}",
                    name=stem,
                    path=path,
                    category=category,
                    license="user-managed",
                    tags=["local", "meme", *word_tags, *alias_tags],
                )
            )
        return results
