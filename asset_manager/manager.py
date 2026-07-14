"""AssetManager — sync, index, verify, and RESOLVE symbolic names to files.

The editor calls :meth:`resolve` with a symbolic name (``emoji_fire``,
``impact_bass``); the manager maps it to a real :class:`~asset_manager.models.Asset`
via the SQLite index — the editor never sees a filename. ``sync`` pulls new
assets from registered sources, ``verify`` prunes corrupt/missing/duplicate
files, and everything is mirrored to ``asset_manifest.json`` for humans.
"""

from __future__ import annotations

import hashlib
import logging
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from asset_manager.models import Asset, AssetManifest
from asset_manager.sources.base import AssetSource, FetchResult, SourceError
from asset_manager.sources.giphy import GiphySource
from asset_manager.sources.google_fonts import GoogleFontsSource
from asset_manager.sources.heroicons import HeroiconsSource
from asset_manager.sources.kenney import KenneySource
from asset_manager.sources.planned import (
    FreesoundSource,
    OpenGameArtSource,
    PixabaySource,
    SvgRepoSource,
    TenorSource,
)
from asset_manager.sources.twemoji import TwemojiSource

logger = logging.getLogger(__name__)

# Keyed sources read their API keys from the environment; the project keeps
# keys in the gitignored .env, so load it when available (best-effort).
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# The full category tree (created on first sync). Sources write into these.
_CATEGORIES = [
    "audio/impacts", "audio/bass", "audio/whoosh", "audio/ui",
    "audio/meme", "audio/explosion",
    "gifs/reactions", "gifs/memes",
    "stickers/emojis", "stickers/reactions", "stickers/arrows",
    "fonts", "icons", "particles", "overlays", "luts",
    "transitions", "comic", "camera",
]

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


class AssetManagerError(RuntimeError):
    """Raised on unrecoverable manager errors (db/registry)."""


class AssetManager:
    """Owns the asset library: its files, its index, and name resolution."""

    def __init__(self, assets_root: Path | None = None) -> None:
        self._root = (assets_root or Path(__file__).resolve().parents[1] / "assets").resolve()
        self._repo_root = self._root.parent
        self._db_path = self._root / "asset_index.db"
        self._manifest_path = self._root / "asset_manifest.json"
        self._cache = self._root / "_cache"
        # Registry: every source is listed; keyed ones are skipped unless enabled.
        self._sources: dict[str, AssetSource] = {
            s.name: s
            for s in (
                TwemojiSource(),
                GoogleFontsSource(), KenneySource(), HeroiconsSource(),
                SvgRepoSource(), OpenGameArtSource(),
                PixabaySource(), FreesoundSource(), GiphySource(), TenorSource(),
            )
        }

    # -- Public API ---------------------------------------------------------- #

    @property
    def sources(self) -> dict[str, AssetSource]:
        return self._sources

    def sync(
        self, source_names: list[str] | None = None, *, include_keyed: bool = False
    ) -> dict[str, int]:
        """Download new assets from the selected sources; refresh the manifest.

        Keyed sources are skipped unless ``include_keyed`` and their key env var
        is set. Returns ``{source_name: assets_registered}``.
        """
        self._ensure_dirs()
        conn = self._connect()
        try:
            names = source_names or list(self._sources)
            summary: dict[str, int] = {}
            for name in names:
                source = self._sources.get(name)
                if source is None:
                    logger.warning("Unknown source %r; skipping", name)
                    continue
                import os

                if source.requires_key and not (
                    include_keyed and source.key_env and os.environ.get(source.key_env)
                ):
                    logger.info("Skipping keyed source %r (no key / not enabled)", name)
                    summary[name] = 0
                    continue
                try:
                    results = source.fetch(self._root)
                except SourceError as exc:
                    logger.warning("Source %r fetch failed: %s", name, exc)
                    summary[name] = 0
                    continue
                registered = sum(self._register(conn, source.name, r) for r in results)
                conn.commit()
                summary[name] = registered
                logger.info("Synced %r: %d asset(s)", name, registered)
            self.rebuild_manifest(conn)
            return summary
        finally:
            conn.close()

    def rebuild_manifest(self, conn: sqlite3.Connection | None = None) -> Path:
        """Regenerate ``asset_manifest.json`` from the index."""
        own = conn is None
        conn = conn or self._connect()
        try:
            assets = self._all_assets(conn)
            sources: dict[str, int] = {}
            categories: dict[str, int] = {}
            for a in assets:
                sources[a.source] = sources.get(a.source, 0) + 1
                categories[a.category] = categories.get(a.category, 0) + 1
            manifest = AssetManifest(
                generated_at=datetime.now(timezone.utc),
                count=len(assets),
                sources=sources,
                categories=categories,
                assets=assets,
            )
            self._manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            logger.info("Wrote manifest (%d assets) -> %s", len(assets), self._manifest_path)
            return self._manifest_path
        finally:
            if own:
                conn.close()

    def search(
        self, tags: list[str], *, category: str | None = None, limit: int = 50
    ) -> list[Asset]:
        """Assets whose tags include ALL of ``tags`` (optionally within a category)."""
        tags = [t.strip().lower() for t in tags if t.strip()]
        conn = self._connect()
        try:
            if not tags:
                rows = conn.execute(
                    "SELECT id FROM assets"
                    + (" WHERE category LIKE ?" if category else "")
                    + " ORDER BY id LIMIT ?",
                    ((f"{category}%", limit) if category else (limit,)),
                ).fetchall()
            else:
                placeholders = ",".join("?" * len(tags))
                params: list = [*tags, len(tags)]
                sql = (
                    f"SELECT a.id FROM assets a JOIN asset_tags t ON a.id = t.asset_id "
                    f"WHERE t.tag IN ({placeholders}) "
                )
                if category:
                    sql += "AND a.category LIKE ? "
                    params.append(f"{category}%")
                sql += (
                    "GROUP BY a.id HAVING COUNT(DISTINCT t.tag) = ? ORDER BY a.id LIMIT ?"
                )
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
            return [self._load_asset(conn, r[0]) for r in rows]
        finally:
            conn.close()

    def random(self, category: str) -> Asset | None:
        """A random asset from a category (prefix match, e.g. ``audio`` or ``audio/bass``)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id FROM assets WHERE category LIKE ? ORDER BY RANDOM() LIMIT 1",
                (f"{category}%",),
            ).fetchone()
            return self._load_asset(conn, row[0]) if row else None
        finally:
            conn.close()

    def resolve(
        self, symbolic: str, *, category: str | None = None, pick_random: bool = False
    ) -> Asset | None:
        """Map a symbolic name (``emoji_fire``, ``comic_boom``) to an asset.

        The name is split into tokens; assets matching ALL tokens as tags win
        (deterministic by id, or random with ``pick_random``). If none match all,
        the best partial match (most tokens, category-prefix tiebreak) is used.
        ``category`` scopes matching to a category prefix — the editor uses this
        to keep e.g. SFX lookups inside ``audio/`` (tags collide across kinds:
        a font can be tagged "impact" too).
        """
        tokens = [t for t in symbolic.strip().lower().replace("-", "_").split("_") if t]
        if not tokens:
            return None
        exact = self.search(tokens, category=category, limit=100)
        if exact:
            return random.choice(exact) if pick_random else exact[0]

        # Partial: rank every asset by how many tokens it matches.
        conn = self._connect()
        try:
            best: tuple[int, int, str] | None = None  # (matches, cat_bonus, id)
            for a in self._all_assets(conn):
                if category and not a.category.startswith(category):
                    continue
                tagset = set(a.tags)
                matches = sum(1 for t in tokens if t in tagset)
                if matches == 0:
                    continue
                cat_bonus = 1 if a.category.split("/")[-1].startswith(tokens[0]) else 0
                cand = (matches, cat_bonus, a.id)
                if best is None or cand[:2] > best[:2]:
                    best = cand
            return self._load_asset(conn, best[2]) if best else None
        finally:
            conn.close()

    def verify(self) -> dict[str, list[str]]:
        """Drop rows for missing/corrupt files; report duplicate-content assets."""
        conn = self._connect()
        report: dict[str, list[str]] = {"missing": [], "corrupt": [], "duplicates": []}
        try:
            seen_hash: dict[str, str] = {}
            for a in self._all_assets(conn):
                fpath = self._repo_root / a.path
                if not fpath.is_file():
                    report["missing"].append(a.id)
                    self._delete(conn, a.id)
                    continue
                if self._sha256(fpath) != a.hash:
                    report["corrupt"].append(a.id)
                    self._delete(conn, a.id)
                    fpath.unlink(missing_ok=True)
                    continue
                if a.hash in seen_hash:
                    report["duplicates"].append(f"{a.id} == {seen_hash[a.hash]}")
                else:
                    seen_hash[a.hash] = a.id
            conn.commit()
            self.rebuild_manifest(conn)
            return report
        finally:
            conn.close()

    def stats(self) -> dict[str, object]:
        conn = self._connect()
        try:
            total = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            by_cat = dict(
                conn.execute(
                    "SELECT category, COUNT(*) FROM assets GROUP BY category ORDER BY category"
                ).fetchall()
            )
            by_src = dict(
                conn.execute(
                    "SELECT source, COUNT(*) FROM assets GROUP BY source"
                ).fetchall()
            )
            return {"total": total, "by_category": by_cat, "by_source": by_src}
        finally:
            conn.close()

    # -- Registration / db --------------------------------------------------- #

    def _register(self, conn: sqlite3.Connection, source: str, r: FetchResult) -> int:
        """Insert one fetched file into the index; dedup by content hash."""
        if not r.path.is_file():
            logger.warning("Fetched file missing on disk: %s", r.path)
            return 0
        digest = self._sha256(r.path)
        dup = conn.execute(
            "SELECT id FROM assets WHERE hash = ? AND id != ?", (digest, r.id)
        ).fetchone()
        if dup:
            logger.info("Duplicate content, skipping %s (== %s)", r.id, dup[0])
            r.path.unlink(missing_ok=True)
            return 0

        preview_rel = None
        if r.preview and r.preview.is_file():
            preview_rel = r.preview.relative_to(self._repo_root).as_posix()
        thumb = self._cache_thumbnail(r.id, r.path)
        if thumb is not None:
            preview_rel = thumb.relative_to(self._repo_root).as_posix()

        conn.execute(
            "INSERT OR REPLACE INTO assets"
            "(id,name,source,license,category,path,preview,hash,resolution,duration)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                r.id, r.name, source, r.license, r.category,
                r.path.relative_to(self._repo_root).as_posix(),
                preview_rel, digest, r.resolution, r.duration,
            ),
        )
        conn.execute("DELETE FROM asset_tags WHERE asset_id = ?", (r.id,))
        conn.executemany(
            "INSERT OR IGNORE INTO asset_tags(asset_id,tag) VALUES (?,?)",
            [(r.id, t.strip().lower()) for t in r.tags if t.strip()],
        )
        return 1

    def _connect(self) -> sqlite3.Connection:
        self._root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
              id TEXT PRIMARY KEY, name TEXT, source TEXT, license TEXT,
              category TEXT, path TEXT UNIQUE, preview TEXT, hash TEXT,
              resolution TEXT, duration REAL
            );
            CREATE TABLE IF NOT EXISTS asset_tags (
              asset_id TEXT, tag TEXT, PRIMARY KEY (asset_id, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_assets_category ON assets(category);
            CREATE INDEX IF NOT EXISTS idx_assets_hash ON assets(hash);
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON asset_tags(tag);
            """
        )
        return conn

    def _delete(self, conn: sqlite3.Connection, asset_id: str) -> None:
        conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        conn.execute("DELETE FROM asset_tags WHERE asset_id = ?", (asset_id,))

    def _all_assets(self, conn: sqlite3.Connection) -> list[Asset]:
        ids = [r[0] for r in conn.execute("SELECT id FROM assets ORDER BY id").fetchall()]
        return [self._load_asset(conn, i) for i in ids]

    def _load_asset(self, conn: sqlite3.Connection, asset_id: str) -> Asset:
        row = conn.execute(
            "SELECT id,name,source,license,category,path,preview,hash,resolution,duration"
            " FROM assets WHERE id = ?",
            (asset_id,),
        ).fetchone()
        tags = [
            t[0] for t in conn.execute(
                "SELECT tag FROM asset_tags WHERE asset_id = ? ORDER BY tag", (asset_id,)
            ).fetchall()
        ]
        return Asset(
            id=row[0], name=row[1], source=row[2], license=row[3], category=row[4],
            path=row[5], preview=row[6], hash=row[7], resolution=row[8],
            duration=row[9], tags=tags,
        )

    # -- Assets / files ------------------------------------------------------ #

    def _ensure_dirs(self) -> None:
        for cat in _CATEGORIES:
            (self._root / cat).mkdir(parents=True, exist_ok=True)
        (self._cache / "thumbnails").mkdir(parents=True, exist_ok=True)
        (self._cache / "waveforms").mkdir(parents=True, exist_ok=True)

    def _cache_thumbnail(self, asset_id: str, path: Path) -> Path | None:
        """A 128px thumbnail for image assets (Pillow); None if not an image."""
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            return None
        try:
            from PIL import Image

            dest = self._cache / "thumbnails" / f"{asset_id.replace(':', '_')}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as img:
                img = img.convert("RGBA")
                img.thumbnail((128, 128))
                img.save(dest)
            return dest
        except Exception as exc:  # noqa: BLE001 — thumbnails are best-effort
            logger.debug("Thumbnail failed for %s: %s", asset_id, exc)
            return None

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
