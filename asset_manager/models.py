"""Pydantic models: one asset's metadata + the manifest that lists them all."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = "1.0"

_STRICT_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Asset(BaseModel):
    """One downloaded asset and everything needed to find, license, and use it.

    ``path`` (and ``preview``) are repo-relative POSIX strings so the manifest is
    portable. ``id`` is stable (``<source>:<slug>``) so re-syncs are idempotent.
    """

    model_config = _STRICT_CONFIG

    id: str
    name: str
    source: str
    license: str
    tags: list[str] = Field(default_factory=list)
    category: str            # e.g. "stickers/emojis", "audio/impacts"
    path: str                # repo-relative path to the asset file
    preview: str | None = None   # repo-relative path to a thumbnail, if any
    hash: str                # sha256 of the asset file
    resolution: str | None = None   # "72x72" for images/video, else None
    duration: float | None = None   # seconds for audio/video/gif, else None


class AssetManifest(BaseModel):
    """The full, human-readable index of managed assets (rebuilt from the db)."""

    model_config = _STRICT_CONFIG

    schema_version: str = MANIFEST_SCHEMA_VERSION
    generated_at: datetime
    count: int
    sources: dict[str, int] = Field(default_factory=dict)   # source -> asset count
    categories: dict[str, int] = Field(default_factory=dict)  # category -> count
    assets: list[Asset]
