"""The source-adapter contract shared by every provider."""

from __future__ import annotations

import logging
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_UA = {"User-Agent": "Mozilla/5.0 (AssetManager; +https://example.local)"}


@dataclass
class FetchResult:
    """What a source hands back for one file, before hashing/registration.

    The source has already written ``path`` to disk (under the category dir);
    the manager hashes it, dedups, and turns it into an :class:`~asset_manager.models.Asset`.
    """

    id: str
    name: str
    path: Path
    category: str
    license: str
    tags: list[str] = field(default_factory=list)
    resolution: str | None = None
    duration: float | None = None
    preview: Path | None = None


class SourceError(RuntimeError):
    """Raised when a source can't fetch (network, missing key, bad response)."""


class AssetSource(ABC):
    """Base class for every asset provider."""

    #: short, stable id used on the CLI and in ``Asset.source``
    name: str = "base"
    #: human note on where assets come from / their licensing
    license_note: str = ""
    #: True if the adapter needs an API key (read from the environment)
    requires_key: bool = False
    #: env var holding the key, when ``requires_key``
    key_env: str | None = None

    @abstractmethod
    def fetch(self, dest_root: Path) -> list[FetchResult]:
        """Download this source's assets under ``dest_root`` and describe them.

        ``dest_root`` is the ``assets/`` root; a source writes into its own
        category subfolders (e.g. ``dest_root / "stickers" / "emojis"``).
        """

    # -- Shared helpers ------------------------------------------------------ #

    @staticmethod
    def _download(url: str, dest: Path, *, timeout: int = 30) -> bool:
        """Fetch ``url`` to ``dest`` (skips if present). Returns True on write."""
        if dest.exists() and dest.stat().st_size > 0:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001 — network is inherently flaky
            raise SourceError(f"download failed: {url} ({exc})") from exc
        if not data:
            raise SourceError(f"empty response: {url}")
        dest.write_bytes(data)
        return True
