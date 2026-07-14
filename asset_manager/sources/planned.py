"""Declared-but-not-yet-implemented source adapters (honest scaffolds).

Each is registered so ``sync`` lists it and its licensing/key status is visible,
but ``fetch`` raises until implemented. SVG Repo and OpenGameArt are DEFERRED:
neither documents an official bulk/download API and their terms don't clearly
permit automated downloading — Heroicons/Kenney/Twemoji cover the same needs.
Keyed ones (Pixabay/Freesound/Tenor) await user API keys. Implementing one =
filling in its ``fetch`` — no manager or editor changes.
"""

from __future__ import annotations

from pathlib import Path

from asset_manager.sources.base import AssetSource, FetchResult, SourceError


class _PlannedSource(AssetSource):
    def fetch(self, dest_root: Path) -> list[FetchResult]:
        raise SourceError(
            f"source '{self.name}' is registered but not implemented yet "
            f"({'needs API key ' + (self.key_env or '') if self.requires_key else 'keyless'})"
        )


# -- Deferred: no official API and bulk-download permission unclear ---------- #
class SvgRepoSource(_PlannedSource):
    name = "svgrepo"
    license_note = "SVG Repo (per-icon licenses vary; use only API/permitted downloads)"
    requires_key = False


class OpenGameArtSource(_PlannedSource):
    name = "opengameart"
    license_note = "OpenGameArt (per-asset licenses vary; respect each asset's terms)"
    requires_key = False


# -- Keyed (await user API keys) --------------------------------------------- #
class PixabaySource(_PlannedSource):
    name = "pixabay"
    license_note = "Pixabay Content License; official API"
    requires_key = True
    key_env = "PIXABAY_API_KEY"


class FreesoundSource(_PlannedSource):
    name = "freesound"
    license_note = "Freesound (CC0/CC-BY per sound); official API (OAuth/token)"
    requires_key = True
    key_env = "FREESOUND_API_KEY"


class TenorSource(_PlannedSource):
    name = "tenor"
    license_note = "Tenor API (respect API terms)"
    requires_key = True
    key_env = "TENOR_API_KEY"
