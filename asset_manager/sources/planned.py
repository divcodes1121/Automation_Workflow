"""Declared-but-not-yet-implemented source adapters (honest scaffolds).

Each is registered so ``sync`` lists it and its licensing/key status is visible,
but ``fetch`` raises until the per-source specifics are wired (pack/family/icon
lists, API pagination). Keyless ones come next; keyed ones await user API keys.
Implementing one = filling in its ``fetch`` — no manager or editor changes.
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


# -- Keyless (implement next) ------------------------------------------------ #
class GoogleFontsSource(_PlannedSource):
    name = "google_fonts"
    license_note = "Google Fonts (mostly OFL); font files from github.com/google/fonts"
    requires_key = False


class KenneySource(_PlannedSource):
    name = "kenney"
    license_note = "Kenney game assets, CC0; per-pack zip downloads from kenney.nl"
    requires_key = False


class HeroiconsSource(_PlannedSource):
    name = "heroicons"
    license_note = "Heroicons, MIT; SVGs from github.com/tailwindlabs/heroicons"
    requires_key = False


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


class GiphySource(_PlannedSource):
    name = "giphy"
    license_note = "GIPHY API (respect API terms)"
    requires_key = True
    key_env = "GIPHY_API_KEY"


class TenorSource(_PlannedSource):
    name = "tenor"
    license_note = "Tenor API (respect API terms)"
    requires_key = True
    key_env = "TENOR_API_KEY"
