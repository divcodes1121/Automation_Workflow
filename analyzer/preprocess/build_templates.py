"""Build the card template library from bundled Clash Royale art (slice 2A).

:class:`TemplateBuilder` preprocesses every card PNG under the configured source
directories into a normalized, cached representation the detectors will match
against, and writes a manifest cataloging them.

For each source PNG it produces (all normalized to a canonical height, width
preserving the source aspect ratio):

* ``rgba``  -- resized colour + alpha (RGBA),
* ``gray``  -- grayscale, for plain template matching,
* ``mask``  -- 8-bit alpha mask, for masked matching (ignores transparent art),
* ``edges`` -- Canny edges, in case edge matching beats intensity matching.

These are stored together in a compressed ``.npz`` (with an embedded schema
version) under a version-stamped cache directory so a future cache format can
coexist. OpenCV and numpy are imported at module load -- this is the CV module;
the CLI imports it lazily so the rest of the package imports without them.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import (
    TEMPLATE_LIBRARY_SCHEMA_VERSION,
    CardTemplate,
    CardVariant,
    TemplateLibrary,
)

logger = logging.getLogger(__name__)

_IMAGE_SUFFIX = ".png"


class TemplateBuildError(Exception):
    """Raised when the template library cannot be built (e.g. missing source)."""


class TemplateBuilder:
    """Preprocesses card art into a cached, catalogued template library.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~analyzer.config.get_analyzer_settings`).
    """

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self._settings: AnalyzerSettings = settings or get_analyzer_settings()

    # -- Public API -----------------------------------------------------------

    def build(self, dry_run: bool = False) -> TemplateLibrary:
        """Build (or, when ``dry_run``, just plan) the template library.

        Raises
        ------
        TemplateBuildError
            If a configured source directory does not exist.
        """
        sources = [
            (self._settings.cards_dir, CardVariant.BASE),
            (self._settings.evolutions_dir, CardVariant.EVOLUTION),
            (self._settings.heroes_dir, CardVariant.HERO),
        ]
        for directory, _variant in sources:
            if not directory.is_dir():
                raise TemplateBuildError(f"Source directory not found: {directory}")

        # Base slugs let us flag evolutions that have no matching base card.
        base_slugs = {p.stem for p in self._pngs(self._settings.cards_dir)}

        templates: list[CardTemplate] = []
        skipped: list[str] = []
        cache_root = self._settings.cache_root()

        for directory, variant in sources:
            for png in self._pngs(directory):
                try:
                    template = self._process(
                        png, variant, base_slugs, cache_root, dry_run=dry_run
                    )
                except _ImageReadError as exc:
                    logger.warning("Skipping unreadable template %s: %s", png.name, exc)
                    skipped.append(f"{png.name}: {exc}")
                    continue
                templates.append(template)

        return self._catalog(templates, skipped)

    def save(self, library: TemplateLibrary, destination: Path | None = None) -> Path:
        """Write the manifest JSON. Returns its path."""
        dest = destination or self._settings.cache_root() / "template_manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(library.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved template manifest (%d cards) to %s", library.card_count, dest)
        return dest

    # -- Internals ------------------------------------------------------------

    @staticmethod
    def _pngs(directory: Path) -> list[Path]:
        """Source PNGs in a directory, sorted for deterministic ordering."""
        return sorted(p for p in directory.iterdir() if p.suffix.lower() == _IMAGE_SUFFIX)

    def _process(
        self,
        png: Path,
        variant: CardVariant,
        base_slugs: set[str],
        cache_root: Path,
        *,
        dry_run: bool,
    ) -> CardTemplate:
        """Preprocess one PNG into a :class:`CardTemplate` (+ cache write)."""
        image = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise _ImageReadError("cv2 could not decode the file")

        source_h, source_w = image.shape[:2]
        bgr, mask = self._split_bgr_alpha(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Normalize to the canonical height, preserving aspect ratio.
        target_h = self._settings.template_height
        target_w = max(1, round(source_w * target_h / source_h))
        gray_r = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask_r = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgba_r = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if image.shape[2] == 4
            else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA),
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        edges_r = cv2.Canny(gray_r, 50, 150)

        slug = png.stem
        cache_path = cache_root / variant.value / f"{slug}.npz"
        if not dry_run:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                version=np.asarray(TEMPLATE_LIBRARY_SCHEMA_VERSION),
                gray=gray_r,
                rgba=rgba_r,
                mask=mask_r,
                edges=edges_r,
            )

        has_base_match = variant is CardVariant.BASE or slug in base_slugs
        return CardTemplate(
            slug=slug,
            display_name=slug.replace("-", " ").title(),
            variant=variant,
            source_path=png,
            cache_path=cache_path,
            source_hash=self._sha256(png),
            source_width=source_w,
            source_height=source_h,
            template_width=target_w,
            template_height=target_h,
            has_base_match=has_base_match,
        )

    @staticmethod
    def _split_bgr_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(bgr, alpha_mask)`` for a BGR/BGRA/gray OpenCV image."""
        if image.ndim == 2:  # grayscale source
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            return bgr, mask
        if image.shape[2] == 4:  # BGRA
            bgr = image[:, :, :3]
            mask = image[:, :, 3].copy()
            return bgr, mask
        bgr = image[:, :, :3]  # BGR, fully opaque
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        return bgr, mask

    @staticmethod
    def _sha256(path: Path) -> str:
        """SHA-256 of a file's bytes."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _catalog(
        self, templates: list[CardTemplate], skipped: list[str]
    ) -> TemplateLibrary:
        """Assemble the :class:`TemplateLibrary` (counts + stats) from templates."""
        variant_counts: dict[str, int] = {}
        for tpl in templates:
            variant_counts[tpl.variant.value] = variant_counts.get(tpl.variant.value, 0) + 1

        if templates:
            avg_w = round(sum(t.template_width for t in templates) / len(templates))
            avg_h = round(sum(t.template_height for t in templates) / len(templates))
            largest = max(templates, key=lambda t: t.template_width * t.template_height)
            smallest = min(templates, key=lambda t: t.template_width * t.template_height)
            largest_slug: str | None = largest.slug
            smallest_slug: str | None = smallest.slug
        else:
            avg_w = avg_h = 0
            largest_slug = smallest_slug = None

        return TemplateLibrary(
            built_at=datetime.now(timezone.utc),
            template_height=self._settings.template_height,
            card_count=len(templates),
            variant_counts=variant_counts,
            average_width=avg_w,
            average_height=avg_h,
            largest_template=largest_slug,
            smallest_template=smallest_slug,
            templates=templates,
            skipped=skipped,
        )


class _ImageReadError(Exception):
    """Internal: a single template image could not be read/decoded."""
