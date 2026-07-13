"""In-memory index over the 2A card-template cache, with card matching.

:class:`TemplateIndex` loads the preprocessed templates (grayscale + alpha mask)
produced by ``build-templates`` and identifies an image region (a hand slot) by
matching it against all of them.

Two matchers are available (:class:`~analyzer.models.MatchingMethod`):

* ``ORB`` (default) -- feature/descriptor matching. Robust to the scale and
  framing differences between the official card art and the in-game hand-card
  render. This is what works on real footage.
* ``TEMPLATE`` -- masked grayscale cross-correlation. Works only when the art is
  already aligned/scaled (e.g. synthetic frames); it fails on real footage.

ORB descriptors are computed for every template at load time from the cached
grayscale + mask (no cache-format change needed). cv2/numpy are imported at
module load; callers that must import without them (the CLI) import this lazily.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from analyzer.config import AnalyzerSettings, get_analyzer_settings
from analyzer.models import MatchingMethod

logger = logging.getLogger(__name__)

# Canonical size ORB operates on (both templates and slots are resized to this).
_ORB_W, _ORB_H = 160, 190
# Fraction of the template border cropped before ORB, to drop the rounded frame.
_ORB_BORDER = 0.10


@dataclass(frozen=True)
class _Template:
    slug: str
    variant: str
    gray: np.ndarray  # uint8 HxW
    mask: np.ndarray  # uint8 HxW (alpha)
    orb_des: np.ndarray | None  # ORB descriptors (or None if too few features)


class TemplateIndexError(Exception):
    """Raised when the template cache is missing or unreadable."""


class TemplateIndex:
    """Loads the cached card templates and identifies slots against them."""

    def __init__(self, templates: list[_Template], settings: AnalyzerSettings) -> None:
        self._templates = templates
        self._settings = settings
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._ratio = settings.orb_ratio

    def __len__(self) -> int:
        return len(self._templates)

    # -- Loading --------------------------------------------------------------

    @classmethod
    def load(cls, settings: AnalyzerSettings | None = None) -> TemplateIndex:
        """Load the template library from the 2A cache (+ compute ORB descriptors).

        Raises
        ------
        TemplateIndexError
            If the manifest or any cached array is missing (run ``build-templates``).
        """
        settings = settings or get_analyzer_settings()
        manifest_path = settings.cache_root() / "template_manifest.json"
        if not manifest_path.is_file():
            raise TemplateIndexError(
                f"Template cache not found at {manifest_path}. Run "
                f"`python -m analyzer.main build-templates` first."
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise TemplateIndexError(f"Could not read template manifest: {exc}") from exc

        orb = cv2.ORB_create(nfeatures=settings.orb_nfeatures)
        templates: list[_Template] = []
        for entry in manifest.get("templates", []):
            cache_path = Path(entry["cache_path"])
            try:
                data = np.load(cache_path, allow_pickle=False)
                gray, mask = data["gray"], data["mask"]
            except (OSError, KeyError, ValueError) as exc:
                raise TemplateIndexError(
                    f"Could not load cached template {cache_path}: {exc}"
                ) from exc
            g, m = _orb_prep(gray, mask)
            _, des = orb.detectAndCompute(g, m)
            templates.append(
                _Template(entry["slug"], entry["variant"], gray, mask, des)
            )

        if not templates:
            raise TemplateIndexError("Template cache is empty; rebuild it.")
        logger.info("Loaded %d card templates (ORB descriptors ready)", len(templates))
        return cls(templates, settings)

    # -- Identification -------------------------------------------------------

    def identify(
        self, slot_gray: np.ndarray, method: MatchingMethod | None = None
    ) -> tuple[str | None, str | None, float, bool]:
        """Identify a hand-slot crop.

        Returns ``(slug, variant, score, matched)``. ``slug``/``variant`` are the
        best candidate; ``matched`` says whether it clears the acceptance
        thresholds (else a caller should treat the slot as empty/unknown, though
        the best-guess slug is still returned for diagnostics). ``score`` is the
        ORB good-match count, or the correlation coefficient for ``TEMPLATE``.
        """
        if slot_gray.ndim != 2:
            raise ValueError("slot_gray must be a single-channel (grayscale) image")
        method = method or MatchingMethod(self._settings.matching_method)

        if method is MatchingMethod.TEMPLATE:
            slug, variant, score = self._match_template(slot_gray)
            return slug, variant, score, score >= self._settings.hand_match_threshold
        if method is MatchingMethod.ORB:
            return self._identify_orb(slot_gray)
        raise NotImplementedError(f"Matching method {method.value!r} not implemented.")

    def _identify_orb(
        self, slot_gray: np.ndarray
    ) -> tuple[str | None, str | None, float, bool]:
        # Slots are resized directly (no border crop) -- the in-game frame edge
        # carries some signal and matches the validated experiment.
        g = cv2.resize(slot_gray, (_ORB_W, _ORB_H))
        orb = cv2.ORB_create(nfeatures=self._settings.orb_nfeatures)
        _, des = orb.detectAndCompute(g, None)
        best_c = second_c = 0
        best_slug = best_var = None
        for tpl in self._templates:
            c = self._good_matches(des, tpl.orb_des)
            if c > best_c:
                second_c = best_c
                best_c, best_slug, best_var = c, tpl.slug, tpl.variant
            elif c > second_c:
                second_c = c
        matched = (
            best_c >= self._settings.orb_min_matches
            and (best_c - second_c) >= self._settings.orb_min_margin
        )
        if matched:
            return best_slug, best_var, float(best_c), True
        return best_slug, best_var, float(best_c), False

    def _good_matches(self, des1: np.ndarray | None, des2: np.ndarray | None) -> int:
        """Lowe-ratio-test good matches between two ORB descriptor sets."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
        good = 0
        for pair in self._bf.knnMatch(des1, des2, k=2):
            if len(pair) == 2 and pair[0].distance < self._ratio * pair[1].distance:
                good += 1
        return good

    def _match_template(self, slot_gray: np.ndarray) -> tuple[str, str, float]:
        """Best (slug, variant, correlation) via masked grayscale correlation."""
        h, w = slot_gray.shape[:2]
        slot = slot_gray.astype(np.float32)
        best_slug, best_variant, best_score = "", "", -1.0
        for tpl in self._templates:
            tpl_gray = cv2.resize(tpl.gray, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
            tpl_mask = cv2.resize(tpl.mask, (w, h), interpolation=cv2.INTER_AREA)
            opaque = tpl_mask > 0
            if int(opaque.sum()) < 16:
                continue
            a = slot[opaque] - slot[opaque].mean()
            b = tpl_gray[opaque] - tpl_gray[opaque].mean()
            da, db = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
            score = 0.0 if da < 1e-6 or db < 1e-6 else float(a @ b / (da * db))
            if score > best_score:
                best_score, best_slug, best_variant = score, tpl.slug, tpl.variant
        return best_slug, best_variant, best_score

    def match(
        self, slot_gray: np.ndarray, method: MatchingMethod = MatchingMethod.TEMPLATE
    ) -> tuple[str, str, float]:
        """Best ``(slug, variant, score)`` without thresholding (diagnostics/tests)."""
        if method is MatchingMethod.TEMPLATE:
            return self._match_template(slot_gray)
        slug, variant, score, _ = self._identify_orb(slot_gray)
        return slug or "", variant or "", score


def _orb_prep(gray: np.ndarray, mask: np.ndarray | None):
    """Center-crop the frame border and resize to the canonical ORB size."""
    h, w = gray.shape[:2]
    by, bx = int(_ORB_BORDER * h), int(_ORB_BORDER * w)
    g = cv2.resize(gray[by : h - by, bx : w - bx], (_ORB_W, _ORB_H))
    m = None
    if mask is not None:
        m = cv2.resize(mask[by : h - by, bx : w - bx], (_ORB_W, _ORB_H))
        m = ((m > 0).astype(np.uint8)) * 255
    return g, m
