"""Thumbnail rendering — execute a ThumbnailPlan into a PNG (Feature 9B).

:class:`ThumbnailRenderer` is a renderer, not a designer: FFmpeg extracts the
requested frame, Pillow composites a **fixed, config-driven template** (blur, crop,
title, highlight, badge, glow). No AI/OCR/detection/content-aware layout — it
executes the plan exactly. Layout positions come from module-level constants (the
"classic" template) so tweaks don't require touching drawing code.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.models import ThumbnailCropMode, ThumbnailPlan, ThumbnailResult

logger = logging.getLogger(__name__)

# -- "classic" template constants (fractions of target size / pixels) -------- #
SCRIM_HEIGHT_RATIO = 0.42        # bottom gradient scrim covers this fraction
SCRIM_OPACITY = 200              # 0..255
GLOW_RADIUS = 6                  # gaussian blur for the text shadow
STROKE_WIDTH_TITLE = 3
STROKE_WIDTH_HIGHLIGHT = 5
STROKE_WIDTH_BADGE = 2
BADGE_PADDING = 18               # px inside the badge pill
LINE_SPACING_RATIO = 1.12        # line height multiplier

_NAMED_RGB = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "yellow": (255, 221, 0),
    "red": (230, 40, 40),
    "green": (40, 200, 90),
    "blue": (60, 120, 240),
    "gold": (255, 200, 40),
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class ThumbnailRenderingError(Exception):
    """Base class for all errors raised by :class:`ThumbnailRenderer`."""


class ThumbnailRenderError(ThumbnailRenderingError):
    """Raised when a thumbnail cannot be rendered (bad inputs / FFmpeg / Pillow)."""


class ThumbnailRenderer:
    """Renders a :class:`ThumbnailPlan` into a PNG.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying fonts/sizes/colours.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()

    # -- Public API -----------------------------------------------------------

    def _layout(self, plan: ThumbnailPlan) -> dict[str, Any]:
        """Validate inputs + fonts and compute layout boxes. No canvas/FFmpeg."""
        source = Path(plan.source_video)
        if not source.is_file():
            raise ThumbnailRenderError(f"Base video not found: {source}")

        margin = plan.safe_area_margin
        max_text_w = plan.target_width - 2 * margin

        highlight_font = self._load_font(self._settings.thumbnail_highlight_font_size)
        title_font = self._load_font(self._settings.thumbnail_title_font_size)
        badge_font = self._load_font(self._settings.thumbnail_badge_font_size)

        layout: dict[str, Any] = {
            "frame_timestamp": plan.target_frame_timestamp_seconds,
            "target_width": plan.target_width,
            "target_height": plan.target_height,
            "highlight_box": None,
            "title_box": None,
            "badge_box": None,
        }

        if plan.highlight_text:
            lines = _wrap(plan.highlight_text, highlight_font, max_text_w)
            w, h = _block_size(lines, highlight_font)
            layout["highlight_box"] = {"x": margin, "y": margin, "w": w, "h": h,
                                       "lines": lines}

        if plan.title_text:
            lines = _wrap(plan.title_text, title_font, max_text_w)
            w, h = _block_size(lines, title_font)
            y = plan.target_height - margin - h
            layout["title_box"] = {"x": margin, "y": y, "w": w, "h": h, "lines": lines}

        if plan.badge_text:
            bw = int(_text_width(plan.badge_text, badge_font)) + 2 * BADGE_PADDING
            bh = _line_height(badge_font) + BADGE_PADDING
            layout["badge_box"] = {
                "x": plan.target_width - margin - bw, "y": margin, "w": bw, "h": bh,
                "text": plan.badge_text,
            }

        layout["_fonts"] = {"highlight": highlight_font, "title": title_font, "badge": badge_font}
        return layout

    def output_path(self, plan: ThumbnailPlan, output: Path | None) -> Path:
        """Resolve the output PNG path (default ``edited/<slug>.thumbnail.png``)."""
        if output is not None:
            return Path(output).resolve()
        slug = _slugify(plan.project_id or plan.title)
        return (self._settings.edited_dir / f"{slug}.thumbnail.png").resolve()

    def save_layout(self, layout: dict[str, Any], destination: Path) -> Path:
        """Write the layout (minus font objects) to a debug JSON file."""
        serialisable = {k: v for k, v in layout.items() if not k.startswith("_")}
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
        return destination

    def render(self, plan: ThumbnailPlan, output: Path | None = None) -> ThumbnailResult:
        """Render ``plan`` into a PNG and return a :class:`ThumbnailResult`.

        Raises
        ------
        ThumbnailRenderError
            On missing inputs or an FFmpeg/Pillow failure.
        """
        from PIL import Image, ImageDraw, ImageFilter

        started = time.perf_counter()
        layout = self._layout(plan)
        out = self.output_path(plan, output)
        out.parent.mkdir(parents=True, exist_ok=True)

        requested = plan.target_frame_timestamp_seconds
        with tempfile.TemporaryDirectory(prefix="thumb_") as tmp:
            frame_path = Path(tmp) / "frame.png"
            self._extract_frame(Path(plan.source_video), requested, frame_path)
            try:
                base = Image.open(frame_path).convert("RGBA")
            except Exception as exc:  # noqa: BLE001 — Pillow open failure.
                raise ThumbnailRenderError(f"Could not open extracted frame: {exc}") from exc

        canvas = _fit(base, plan.target_width, plan.target_height, plan.crop_mode)
        if plan.blur_background and self._settings.thumbnail_blur_radius > 0:
            canvas = canvas.filter(
                ImageFilter.GaussianBlur(self._settings.thumbnail_blur_radius)
            )

        # Bottom scrim for legibility.
        scrim_h = int(plan.target_height * SCRIM_HEIGHT_RATIO)
        scrim = Image.new("RGBA", (plan.target_width, scrim_h), (0, 0, 0, 0))
        for row in range(scrim_h):
            alpha = int(SCRIM_OPACITY * (row / scrim_h))
            ImageDraw.Draw(scrim).line(
                [(0, row), (plan.target_width, row)], fill=(0, 0, 0, alpha)
            )
        canvas.alpha_composite(scrim, (0, plan.target_height - scrim_h))

        fill = _colour(self._settings.thumbnail_text_colour)
        outline = _colour(self._settings.thumbnail_text_outline_colour)
        fonts = layout["_fonts"]
        draw = ImageDraw.Draw(canvas)

        if layout["highlight_box"]:
            box = layout["highlight_box"]
            _draw_block(canvas, draw, box["lines"], fonts["highlight"], box["x"], box["y"],
                        fill, outline, STROKE_WIDTH_HIGHLIGHT, plan.glow)
        if layout["title_box"]:
            box = layout["title_box"]
            _draw_block(canvas, draw, box["lines"], fonts["title"], box["x"], box["y"],
                        fill, outline, STROKE_WIDTH_TITLE, plan.glow)
        if layout["badge_box"]:
            box = layout["badge_box"]
            draw.rounded_rectangle(
                [box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]],
                radius=12, fill=_colour(self._settings.thumbnail_text_outline_colour) + (210,),
            )
            draw.text((box["x"] + BADGE_PADDING, box["y"] + BADGE_PADDING // 2),
                      box["text"], font=fonts["badge"], fill=fill,
                      stroke_width=STROKE_WIDTH_BADGE, stroke_fill=outline)

        canvas.convert("RGB").save(out, format="PNG")
        self.save_layout(layout, layout_path(out))

        elapsed = time.perf_counter() - started
        logger.info("Rendered thumbnail -> %s (%.2fs)", out, elapsed)
        return ThumbnailResult(
            output_file=out,
            width=plan.target_width,
            height=plan.target_height,
            requested_frame_timestamp_seconds=requested,
            actual_frame_timestamp_seconds=requested,
            elapsed_seconds=round(elapsed, 3),
        )

    # -- Internals ------------------------------------------------------------

    def _extract_frame(self, video: Path, timestamp: float, destination: Path) -> None:
        """Extract a single frame at ``timestamp`` via FFmpeg."""
        argv = [
            str(self._settings.ffmpeg_path), "-y",
            "-ss", f"{timestamp}",
            "-i", str(video.resolve()),
            "-frames:v", "1",
            str(destination),
        ]
        completed = subprocess.run(argv, capture_output=True, text=True)
        if completed.returncode != 0 or not destination.is_file():
            tail = (completed.stderr or "").strip()[-1500:]
            raise ThumbnailRenderError(
                f"FFmpeg failed to extract frame at {timestamp}s.\n{tail}"
            )

    def _load_font(self, size: int):
        """Load a font: configured -> bundled Noto Sans Bold -> Pillow default."""
        from PIL import ImageFont

        try:
            return ImageFont.truetype(self._settings.thumbnail_font, size)
        except OSError:
            pass
        bundled = self._settings.assets_dir / "fonts" / "NotoSans-Bold.ttf"
        try:
            return ImageFont.truetype(str(bundled), size)
        except OSError:
            logger.warning(
                "Thumbnail font %r and bundled font unavailable; using Pillow default.",
                self._settings.thumbnail_font,
            )
            try:
                return ImageFont.load_default(size)
            except TypeError:  # older Pillow
                return ImageFont.load_default()


# -- Pure helpers ------------------------------------------------------------ #
def _fit(image, width: int, height: int, mode: ThumbnailCropMode):
    """Fit ``image`` to ``width x height`` per crop mode. Returns RGBA."""
    from PIL import Image

    src_w, src_h = image.size
    if mode == ThumbnailCropMode.STRETCH:
        return image.resize((width, height), Image.LANCZOS)
    if mode == ThumbnailCropMode.CONTAIN:
        scale = min(width / src_w, height / src_h)
        resized = image.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.LANCZOS)
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        canvas.alpha_composite(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas
    # COVER (default): scale to fill, centre-crop.
    scale = max(width / src_w, height / src_h)
    resized = image.resize((max(1, int(src_w * scale)), max(1, int(src_h * scale))), Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _draw_block(canvas, draw, lines, font, x, y, fill, outline, stroke_w, glow) -> None:
    """Draw multiline text with an outline and optional soft glow/shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    line_h = _line_height(font)
    if glow:
        shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        yy = y
        for line in lines:
            sdraw.text((x, yy), line, font=font, fill=(0, 0, 0, 230),
                       stroke_width=stroke_w, stroke_fill=(0, 0, 0, 230))
            yy += line_h
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(GLOW_RADIUS)))
    yy = y
    for line in lines:
        draw.text((x, yy), line, font=font, fill=fill + (255,),
                  stroke_width=stroke_w, stroke_fill=outline + (255,))
        yy += line_h


def _line_height(font) -> int:
    ascent, descent = font.getmetrics()
    return int((ascent + descent) * LINE_SPACING_RATIO)


def _text_width(text: str, font) -> float:
    return font.getlength(text)


def _block_size(lines, font) -> tuple[int, int]:
    width = int(max((font.getlength(line) for line in lines), default=0))
    return width, _line_height(font) * len(lines)


def _wrap(text: str, font, max_width: int) -> list[str]:
    """Greedy word-wrap ``text`` to ``max_width`` pixels."""
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _colour(value: str) -> tuple[int, int, int]:
    """Parse a colour name / ``#RRGGBB`` into an RGB tuple."""
    text = value.strip().lower()
    if text.startswith("#") and len(text) == 7:
        return int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    return _NAMED_RGB.get(text, (255, 255, 255))


def layout_path(png_path: Path) -> Path:
    """Sidecar layout-debug path next to the PNG (``…_layout.json``)."""
    return png_path.with_name(f"{png_path.stem}_layout.json")


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "thumbnail"
