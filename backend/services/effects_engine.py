"""Effects Engine / Editing Language (Phase 4): data-driven, VARIED edits.

The editor never hardcodes "Rocket gets a zoom". Each clip has a ROLE
(hook/beat/flash/hero/victory); a role maps to a *list* of recipe VARIANTS in
``backend/recipes/highlight_effects.json`` (pure data). :meth:`plan_chains`
cycles through a role's variants — seeded per video — so repeated beats (e.g.
four Rockets) each get a different edit ON PURPOSE, while the hook stays grand
and the hero stays the payoff. This is the "controlled variation" that separates
a crafted edit from a preset applied on repeat.

Selection (editorial context) lives here; a chosen recipe is *compiled* into an
FFmpeg filter chain by :meth:`_compile`. A STYLE PACK (Esports / Meme /
Cinematic) is simply a different recipe file: ``EffectsEngine(recipes_path=...)``.

Primitive set (pure FFmpeg, only the bundled font as an asset): ``zoom``
(punch-in), ``shake`` (jitter, needs a zoom for margin), ``flash`` (brightness
pulse), ``callout`` (pop-text). Freeze frames, speed ramps, rewinds and SFX are
later slices — SFX also needs sourced copyright-free audio.

Timing: every effect fires at ``offset`` seconds from the clip's IMPACT ANCHOR
(card plays: event moment + payoff lag; phase flashes: the phase moment;
victory: just after the cut).
"""

from __future__ import annotations

import json
import logging
import zlib
from pathlib import Path
from typing import Any

from backend.models import HighlightClip, HighlightRole

logger = logging.getLogger(__name__)

_RECIPES_PATH = Path(__file__).resolve().parents[1] / "recipes" / "highlight_effects.json"
_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NotoSans-Bold.ttf"

# Seconds between a play's recorded timestamp and its visible payoff (the
# recorded event trails the real placement; the impact lands just after).
_IMPACT_LAG_S = 1.3
_VICTORY_ANCHOR_S = 0.4  # anchor just after the cut to the end screen
_CALLOUT_FONT_SIZE = 150
_CALLOUT_Y_FRAC = 0.18   # vertical position of pop text (fraction of height)


class EffectsError(ValueError):
    """Raised when the recipe file is missing/invalid or a chain can't build."""


class EffectsEngine:
    """Compiles per-clip effect recipes into FFmpeg filter-chain snippets."""

    def __init__(self, recipes_path: Path | None = None) -> None:
        self._recipes_path = recipes_path or _RECIPES_PATH
        self._recipes, self._role_variants = self._load()

    def _load(self) -> tuple[dict[str, Any], dict[str, list[str]]]:
        if not self._recipes_path.is_file():
            raise EffectsError(f"effects recipe file not found: {self._recipes_path}")
        try:
            data = json.loads(self._recipes_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise EffectsError(f"invalid effects recipe JSON: {exc}") from exc
        recipes = data.get("recipes") or {}
        role_variants = data.get("role_variants") or {}
        if not recipes or not role_variants:
            raise EffectsError("effects recipe file needs 'recipes' and 'role_variants'")
        return recipes, role_variants

    # -- Public API ----------------------------------------------------------- #

    def plan_chains(
        self,
        clips: list[HighlightClip],
        width: int,
        height: int,
        *,
        seed: int = 0,
    ) -> dict[int, str]:
        """Select + compile a filter chain per clip, with controlled variation.

        For each ROLE the variants are cycled through in order (offset by
        ``seed`` so different videos differ), so repeated beats never look
        identical. Returns ``{clip_index: ffmpeg_chain}`` with dimensions
        already substituted.
        """
        role_position: dict[str, int] = {}
        chains: dict[int, str] = {}
        for i, clip in enumerate(clips):
            role = clip.role.value
            variants = self._role_variants.get(role) or []
            if not variants:
                chains[i] = ""
                continue
            pos = role_position.get(role, 0)
            role_position[role] = pos + 1
            recipe_name = variants[(seed + pos) % len(variants)]
            recipe = self._recipes.get(recipe_name)
            chain = self._compile(clip, recipe) if recipe else ""
            chains[i] = chain.replace("{W}", str(width)).replace("{H}", str(height))
        return chains

    @staticmethod
    def stable_seed(text: str) -> int:
        """A deterministic per-video seed so variation is reproducible."""
        return zlib.crc32(text.encode("utf-8"))

    # -- Compilation ---------------------------------------------------------- #

    def _compile(self, clip: HighlightClip, recipe: dict[str, Any]) -> str:
        """FFmpeg filter snippet (prefixed with ',') for one clip+recipe, or ''."""
        if not recipe:
            return ""
        anchor = self._anchor(clip)
        filters: list[str] = []
        zoom_window: tuple[float, float] | None = None

        for effect in recipe.get("effects", []):
            etype = effect.get("type")
            start = _clamp(anchor + float(effect.get("offset", 0.0)), 0.0, clip.duration_seconds)
            end = _clamp(start + float(effect.get("duration", 0.5)), 0.0, clip.duration_seconds)
            if end <= start:
                continue
            intensity = float(effect.get("intensity", 0.0))

            if etype == "zoom":
                zoom_window = (start, end)
                # Shake needs the zoom margin; the crop below applies both.
            elif etype == "flash":
                filters.append(
                    f"eq=brightness={intensity}:enable='between(t,{start:.3f},{end:.3f})'"
                )
            elif etype == "callout":
                text = str(effect.get("text", "{label}")).replace("{label}", clip.label)
                filters.append(self._callout(text, start, end))
            elif etype == "shake":
                pass  # folded into the zoom/crop below
            else:
                logger.warning("Unknown effect type %r in a recipe", etype)

        # zoom (+ optional shake jitter inside the zoom margin) as one
        # scale+crop pair. Shake without zoom has no margin -> skipped.
        shake = next(
            (e for e in recipe.get("effects", []) if e.get("type") == "shake"), None
        )
        if zoom_window is not None:
            za, zb = zoom_window
            zoom_expr = f"(1+{self._zoom_intensity(recipe):.3f}*between(t,{za:.3f},{zb:.3f}))"
            scale = f"scale=eval=frame:w='trunc(iw*{zoom_expr}/2)*2':h=-2"
            jx = jy = ""
            if shake is not None:
                s_start = _clamp(anchor + float(shake.get("offset", 0.0)), za, zb)
                s_end = _clamp(s_start + float(shake.get("duration", 0.5)), za, zb)
                amp = float(shake.get("intensity", 10.0))
                if s_end > s_start:
                    win = f"between(t,{s_start:.3f},{s_end:.3f})"
                    jx = f"+{win}*{amp:.1f}*sin(2*PI*t*13)"
                    jy = f"+{win}*{amp:.1f}*cos(2*PI*t*11)"
            crop = (
                f"crop=w={{W}}:h={{H}}"
                f":x='(in_w-out_w)/2{jx}':y='(in_h-out_h)/2{jy}'"
            )
            filters.insert(0, f"{scale},{crop}")

        return ("," + ",".join(filters)) if filters else ""

    # -- Internals ------------------------------------------------------------ #

    def _zoom_intensity(self, recipe: dict[str, Any]) -> float:
        zoom = next((e for e in recipe.get("effects", []) if e.get("type") == "zoom"), None)
        return float(zoom.get("intensity", 0.1)) if zoom else 0.1

    @staticmethod
    def _anchor(clip: HighlightClip) -> float:
        """The clip-relative moment effects centre on."""
        if clip.role == HighlightRole.VICTORY:
            return _VICTORY_ANCHOR_S
        event_in_clip = clip.event_timestamp_seconds - clip.source_start_seconds
        if clip.role == HighlightRole.FLASH:
            return _clamp(event_in_clip, 0.0, clip.duration_seconds)
        return _clamp(event_in_clip + _IMPACT_LAG_S, 0.0, clip.duration_seconds)

    @staticmethod
    def _callout(text: str, start: float, end: float) -> str:
        if not _FONT_PATH.is_file():
            raise EffectsError(f"bundled callout font not found: {_FONT_PATH}")
        # RELATIVE path, no drive colon: this ffmpeg build's filter parser
        # splits on ':' even inside quotes (same Windows gotcha 8B hit), so the
        # renderer runs ffmpeg with cwd = project root and this resolves there.
        font = "assets/fonts/NotoSans-Bold.ttf"
        fade = 0.12
        # Plain comma is safe here: the surrounding single quotes protect it at
        # graph-parse level (same pattern as enable='between(t,a,b)').
        alpha = f"min(1,(t-{start:.3f})/{fade})"
        return (
            f"drawtext=fontfile={font}:text='{text}'"
            f":fontsize={_CALLOUT_FONT_SIZE}:fontcolor=white"
            f":borderw=10:bordercolor=black"
            f":x=(w-text_w)/2:y=h*{_CALLOUT_Y_FRAC}"
            f":alpha='{alpha}':enable='between(t,{start:.3f},{end:.3f})'"
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
