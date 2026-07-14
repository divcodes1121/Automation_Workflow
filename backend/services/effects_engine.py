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

Primitive set: in-clip FFmpeg filters — ``zoom`` (punch-in), ``shake`` (jitter,
needs a zoom for margin), ``flash`` (brightness pulse), ``callout`` (pop-text);
plus two renderer-level cues that :meth:`plan_chains` returns alongside the
chains — ``sfx`` (a meme sound mixed OVER the game audio) and ``meme`` (a
meme-video INTERRUPT: the game freezes on its last frame while the meme plays
over/instead of it, then the reel resumes). Speed ramps and rewinds are later
slices.

Timing: every effect fires at ``offset`` seconds from the clip's IMPACT ANCHOR
(card plays: event moment + payoff lag; phase flashes: the phase moment;
victory: just after the cut).
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.models import HighlightClip, HighlightRole

logger = logging.getLogger(__name__)

_RECIPES_PATH = Path(__file__).resolve().parents[1] / "recipes" / "highlight_effects.json"
_FONT_PATH = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "NotoSans-Bold.ttf"
_SFX_DIR = Path(__file__).resolve().parents[2] / "Memes" / "SFX"
_MEME_DIR = Path(__file__).resolve().parents[2] / "Memes" / "IMGS"

_DEFAULT_MEME_DURATION_S = 2.8  # a meme interrupt stays punchy


@dataclass(frozen=True)
class SfxCue:
    """A meme sound to mix over the reel audio at a given reel position."""

    file: Path
    position_seconds: float  # where in the FINAL reel timeline it starts
    volume: float


@dataclass(frozen=True)
class MemeCue:
    """A meme-video INTERRUPT spliced into the reel right after a clip.

    The game freezes on ``freeze_at_seconds`` (the clip's last source frame) for
    ``duration`` seconds while the meme plays with its own audio. ``mode`` picks
    the treatment: ``subject`` AI-mats the subject (background removed) over the
    frozen frame — works on any clip with a separable subject; ``overlay``
    chroma-keys the subject over the frozen frame (fast, for real green/white
    screens); ``cutaway`` replaces the whole frame. Key params are only used by
    ``overlay``; ``fps`` only by ``subject``.
    """

    file: Path
    mode: str  # "subject" | "overlay" | "cutaway"
    after_index: int          # splice in right after this clip
    freeze_at_seconds: float  # source time of the frame to hold behind an overlay
    duration: float
    volume: float
    key_color: str = "white"
    similarity: float = 0.12
    blend: float = 0.08
    fps: float = 15.0  # matte frame rate for 'subject' mode

# Seconds between a play's recorded timestamp and its visible payoff (the
# recorded event trails the real placement; the impact lands just after).
_IMPACT_LAG_S = 1.3
_VICTORY_ANCHOR_S = 0.4  # anchor just after the cut to the end screen
_CALLOUT_FONT_SIZE = 150
_CALLOUT_Y_FRAC = 0.18   # vertical position of pop text (fraction of height)
_CALLOUT_FADE_S = 0.10   # opacity fade-in for a caption


class EffectsError(ValueError):
    """Raised when the recipe file is missing/invalid or a chain can't build."""


class EffectsEngine:
    """Compiles per-clip effect recipes into FFmpeg filter-chain snippets."""

    def __init__(self, recipes_path: Path | None = None) -> None:
        self._recipes_path = recipes_path or _RECIPES_PATH
        (
            self._recipes,
            self._role_variants,
            self._sfx_library,
            self._meme_library,
        ) = self._load()

    def _load(
        self,
    ) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
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
        return (
            recipes,
            role_variants,
            data.get("sfx_library") or {},
            data.get("meme_library") or {},
        )

    # -- Public API ----------------------------------------------------------- #

    def plan_chains(
        self,
        clips: list[HighlightClip],
        width: int,
        height: int,
        *,
        seed: int = 0,
        memes: bool = True,
    ) -> tuple[dict[int, str], list[SfxCue], list[MemeCue]]:
        """Select + compile per-clip video chains and collect SFX + meme cues.

        For each ROLE the variants are cycled (offset by ``seed`` so different
        videos differ), so repeated beats never look identical. Returns
        ``({clip_index: ffmpeg_chain}, [SfxCue, ...], [MemeCue, ...])`` with
        dimensions already substituted; SFX cues carry their absolute position in
        the (pre-meme) reel timeline and meme cues carry the clip they follow.
        The renderer shifts SFX past any spliced-in memes.
        """
        role_position: dict[str, int] = {}
        sfx_position: dict[str, int] = {}
        chains: dict[int, str] = {}
        cues: list[SfxCue] = []
        meme_cues: list[MemeCue] = []
        reel_offset = 0.0
        for i, clip in enumerate(clips):
            role = clip.role.value
            variants = self._role_variants.get(role) or []
            if variants:
                pos = role_position.get(role, 0)
                role_position[role] = pos + 1
                recipe = self._recipes.get(variants[(seed + pos) % len(variants)])
                if recipe:
                    chain = self._compile(clip, recipe)
                    chains[i] = chain.replace("{W}", str(width)).replace("{H}", str(height))
                    cues.extend(self._sfx_cues(clip, recipe, reel_offset, seed, sfx_position))
                    if memes:
                        meme_cues.extend(self._meme_cues(clip, recipe, i))
                else:
                    chains[i] = ""
            else:
                chains[i] = ""
            reel_offset += clip.duration_seconds
        return chains, cues, meme_cues

    def _meme_cues(
        self, clip: HighlightClip, recipe: dict[str, Any], clip_index: int
    ) -> list[MemeCue]:
        """Resolve a recipe's ``meme`` effects into positioned interrupt cues."""
        cues: list[MemeCue] = []
        for effect in recipe.get("effects", []):
            if effect.get("type") != "meme":
                continue
            asset = str(effect.get("asset", ""))
            entry = self._meme_library.get(asset)
            if not entry:
                logger.warning("No meme asset %r in meme_library", asset)
                continue
            path = _MEME_DIR / str(entry.get("file", ""))
            if not path.is_file():
                logger.warning("Meme file missing: %s", path)
                continue
            mode = str(entry.get("mode", "cutaway"))
            if mode not in ("subject", "overlay", "cutaway"):
                logger.warning("Meme %r has unknown mode %r; skipping", asset, mode)
                continue
            duration = float(
                effect.get("duration", entry.get("max_duration", _DEFAULT_MEME_DURATION_S))
            )
            cues.append(
                MemeCue(
                    file=path,
                    mode=mode,
                    after_index=clip_index,
                    freeze_at_seconds=clip.source_end_seconds,
                    duration=duration,
                    volume=float(entry.get("volume", 1.0)),
                    key_color=str(entry.get("key_color", "white")),
                    similarity=float(entry.get("similarity", 0.12)),
                    blend=float(entry.get("blend", 0.08)),
                    fps=float(entry.get("fps", 15.0)),
                )
            )
        return cues

    def _sfx_cues(
        self,
        clip: HighlightClip,
        recipe: dict[str, Any],
        reel_offset: float,
        seed: int,
        sfx_position: dict[str, int],
    ) -> list[SfxCue]:
        """Resolve a recipe's ``sfx`` effects into positioned reel cues."""
        anchor = self._anchor(clip)
        cues: list[SfxCue] = []
        for effect in recipe.get("effects", []):
            if effect.get("type") != "sfx":
                continue
            category = str(effect.get("category", ""))
            files = self._sfx_library.get(category) or []
            if not files:
                logger.warning("No SFX in category %r", category)
                continue
            # Rotate within the category (seeded) so repeats vary.
            pos = sfx_position.get(category, 0)
            sfx_position[category] = pos + 1
            path = _SFX_DIR / files[(seed + pos) % len(files)]
            if not path.is_file():
                logger.warning("SFX file missing: %s", path)
                continue
            start = max(0.0, reel_offset + anchor + float(effect.get("offset", 0.0)))
            cues.append(SfxCue(path, round(start, 3), float(effect.get("volume", 0.8))))
        return cues

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
                filters.append(
                    self._callout(
                        text, start, end,
                        animation=str(effect.get("animation", "pop")),
                        color=str(effect.get("color", "white")),
                        base_size=effect.get("size"),
                        y_frac=effect.get("y"),
                    )
                )
            elif etype in ("shake", "sfx", "meme"):
                # shake folds into the zoom/crop below; sfx is audio and meme is
                # a spliced interrupt — both handled outside _compile.
                pass
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
    def _callout(
        text: str,
        start: float,
        end: float,
        *,
        animation: str = "pop",
        color: str = "white",
        base_size: int | None = None,
        y_frac: float | None = None,
    ) -> str:
        """A bold animated caption (drawtext) — the hyper-gaming style's #1 pillar.

        ``animation`` drives entrance motion, all via per-frame drawtext
        expressions (no extra layers): ``pop``/``punch`` scale-overshoot in
        (this ffmpeg animates ``fontsize`` over ``t``), ``slide`` eases in from
        the right, ``rise`` eases up from below, ``fade`` just opacity-fades.
        ``color`` is any ffmpeg colour (name or ``0xRRGGBB``).
        """
        if not _FONT_PATH.is_file():
            raise EffectsError(f"bundled callout font not found: {_FONT_PATH}")
        # RELATIVE path, no drive colon: this ffmpeg build's filter parser
        # splits on ':' even inside quotes (same Windows gotcha 8B hit), so the
        # renderer runs ffmpeg with cwd = project root and this resolves there.
        font = "assets/fonts/NotoSans-Bold.ttf"
        s = float(start)
        base = int(base_size or _CALLOUT_FONT_SIZE)
        yf = float(y_frac if y_frac is not None else _CALLOUT_Y_FRAC)
        target_y = f"h*{yf:.3f}"
        center_x = "(w-text_w)/2"
        # Plain commas are safe inside the single-quoted arg values below (same
        # pattern as enable='between(t,a,b)').
        alpha = f"min(1,(t-{s:.3f})/{_CALLOUT_FADE_S})"
        fontsize, x_expr, y_expr = str(base), center_x, target_y

        if animation in ("pop", "punch"):
            din = 0.22 if animation == "punch" else 0.30
            # Ease-out-back overshoot: f(0)=0, f(1)=1, peaks ~1.1-1.15 between.
            over = 2.4 if animation == "punch" else 1.70158
            p = f"clip((t-{s:.3f})/{din},0,1)"
            scale = f"(1+{over + 1:.5f}*pow({p}-1,3)+{over:.5f}*pow({p}-1,2))"
            fontsize = f"max(8,{base}*{scale})"
        elif animation == "slide":
            po = f"(1-pow(1-clip((t-{s:.3f})/0.35,0,1),3))"  # ease-out cubic
            x_expr = f"(w+({center_x}-(w))*{po})"            # from off-right to centre
        elif animation == "rise":
            po = f"(1-pow(1-clip((t-{s:.3f})/0.35,0,1),3))"
            y_expr = f"(h+({target_y}-(h))*{po})"            # from below to target

        return (
            f"drawtext=fontfile={font}:text='{text}'"
            f":fontsize='{fontsize}':fontcolor={color}"
            f":borderw=12:bordercolor=black:shadowcolor=black@0.5:shadowx=4:shadowy=4"
            f":x='{x_expr}':y='{y_expr}'"
            f":alpha='{alpha}':enable='between(t,{s:.3f},{end:.3f})'"
        )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
