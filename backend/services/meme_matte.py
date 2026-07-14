"""AI subject matting for meme overlays (Phase 4): background removal.

The ``subject`` meme mode isolates the moving subject of a meme clip and drops
its background entirely — no green screen required — so the subject can be
composited over the frozen Clash Royale arena. This is the general case that
chroma-key can't cover (arbitrary backgrounds, no keyable colour).

Matting is done per frame by ``rembg`` (an ONNX model — no torch, so it stays in
the backend venv; imported lazily so the backend imports without it). Because
inference is ~1s/frame, the result is CACHED once per (clip, model, fps,
duration) as an alpha-carrying ``.mov`` (QuickTime RLE keeps a real alpha
channel) with the meme's own audio; the renderer just overlays that.

NOTE the subject must actually BE separable: a clip with a clearly-lit person
mats cleanly, but a dark/stylised montage with no distinct subject yields an
empty matte — for those, use ``cutaway`` (full-frame) instead.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import tempfile
from pathlib import Path

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Human-subject model — best for the person memes we key. rembg resizes
# internally, so input frame size barely affects speed.
_DEFAULT_MODEL = "u2net_human_seg"
_DEFAULT_FPS = 15.0  # memes tolerate a lower rate; the game frame behind is still
_CACHE_VERSION = "v1"


class MemeMatteError(RuntimeError):
    """Raised when matting is unavailable or a matte can't be produced."""


class MemeMatteService:
    """Turns a meme clip into a cached, background-removed alpha video."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._session = None  # lazy rembg session (one model load, reused)
        self._model = _DEFAULT_MODEL

    def _cache_dir(self) -> Path:
        # Under Memes/ so it inherits the folder's gitignore (like the assets).
        return self._settings.project_root / "Memes" / "cache" / _CACHE_VERSION

    def matte(
        self,
        meme_file: Path,
        *,
        duration: float,
        fps: float = _DEFAULT_FPS,
        model: str = _DEFAULT_MODEL,
    ) -> Path:
        """Return a cached alpha ``.mov`` of the meme's first ``duration`` seconds.

        The background is removed frame-by-frame; the meme's original audio is
        preserved. Reuses the cache when the (clip bytes, model, fps, duration)
        all match, so a given meme is only matted once.
        """
        if not meme_file.is_file():
            raise MemeMatteError(f"meme clip not found: {meme_file}")

        key = self._cache_key(meme_file, duration, fps, model)
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / f"{meme_file.stem}_{model}_{fps:g}_{duration:g}_{key}.mov"
        if dest.is_file():
            logger.info("Reusing cached meme matte: %s", dest.name)
            return dest

        logger.info("Matting meme subject (%s, %.1fs @ %gfps) ...", meme_file.name, duration, fps)
        with tempfile.TemporaryDirectory(prefix="meme_matte_") as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw"
            cut = tmp_path / "cut"
            raw.mkdir()
            cut.mkdir()
            self._extract_frames(meme_file, duration, fps, raw)
            frames = sorted(raw.glob("f*.png"))
            if not frames:
                raise MemeMatteError(f"no frames extracted from {meme_file}")
            kept = self._matte_frames(frames, cut, model)
            if kept <= 0.001:
                raise MemeMatteError(
                    f"the matte for {meme_file.name} is empty — this clip has no "
                    "separable subject; use meme mode 'cutaway' instead"
                )
            self._encode(cut, meme_file, duration, fps, dest)
        logger.info("Cached meme matte -> %s", dest.name)
        return dest

    # -- Internals ----------------------------------------------------------- #

    def _cache_key(self, meme_file: Path, duration: float, fps: float, model: str) -> str:
        h = hashlib.sha256()
        h.update(meme_file.read_bytes())
        h.update(f"|{model}|{fps}|{duration}|{_CACHE_VERSION}".encode())
        return h.hexdigest()[:12]

    def _extract_frames(self, meme_file: Path, duration: float, fps: float, out: Path) -> None:
        completed = subprocess.run(
            [
                self._settings.ffmpeg_path, "-y", "-t", f"{duration:.3f}",
                "-i", str(meme_file), "-vf", f"fps={fps:g}",
                str(out / "f%05d.png"),
            ],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise MemeMatteError(
                f"frame extraction failed (exit {completed.returncode}): "
                f"{(completed.stderr or '').strip()[-500:]}"
            )

    def _matte_frames(self, frames: list[Path], out: Path, model: str) -> float:
        """Background-remove each frame; returns the mean kept-alpha fraction."""
        remove, session = self._rembg(model)
        import numpy as np
        from PIL import Image

        kept_total = 0.0
        for frame in frames:
            with Image.open(frame) as img:
                cut = remove(img.convert("RGBA"), session=session)
            cut.save(out / frame.name)
            kept_total += float((np.asarray(cut.split()[-1]) > 20).mean())
        return kept_total / len(frames)

    def _rembg(self, model: str):
        """Lazily import rembg and build (reuse) the model session."""
        try:
            from rembg import new_session, remove
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise MemeMatteError(
                "meme mode 'subject' needs the 'rembg' package "
                "(pip install rembg onnxruntime)"
            ) from exc
        if self._session is None or self._model != model:
            self._session = new_session(model)
            self._model = model
        return remove, self._session

    def _encode(
        self, cut_dir: Path, meme_file: Path, duration: float, fps: float, dest: Path
    ) -> None:
        """Encode the matted PNG sequence + original audio to an alpha .mov."""
        completed = subprocess.run(
            [
                self._settings.ffmpeg_path, "-y",
                "-framerate", f"{fps:g}", "-i", str(cut_dir / "f%05d.png"),
                "-t", f"{duration:.3f}", "-i", str(meme_file),
                "-map", "0:v", "-map", "1:a?",
                # QuickTime RLE keeps a genuine alpha channel (argb).
                "-c:v", "qtrle", "-pix_fmt", "argb",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                str(dest),
            ],
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            dest.unlink(missing_ok=True)
            raise MemeMatteError(
                f"matte encode failed (exit {completed.returncode}): "
                f"{(completed.stderr or '').strip()[-500:]}"
            )
