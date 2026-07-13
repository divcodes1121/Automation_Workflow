"""Environment-driven configuration for the Gameplay Analyzer.

Mirrors the pattern of :mod:`backend.config` (env-overridable paths resolved
against the repo root, a single frozen :class:`AnalyzerSettings` cached by
:func:`get_analyzer_settings`) but is intentionally a **separate** module so the
analyzer never imports ``backend``. The two subsystems share only the on-disk
``.env`` file and the ``assets/`` directory, never code.

Only the knobs the current slice (2A -- template library) needs are defined
here. Detector-specific settings (match thresholds, ROI boxes, frame sampling
rate) are added by their own slices when real footage exists to tune them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# Repository root = two levels up from this file (…/CR AI Workflow/analyzer/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache layout version. Bumped when the .npz format changes so a new build can
# live beside older caches (analyzer/cache/templates/v1, .../v2, …).
CACHE_VERSION = "v1"

# Frame-extraction layout version. Bumped when the frames/manifest format
# changes; also recorded in the manifest for validated cache reuse.
FRAMES_VERSION = "v1"


def _env_path(var: str, default: Path) -> Path:
    """Return an absolute path from ``var``, falling back to ``default``.

    Relative environment values are resolved against the project root so
    configuration stays portable across machines.
    """
    raw = os.getenv(var)
    if raw is None or not raw.strip():
        return default
    candidate = Path(raw.strip())
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate)


def _env_bool(var: str, default: bool) -> bool:
    """Parse a boolean environment variable (1/true/yes/on)."""
    raw = os.getenv(var)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class AnalyzerSettings(BaseModel):
    """Immutable, validated view of the analyzer's runtime configuration.

    Use :func:`get_analyzer_settings` rather than constructing this directly so
    the whole process shares one instance.
    """

    model_config = ConfigDict(frozen=True)

    project_root: Path = _PROJECT_ROOT

    # -- Source card art (bundled assets) -------------------------------------
    assets_dir: Path
    cards_dir: Path
    evolutions_dir: Path
    heroes_dir: Path  # in-battle evolution artwork (gold border)

    # -- External tools (off PATH on this machine; paths come from .env) -------
    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")

    # -- Analyzer working directories -----------------------------------------
    # Binary template cache (gitignored, rebuildable). The versioned subdir is
    # composed from CACHE_VERSION at use-time, not stored here.
    template_cache_dir: Path
    # Extracted-frame cache (gitignored). Versioned subdir from FRAMES_VERSION.
    frames_cache_dir: Path
    # Where future detector slices will write gameplay_analysis.json -- the
    # single handoff to the production pipeline.
    analysis_output_dir: Path

    # -- Template preprocessing -----------------------------------------------
    # Cached templates are normalized to this height; width preserves each
    # source image's aspect ratio. Detectors rescale to the on-screen card size
    # at match time, so this is only a storage baseline.
    template_height: int = Field(default=168, gt=0)

    # -- Frame extraction -----------------------------------------------------
    frame_sample_fps: float = Field(default=5.0, gt=0)
    frame_image_format: str = Field(default="png")  # "png" | "jpg"
    frame_jpeg_quality: int = Field(default=90, ge=1, le=100)
    frame_png_compression: int = Field(default=3, ge=0, le=9)
    frame_keep_existing: bool = Field(default=False)

    # -- Calibration + detection (2C-2H) --------------------------------------
    calibration_profiles_dir: Path
    active_profile: str = Field(default="default")
    play_stability_frames: int = Field(default=2, ge=1)
    # Matcher: ORB feature matching is the default -- it is robust to the
    # scale/framing differences between the official card art and the in-game
    # hand-card render (plain grayscale template correlation fails on real
    # footage). "template" is kept for synthetic/aligned cases.
    matching_method: str = Field(default="orb")  # "orb" | "template"
    hand_match_threshold: float = Field(default=0.6, ge=0.0, le=1.0)  # template method
    # ORB knobs. A slot is accepted when the best template has >= min_matches
    # good descriptor matches and beats the runner-up by >= min_margin.
    orb_nfeatures: int = Field(default=400, gt=0)
    orb_ratio: float = Field(default=0.78, gt=0.0, le=1.0)
    orb_min_matches: int = Field(default=15, ge=1)
    orb_min_margin: int = Field(default=4, ge=0)

    # -- Deck reconstruction (2I) ---------------------------------------------
    # A card must be observed this many times to count as a deck card (filters
    # one-off misdetections); "strong" is the count at which a card reaches full
    # per-card confidence.
    deck_min_observations: int = Field(default=3, ge=1)
    deck_strong_observations: int = Field(default=6, ge=1)
    # Per-card confidence = (count/strong) x (avg_match/quality_norm), each capped
    # at 1. Cards below deck_review_confidence are flagged "needs review".
    deck_quality_norm: float = Field(default=40.0, gt=0)
    deck_review_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    # Event-driven optimization: when enabled, a hand SLOT whose pixels are
    # (near-)unchanged from the previous frame reuses the previous card instead
    # of re-running ORB. hand_change_threshold is the mean-abs pixel difference
    # (0-255, on a 12x12 downscale) above which a slot is considered changed.
    hand_skip_unchanged: bool = Field(default=False)
    hand_change_threshold: float = Field(default=6.0, ge=0.0)

    def cache_root(self) -> Path:
        """Versioned template-cache root: ``template_cache_dir/<CACHE_VERSION>``."""
        return self.template_cache_dir / CACHE_VERSION

    def frames_root(self) -> Path:
        """Versioned frame-cache root: ``frames_cache_dir/<FRAMES_VERSION>``."""
        return self.frames_cache_dir / FRAMES_VERSION

    def ensure_directories(self) -> None:
        """Create the analyzer's working directories if they do not exist."""
        for path in (self.cache_root(), self.frames_root(), self.analysis_output_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_analyzer_settings() -> AnalyzerSettings:
    """Load ``.env`` and build the process-wide :class:`AnalyzerSettings`.

    Cached so the ``.env`` file is read exactly once per process.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    assets_dir = _env_path("ASSETS_DIR", _PROJECT_ROOT / "assets")

    return AnalyzerSettings(
        project_root=_PROJECT_ROOT,
        assets_dir=assets_dir,
        cards_dir=_env_path("ANALYZER_CARDS_DIR", assets_dir / "cards"),
        evolutions_dir=_env_path("ANALYZER_EVOLUTIONS_DIR", assets_dir / "Evolutions"),
        heroes_dir=_env_path("ANALYZER_HEROES_DIR", assets_dir / "Heroes"),
        ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
        ffprobe_path=os.getenv("FFPROBE_PATH", "ffprobe"),
        template_cache_dir=_env_path(
            "ANALYZER_TEMPLATE_CACHE_DIR", _PROJECT_ROOT / "analyzer" / "cache" / "templates"
        ),
        frames_cache_dir=_env_path(
            "ANALYZER_FRAMES_CACHE_DIR", _PROJECT_ROOT / "analyzer" / "cache" / "frames"
        ),
        analysis_output_dir=_env_path(
            "ANALYZER_ANALYSIS_DIR", _PROJECT_ROOT / "gameplay" / "analysis"
        ),
        template_height=int(os.getenv("ANALYZER_TEMPLATE_HEIGHT", "168")),
        frame_sample_fps=float(os.getenv("ANALYZER_SAMPLE_FPS", "5.0")),
        frame_image_format=os.getenv("ANALYZER_FRAME_FORMAT", "png"),
        frame_jpeg_quality=int(os.getenv("ANALYZER_JPEG_QUALITY", "90")),
        frame_png_compression=int(os.getenv("ANALYZER_PNG_COMPRESSION", "3")),
        frame_keep_existing=_env_bool("ANALYZER_KEEP_FRAMES", False),
        calibration_profiles_dir=_env_path(
            "ANALYZER_PROFILES_DIR", _PROJECT_ROOT / "analyzer" / "calibration" / "profiles"
        ),
        active_profile=os.getenv("ANALYZER_PROFILE", "default"),
        play_stability_frames=int(os.getenv("ANALYZER_PLAY_STABILITY_FRAMES", "2")),
        matching_method=os.getenv("ANALYZER_MATCHING_METHOD", "orb"),
        hand_match_threshold=float(os.getenv("ANALYZER_HAND_THRESHOLD", "0.6")),
        orb_nfeatures=int(os.getenv("ANALYZER_ORB_NFEATURES", "400")),
        orb_ratio=float(os.getenv("ANALYZER_ORB_RATIO", "0.78")),
        orb_min_matches=int(os.getenv("ANALYZER_ORB_MIN_MATCHES", "15")),
        orb_min_margin=int(os.getenv("ANALYZER_ORB_MIN_MARGIN", "4")),
        deck_min_observations=int(os.getenv("ANALYZER_DECK_MIN_OBS", "3")),
        deck_strong_observations=int(os.getenv("ANALYZER_DECK_STRONG_OBS", "6")),
        deck_quality_norm=float(os.getenv("ANALYZER_DECK_QUALITY_NORM", "40")),
        deck_review_confidence=float(os.getenv("ANALYZER_DECK_REVIEW_CONF", "0.6")),
        hand_skip_unchanged=_env_bool("ANALYZER_HAND_SKIP_UNCHANGED", False),
        hand_change_threshold=float(os.getenv("ANALYZER_HAND_CHANGE_THRESHOLD", "2.0")),
    )
