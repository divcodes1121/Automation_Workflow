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

    # -- 2G Match state (timer / elixir) --------------------------------------
    # Directory of CR timer digit templates ("<d>_<regime>.png"): committed art
    # (regime = reg | ot, for regulation dark-bg vs overtime red-bg rendering).
    timer_templates_dir: Path
    # Emit at most one MatchState snapshot per this many seconds of footage (the
    # timer + elixir change ~1/sec, so denser sampling only bloats the JSON).
    match_state_interval_s: float = Field(default=1.0, gt=0)
    # A timer read below this per-digit confidence is discarded (time -> None but
    # phase is still reported). Tuned from the read-test: regulation ~0.62-0.99,
    # bad overtime reads ~0.40.
    timer_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Elixir bar reader: skip the leftmost fraction (the "10" cap glyph) then a
    # bar column counts as filled when > elixir_column_fill of it is magenta.
    elixir_left_trim: float = Field(default=0.14, ge=0.0, le=0.5)
    elixir_column_fill: float = Field(default=0.30, ge=0.0, le=1.0)

    # -- 3.3 Battle splitter ---------------------------------------------------
    # Where the per-match clips cut out of a long recording are written.
    split_output_dir: Path
    # Timer sampling rate. 1/sec is plenty: a match lasts minutes and the clock
    # only ticks once a second.
    split_sample_fps: float = Field(default=1.0, gt=0)
    # Readable-clock runs closer than this fuse into one match. Covers the
    # final-minute flicker (gaps <= 8s observed) while staying well under the
    # smallest observed gap BETWEEN matches (35s).
    split_max_gap_s: float = Field(default=10.0, gt=0)
    # Shorter blocks are discarded as accidental captures (replays, instant
    # forfeits). Real matches ran 177-268s; accidents 2s and 15s.
    split_min_battle_s: float = Field(default=60.0, gt=0)
    # A Clash Royale clock never exceeds 3:00; higher reads are misdetections
    # (the crowns banner + confetti can score a spurious 3-glyph match).
    split_max_clock_seconds: int = Field(default=180, gt=0)
    # Across a gap, a real clock ticks down by about the elapsed wall time. This
    # is how much that may disagree before the two runs are treated as separate
    # matches rather than one match seen through a gap.
    split_clock_drift_s: float = Field(default=4.0, ge=0)
    # How far outside the clock to hunt for the loading-screen / lobby cut.
    split_scene_search_s: float = Field(default=14.0, gt=0)
    # scdet score (0-100) above which a frame counts as a real screen transition.
    # Measured: wanted boundaries scored 23-44, ordinary in-app animation <10.
    split_scene_min_score: float = Field(default=15.0, gt=0.0, le=100.0)
    # How far BEFORE the last clock read to start hunting for the end boundary.
    # An overtime clock can freeze on screen and keep reading into the result
    # screen, putting the last read past the real cut. Safe because gameplay
    # produces no cut near split_scene_min_score.
    split_end_lookback_s: float = Field(default=4.0, ge=0.0)
    # Stream copy must start on a keyframe. How far BEFORE the boundary one may
    # sit and still be chosen. Default 0 = never start early, so no frame of the
    # previous screen can survive -- important because frame 0 becomes the clip's
    # poster frame, and a flash of Battle Log there looks like a mistake. The cost
    # is opening up to a keyframe interval (~1s) into the match intro. Raise this
    # to trade a sliver of the previous screen for a more exact opening.
    split_keyframe_backfill_s: float = Field(default=0.0, ge=0.0)
    # Used when no scene cut is found in the search window.
    split_pre_roll_s: float = Field(default=6.0, ge=0.0)
    split_post_roll_s: float = Field(default=6.0, ge=0.0)

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
        split_output_dir=_env_path(
            "ANALYZER_SPLIT_DIR", _PROJECT_ROOT / "gameplay" / "raw"
        ),
        split_sample_fps=float(os.getenv("ANALYZER_SPLIT_SAMPLE_FPS", "1.0")),
        split_max_gap_s=float(os.getenv("ANALYZER_SPLIT_MAX_GAP", "10.0")),
        split_min_battle_s=float(os.getenv("ANALYZER_SPLIT_MIN_BATTLE", "60.0")),
        split_max_clock_seconds=int(os.getenv("ANALYZER_SPLIT_MAX_CLOCK", "180")),
        split_clock_drift_s=float(os.getenv("ANALYZER_SPLIT_CLOCK_DRIFT", "4.0")),
        split_scene_search_s=float(os.getenv("ANALYZER_SPLIT_SCENE_SEARCH", "14.0")),
        split_scene_min_score=float(os.getenv("ANALYZER_SPLIT_SCENE_MIN_SCORE", "15.0")),
        split_end_lookback_s=float(os.getenv("ANALYZER_SPLIT_END_LOOKBACK", "4.0")),
        split_keyframe_backfill_s=float(os.getenv("ANALYZER_SPLIT_KEYFRAME_BACKFILL", "0.0")),
        split_pre_roll_s=float(os.getenv("ANALYZER_SPLIT_PRE_ROLL", "6.0")),
        split_post_roll_s=float(os.getenv("ANALYZER_SPLIT_POST_ROLL", "6.0")),
        template_height=int(os.getenv("ANALYZER_TEMPLATE_HEIGHT", "168")),
        frame_sample_fps=float(os.getenv("ANALYZER_SAMPLE_FPS", "5.0")),
        frame_image_format=os.getenv("ANALYZER_FRAME_FORMAT", "png"),
        frame_jpeg_quality=int(os.getenv("ANALYZER_JPEG_QUALITY", "90")),
        frame_png_compression=int(os.getenv("ANALYZER_PNG_COMPRESSION", "3")),
        frame_keep_existing=_env_bool("ANALYZER_KEEP_FRAMES", False),
        calibration_profiles_dir=_env_path(
            "ANALYZER_PROFILES_DIR", _PROJECT_ROOT / "analyzer" / "calibration" / "profiles"
        ),
        timer_templates_dir=_env_path(
            "ANALYZER_TIMER_TEMPLATES_DIR", _PROJECT_ROOT / "analyzer" / "assets" / "timer_digits"
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
