"""Domain models for the Gameplay Analyzer.

Pure data only -- no I/O, no OpenCV, no import of :mod:`analyzer.config`. Mirrors
the conventions of :mod:`backend.models`: a shared strict config, ``StrEnum``
enums, and an explicit ``schema_version`` constant on each persisted artifact.

Slice 2A defines the **card template library** artifact. The binary template
data (grayscale/rgba/mask/edges arrays) lives in per-card ``.npz`` files on
disk; these models describe the catalog that indexes them.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_STRICT_CONFIG = ConfigDict(
    extra="forbid",
    str_strip_whitespace=True,
    validate_assignment=True,
)

# Schema version stamped onto the template library manifest (and embedded in
# every cached .npz) so a loader can detect a stale cache and rebuild.
TEMPLATE_LIBRARY_SCHEMA_VERSION = "1.0"


class CardVariant(StrEnum):
    """Which art set a template came from.

    The source directory is the only disambiguator between a base card and its
    evolution (evolution files reuse the base slug), so variant is recorded
    explicitly. ``HERO`` is reserved for a future slice -- ``assets/Heroes`` is
    currently ambiguous (its files duplicate base-card slugs) and is not built.
    """

    BASE = "base"
    EVOLUTION = "evolution"
    # In-battle evolution artwork (gold border), shipped under assets/Heroes/.
    # Distinct render from EVOLUTION (pink border) -- both are kept so ORB can
    # match whichever the game shows in hand.
    HERO = "hero"


class CardTemplate(BaseModel):
    """One preprocessed card template indexed in the library catalog."""

    model_config = _STRICT_CONFIG

    slug: str
    display_name: str
    variant: CardVariant
    source_path: Path
    cache_path: Path
    # SHA-256 of the source PNG bytes -- lets a future build rebuild only the
    # templates whose art actually changed.
    source_hash: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    template_width: int = Field(gt=0)
    template_height: int = Field(gt=0)
    # False when an evolution's slug has no matching base card (e.g. the
    # `furnance` typo) -- a data-quality flag surfaced at build time.
    has_base_match: bool = True
    # Reserved for a future descriptor-based matcher (ORB/AKAZE/SIFT). Kept None
    # today so adding descriptor support later is not a schema change.
    descriptors: list | None = None
    keypoints: list | None = None


class TemplateLibrary(BaseModel):
    """Catalog of every preprocessed card template (the build artifact)."""

    model_config = _STRICT_CONFIG

    schema_version: str = TEMPLATE_LIBRARY_SCHEMA_VERSION
    built_at: datetime
    template_height: int = Field(gt=0)
    card_count: int = Field(ge=0)
    variant_counts: dict[str, int]
    # Sanity-check statistics over the cached templates.
    average_width: int = Field(ge=0)
    average_height: int = Field(ge=0)
    largest_template: str | None = None
    smallest_template: str | None = None
    templates: list[CardTemplate]
    skipped: list[str] = Field(default_factory=list)


# Schema version stamped onto the frames manifest.
FRAMES_MANIFEST_SCHEMA_VERSION = "1.0"


class ExtractedFrame(BaseModel):
    """One frame sampled from a recording."""

    model_config = _STRICT_CONFIG

    index: int = Field(ge=0)  # 0-based output order
    filename: str  # f<source_frame:06d>.<ext>
    source_frame: int = Field(ge=0)  # frame number in the original video
    timestamp_seconds: float = Field(ge=0)
    sha256: str  # checksum of the written frame (corruption / dedup checks)
    # Reserved for later slices so the ROI/detector work needs no schema change.
    roi: dict | None = None
    analysis: dict | None = None
    notes: str | None = None


class FramesManifest(BaseModel):
    """Self-describing, reproducible record of a frame-extraction run.

    Carries enough to (a) validate a cache before reuse -- source hash,
    extraction version, and settings -- and (b) reproduce/debug the extraction
    -- the exact ffmpeg version and command.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = FRAMES_MANIFEST_SCHEMA_VERSION
    extraction_version: str
    video: str
    video_path: Path
    video_sha256: str
    video_modified_at: datetime
    source_fps: float = Field(gt=0)
    duration_seconds: float = Field(ge=0)
    sample_fps: float = Field(gt=0)  # effective (clamped to source_fps)
    image_format: str
    # Reproducibility: the settings + tool that produced these frames.
    png_compression: int = Field(ge=0, le=9)
    jpeg_quality: int = Field(ge=1, le=100)
    keep_existing: bool
    ffmpeg_version: str
    ffmpeg_command: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int = Field(ge=0)
    # Sanity-check statistics over the sampled timestamps.
    first_timestamp: float = Field(ge=0)
    last_timestamp: float = Field(ge=0)
    average_spacing: float = Field(ge=0)
    frames_dir: Path
    frames: list[ExtractedFrame]
    extracted_at: datetime


# --------------------------------------------------------------------------- #
# Detection engine (2C-2H)
# --------------------------------------------------------------------------- #


class MatchingMethod(StrEnum):
    """Card-matching algorithm. Only ``TEMPLATE`` is implemented today; the
    others reserve the seam so a better matcher can be swapped in later without
    touching any detector."""

    TEMPLATE = "template"
    ORB = "orb"
    AKAZE = "akaze"


# -- 2C Calibration ---------------------------------------------------------- #

CALIBRATION_PROFILE_SCHEMA_VERSION = "1.0"


class ROI(BaseModel):
    """A named region of interest, stored as fractions of the frame (0..1) so a
    profile is resolution-independent."""

    model_config = _STRICT_CONFIG

    name: str
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Return ``(x, y, w, h)`` in pixels for a given frame size."""
        x = round(self.x * frame_width)
        y = round(self.y * frame_height)
        w = round(self.w * frame_width)
        h = round(self.h * frame_height)
        return x, y, w, h


class CalibrationProfile(BaseModel):
    """Where every UI element sits, for one capture layout/device.

    Coordinates ship as **placeholders** (``is_placeholder=True``) until tuned
    against a real frame from that device via the ``calibrate`` command.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = CALIBRATION_PROFILE_SCHEMA_VERSION
    name: str
    reference_width: int = Field(gt=0)
    reference_height: int = Field(gt=0)
    rois: dict[str, ROI]
    is_placeholder: bool = True
    notes: str | None = None


# -- 2D Hand detection ------------------------------------------------------- #


class HandSlotReading(BaseModel):
    """The card identified in one of the four hand slots on a single frame."""

    model_config = _STRICT_CONFIG

    slot: int = Field(ge=1, le=4)
    card: str | None = None
    variant: str | None = None
    score: float
    matched: bool


class HandReading(BaseModel):
    """The four-slot hand read from one frame."""

    model_config = _STRICT_CONFIG

    source_frame: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    slots: list[HandSlotReading]


# -- 2E Play detection ------------------------------------------------------- #


class PlayEvent(BaseModel):
    """A card leaving the hand -- i.e. it was played."""

    model_config = _STRICT_CONFIG

    source_frame: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    card: str
    variant: str | None = None
    slot: int = Field(ge=1, le=4)
    score: float  # departing card's match confidence


# -- 2I Deck reconstruction + analyzer metrics ------------------------------- #

DECK_SIZE = 8  # a Clash Royale deck is always exactly 8 cards


class DeckCard(BaseModel):
    """One card discovered in a player's deck, with observation confidence."""

    model_config = _STRICT_CONFIG

    slug: str
    variant: str
    first_seen_time: float = Field(ge=0)
    last_seen_time: float = Field(ge=0)
    observation_count: int = Field(ge=1)
    average_match_score: float = Field(ge=0)
    matching_method: str
    confidence: float = Field(ge=0.0, le=1.0)  # observation count x match quality
    needs_review: bool = False  # low confidence -> flag rather than silently accept


class ReconstructedDeck(BaseModel):
    """A player's deck rebuilt from hand observations across the match."""

    model_config = _STRICT_CONFIG

    side: str  # "player" | "opponent"
    cards: list[DeckCard]  # confirmed cards, ranked by confidence, <= DECK_SIZE
    complete: bool
    completion_percent: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)  # overall deck confidence


class AnalyzerMetrics(BaseModel):
    """Per-run analyzer performance + quality metrics."""

    model_config = _STRICT_CONFIG

    frames_processed: int = Field(ge=0)
    frames_skipped: int = Field(ge=0)
    slots_analyzed: int = Field(ge=0)
    slots_skipped: int = Field(default=0, ge=0)  # ORB reused (unchanged slot)
    cards_identified: int = Field(ge=0)
    unknown_slots: int = Field(ge=0)
    average_orb_matches: float = Field(ge=0)
    average_confidence: float = Field(ge=0.0, le=1.0)
    processing_seconds: float = Field(ge=0)
    fps_processed: float = Field(ge=0)
    matching_method: str


# -- 2G Match state (timer / elixir / crowns) -------------------------------- #


class MatchPhase(StrEnum):
    """The tempo phase of a Clash Royale match.

    Detected from signals we can read reliably: the timer background turns RED in
    overtime (a clean colour signal), while regulation stays dark. ``UNKNOWN`` is
    used when the timer is unreadable on a frame.
    """

    REGULATION = "regulation"
    OVERTIME = "overtime"
    UNKNOWN = "unknown"


class MatchState(BaseModel):
    """A snapshot of the scoreboard-level match context at one instant.

    These form a **timeline** (``GameplayAnalysis.match_states``) sampled roughly
    once per second; each :class:`GameEvent` references the nearest entry by index
    (``match_state_ref``) rather than duplicating the context on every event.

    Fields are optional/None when their detector could not read that frame (e.g.
    the timer during the red-on-red overtime endgame, or crowns which are not yet
    detected) -- honest gaps over fabricated values.
    """

    model_config = _STRICT_CONFIG

    index: int = Field(ge=0)  # position in the timeline (the ref target)
    timestamp_seconds: float = Field(ge=0)
    source_frame: int = Field(ge=0)
    time_remaining: str | None = None  # "M:SS" as shown on the clock
    time_remaining_seconds: int | None = Field(default=None, ge=0)
    phase: MatchPhase = MatchPhase.UNKNOWN
    # Elixir generation rate: 1x, 2x (regulation final minute + overtime). 3x is
    # not claimed without a dedicated signal (left None rather than guessed).
    elixir_multiplier: int | None = Field(default=None, ge=1, le=3)
    player_elixir: int | None = Field(default=None, ge=0, le=10)
    opponent_elixir: int | None = Field(default=None, ge=0, le=10)
    # Crowns require tower-destruction detection (CrownDetector, deferred) -> None.
    player_crowns: int | None = Field(default=None, ge=0, le=3)
    opponent_crowns: int | None = Field(default=None, ge=0, le=3)
    timer_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


# -- 2H Event builder / final artifact --------------------------------------- #

GAMEPLAY_ANALYSIS_SCHEMA_VERSION = "1.2"


class GameEvent(BaseModel):
    """One structured gameplay event (the unit Claude reasons over)."""

    model_config = _STRICT_CONFIG

    event_id: str  # stable handle, e.g. "play_000123"
    sequence_number: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    source_frame: int = Field(ge=0)
    type: str = "card_played"
    card: str
    variant: str | None = None
    slot: int = Field(ge=1, le=4)
    confidence: float | None = None
    # Filled by later slices (2F arena lane).
    lane: str | None = None
    # Index into GameplayAnalysis.match_states of the nearest state (2G), or None
    # if no timeline exists. Referenced, not embedded, to avoid duplicating the
    # match context across dozens of events.
    match_state_ref: int | None = Field(default=None, ge=0)
    context: dict | None = None
    notes: str | None = None


class GameplayAnalysis(BaseModel):
    """The analyzer's output artifact -- the handoff to script generation."""

    model_config = _STRICT_CONFIG

    schema_version: str = GAMEPLAY_ANALYSIS_SCHEMA_VERSION
    video: str
    video_sha256: str
    source_fps: float = Field(gt=0)
    duration_seconds: float = Field(ge=0)
    sample_fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    profile_name: str
    events: list[GameEvent]
    # Match-state timeline (2G): ~1 snapshot/sec of timer/phase/elixir. Events
    # point into this list via ``match_state_ref``.
    match_states: list[MatchState] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
    # Reconstructed decks (2I) + run metrics.
    player_deck: ReconstructedDeck | None = None
    opponent_deck: ReconstructedDeck | None = None
    metrics: AnalyzerMetrics | None = None
    # Reserved for later slices; declared now to keep the schema stable.
    winner: str | None = None
