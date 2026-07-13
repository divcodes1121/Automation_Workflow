"""Domain models for the AI Creator Studio.

These Pydantic models describe a single YouTube *project*: the long-form video
plus its derived vertical shorts, together with all the metadata required to
script, voice, render and (eventually) upload it.

Design notes
------------
* The models are **pure**: they perform no I/O and import nothing from
  :mod:`backend.config`. This keeps the domain layer independent and trivially
  unit-testable.
* ``model_config`` uses ``extra="forbid"`` so an unexpected key in the source
  JSON becomes a validation error instead of being silently dropped — critical
  for a data pipeline where a typo must never pass unnoticed.
* Enumerations (``VideoCategory``) are seeded with common values but are cheap
  to extend as the channel grows; free-form strings (``language``,
  ``voice_style``) are validated for shape only, because their allowed sets are
  owned by external services (YouTube, ElevenLabs) and will be enforced there.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Shared, strict configuration applied to every model in the domain layer.
_STRICT_CONFIG = ConfigDict(
    extra="forbid",           # Reject unknown keys instead of ignoring them.
    str_strip_whitespace=True,  # Normalise incidental whitespace from authors.
    validate_assignment=True,   # Keep instances valid after mutation.
)


class VideoCategory(StrEnum):
    """YouTube-facing content category for a Clash Royale video.

    Seeded with the categories this channel produces today. Extend as needed;
    values are strings so they serialise cleanly to/from JSON.
    """

    GAMEPLAY = "gameplay"
    DECK_GUIDE = "deck_guide"
    STRATEGY = "strategy"
    NEWS_UPDATE = "news_update"
    ENTERTAINMENT = "entertainment"


class ProjectStatus(StrEnum):
    """Lifecycle state of a project as it moves through the pipeline.

    Tracking state explicitly on the model (rather than inferring it from which
    files happen to exist on disk) keeps orchestration decisions unambiguous.
    """

    PENDING = "pending"
    SCRIPT_READY = "script_ready"
    GAMEPLAY_READY = "gameplay_ready"
    EDITING = "editing"
    THUMBNAIL_READY = "thumbnail_ready"
    UPLOADING = "uploading"
    PUBLISHED = "published"


class GameplayMetadata(BaseModel):
    """Technical metadata extracted from a gameplay video via ``ffprobe``.

    Produced by :class:`~backend.services.metadata.MetadataService`. Every later
    stage (editor, shorts, subtitles, upload, quality checks) consumes this
    object, so it is a first-class domain entity rather than a service-local
    detail. The model is pure: it performs no I/O and does not know how it was
    obtained or where it will be persisted.

    Attributes
    ----------
    source_file:
        Path to the analysed video file.
    file_size_bytes:
        Size of the file on disk, in bytes.
    video_hash:
        SHA-256 hex digest of the file's contents. Enables duplicate detection
        ("already processed? → skip") when footage is re-recorded or copied.
    container_format:
        Container/format name reported by ffprobe (e.g. ``"mov,mp4,m4a,..."``).
    duration_seconds:
        Total duration in seconds.
    video_codec:
        Codec of the primary video stream (e.g. ``"h264"``).
    width, height:
        Frame dimensions in pixels.
    fps:
        Frames per second of the primary video stream.
    bitrate_bps:
        Overall bitrate in bits per second, if reported by ffprobe.
    creation_date:
        Container creation timestamp, if present in the file's metadata.
    analyzed_at:
        Timestamp of when this metadata was produced.
    """

    model_config = _STRICT_CONFIG

    source_file: Path
    file_size_bytes: int = Field(ge=0)
    video_hash: str = Field(min_length=1)
    container_format: str
    duration_seconds: float = Field(ge=0)
    video_codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(ge=0)
    bitrate_bps: int | None = Field(default=None, ge=0)
    creation_date: datetime | None = None
    analyzed_at: datetime

    @property
    def resolution(self) -> str:
        """Human-readable resolution string, e.g. ``"1920x1080"``."""
        return f"{self.width}x{self.height}"


class Short(BaseModel):
    """A single vertical short derived from the long-form video.

    Attributes
    ----------
    title:
        Short-facing title (shown on the Shorts shelf).
    script:
        The full narration/caption script for the short.
    hook:
        The opening line intended to stop the scroll in the first seconds.
    duration_seconds:
        Target duration; must be positive and within YouTube's 60s Shorts cap.
    """

    model_config = _STRICT_CONFIG

    title: str = Field(min_length=1, max_length=100)
    script: str = Field(min_length=1)
    hook: str = Field(min_length=1, max_length=200)
    duration_seconds: int = Field(gt=0, le=60)


class SegmentImportance(StrEnum):
    """Editorial weight of a narration segment.

    Used later by the editor and Shorts extractor to prioritise which moments
    deserve emphasis or become standalone clips.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NarrationSegment(BaseModel):
    """An *untimed* unit of narration: what is said and what to show.

    This is the input to timeline construction. It maps one-to-one onto the
    authored/Claude "rich" JSON format (hence the ``voice`` field name), and is
    also what :class:`~backend.services.script_splitter.DefaultScriptSplitter`
    produces when deriving segments from a plain ``long_script``.

    Attributes
    ----------
    id:
        Optional author-supplied ordinal. The timeline assigns its own
        authoritative ids, so this is informational only.
    voice:
        The narration text for this segment.
    visual:
        Optional description of the intended on-screen gameplay/visual.
    importance:
        Optional editorial weight (see :class:`SegmentImportance`).
    """

    model_config = _STRICT_CONFIG

    id: int | None = None
    voice: str = Field(min_length=1)
    visual: str | None = None
    importance: SegmentImportance | None = None


class Project(BaseModel):
    """A complete YouTube project: the long-form video and its shorts.

    This is the aggregate root passed between pipeline stages. It carries every
    piece of author-supplied metadata needed to produce and publish a video.

    Attributes
    ----------
    title, description:
        Long-form video metadata.
    tags:
        Non-empty list of search/discovery tags.
    thumbnail_prompt:
        Text prompt handed to the (future) image-generation module.
    long_script:
        Full narration script for the long-form video.
    voice_style:
        Free-form voice style identifier resolved by the voice module
        (e.g. an ElevenLabs voice/style name).
    upload_time:
        Intended publish time (timezone-aware recommended).
    category:
        High-level content category (see :class:`VideoCategory`).
    language:
        BCP-47 / ISO language code for the audio and metadata (e.g. ``"en"``).
    shorts:
        Zero or more derived vertical shorts.
    segments:
        Optional authored narration segments (the "rich" format). When present,
        the timeline builder uses them verbatim instead of splitting
        ``long_script``. Reserved for Claude's richer output; ``None`` today.
    project_id:
        Stable unique identifier for the project. Reserved for later use.
    created_at:
        When the project was first created. Reserved for later use.
    status:
        Current pipeline lifecycle state (see :class:`ProjectStatus`).
    version:
        Monotonic revision number of this project definition.
    """

    model_config = _STRICT_CONFIG

    title: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=5000)
    tags: list[str] = Field(min_length=1)
    thumbnail_prompt: str = Field(min_length=1)
    long_script: str = Field(min_length=1)
    voice_style: str = Field(min_length=1)
    upload_time: datetime
    category: VideoCategory
    language: str = Field(min_length=2, max_length=10)
    shorts: list[Short] = Field(default_factory=list)
    segments: list[NarrationSegment] | None = None

    # -- Reserved lifecycle fields (optional today; drive orchestration later) -
    project_id: str | None = None
    created_at: datetime | None = None
    status: ProjectStatus = ProjectStatus.PENDING
    version: int = Field(default=1, ge=1)

    @field_validator("tags")
    @classmethod
    def _tags_must_be_non_empty(cls, value: list[str]) -> list[str]:
        """Ensure no tag is blank after whitespace stripping."""
        cleaned = [tag.strip() for tag in value]
        if any(not tag for tag in cleaned):
            raise ValueError("tags must not contain empty strings")
        return cleaned

    @property
    def short_count(self) -> int:
        """Number of shorts attached to this project."""
        return len(self.shorts)


# Schema version for the Phase-3 Claude prompt artifact.
GENERATED_PROMPT_SCHEMA_VERSION = "1.0"


class GeneratedPrompt(BaseModel):
    """A ready-to-send Claude prompt that turns a match analysis into a project.

    Phase 3's deterministic core (see
    :class:`~backend.services.project_generator.ProjectGenerator`): it distills a
    ``gameplay_analysis.json`` into ``prompt`` — the exact text a human pastes
    into Claude to get back a valid ``project.json``. Building the *prompt* is
    pure and free; the Claude call itself is a swappable step (manual paste today,
    an API adapter later). The extra fields are a distilled record of what the
    prompt was built from, for inspection/regression.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = GENERATED_PROMPT_SCHEMA_VERSION
    generated_at: datetime
    source_analysis: str  # analysis file (or video) the prompt was built from
    video: str
    player_deck: list[str]
    opponent_deck: list[str]
    play_count: int
    prompt: str  # the full text to paste into Claude


# Schema version for the highlight-edit plan artifact (gaming-highlights mode).
HIGHLIGHT_PLAN_SCHEMA_VERSION = "1.1"


class HighlightRole(StrEnum):
    """A clip's job in the reel's story arc (drives pacing now, effects later).

    ``HOOK`` opens on the first signature play, ``BEAT`` escalates through the
    middle plays with tight windows, ``FLASH`` is a sub-2s phase-change insert
    (Double Elixir / Overtime), ``HERO`` is the payoff play with a longer
    window, ``VICTORY`` is the end screen.
    """

    HOOK = "hook"
    BEAT = "beat"
    FLASH = "flash"
    HERO = "hero"
    VICTORY = "victory"


class HighlightClip(BaseModel):
    """One event-synced clip window cut from the original gameplay recording.

    The window is placed around the *real* event timestamp from the analyzer
    (``gameplay_analysis.json``), not a linear slice — this is the core of the
    "gaming highlights" mode. ``label`` is a viewer-facing tag (e.g. "ROCKET",
    "OVERTIME") that later slices turn into an on-screen caption; ``role`` is
    the clip's job in the story arc (later slices apply per-role effects).
    """

    model_config = _STRICT_CONFIG

    index: int = Field(ge=0)  # order in the reel
    role: HighlightRole = HighlightRole.BEAT
    event_timestamp_seconds: float = Field(ge=0)  # the moment in the source video
    card: str | None = None
    phase: str | None = None
    label: str  # viewer-facing tag for a future caption
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    duration_seconds: float = Field(gt=0)


class HighlightPlan(BaseModel):
    """A gameplay-only highlight edit: which windows to cut and why.

    Built by :class:`~backend.services.highlight_editor.HighlightEditor` from a
    match analysis. Mode ``gameplay_only`` keeps the ORIGINAL Clash Royale audio
    and adds no narration — the analyzer's events drive the *editing* instead of
    a voiceover. Captions/effects (later slices) layer on top of these clips.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = HIGHLIGHT_PLAN_SCHEMA_VERSION
    generated_at: datetime
    mode: str = "gameplay_only"
    source_analysis: str
    video: str
    clip_count: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    clips: list[HighlightClip]


class TimelineTiming(BaseModel):
    """Timing of a single timeline segment, keeping estimates and actuals apart.

    Estimated values are computed at build time (from word count and a speaking
    rate) because no audio exists yet. Once voice is generated and analysed, the
    ``actual_*`` fields are populated to *enrich* — never overwrite — the
    estimates, which makes drift between plan and reality easy to inspect.
    """

    model_config = _STRICT_CONFIG

    estimated_start_seconds: float = Field(ge=0)
    estimated_end_seconds: float = Field(ge=0)
    actual_start_seconds: float | None = Field(default=None, ge=0)
    actual_end_seconds: float | None = Field(default=None, ge=0)

    @property
    def estimated_duration_seconds(self) -> float:
        """Estimated duration (end − start)."""
        return self.estimated_end_seconds - self.estimated_start_seconds

    @property
    def actual_duration_seconds(self) -> float | None:
        """Measured duration, or ``None`` until both actual bounds are set."""
        if self.actual_start_seconds is None or self.actual_end_seconds is None:
            return None
        return self.actual_end_seconds - self.actual_start_seconds

    @property
    def effective_start_seconds(self) -> float:
        """Actual start if known, otherwise the estimate — one place to read."""
        return (
            self.actual_start_seconds
            if self.actual_start_seconds is not None
            else self.estimated_start_seconds
        )

    @property
    def effective_end_seconds(self) -> float:
        """Actual end if known, otherwise the estimate — one place to read."""
        return (
            self.actual_end_seconds
            if self.actual_end_seconds is not None
            else self.estimated_end_seconds
        )


class TimelineSegment(BaseModel):
    """A :class:`NarrationSegment` placed on the video timeline.

    Composition (rather than flattening) preserves the distinction between the
    untimed narration content and its timed placement.

    Attributes
    ----------
    id:
        1-based ordinal for human-readable display and ordering.
    segment_uuid:
        Stable identifier that downstream artifacts (voice clip, subtitle,
        gameplay selection, thumbnail) reference. Survives insertion/reordering,
        unlike ``id``.
    timing:
        Estimated (and later actual) placement in seconds.
    narration:
        The narration content for this segment.
    gameplay_hint:
        Reserved. A later editor can resolve ``narration.visual`` into a concrete
        clip selection hint without any model change. ``None`` today.
    """

    model_config = _STRICT_CONFIG

    id: int = Field(ge=1)
    segment_uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timing: TimelineTiming
    narration: NarrationSegment
    gameplay_hint: str | None = None

    @property
    def voice(self) -> str:
        """Narration text for this segment."""
        return self.narration.voice

    @property
    def visual(self) -> str | None:
        """Authored visual description for this segment, if any."""
        return self.narration.visual


class Timeline(BaseModel):
    """The master timeline for a project's long-form video.

    Produced by :class:`~backend.services.timeline.TimelineBuilderService` and
    consumed by every later stage (voice, editor, subtitles, thumbnail, shorts).
    Pure data — no I/O.

    Attributes
    ----------
    title:
        Project title this timeline belongs to.
    project_id:
        Owning project's id, if assigned.
    segments:
        Ordered timeline segments.
    total_duration_seconds:
        Estimated total duration (the last segment's estimated end).
    words_per_minute:
        Speaking rate used to estimate timings.
    is_estimated:
        ``True`` while timings are estimates; becomes ``False`` once every
        segment carries measured actuals.
    generated_at:
        When this timeline was built.
    """

    model_config = _STRICT_CONFIG

    title: str
    project_id: str | None = None
    segments: list[TimelineSegment]
    total_duration_seconds: float = Field(ge=0)
    words_per_minute: float = Field(gt=0)
    is_estimated: bool = True
    generated_at: datetime

    @property
    def segment_count(self) -> int:
        """Number of segments in the timeline."""
        return len(self.segments)


# Schema version stamped onto every NarrationPackage so future consumers can
# detect and migrate older payloads.
NARRATION_SCHEMA_VERSION = "1.0"


class NarrationEmotion(StrEnum):
    """Closed vocabulary of narration emotions.

    A fixed set (rather than free-form strings) lets a future TTS engine map each
    emotion to consistent voice settings without guessing at synonyms like
    "happy" / "energetic" / "excite".
    """

    NEUTRAL = "neutral"
    EXCITED = "excited"
    SERIOUS = "serious"
    SUSPENSE = "suspense"
    CELEBRATION = "celebration"
    CALM = "calm"


class PreparedNarrationSegment(BaseModel):
    """One narration segment prepared for (future) text-to-speech.

    Produced by :class:`~backend.services.narration.NarrationService` from a
    :class:`TimelineSegment`. It carries the original text, a cleaned/TTS-ready
    variant, timing/delivery hints, and reserved slots that a later voice
    provider fills in. Pure data — no I/O.

    Attributes
    ----------
    segment_uuid:
        Stable identifier copied from the timeline segment; the shared key that
        ties narration, audio, subtitles and gameplay together.
    voice_text:
        The original narration text, never mutated.
    cleaned_text:
        Whitespace/punctuation-normalised, abbreviation-expanded text intended
        for the TTS engine to read.
    estimated_duration_seconds:
        Estimated spoken duration, copied from the timeline (pre-audio estimate).
    speech_rate:
        Relative speaking-rate multiplier (1.0 = engine default).
    pause_before_seconds, pause_after_seconds:
        Silence to insert around this segment. Zero until a later stage sets them.
    emphasis_words:
        Words to emphasise. Empty until a later stage populates it.
    emotion:
        Optional delivery emotion (see :class:`NarrationEmotion`).
    output_audio:
        Path to the generated audio for this segment. ``None`` until a TTS
        provider fills it (Feature 5).
    provider_metadata:
        Reserved, provider-specific data (e.g. ``voice_id``/``sample_rate``/
        ``provider``). Empty today; keeps the schema stable when a provider adds
        its own fields.
    """

    model_config = _STRICT_CONFIG

    segment_uuid: str = Field(min_length=1)
    voice_text: str = Field(min_length=1)
    cleaned_text: str = Field(min_length=1)
    estimated_duration_seconds: float = Field(ge=0)
    speech_rate: float = Field(default=1.0, gt=0)
    pause_before_seconds: float = Field(default=0.0, ge=0)
    pause_after_seconds: float = Field(default=0.0, ge=0)
    emphasis_words: list[str] = Field(default_factory=list)
    emotion: NarrationEmotion | None = None
    output_audio: Path | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class NarrationPackage(BaseModel):
    """Provider-neutral bundle of prepared narration for a project.

    The single artifact any voice provider (Kokoro, Piper, ElevenLabs, …) can
    consume to generate audio, without changing the schema. Pure data — no I/O.

    Attributes
    ----------
    project_id:
        Owning project's id, if assigned.
    title:
        Project title (for display and the output filename).
    schema_version:
        Version of this package's schema (see :data:`NARRATION_SCHEMA_VERSION`).
    generated_at:
        When this package was prepared.
    segments:
        Ordered prepared narration segments.
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = NARRATION_SCHEMA_VERSION
    generated_at: datetime
    segments: list[PreparedNarrationSegment]

    @property
    def segment_count(self) -> int:
        """Number of prepared segments."""
        return len(self.segments)

    @property
    def total_estimated_duration_seconds(self) -> float:
        """Estimated total runtime including inter-segment pauses."""
        return sum(
            seg.pause_before_seconds
            + seg.estimated_duration_seconds
            + seg.pause_after_seconds
            for seg in self.segments
        )


# Schema version stamped onto every GeneratedNarration manifest.
GENERATED_NARRATION_SCHEMA_VERSION = "1.0"


class WordTiming(BaseModel):
    """Start/end time of a single spoken word, measured from real audio.

    Produced by the TTS provider (Kokoro exposes these natively). Feeds subtitle
    alignment and Feature 6 (audio-timeline enrichment). Times are relative to
    the start of the owning segment's audio file.
    """

    model_config = _STRICT_CONFIG

    text: str | None = None
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)


class GeneratedNarrationSegment(BaseModel):
    """A synthesised audio clip for one narration segment.

    Attributes
    ----------
    segment_uuid:
        Stable id shared with the prepared-narration/timeline segment.
    index:
        1-based ordinal; matches the ``NNN.wav`` filename.
    audio_file:
        Path to this segment's generated WAV (audio is **not** concatenated).
    duration_seconds:
        Measured audio duration.
    sample_rate:
        Audio sample rate in Hz.
    synthesis_seconds:
        Wall-clock time the provider took to synthesise this segment.
    word_timings:
        Word-level timings measured from the audio (may be empty if the provider
        does not expose them).
    """

    model_config = _STRICT_CONFIG

    segment_uuid: str = Field(min_length=1)
    index: int = Field(ge=1)
    audio_file: Path
    duration_seconds: float = Field(ge=0)
    sample_rate: int = Field(gt=0)
    synthesis_seconds: float = Field(ge=0)
    word_timings: list[WordTiming] = Field(default_factory=list)


class GeneratedNarration(BaseModel):
    """Manifest of synthesised narration audio for a project.

    Produced by :class:`~backend.services.speech.SpeechSynthesisService`. Ties
    each narration segment to its generated WAV plus measured timings, and
    records provider-specific detail in :attr:`provider_data`. Pure data — no I/O.

    Attributes
    ----------
    provider:
        Synthesis provider used (e.g. ``"kokoro"``).
    voice:
        Voice identifier used for synthesis.
    sample_rate:
        Sample rate of the generated audio, in Hz.
    segments:
        Ordered synthesised segments.
    provider_data:
        Free-form provider-specific metadata (voice id, model version, sample
        rate, synthesis timings, …).
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = GENERATED_NARRATION_SCHEMA_VERSION
    generated_at: datetime
    provider: str
    voice: str
    sample_rate: int = Field(gt=0)
    segments: list[GeneratedNarrationSegment]
    provider_data: dict[str, Any] = Field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        """Number of synthesised segments."""
        return len(self.segments)

    @property
    def total_audio_seconds(self) -> float:
        """Total duration of all generated audio."""
        return sum(seg.duration_seconds for seg in self.segments)

    @property
    def total_synthesis_seconds(self) -> float:
        """Total wall-clock synthesis time across segments."""
        return sum(seg.synthesis_seconds for seg in self.segments)

    @property
    def realtime_factor(self) -> float | None:
        """Synthesis time / audio duration (``<1`` = faster than realtime)."""
        audio = self.total_audio_seconds
        return round(self.total_synthesis_seconds / audio, 3) if audio else None


# Schema version stamped onto every ExecutionTimeline.
EXECUTION_TIMELINE_SCHEMA_VERSION = "1.0"


class ExecutionSegment(BaseModel):
    """A fully-resolved segment ready for the production pipeline.

    The merge of a timeline segment (content + estimated timing), its prepared
    narration (cleaned text, delivery hints) and its synthesised audio (measured
    timing, word timings). Downstream stages (editor, subtitles, thumbnail,
    shorts, upload) read *only* this — never the upstream artifacts. Pure data.

    Attributes
    ----------
    segment_uuid:
        Stable id shared across timeline / narration / audio.
    index:
        1-based ordinal.
    narration:
        Untimed narration content (voice text, intended visual, importance).
    cleaned_text:
        TTS-ready text that was actually spoken.
    speech_rate:
        Delivery rate multiplier used for this segment.
    emotion:
        Optional delivery emotion.
    timing:
        Estimated *and* measured (``actual_*``) placement in seconds.
    audio_offset_seconds:
        Absolute start of this segment's audio on the assembled timeline
        (equals ``timing.actual_start_seconds``). Downstream code turns a
        segment-relative word time into an absolute one via
        ``audio_offset_seconds + word.start_seconds``.
    actual_duration_seconds:
        Measured audio duration (stored for convenience; equals
        ``timing.actual_end - timing.actual_start``).
    audio_file:
        Path to this segment's synthesised WAV.
    sample_rate:
        Audio sample rate in Hz.
    word_timings:
        Word-level timings, **relative to this segment's audio**.
    provider:
        Synthesis provider that produced the audio.
    """

    model_config = _STRICT_CONFIG

    segment_uuid: str = Field(min_length=1)
    index: int = Field(ge=1)
    narration: NarrationSegment
    cleaned_text: str = Field(min_length=1)
    speech_rate: float = Field(gt=0)
    emotion: NarrationEmotion | None = None
    timing: TimelineTiming
    audio_offset_seconds: float = Field(ge=0)
    actual_duration_seconds: float = Field(ge=0)
    audio_file: Path
    sample_rate: int = Field(gt=0)
    word_timings: list[WordTiming] = Field(default_factory=list)
    provider: str = Field(min_length=1)


class ExecutionTimeline(BaseModel):
    """The authoritative timeline the whole production pipeline executes from.

    Produced by :class:`~backend.services.synchronizer.TimelineSynchronizer` by
    merging the estimated timeline, prepared narration and synthesised audio into
    one validated object. Named for its role, not how it was produced. Pure data.

    Attributes
    ----------
    provider:
        Synthesis provider used across the timeline.
    sample_rate:
        Uniform audio sample rate in Hz.
    segments:
        Ordered, fully-resolved execution segments.
    is_synchronized:
        ``True`` once every segment carries measured timing.
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = EXECUTION_TIMELINE_SCHEMA_VERSION
    generated_at: datetime
    provider: str
    sample_rate: int = Field(gt=0)
    segments: list[ExecutionSegment]
    is_synchronized: bool = True

    @property
    def segment_count(self) -> int:
        """Number of execution segments."""
        return len(self.segments)

    @property
    def total_actual_duration_seconds(self) -> float:
        """Total measured runtime (end of the last segment's audio)."""
        return self.segments[-1].timing.effective_end_seconds if self.segments else 0.0


# Schema version stamped onto every EditPlan.
EDIT_PLAN_SCHEMA_VERSION = "1.0"


class EditPlanSource(BaseModel):
    """A gameplay source video referenced by an edit plan.

    Carries just enough for the renderer to locate and size the footage; the
    per-segment trims live on :class:`EditPlanSegment`.
    """

    model_config = _STRICT_CONFIG

    file: Path
    duration_seconds: float = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(ge=0)


class EditPlanSegment(BaseModel):
    """One gameplay range mapped onto one narration segment's slot.

    Describes *what to show* under a segment's audio without doing any encoding:
    a contiguous ``[source_start, source_end]`` range of ``source_file`` placed at
    ``[target_start, target_end]`` on the final timeline. The renderer (7B) reads
    this; the planner (7A) never touches FFmpeg.

    Attributes
    ----------
    segment_uuid:
        Stable id shared with the execution timeline / narration / audio.
    index:
        1-based ordinal.
    source_file:
        Gameplay video this segment is cut from.
    source_start_seconds, source_end_seconds:
        Trim range within ``source_file``.
    target_start_seconds, target_end_seconds:
        Placement on the final video timeline (= the segment's measured audio span).
    duration_seconds:
        Segment length (``target_end - target_start`` == source range length).
    audio_file:
        The narration WAV to play under this gameplay range.
    narration_visual:
        Optional authored visual hint (for the renderer/debugging).
    effects:
        Reserved for the renderer/future (zooms, crops, transitions). Empty now.
    """

    model_config = _STRICT_CONFIG

    segment_uuid: str = Field(min_length=1)
    index: int = Field(ge=1)
    source_file: Path
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(ge=0)
    target_start_seconds: float = Field(ge=0)
    target_end_seconds: float = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    audio_file: Path
    narration_visual: str | None = None
    effects: dict[str, Any] = Field(default_factory=dict)


class EditPlan(BaseModel):
    """A renderable plan: which gameplay to show under each narration segment.

    Produced by :class:`~backend.services.planner.GameplayPlanner` from an
    :class:`ExecutionTimeline` plus gameplay footage. It is a **pure plan** — an
    inspectable artifact the renderer (7B) turns into video. No audio/video data,
    just references and ranges. Pure data — no I/O.
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = EDIT_PLAN_SCHEMA_VERSION
    generated_at: datetime
    source_videos: list[EditPlanSource]
    segments: list[EditPlanSegment]

    @property
    def segment_count(self) -> int:
        """Number of planned segments."""
        return len(self.segments)

    @property
    def total_duration_seconds(self) -> float:
        """Total planned runtime (end of the last segment's target span)."""
        return self.segments[-1].target_end_seconds if self.segments else 0.0


# Schema + algorithm versions stamped onto every ThumbnailPlan.
THUMBNAIL_PLAN_SCHEMA_VERSION = "1.0"
THUMBNAIL_PLANNER_VERSION = "1.0"


class ThumbnailCropMode(StrEnum):
    """How the extracted frame is fitted to the thumbnail canvas."""

    COVER = "cover"
    CONTAIN = "contain"
    STRETCH = "stretch"


class FrameSelectionReason(StrEnum):
    """Why the planner chose a particular frame (for traceability)."""

    IMPORTANCE = "importance"
    VISUAL = "visual"
    FALLBACK = "fallback"


class FrameSource(StrEnum):
    """Where the thumbnail background comes from (always VIDEO today)."""

    VIDEO = "video"
    IMAGE = "image"
    GENERATED = "generated"


class ThumbnailPlan(BaseModel):
    """A renderable plan for a video thumbnail.

    Produced by :class:`~backend.services.thumbnail_planner.ThumbnailPlanner` from
    an :class:`ExecutionTimeline` + the base video. Self-contained (carries the
    source video's metadata so the renderer needn't re-probe) and
    renderer-agnostic (no image-library specifics). Pure data — no I/O.
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = THUMBNAIL_PLAN_SCHEMA_VERSION
    planner_version: str = THUMBNAIL_PLANNER_VERSION
    variant: str = "default"
    generated_at: datetime

    # -- Source (self-contained; renderer needn't re-probe) -------------------
    source_video: Path
    frame_source: FrameSource = FrameSource.VIDEO
    video_duration_seconds: float = Field(ge=0)
    video_width: int = Field(gt=0)
    video_height: int = Field(gt=0)
    video_fps: float = Field(ge=0)

    # -- Frame choice ---------------------------------------------------------
    target_frame_timestamp_seconds: float = Field(ge=0)
    selection_reason: FrameSelectionReason
    source_segment_uuid: str | None = None

    # -- Composition ----------------------------------------------------------
    target_width: int = Field(gt=0)
    target_height: int = Field(gt=0)
    crop_mode: ThumbnailCropMode = ThumbnailCropMode.COVER
    title_text: str
    highlight_text: str | None = None
    badge_text: str | None = None
    arrow_position: str | None = None
    blur_background: bool = True
    glow: bool = True
    safe_area_margin: int = Field(default=0, ge=0)

    # -- Reserved renderer-only fields (None today) ---------------------------
    character_image: Path | None = None
    card_image: Path | None = None
    background_frame_path: Path | None = None
    overlay_template: str | None = None


class ThumbnailResult(BaseModel):
    """Summary of a rendered thumbnail image (Feature 9B)."""

    model_config = _STRICT_CONFIG

    output_file: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    requested_frame_timestamp_seconds: float = Field(ge=0)
    actual_frame_timestamp_seconds: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)


# Schema version stamped onto every UploadResult.
UPLOAD_RESULT_SCHEMA_VERSION = "1.0"


class UploadResult(BaseModel):
    """Outcome of a YouTube upload (Feature 10).

    The stable domain model n8n reads; the untouched API payload is archived
    separately as ``youtube_response.json``. Pure data — no I/O.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = UPLOAD_RESULT_SCHEMA_VERSION
    video_id: str
    url: str
    title: str
    privacy: str
    status: str
    upload_time: datetime
    publish_time: datetime | None = None
    processing_status: str | None = None
    thumbnail_uploaded: bool = False
    elapsed_seconds: float = Field(ge=0)


class RenderResult(BaseModel):
    """Summary of a completed video render.

    Describes the produced ``video.mp4`` (the actual artifact) plus render
    performance. Pure data — no I/O.
    """

    model_config = _STRICT_CONFIG

    output_file: Path
    duration_seconds: float = Field(ge=0)
    video_codec: str
    audio_codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(ge=0)
    segment_count: int = Field(ge=0)
    input_clip_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)

    @property
    def resolution(self) -> str:
        """Human-readable resolution string, e.g. ``"1280x720"``."""
        return f"{self.width}x{self.height}"

    @property
    def render_speed(self) -> float | None:
        """Output duration / render time (``>1`` = faster than realtime)."""
        if self.elapsed_seconds <= 0:
            return None
        return round(self.duration_seconds / self.elapsed_seconds, 2)


# Schema version stamped onto every SubtitleTrack.
SUBTITLE_SCHEMA_VERSION = "1.0"


class SubtitleCue(BaseModel):
    """A single subtitle cue: some text shown over a time span.

    Times are **absolute** on the final video timeline (seconds). Pure data.
    """

    model_config = _STRICT_CONFIG

    index: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str = Field(min_length=1)
    # Reserved for future multi-speaker/commentary videos; unused today.
    speaker: str | None = None


class SubtitleTrack(BaseModel):
    """An ordered set of subtitle cues for a project (rendered to SRT).

    Produced by :class:`~backend.services.subtitles.SubtitleGenerator` from an
    :class:`ExecutionTimeline`. Pure data — no I/O.
    """

    model_config = _STRICT_CONFIG

    project_id: str | None = None
    title: str
    schema_version: str = SUBTITLE_SCHEMA_VERSION
    format: str = "srt"
    generated_at: datetime
    cues: list[SubtitleCue]

    @property
    def cue_count(self) -> int:
        """Number of cues in the track."""
        return len(self.cues)

    @property
    def total_duration_seconds(self) -> float:
        """End of the last cue."""
        return self.cues[-1].end_seconds if self.cues else 0.0


# Schema version stamped onto every RunResult (the whole-pipeline record).
RUN_RESULT_SCHEMA_VERSION = "1.0"


class RunResult(BaseModel):
    """Outcome of a full pipeline run (Feature 11).

    The single artifact that ``run`` (and therefore n8n) reads to learn what a
    whole-pipeline invocation produced: which stages completed, how long each
    took, where the final media landed, and the upload outcome if one was
    requested. ``artifacts`` points at each stage's own versioned artifact
    rather than duplicating their contents. Pure data -- no I/O.
    """

    model_config = _STRICT_CONFIG

    schema_version: str = RUN_RESULT_SCHEMA_VERSION
    project_id: str
    title: str
    completed_stages: list[str]
    artifacts: dict[str, str]
    stage_timings: dict[str, float]
    video_path: Path
    thumbnail_path: Path
    requested_upload: bool = False
    uploaded: bool = False
    upload: UploadResult | None = None
    total_elapsed_seconds: float = Field(ge=0)
    created_at: datetime
