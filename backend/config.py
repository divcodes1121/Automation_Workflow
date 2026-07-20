"""Environment-driven configuration for the AI Creator Studio.

Responsibilities
----------------
* Load environment variables from the project-root ``.env`` file.
* Expose a single, cached :class:`Settings` instance (the composition root's
  source of truth) via :func:`get_settings`.
* Hold **future** third-party credentials (YouTube, OpenAI, Claude, Gemini,
  ElevenLabs) as :class:`~pydantic.SecretStr` so they are never printed,
  logged, or leaked through ``repr``/tracebacks.
* Resolve every working directory of the studio, all overridable via env,
  none hardcoded, and create them on demand.

Nothing here performs feature work; this module only *describes where things
live and how to authenticate*. Downstream modules depend on it, never the
reverse.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr

# Repository root = two levels up from this file (…/CR AI Workflow/backend/config.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var: str, default: Path) -> Path:
    """Return an absolute path from ``var``, falling back to ``default``.

    Relative values in the environment are resolved against the project root so
    configuration stays portable across machines.
    """
    raw = os.getenv(var)
    if raw is None or not raw.strip():
        return default
    candidate = Path(raw.strip())
    return candidate if candidate.is_absolute() else (_PROJECT_ROOT / candidate)


def _env_secret(var: str) -> SecretStr | None:
    """Wrap an optional environment secret in :class:`SecretStr`."""
    raw = os.getenv(var)
    return SecretStr(raw) if raw and raw.strip() else None


def _env_bool(var: str, default: bool) -> bool:
    """Parse a boolean environment variable (1/true/yes/on)."""
    raw = os.getenv(var)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class MissingSecretError(RuntimeError):
    """Raised when a required credential is requested but not configured."""


class Settings(BaseModel):
    """Immutable, validated view of the studio's runtime configuration.

    Instances are frozen: configuration is read once at startup and never
    mutated at runtime. Use :func:`get_settings` rather than constructing this
    directly, so the whole process shares one instance.
    """

    # SecretStr fields already redact themselves; freezing prevents accidental
    # in-flight edits to paths or keys.
    model_config = ConfigDict(frozen=True)

    # -- Base locations -------------------------------------------------------
    project_root: Path = _PROJECT_ROOT

    # -- Working directories (all env-overridable, defaults under the repo) ----
    gameplay_raw_dir: Path
    # Drop folder: recordings placed here are picked up by `auto` (Phase 3.3).
    incoming_dir: Path
    gameplay_processed_dir: Path
    gameplay_archive_dir: Path
    gameplay_shorts_dir: Path
    gameplay_metadata_dir: Path
    scripts_dir: Path
    edited_dir: Path
    uploads_dir: Path
    voice_dir: Path
    assets_dir: Path
    config_dir: Path
    templates_dir: Path
    prompts_dir: Path
    logs_dir: Path
    output_dir: Path
    projects_dir: Path

    # -- External tool paths --------------------------------------------------
    ffmpeg_path: str = Field(default="ffmpeg")
    ffprobe_path: str = Field(default="ffprobe")

    # -- Timeline estimation --------------------------------------------------
    # Speaking rate used to estimate segment durations before real voice audio
    # exists. Overridable per channel/voice via NARRATION_WPM.
    narration_wpm: float = Field(default=150.0, gt=0)

    # -- Video rendering (FFmpeg) ---------------------------------------------
    # x264 quality knobs for the renderer. Codecs are fixed (libx264/aac).
    render_crf: int = Field(default=18, ge=0, le=51)
    render_preset: str = Field(default="medium")

    # -- Subtitles ------------------------------------------------------------
    # Caption chunking/readability limits.
    subtitle_max_chars: int = Field(default=42, gt=0)
    subtitle_max_line_duration: float = Field(default=4.0, gt=0)
    subtitle_min_duration: float = Field(default=0.8, gt=0)
    subtitle_max_cps: float = Field(default=20.0, gt=0)
    # Burn-in style (ASS). Colours accept names / #RRGGBB / &HAABBGGRR.
    subtitle_font: str = Field(default="Noto Sans")
    subtitle_font_size: int = Field(default=54, gt=0)
    subtitle_outline: int = Field(default=3, ge=0)
    subtitle_shadow: int = Field(default=1, ge=0)
    subtitle_primary_colour: str = Field(default="white")
    subtitle_outline_colour: str = Field(default="black")
    subtitle_margin_v: int = Field(default=70, ge=0)

    # -- Thumbnail ------------------------------------------------------------
    thumbnail_width: int = Field(default=1280, gt=0)
    thumbnail_height: int = Field(default=720, gt=0)
    thumbnail_safe_area_margin: int = Field(default=60, ge=0)
    thumbnail_blur_background: bool = Field(default=True)
    thumbnail_glow: bool = Field(default=True)
    thumbnail_badge_text: str = Field(default="")
    thumbnail_fallback_position: float = Field(default=0.4, ge=0.0, le=1.0)
    # -- Thumbnail rendering (Pillow) -----------------------------------------
    thumbnail_font: str = Field(default="arialbd.ttf")
    thumbnail_title_font_size: int = Field(default=60, gt=0)
    thumbnail_highlight_font_size: int = Field(default=96, gt=0)
    thumbnail_badge_font_size: int = Field(default=40, gt=0)
    thumbnail_text_colour: str = Field(default="white")
    thumbnail_text_outline_colour: str = Field(default="black")
    thumbnail_blur_radius: int = Field(default=8, ge=0)
    thumbnail_template: str = Field(default="classic")

    # -- YouTube upload (OAuth) ------------------------------------------------
    youtube_client_secret_file: Path
    youtube_token_file: Path
    # Fixed OAuth loopback port, so the consent URL is stable and reusable
    # rather than changing on every run.
    youtube_oauth_port: int = Field(default=8765, gt=0, le=65535)
    youtube_default_privacy: str = Field(default="private")
    # Scheduled publishing, as IST wall-clock times (India is UTC+5:30, no DST).
    # The long-form lands at the evening peak; the shorts are spread across the
    # day so a single batch does not compete with itself for one session.
    # Facts about the channel that NOTHING in the footage reveals -- the author
    # asserts them. Title templates needing them stay unused while these are
    # blank, so an unset rank can never become a false claim.
    channel_rank: str = Field(default="")
    channel_season: str = Field(default="")
    publish_long_at: str = Field(default="20:00")
    publish_shorts_at: str = Field(default="13:00,18:00,21:30")
    youtube_category_id: str = Field(default="20")  # Gaming

    # -- Speech synthesis (Kokoro sidecar) ------------------------------------
    # Kokoro runs in a dedicated Python 3.12 environment (it is incompatible
    # with the backend's 3.13). The backend never imports it; it invokes the
    # worker script via this interpreter. Defaults reuse the Feature-5a spike
    # venv; repoint at a dedicated production env via KOKORO_PYTHON.
    kokoro_python: Path
    kokoro_worker: Path
    kokoro_voice: str = Field(default="af_heart")
    kokoro_lang: str = Field(default="a")

    # -- Credentials (populated in later phases; optional today) --------------
    youtube_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    claude_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    elevenlabs_api_key: SecretStr | None = None

    def ensure_directories(self) -> None:
        """Create every configured working directory if it does not yet exist.

        Lets downstream modules assume their input/output folders are present
        without each re-implementing the check.
        """
        for path in (
            self.gameplay_raw_dir,
            self.incoming_dir,
            self.gameplay_processed_dir,
            self.gameplay_archive_dir,
            self.gameplay_shorts_dir,
            self.gameplay_metadata_dir,
            self.scripts_dir,
            self.edited_dir,
            self.uploads_dir,
            self.voice_dir,
            self.assets_dir,
            self.config_dir,
            self.templates_dir,
            self.prompts_dir,
            self.logs_dir,
            self.output_dir,
            self.projects_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def require_secret(self, name: str) -> str:
        """Return the plaintext value of a required credential.

        Parameters
        ----------
        name:
            Attribute name of the credential, e.g. ``"openai_api_key"``.

        Raises
        ------
        MissingSecretError
            If the credential is not configured. This is the single, explicit
            place a missing key fails — loudly and with a helpful message.
        """
        secret = getattr(self, name, None)
        if not isinstance(secret, SecretStr):
            raise MissingSecretError(
                f"Credential {name!r} is not configured. "
                f"Set the matching variable in your .env file."
            )
        return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load ``.env`` and build the process-wide :class:`Settings` singleton.

    Cached so the ``.env`` file is read exactly once per process. n8n and the
    CLI both obtain configuration through this function.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    gameplay_root = _env_path("GAMEPLAY_DIR", _PROJECT_ROOT / "gameplay")

    return Settings(
        project_root=_PROJECT_ROOT,
        gameplay_raw_dir=_env_path("GAMEPLAY_RAW_DIR", gameplay_root / "raw"),
        incoming_dir=_env_path("GAMEPLAY_INCOMING_DIR", gameplay_root / "incoming"),
        gameplay_processed_dir=_env_path(
            "GAMEPLAY_PROCESSED_DIR", gameplay_root / "processed"
        ),
        gameplay_archive_dir=_env_path(
            "GAMEPLAY_ARCHIVE_DIR", gameplay_root / "archive"
        ),
        gameplay_shorts_dir=_env_path("GAMEPLAY_SHORTS_DIR", gameplay_root / "shorts"),
        gameplay_metadata_dir=_env_path("METADATA_DIR", gameplay_root / "metadata"),
        scripts_dir=_env_path("SCRIPTS_DIR", _PROJECT_ROOT / "scripts"),
        edited_dir=_env_path("EDITED_DIR", _PROJECT_ROOT / "edited"),
        uploads_dir=_env_path("UPLOADS_DIR", _PROJECT_ROOT / "uploads"),
        voice_dir=_env_path("VOICE_DIR", _PROJECT_ROOT / "voice"),
        assets_dir=_env_path("ASSETS_DIR", _PROJECT_ROOT / "assets"),
        config_dir=_env_path("CONFIG_DIR", _PROJECT_ROOT / "config"),
        templates_dir=_env_path("TEMPLATES_DIR", _PROJECT_ROOT / "templates"),
        prompts_dir=_env_path("PROMPTS_DIR", _PROJECT_ROOT / "prompts"),
        logs_dir=_env_path("LOGS_DIR", _PROJECT_ROOT / "logs"),
        output_dir=_env_path("OUTPUT_DIR", _PROJECT_ROOT / "backend" / "output"),
        projects_dir=_env_path("PROJECTS_DIR", _PROJECT_ROOT / "projects"),
        ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
        ffprobe_path=os.getenv("FFPROBE_PATH", "ffprobe"),
        narration_wpm=float(os.getenv("NARRATION_WPM", "150")),
        render_crf=int(os.getenv("RENDER_CRF", "18")),
        render_preset=os.getenv("RENDER_PRESET", "medium"),
        subtitle_max_chars=int(os.getenv("SUBTITLE_MAX_CHARS", "42")),
        subtitle_max_line_duration=float(os.getenv("SUBTITLE_MAX_LINE_DURATION", "4.0")),
        subtitle_min_duration=float(os.getenv("SUBTITLE_MIN_DURATION", "0.8")),
        subtitle_max_cps=float(os.getenv("SUBTITLE_MAX_CPS", "20.0")),
        subtitle_font=os.getenv("SUBTITLE_FONT", "Noto Sans"),
        subtitle_font_size=int(os.getenv("SUBTITLE_FONT_SIZE", "54")),
        subtitle_outline=int(os.getenv("SUBTITLE_OUTLINE", "3")),
        subtitle_shadow=int(os.getenv("SUBTITLE_SHADOW", "1")),
        subtitle_primary_colour=os.getenv("SUBTITLE_PRIMARY_COLOUR", "white"),
        subtitle_outline_colour=os.getenv("SUBTITLE_OUTLINE_COLOUR", "black"),
        subtitle_margin_v=int(os.getenv("SUBTITLE_MARGIN_V", "70")),
        thumbnail_width=int(os.getenv("THUMBNAIL_WIDTH", "1280")),
        thumbnail_height=int(os.getenv("THUMBNAIL_HEIGHT", "720")),
        thumbnail_safe_area_margin=int(os.getenv("THUMBNAIL_SAFE_AREA_MARGIN", "60")),
        thumbnail_blur_background=_env_bool("THUMBNAIL_BLUR_BACKGROUND", True),
        thumbnail_glow=_env_bool("THUMBNAIL_GLOW", True),
        thumbnail_badge_text=os.getenv("THUMBNAIL_BADGE_TEXT", ""),
        thumbnail_fallback_position=float(os.getenv("THUMBNAIL_FALLBACK_POSITION", "0.4")),
        thumbnail_font=os.getenv("THUMBNAIL_FONT", "arialbd.ttf"),
        thumbnail_title_font_size=int(os.getenv("THUMBNAIL_TITLE_FONT_SIZE", "60")),
        thumbnail_highlight_font_size=int(os.getenv("THUMBNAIL_HIGHLIGHT_FONT_SIZE", "96")),
        thumbnail_badge_font_size=int(os.getenv("THUMBNAIL_BADGE_FONT_SIZE", "40")),
        thumbnail_text_colour=os.getenv("THUMBNAIL_TEXT_COLOUR", "white"),
        thumbnail_text_outline_colour=os.getenv("THUMBNAIL_TEXT_OUTLINE_COLOUR", "black"),
        thumbnail_blur_radius=int(os.getenv("THUMBNAIL_BLUR_RADIUS", "8")),
        thumbnail_template=os.getenv("THUMBNAIL_TEMPLATE", "classic"),
        youtube_client_secret_file=_env_path(
            "YOUTUBE_CLIENT_SECRET_FILE", _PROJECT_ROOT / "config" / "youtube_client_secret.json"
        ),
        youtube_token_file=_env_path(
            "YOUTUBE_TOKEN_FILE", _PROJECT_ROOT / "config" / "youtube_token.json"
        ),
        youtube_oauth_port=int(os.getenv("YOUTUBE_OAUTH_PORT", "8765")),
        youtube_default_privacy=os.getenv("YOUTUBE_DEFAULT_PRIVACY", "private"),
        channel_rank=os.getenv("CHANNEL_RANK", ""),
        channel_season=os.getenv("CHANNEL_SEASON", ""),
        publish_long_at=os.getenv("PUBLISH_LONG_AT", "20:00"),
        publish_shorts_at=os.getenv("PUBLISH_SHORTS_AT", "13:00,18:00,21:30"),
        youtube_category_id=os.getenv("YOUTUBE_CATEGORY_ID", "20"),
        kokoro_python=_env_path(
            "KOKORO_PYTHON",
            _PROJECT_ROOT / "experiments" / "kokoro" / ".venv312" / "Scripts" / "python.exe",
        ),
        kokoro_worker=_env_path(
            "KOKORO_WORKER", _PROJECT_ROOT / "tts_worker" / "kokoro_worker.py"
        ),
        kokoro_voice=os.getenv("KOKORO_VOICE", "af_heart"),
        kokoro_lang=os.getenv("KOKORO_LANG", "a"),
        youtube_api_key=_env_secret("YOUTUBE_API_KEY"),
        openai_api_key=_env_secret("OPENAI_API_KEY"),
        claude_api_key=_env_secret("CLAUDE_API_KEY"),
        gemini_api_key=_env_secret("GEMINI_API_KEY"),
        elevenlabs_api_key=_env_secret("ELEVENLABS_API_KEY"),
    )
