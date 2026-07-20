"""YouTube upload — publish the finished video + thumbnail (Feature 10).

:class:`YouTubeUploader` is **upload-only**: no AI, no metadata generation, no
editing, no scheduling. Metadata comes from the :class:`~backend.models.Project`;
privacy/category from config. It follows the ``prepare() -> --dry-run -> upload()``
pattern used by the render/burn/thumbnail stages.

Authentication uses the OAuth **InstalledApp flow** (uploading to a personal
channel requires user consent): the first run opens a browser once and caches a
token; later runs reuse/refresh it. The Google client libraries are imported
lazily inside methods so the rest of the backend imports without them.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import Settings, get_settings
from backend.models import Project, UploadResult

logger = logging.getLogger(__name__)

# Minimal scope needed to insert a video and set its thumbnail.
YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"

# Resumable-upload chunk size. Must be a multiple of 256 KB. 8 MB keeps each
# request short enough to survive a flaky link while staying efficient on a
# multi-gigabyte long-form video.
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_CHUNK_RETRIES = 5
# Transport-level failures that a resumable session can simply continue past.
_RETRYABLE = (ConnectionError, TimeoutError, OSError)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class UploadError(Exception):
    """Base class for all errors raised by :class:`YouTubeUploader`."""


class UploadRequestError(UploadError):
    """Raised when the upload request is invalid (e.g. missing files)."""


class UploadAuthError(UploadError):
    """Raised when OAuth credentials are missing or authentication fails."""


class YouTubeUploader:
    """Uploads a video + thumbnail to YouTube.

    Parameters
    ----------
    settings:
        Optional configuration override (defaults to
        :func:`~backend.config.get_settings`), supplying credential paths,
        default privacy and category.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings or get_settings()
        self._last_response: dict[str, Any] | None = None

    # -- Public API -----------------------------------------------------------

    def credentials_present(self) -> bool:
        """Whether the OAuth client-secret file exists."""
        return Path(self._settings.youtube_client_secret_file).is_file()

    def prepare(
        self,
        project: Project,
        video: Path,
        thumbnail: Path,
        privacy: str | None = None,
    ) -> dict[str, Any]:
        """Validate inputs and build the YouTube request from a Project.

        Thin wrapper over :meth:`prepare_video` for the narrated pipeline, whose
        metadata lives on the :class:`~backend.models.Project`.
        """
        return self.prepare_video(
            video,
            title=project.title,
            description=project.description,
            tags=project.tags,
            language=project.language,
            thumbnail=thumbnail,
            privacy=privacy,
        )

    def prepare_video(
        self,
        video: Path,
        *,
        title: str,
        description: str,
        tags: list[str],
        language: str = "en",
        thumbnail: Path | None = None,
        privacy: str | None = None,
        publish_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Validate inputs and build the YouTube request. No API/auth.

        Metadata is passed directly rather than through a ``Project`` so gameplay
        clips can be uploaded without inventing narration fields (``long_script``,
        ``voice_style``) that do not apply to them.

        ``thumbnail`` is optional: YouTube does not take custom thumbnails for
        Shorts, and an auto-selected frame is better than a fabricated one.

        Raises
        ------
        UploadRequestError
            If the video (or a supplied thumbnail) is missing.
        """
        problems: list[str] = []
        if not Path(video).is_file():
            problems.append(f"video not found: {video}")
        if thumbnail is not None and not Path(thumbnail).is_file():
            problems.append(f"thumbnail not found: {thumbnail}")
        if problems:
            raise UploadRequestError("Cannot upload:\n- " + "\n- ".join(problems))

        resolved_privacy = privacy or self._settings.youtube_default_privacy
        body = {
            "snippet": {
                # YouTube rejects titles over 100 chars outright.
                "title": title[:100],
                "description": description[:5000],
                "tags": tags,
                "categoryId": self._settings.youtube_category_id,
                "defaultLanguage": language,
                # Declaring the audio language is an accessibility signal: it is
                # what lets YouTube offer auto-captions and translated metadata.
                # Without it a video is treated as language-unknown.
                "defaultAudioLanguage": language,
            },
            "status": {
                "privacyStatus": resolved_privacy,
                # Never "made for kids": that setting strips comments, disables
                # personalised ads and removes the video from most surfaces.
                "selfDeclaredMadeForKids": False,
            },
        }
        if publish_at is not None:
            # Scheduled publishing: YouTube requires the video to be inserted as
            # private, then flips it public at this instant. This is what lets the
            # heavy processing/upload run whenever the machine is free while the
            # video still goes live at the audience's peak hour.
            if publish_at.tzinfo is None:
                raise UploadRequestError("publish_at must be timezone-aware")
            body["status"]["privacyStatus"] = "private"
            body["status"]["publishAt"] = (
                publish_at.astimezone(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
        return {
            "body": body,
            "video": str(Path(video)),
            "thumbnail": str(Path(thumbnail)) if thumbnail is not None else None,
            "privacy": resolved_privacy,
        }

    def upload(
        self,
        project: Project,
        video: Path,
        thumbnail: Path,
        privacy: str | None = None,
    ) -> UploadResult:
        """Upload the video + thumbnail and return an :class:`UploadResult`.

        Raises
        ------
        UploadRequestError, UploadAuthError, UploadError
        """
        return self.upload_request(self.prepare(project, video, thumbnail, privacy))

    def upload_request(self, request: dict[str, Any]) -> UploadResult:
        """Execute a request built by :meth:`prepare`/:meth:`prepare_video`."""
        title = request["body"]["snippet"]["title"]
        youtube = self._service()

        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        started = time.perf_counter()
        try:
            # Chunked, not chunksize=-1. Sending the whole file as one request has
            # no resume point, so a single network blip loses everything -- a
            # 1.4 GB upload died on ConnectionResetError that way. Chunking lets
            # the resumable protocol retry just the failed slice.
            media = MediaFileUpload(
                request["video"],
                chunksize=_UPLOAD_CHUNK_BYTES,
                resumable=True,
                mimetype="video/*",
            )
            insert = youtube.videos().insert(
                part="snippet,status", body=request["body"], media_body=media
            )
            response = None
            attempts = 0
            while response is None:
                try:
                    status, response = insert.next_chunk(num_retries=_CHUNK_RETRIES)
                    attempts = 0
                    if status:
                        logger.info("Upload progress: %d%%", int(status.progress() * 100))
                except _RETRYABLE as exc:
                    # The connection dropped mid-chunk. The resumable session is
                    # still valid, so backing off and continuing resumes from the
                    # last acknowledged byte rather than restarting the file.
                    attempts += 1
                    if attempts > _CHUNK_RETRIES:
                        raise UploadError(
                            f"upload failed after {_CHUNK_RETRIES} retries: {exc}"
                        ) from exc
                    delay = 2 ** attempts
                    logger.warning(
                        "Connection dropped mid-upload (%s); retry %d/%d in %ds",
                        type(exc).__name__, attempts, _CHUNK_RETRIES, delay,
                    )
                    time.sleep(delay)
        except HttpError as exc:
            raise UploadError(f"YouTube video insert failed: {exc}") from exc

        video_id = response["id"]
        thumbnail_uploaded = False
        if request.get("thumbnail"):
            try:
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(request["thumbnail"], mimetype="image/png"),
                ).execute()
                thumbnail_uploaded = True
            except HttpError as exc:  # non-fatal — video is already up.
                logger.warning("Thumbnail set failed for %s: %s", video_id, exc)

        elapsed = time.perf_counter() - started
        self._last_response = response
        upload_status = response.get("status", {}).get("uploadStatus", "uploaded")
        result = UploadResult(
            video_id=video_id,
            url=f"https://youtu.be/{video_id}",
            title=title,
            privacy=request["privacy"],
            status="uploaded",
            upload_time=datetime.now(timezone.utc),
            processing_status=upload_status,
            thumbnail_uploaded=thumbnail_uploaded,
            elapsed_seconds=round(elapsed, 3),
        )
        logger.info("Uploaded %r -> %s (%.1fs)", title, result.url, elapsed)
        return result

    def save(self, result: UploadResult, destination: Path | None = None) -> Path:
        """Write ``upload_result.json`` (+ the raw ``youtube_response.json``)."""
        slug = _slugify(result.title)
        dest = destination or self._settings.edited_dir / f"{slug}.upload_result.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if self._last_response is not None:
            raw = dest.parent / f"{slug}.youtube_response.json"
            raw.write_text(
                json.dumps(self._last_response, indent=2, default=str), encoding="utf-8"
            )
        logger.info("Saved upload result for %r to %s", result.title, dest)
        return dest

    def save_request(self, request: dict[str, Any], destination: Path) -> Path:
        """Write the built request to JSON (for ``--dry-run``)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(request, indent=2), encoding="utf-8")
        return destination

    # -- Auth / service -------------------------------------------------------

    def _service(self):
        """Build an authenticated YouTube Data API v3 client."""
        from googleapiclient.discovery import build

        return build("youtube", "v3", credentials=self._authenticate())

    def _authenticate(self):
        """Load/refresh cached credentials, or run the consent flow once."""
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        token = Path(self._settings.youtube_token_file)
        secret = Path(self._settings.youtube_client_secret_file)

        creds = None
        if token.is_file():
            creds = Credentials.from_authorized_user_file(str(token), [YOUTUBE_UPLOAD_SCOPE])
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token.write_text(creds.to_json(), encoding="utf-8")
            return creds

        if not secret.is_file():
            raise UploadAuthError(
                f"OAuth client secret not found at {secret}. Create a Desktop OAuth "
                f"client for the YouTube Data API v3 in Google Cloud Console and save "
                f"its JSON there (or set YOUTUBE_CLIENT_SECRET_FILE)."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(secret), [YOUTUBE_UPLOAD_SCOPE])
        creds = flow.run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json(), encoding="utf-8")
        return creds


def _slugify(value: str) -> str:
    """Produce a filesystem-safe slug, e.g. ``"Evo Pekka 001" -> "evo_pekka_001"``."""
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "upload"
