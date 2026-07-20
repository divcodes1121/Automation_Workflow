"""Publish metadata (Phase 3.4): titles, descriptions and tags for uploads.

Built entirely from what the analyzer actually observed, because the alternative
-- inventing plausible-sounding hype -- would put claims on a real channel that
the footage may not support.

**What the analyzer can back up:** both reconstructed decks, which card a short is
built around, whether the match reached overtime, match durations, and how many
matches a session held.

**What it cannot:** who won. Crown/winner detection is deferred, so
``GameplayAnalysis.winner`` is always ``None``. Nothing here may claim a win,
a loss, or a crown count -- not even implicitly ("INSANE COMEBACK") -- because
there is no signal behind it. Overtime is the one bit of drama that is genuinely
grounded, and it is used only when the timeline really shows it.

Low-confidence cards (the analyzer's own ``needs_review`` flag) are dropped from
published deck lists rather than risk naming a card that was misdetected.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.services.highlight_editor import HIGHLIGHT_CARDS

logger = logging.getLogger(__name__)

# A phase must hold this many consecutive ~1s states to count. The match-intro
# screen can misread as "overtime" for a couple of unreadable frames.
_SUSTAIN_STATES = 5
_MAX_TAGS = 30
# YouTube caps the tags FIELD at 500 characters and rejects the whole insert
# with `invalidTags` when it is exceeded. Kept below 500 for headroom.
_TAGS_BUDGET = 460


@dataclass(frozen=True)
class VideoMetadata:
    """Everything the uploader needs to describe one video."""

    title: str
    description: str
    tags: list[str] = field(default_factory=list)


_TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "recipes" / "title_templates.json"


def _load_templates() -> dict[str, Any]:
    """The author's title library (see backend/recipes/title_templates.json)."""
    try:
        return json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read title templates (%s); falling back", exc)
        return {"long": [], "short": []}


def pick_title(
    kind: str,
    facts: dict[str, str],
    *,
    seed: int = 0,
    fallback: str = "",
    avoid: set[str] | None = None,
) -> str:
    """Choose a title from the library that the available facts can fill.

    Titles come only from the curated library. A template is eligible when every
    one of its ``requires`` keys is present in ``facts``; templates marked
    ``unverifiable`` therefore never qualify, because no fact ever supplies that
    key. That is deliberate -- the pipeline cannot measure a win rate, a trophy
    count, or even who won, so those titles stay available for manual use rather
    than being asserted automatically.

    Selection rotates deterministically on ``seed`` so consecutive uploads differ
    without being random (the same video always yields the same title).
    """
    templates = _load_templates().get(kind, [])
    eligible = [
        t for t in templates
        if all(key in facts for key in t.get("requires", []))
    ]
    if not eligible:
        return fallback

    # Two videos in one day's batch must not share a title -- they would compete
    # with each other in search and read as spam. Walk the rotation until an
    # unused one turns up; different videos can have different eligible pools, so
    # equal seeds are not enough to guarantee distinctness.
    taken = avoid or set()
    for step in range(len(eligible)):
        chosen = eligible[(seed + step) % len(eligible)]
        try:
            title = chosen["text"].format(**facts)[:100]
        except KeyError:  # a placeholder with no matching fact
            continue
        if title not in taken:
            return title
    return fallback


def card_name(slug: str) -> str:
    """``"skeleton-barrel"`` -> ``"Skeleton Barrel"``."""
    return " ".join(part.capitalize() for part in slug.split("-"))


def _deck(analysis: dict[str, Any], side: str) -> list[str]:
    """Confident card names from a reconstructed deck (flagged cards dropped)."""
    deck = analysis.get(f"{side}_deck") or {}
    return [
        card_name(c["slug"])
        for c in deck.get("cards", [])
        if c.get("slug") and not c.get("needs_review")
    ]


def _reached_overtime(analysis: dict[str, Any]) -> bool:
    """Did the match genuinely go to overtime (sustained, not a misread)?"""
    run = 0
    for state in analysis.get("match_states") or []:
        run = run + 1 if state.get("phase") == "overtime" else 0
        if run >= _SUSTAIN_STATES:
            return True
    return False


def _clock(seconds: float) -> str:
    """Seconds -> M:SS (YouTube chapter format)."""
    total = int(seconds)
    return f"{total // 60}:{total % 60:02d}"


def _tags(decks: list[str], extra: list[str], featured: list[str] | None = None) -> list[str]:
    """Deduplicated, length-capped tag list, broad -> specific.

    Ordering matters: YouTube weights earlier tags more, so the broad terms that
    establish the topic come first, then the long-tail phrases that actually win
    niche searches ("skeleton barrel cycle deck" beats "gameplay").
    """
    base = [
        "clash royale", "clashroyale", "clash royale gameplay",
        "clash royale ladder", "mobile gaming", "mobile games",
    ]
    # Long-tail combinations built from the cards actually played -- these are
    # what a viewer searching for this specific deck or matchup types.
    longtail: list[str] = []
    for name in featured or []:
        low = name.lower()
        longtail += [low, f"{low} deck", f"clash royale {low}", f"{low} clash royale"]

    seen: list[str] = []
    used = 0
    for tag in base + longtail + extra + [d.lower() for d in decks]:
        if not tag or tag in seen or len(tag) > 30:
            continue
        cost = _tag_cost(tag)
        if used + cost > _TAGS_BUDGET:
            continue  # skip this one but keep trying shorter later tags
        seen.append(tag)
        used += cost
    return seen[:_MAX_TAGS]


def _tag_cost(tag: str) -> int:
    """Characters a tag consumes against YouTube's 500-char tags budget.

    A tag containing a space is counted by YouTube as though it were **quoted**,
    so it costs two more characters than it looks. Missing this rejected a real
    upload with ``invalidTags``: 29 tags measured 477 raw but 527 the way YouTube
    counts them.
    """
    return len(tag) + (2 if " " in tag else 0) + 1  # +1 for the separator


def short_metadata(
    analysis: dict[str, Any],
    signature_card: str | None,
    *,
    index: int,
    avoid: set[str] | None = None,
) -> VideoMetadata:
    """Metadata for one per-match highlight short (a match summary).

    Titles must differ across a day's batch -- three uploads sharing one title
    read as spam and compete with each other in search. The opponent's deck
    supplies that variation and is genuinely detected, so each match is named by
    what it was played against.
    """
    player = _deck(analysis, "player")
    opponent = _deck(analysis, "opponent")
    overtime = _reached_overtime(analysis)

    mine = _featured(analysis, player)
    theirs = _featured(analysis, opponent, opponent_side=True)

    # No outcome words anywhere: the winner is not detected. Overtime is the only
    # drama that is actually evidenced.
    subject = " + ".join(mine) if mine else "Clash Royale"
    # "Skeleton Barrel + Rocket vs Rocket" reads as a mistake. When the opponent
    # leads with a card we also run it is a mirror -- which is a more interesting
    # thing to say anyway, and is what the reconstructed decks actually show.
    if theirs and theirs[0] in mine:
        versus = " (Mirror Match)"
    elif theirs:
        versus = f" vs {theirs[0]}"
    else:
        versus = ""
    prefix = "OVERTIME: " if overtime else ""
    # Titles come from the curated library; this generated one is the fallback
    # used only when no template's requirements can be met.
    generated = f"{prefix}{subject}{versus} | Clash Royale #Shorts"
    facts = build_facts(analysis, mine, theirs, mirror=" (Mirror" in versus)
    title = pick_title("short", facts, seed=index, fallback=generated, avoid=avoid)

    # The first two lines are all that shows in search and suggested-video
    # panels, so the searchable terms go there rather than below the fold.
    matchup = f" against {theirs[0]}" if theirs and " (Mirror" not in versus else ""
    opener = (
        f"{subject} cycle{matchup} in a ranked Clash Royale ladder match."
        if mine else "Ranked Clash Royale ladder match."
    )
    lines = [opener, "Win conditions and spells only - every push that mattered."]
    if overtime:
        lines += ["", "This one went to overtime."]
    if player:
        lines += ["", "MY DECK", ", ".join(player)]
    if opponent:
        lines += ["", "OPPONENT", ", ".join(opponent)]
    lines += [
        "",
        "Clash Royale gameplay with original game audio, no commentary.",
        "",
        # YouTube surfaces the first three hashtags above the title.
        "#Shorts #ClashRoyale #ClashRoyaleShorts",
    ]

    extra = [
        "clash royale shorts", "shorts", "clash royale highlights",
        "clash royale no commentary", "clash royale deck",
    ]
    if overtime:
        extra.append("clash royale overtime")
    return VideoMetadata(
        title=title,
        description="\n".join(lines),
        tags=_tags(player, extra, featured=mine + theirs),
    )


def build_facts(
    analysis: dict[str, Any],
    mine: list[str],
    theirs: list[str],
    *,
    mirror: bool = False,
) -> dict[str, str]:
    """Facts a title template may be filled from -- only things known to be true.

    ``rank`` and ``season`` come from ``.env`` because nothing in the footage
    reveals them; the author asserts them, and templates needing them stay
    ineligible until they do.
    """
    from backend.config import get_settings

    settings = get_settings()
    facts: dict[str, str] = {"year": str(datetime.now(IST).year)}
    if mine:
        facts["deck"] = mine[0]
    # In a mirror the opponent's card IS ours, so "vs {opponent}" would read as
    # "Rocket vs Rocket". Withholding the fact makes those templates ineligible
    # and leaves the mirror-specific ones to win.
    if theirs and not mirror:
        facts["opponent"] = theirs[0]
    if mirror:
        facts["mirror"] = "1"
    if _reached_overtime(analysis):
        facts["overtime"] = "1"
    if settings.channel_rank:
        facts["rank"] = settings.channel_rank
    if settings.channel_season:
        facts["season"] = settings.channel_season
    return facts


def _featured(
    analysis: dict[str, Any], deck: list[str], *, opponent_side: bool = False, limit: int = 2
) -> list[str]:
    """The cards worth naming in a title: the deck's win conditions/spells.

    For the player these are ranked by how often they were actually played; the
    opponent's individual plays are not tracked, so their deck order (which the
    reconstructor sorts by observation count) is used instead.
    """
    candidates = [c for c in deck if c.lower().replace(" ", "-") in HIGHLIGHT_CARDS]
    if opponent_side:
        return candidates[:limit]

    counts: dict[str, int] = {}
    for event in analysis.get("events") or []:
        slug = event.get("card")
        if slug in HIGHLIGHT_CARDS:
            counts[card_name(slug)] = counts.get(card_name(slug), 0) + 1
    ranked = sorted(candidates, key=lambda name: -counts.get(name, 0))
    return [name for name in ranked if counts.get(name)][:limit] or candidates[:limit]


def session_metadata(
    analyses: list[dict[str, Any]],
    clip_durations: list[float],
    *,
    signature_card: str | None = None,
) -> VideoMetadata:
    """Metadata for the merged, gameplay-only long-form video.

    The description carries YouTube chapter markers, computed from the real clip
    durations, so each match is directly seekable. YouTube requires the first
    chapter to be ``0:00``.
    """
    count = len(clip_durations)
    player = _deck(analyses[0], "player") if analyses else []
    anchor = card_name(signature_card) if signature_card else None

    generated = (
        f"{anchor} Ladder Session | {count} Matches | Clash Royale"
        if anchor
        else f"Clash Royale Ladder Session | {count} Matches"
    )
    facts = build_facts(analyses[0] if analyses else {}, [anchor] if anchor else [], [])
    # Rotate on the day so consecutive daily uploads do not share a title.
    title = pick_title(
        "long", facts, seed=datetime.now(IST).timetuple().tm_yday, fallback=generated
    )

    deck_phrase = f"{anchor} cycle" if anchor else "ladder"
    lines = [
        f"{count} full ranked Clash Royale matches with a {deck_phrase} deck, "
        "back to back.",
        "Original game audio, no commentary - just the gameplay.",
        "",
        "CHAPTERS",
    ]
    # YouTube only builds a chapter list if the first stamp is exactly 0:00 and
    # there are at least three of them, each 10s or longer.
    offset = 0.0
    for i, duration in enumerate(clip_durations):
        label = f"Match {i + 1}"
        if i < len(analyses) and _reached_overtime(analyses[i]):
            label += " (overtime)"
        lines.append(f"{_clock(offset)} {label}")
        offset += duration
    if player:
        lines += ["", "MY DECK", ", ".join(player)]
    lines += ["", "#ClashRoyale #ClashRoyaleGameplay #MobileGaming"]

    extra = [
        "clash royale matches", "clash royale no commentary",
        "clash royale full match", "clash royale deck", "clash royale session",
    ]
    featured = [anchor] if anchor else []
    return VideoMetadata(
        title=title, description="\n".join(lines), tags=_tags(player, extra, featured=featured)
    )


# India has never observed daylight saving and has been UTC+5:30 since 1945, so
# a fixed offset is exactly right here -- no tzdata dependency, which matters
# because Windows ships no IANA time-zone database and `zoneinfo` fails on it.
IST = timezone(timedelta(hours=5, minutes=30))


def next_slot(hhmm: str, *, now: datetime | None = None, lead_minutes: int = 30) -> datetime:
    """The next occurrence of an IST wall-clock time like ``"20:00"``.

    Returns today's slot when it is still at least ``lead_minutes`` away,
    otherwise tomorrow's. The lead time exists because YouTube rejects a
    ``publishAt`` in the past, and an upload started minutes before the slot
    would race it.
    """
    current = (now or datetime.now(IST)).astimezone(IST)
    hour, _, minute = hhmm.partition(":")
    slot = current.replace(
        hour=int(hour), minute=int(minute or 0), second=0, microsecond=0
    )
    if slot <= current + timedelta(minutes=lead_minutes):
        slot += timedelta(days=1)
    return slot


def parse_slots(value: str) -> list[str]:
    """``"13:00, 18:00 ,21:30"`` -> ``["13:00", "18:00", "21:30"]``."""
    return [part.strip() for part in value.split(",") if part.strip()]


# YouTube rejects thumbnails over 2 MB.
_THUMBNAIL_MAX_BYTES = 2 * 1024 * 1024
_THUMBNAIL_SUFFIXES = (".png", ".jpg", ".jpeg")


def find_thumbnail(video: Path) -> Path | None:
    """A hand-made thumbnail sitting next to ``video``, if there is one.

    Looks for ``<video stem>.thumbnail.png|jpg`` in the video's own folder, so a
    thumbnail is adopted just by dropping the file in beside the video -- no flag
    and no rename of the video itself. Oversized files are ignored with a warning
    rather than failing the upload, since the video matters more than the image.
    """
    for suffix in _THUMBNAIL_SUFFIXES:
        candidate = video.with_name(f"{video.stem}.thumbnail{suffix}")
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        if size > _THUMBNAIL_MAX_BYTES:
            logger.warning(
                "Ignoring thumbnail %s: %.1f MB exceeds YouTube's 2 MB limit",
                candidate.name, size / 1024 / 1024,
            )
            return None
        return candidate
    return None


def load_analysis(path: Path | str | None) -> dict[str, Any]:
    """Read an analysis JSON, or ``{}`` when unavailable."""
    import json

    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Could not read analysis %s: %s", path, exc)
        return {}
