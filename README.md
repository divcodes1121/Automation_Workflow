# CR AI Workflow

Turns raw Clash Royale screen recordings into edited, publishable video — automatically.

Record a session on your phone, drop the file in a folder, run one command. The
system finds each match inside the recording, cuts it out, merges the matches
into one clean gameplay video, and produces a Shorts-length highlight per match
with the original game audio.

---

## Quick start

```bash
# 1. Drop your recording here
gameplay/incoming/

# 2. Run
python -m backend.main auto --profile iphone_16_pro_max
```

That's it. Everything below is produced for you:

| Output | Where |
|---|---|
| One clip per match (loading screen → crowns) | `gameplay/raw/<name>_game_NN.mp4` |
| All matches back to back, no menus | `gameplay/raw/<name>_merged.mp4` |
| One highlight short per match | `edited/<name>_game_NN.short.mp4` |
| What ran, and what it produced | `edited/<name>.session_result.json` |

To process one specific file instead of the whole inbox:

```bash
python -m backend.main auto path/to/recording.mp4 --profile iphone_16_pro_max
```

Useful flags: `--no-shorts` (split and merge only), `--no-merge`,
`--card rocket` (anchor every short on a specific card).

> **`--profile` matters.** It tells the analyzer where the match clock, hand
> cards and elixir bars sit on screen, which depends on your capture device and
> resolution. A profile calibrated for iPhone 16 Pro Max (1320×2868) ships as
> `iphone_16_pro_max`. For a different device, see *Calibration* below.

---

## Publishing to YouTube

One command produces and publishes a day's batch: **1 long-form video + 1 short
per match**.

```bash
python -m backend.main auto --profile iphone_16_pro_max --upload
```

Uploads land **private and stay private** -- nothing is auto-published. You
make videos public yourself in YouTube Studio. Add `--schedule` to hand that
decision to the IST slots below, or `--privacy public` to publish immediately.

### See what would be posted first

```bash
python -m backend.main publish-preview edited/<name>.session_result.json
```

Prints the exact title, description and tags for every video. Reads local files
only — no credentials, no API calls, nothing uploaded.

### Metadata is generated from what was actually detected

Titles and descriptions are built from the analyzer's real output: both
reconstructed decks, the win conditions and spells played, and whether the match
genuinely reached overtime. The long-form description carries **chapter markers**
computed from the real clip durations, so each match is seekable.

**Nothing claims a result.** Crown/winner detection is not implemented, so
`winner` is always null — no title says "win", "comeback" or "destroyed",
because there is no signal behind it. Cards the analyzer flagged as
low-confidence are dropped from published deck lists rather than risk naming the
wrong card.

### One-time credential setup

```
1. console.cloud.google.com -> new project
2. Enable "YouTube Data API v3"
3. Credentials -> Create -> OAuth client ID -> Desktop app
4. Download the JSON -> save as config/youtube_client_secret.json
```

The first upload opens a browser once for consent; the token then caches to
`config/youtube_token.json` and later runs are non-interactive.

Two limits worth knowing:

- **Quota.** `videos.insert` costs ~1600 units against a default 10,000/day.
  4 uploads/day (1 long + 3 shorts) is ~6,400 — comfortable. Much beyond that
  needs a quota increase request.
- **Unverified apps.** A fresh Google Cloud project can upload, but videos stay
  locked to private until the OAuth app passes verification. Plan on reviewing
  and flipping visibility by hand at first.

### Scheduled publishing (IST)

Uploading and *publishing* are separate. Videos upload private whenever the
machine is free, and YouTube flips them public at a set time via `publishAt`:

| Video | Publish slot (IST) |
|---|---|
| Short 1 | 13:00 |
| Short 2 | 18:00 |
| Long-form | 20:00 |
| Short 3 | 21:30 |

Change them in `.env`:

```
PUBLISH_LONG_AT=20:00
PUBLISH_SHORTS_AT=13:00,18:00,21:30
```

Scheduling is **off by default**; `--schedule` turns it on.

Times are IST (UTC+5:30, fixed — India has no DST, so no `tzdata` dependency).
A slot less than 30 minutes away rolls to the next day, because YouTube rejects
a `publishAt` in the past.

**These slots are a starting hypothesis, not a measured optimum.** They assume an
India-centric gaming audience (evening peak) and deliberately stagger the batch
so four videos don't compete for one browsing session. Replace them with real
data once you have 2-4 weeks of public uploads: **YouTube Studio -> Analytics ->
Audience -> "When your viewers are on YouTube."**

### Disk cleanup

One session costs roughly **6.8 GB**:

| Item | Size |
|---|---|
| Split clips (3) | ~1.4 GB |
| Merged long-form | ~1.5 GB |
| Shorts (3) | ~80 MB |
| Analyzer frame cache | ~2.2 GB |
| Original recording | ~1.6 GB |

Daily, that is ~200 GB/month, so the pipeline can delete it after publishing:

```bash
python -m backend.main auto --upload --cleanup       # keeps the original recording
python -m backend.main auto --upload --cleanup-raw   # deletes that too
```

**Cleanup only runs when every video uploaded.** If any upload failed it is
skipped with a warning, because the footage is the only copy of anything that
did not publish.

**The daily cycle is net-zero on disk.** Everything a run creates, it removes:

| Artifact | Size | After the cycle |
|---|---|---|
| Raw recording | ~1.6 GB | deleted (`--cleanup-raw`) |
| Split clips | ~1.4 GB | deleted |
| Merged long-form | ~1.5 GB | deleted |
| Shorts | ~80 MB | deleted |
| Analyzer frame cache | ~2.2 GB | deleted |
| JSON records | ~20 KB | kept |
| Log file | ~5 KB | kept, pruned after 30 days |

So disk peaks ~6.8 GB mid-run and returns to baseline, growing only ~25 KB a day
(~9 MB a year) in records. What stays permanently is the ~1.4 GB of regression
fixtures (`gameplay/raw/game_01|02|03.mp4`) and the ~12 MB template cache.

> Running `python -m analyzer.main regression` regenerates ~2.2 GB of frame cache
> for those fixtures. Delete `analyzer/cache/frames/v1/*` afterwards to return to
> baseline; it is rebuilt on demand.

`--cleanup-raw` is separate from `--cleanup` on purpose: the original recording
is the one file nothing can rebuild. Every other artifact is derived from it, so
losing it means a video can never be re-cut or re-rendered.

JSON artifacts (split plan, analyses, upload results) are always kept — they are
kilobytes and are the only surviving record of what was published and where.

### Daily automation (Windows Task Scheduler)

`scripts/daily_publish.bat` processes the whole inbox and uploads. Register it:

```bat
schtasks /Create /TN "CR Daily Publish" ^
  /TR "\"C:\path\to\CR AI Workflow\scripts\daily_publish.bat\"" ^
  /SC DAILY /ST 09:00 /RL LIMITED /F
```

Each run logs to `logs/daily_publish_<timestamp>.log` and exits non-zero on
failure. Recordings processed **from the inbox** are moved to
`gameplay/archive/` afterwards, so the next day's run cannot republish the same
session. A recording named explicitly on the command line is left where it is.

The daily job only has work if a recording is waiting — drop one session
(3 matches) into `gameplay/incoming/` and it yields exactly that day's
1 long + 3 shorts.

---

## How the splitting works

The hard part is knowing where each match starts and ends. The trick is that
**Clash Royale only draws the countdown clock during a battle** — so the timer
detector doubles as an "am I in a match?" signal. No new detector, and no
expensive card matching: only the small timer region is decoded.

Three rules turn that raw signal into boundaries, each measured against a real
14.8-minute, 3-match recording:

1. **Gap-merge.** In the final minute the clock reads only every *other* second,
   which would otherwise shatter one match into ~15 fragments. Runs fuse across
   short gaps — but only when the clock actually ticked down by about the elapsed
   time, so an accidentally-opened replay sitting 9s before a real match is *not*
   absorbed.
2. **Minimum duration.** Real matches ran 177–268s; accidental captures were 2s
   and 15s. Anything under 60s is discarded as a replay or abandoned match.
3. **Scene snapping.** The clock marks the battle, but a watchable clip starts on
   the loading screen and ends after the crowns banner. Each boundary extends
   outward to a real screen transition found by a short, windowed FFmpeg
   scene-detect pass.

Cutting is stream-copy — no re-encode, so splitting a session takes seconds.

**Accuracy.** Against hand-cut reference clips the detected boundaries land
within ~1s, and every clip starts on the match intro with no lobby frames and
ends on the winner banner.

---

## Commands

Each stage is independently runnable; `auto` just chains them.

```bash
# Find the matches in a recording (add --plan-only to inspect without cutting)
python -m analyzer.main split <recording.mp4> --profile <device>

# Concatenate a plan's clips into one gameplay-only video
python -m analyzer.main merge <split_plan.json> -o merged.mp4

# Analyze one match -> gameplay_analysis.json (decks, plays, clock, elixir)
python -m analyzer.main analyze <match.mp4> --profile <device> --sample-fps 1

# Cut a highlight reel from an analysis
python -m backend.main highlight <analysis.json> <match.mp4> --no-effects --no-memes

# Protect the frozen deck engine after any analyzer change
python -m analyzer.main regression
```

---

## Architecture

Three packages that deliberately do not import each other. Their only contract is
JSON on disk, so each can evolve without breaking the others.

```
analyzer/       Computer vision (classical, no ML). Reads footage, writes JSON.
                Never imports backend.
backend/        Pipeline + editing. Reads the analyzer's JSON as plain dicts.
asset_manager/  Standalone asset library. Editor resolves assets by symbolic
                name (bass_thud, emoji_fire), never by filename.
```

`auto` drives the analyzer through its CLI rather than importing it, which keeps
that boundary intact.

**What the analyzer extracts:** both players' decks (reconstructed across the
match with confidence scores), every card the player played with a timestamp, and
a ~1/sec match-state timeline (clock, phase, elixir, 1×/2× multiplier).

---

## Calibration

A device profile maps screen regions (timer, hand slots, elixir bars) as
fractions of frame size. Profiles live in `analyzer/calibration/profiles/`.

```bash
# Draw the current profile's regions over a real frame to check the fit
python -m analyzer.main calibrate <frame-or-video> --profile <device>
```

If the split finds no matches, the timer region is almost certainly misaligned —
start there.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
python -m analyzer.main build-templates    # one-time card template cache
```

FFmpeg is required. If it is not on `PATH`, set both in `.env`:

```
FFMPEG_PATH=C:/path/to/ffmpeg.exe
FFPROBE_PATH=C:/path/to/ffprobe.exe
```

---

## Notes

- Shorts currently render with **no effects and no captions** — raw event-synced
  cuts carrying the original Clash Royale audio. The effects engine, animated
  captions and meme layer exist and are wired, but are off in this path.
- Gameplay footage, the asset library and `Memes/` are gitignored (size and
  third-party licensing); everything under them is regenerable or local.
- The narrated long-form pipeline (TTS → subtitles → thumbnail → YouTube upload)
  still exists and is unchanged; see `python -m backend.main --help`.
