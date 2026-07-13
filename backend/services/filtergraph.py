"""Pure FFmpeg filtergraph compilation for the video renderer (Feature 7B).

:class:`FiltergraphBuilder` turns an :class:`~backend.models.EditPlan` into a single
:class:`FFmpegCommand` — inputs, one ``-filter_complex`` graph, and output args — with
**no I/O and no FFmpeg invocation**. The renderer executes it; this module only compiles.

Design highlights:

* **Deduplicated inputs.** Each unique gameplay file appears once as ``-i``; a clip used
  by *k* segments is ``split`` into *k* pads (an FFmpeg filter pad is consumed once). So
  20 segments on one recording produce **one** ``-i match.mp4`` (+ one ``-i`` per WAV),
  not 20 video inputs.
* **One encode.** Trim each range → concat gameplay video, concat narration WAVs, mux,
  encode once (H.264 + AAC). ``-shortest`` guards against tiny A/V duration mismatches.
"""

from __future__ import annotations

import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from backend.models import EditPlan


def _format_fps(fps: float) -> str:
    """Render fps as ``"30"`` for integers, else a trimmed decimal."""
    return str(int(fps)) if float(fps).is_integer() else f"{fps:.5f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class FFmpegCommand:
    """A fully-compiled FFmpeg invocation for one render."""

    argv: list[str]
    filtergraph: str
    inputs: list[Path]
    output: Path

    def command_text(self) -> str:
        """The full command as a single shell-quoted line (for the debug file)."""
        return " ".join(shlex.quote(arg) for arg in self.argv)

    def filtergraph_text(self) -> str:
        """The filter graph with one chain per line (copy-paste-able on error)."""
        return "\n".join(self.filtergraph.split(";"))


class FiltergraphBuilder:
    """Compiles an :class:`EditPlan` into an :class:`FFmpegCommand`. Pure."""

    @staticmethod
    def build(
        plan: EditPlan,
        output: Path,
        *,
        ffmpeg_path: str,
        fps: float,
        crf: int,
        preset: str,
    ) -> FFmpegCommand:
        """Build the FFmpeg command for ``plan``.

        Parameters
        ----------
        plan:
            The edit plan to compile (assumed already validated; must be non-empty).
        output:
            Destination video path.
        ffmpeg_path, fps, crf, preset:
            Renderer settings (fps is the shared source frame rate).
        """
        segments = plan.segments

        # -- Deduplicated input table: unique videos first, then one WAV per segment.
        unique_videos: list[Path] = []
        video_input_index: dict[str, int] = {}
        for segment in segments:
            key = str(segment.source_file)
            if key not in video_input_index:
                video_input_index[key] = len(unique_videos)
                unique_videos.append(Path(segment.source_file))
        num_videos = len(unique_videos)
        inputs: list[Path] = [*unique_videos, *(Path(s.audio_file) for s in segments)]

        # -- Split each clip into as many pads as it is used (>1); else use directly.
        usage = Counter(str(s.source_file) for s in segments)
        fps_str = _format_fps(fps)
        split_chains: list[str] = []
        pad_queues: dict[str, list[str]] = {}
        for key, index in video_input_index.items():
            uses = usage[key]
            if uses == 1:
                pad_queues[key] = [f"{index}:v"]
            else:
                labels = [f"v{index}s{j}" for j in range(uses)]
                split_chains.append(
                    f"[{index}:v]split={uses}" + "".join(f"[{label}]" for label in labels)
                )
                pad_queues[key] = list(labels)

        # -- Per-segment video trims and audio resets.
        video_chains: list[str] = []
        video_labels: list[str] = []
        audio_chains: list[str] = []
        audio_labels: list[str] = []
        for i, segment in enumerate(segments):
            pad = pad_queues[str(segment.source_file)].pop(0)
            vlabel = f"v{i}"
            video_chains.append(
                f"[{pad}]trim=start={segment.source_start_seconds}:"
                f"end={segment.source_end_seconds},setpts=PTS-STARTPTS,"
                f"fps={fps_str},format=yuv420p,setsar=1[{vlabel}]"
            )
            video_labels.append(vlabel)

            alabel = f"a{i}"
            audio_chains.append(f"[{num_videos + i}:a]asetpts=PTS-STARTPTS[{alabel}]")
            audio_labels.append(alabel)

        n = len(segments)
        concat_video = (
            "".join(f"[{label}]" for label in video_labels)
            + f"concat=n={n}:v=1:a=0[outv]"
        )
        concat_audio = (
            "".join(f"[{label}]" for label in audio_labels)
            + f"concat=n={n}:v=0:a=1[outa]"
        )
        filtergraph = ";".join(
            [*split_chains, *video_chains, *audio_chains, concat_video, concat_audio]
        )

        argv: list[str] = [str(ffmpeg_path), "-y"]
        for source in inputs:
            argv += ["-i", str(source)]
        argv += [
            "-filter_complex", filtergraph,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-r", fps_str,
            "-movflags", "+faststart",
            "-shortest",
            str(output),
        ]
        return FFmpegCommand(
            argv=argv, filtergraph=filtergraph, inputs=inputs, output=Path(output)
        )
