from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Tuple

from src.config import (
    VOICEOVER_DUCKING_VOLUME,
    VOICEOVER_AUDIO_FADE,
    INTRO_SFX_VOLUME,
)


def _run(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result


def probe_video(path: Path) -> dict:
    duration_result = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])

    try:
        duration = max(0.0, float(duration_result.stdout.strip()))
    except ValueError as exc:
        raise RuntimeError(f"Nu pot determina durata video pentru {path}") from exc

    audio_result = _run([
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(path),
    ])

    return {"duration": duration, "has_audio": bool(audio_result.stdout.strip())}


def choose_teaser_range(
    clip: dict,
    video_duration: float,
    hook_duration: float,
) -> Tuple[float, float, str]:
    hook_duration = min(max(0.1, hook_duration), max(0.1, video_duration))
    anchors = [item for item in (clip.get("retention_anchors") or []) if isinstance(item, dict)]

    if anchors:
        anchor = max(anchors, key=lambda item: float(item.get("importance", 0) or 0))
        try:
            anchor_start = float(anchor.get("start", 0.0))
            anchor_end = float(anchor.get("end", anchor_start))
        except (TypeError, ValueError):
            anchor_start = anchor_end = 0.0

        center = max(anchor_start, (anchor_start + anchor_end) / 2.0)
        start = center - hook_duration / 2.0
        start = max(0.0, min(max(0.0, video_duration - hook_duration), start))
        return start, min(video_duration, start + hook_duration), "retention_anchor"

    return 0.0, min(video_duration, hook_duration), "clip_start"


def mix_voiceover_hook(
    video_path: Path,
    tts_audio_path: Path,
    output_path: Path,
    hook_duration: float,
    teaser_start: float,
    teaser_end: float,
    ducking_volume: float = VOICEOVER_DUCKING_VOLUME,
    fade_duration: float = VOICEOVER_AUDIO_FADE,
    intro_sfx: list[dict] | None = None,
    sfx_volume: float = INTRO_SFX_VOLUME,
):
    """Mixează teaser video + TTS + SFX opțional, apoi continuă cu clipul original."""
    video_path = Path(video_path)
    tts_audio_path = Path(tts_audio_path)
    output_path = Path(output_path)
    intro_sfx = [item for item in (intro_sfx or []) if item.get("path")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_info = probe_video(video_path)
    video_duration = video_info["duration"]
    has_audio = video_info["has_audio"]

    hook_duration = max(0.1, float(hook_duration))
    teaser_start = max(0.0, min(video_duration, float(teaser_start)))
    teaser_end = max(teaser_start + 0.01, min(video_duration, float(teaser_end)))
    teaser_actual = max(0.01, teaser_end - teaser_start)
    teaser_pad = max(0.0, hook_duration - teaser_actual)
    fade_duration = max(0.0, min(float(fade_duration), hook_duration / 3.0))
    tts_fade_out_start = max(0.0, hook_duration - fade_duration)

    filters = [
        (
            f"[0:v]trim=start={teaser_start:.6f}:end={teaser_end:.6f},"
            f"setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={teaser_pad:.6f}[hookv]"
        ),
        "[0:v]setpts=PTS-STARTPTS[mainv]",
        "[hookv][mainv]concat=n=2:v=1:a=0[vout]",
        (
            f"[1:a]atrim=0:{hook_duration:.6f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=channel_layouts=stereo,"
            f"apad=pad_dur={hook_duration:.6f},atrim=0:{hook_duration:.6f},"
            f"afade=t=out:st={tts_fade_out_start:.6f}:d={fade_duration:.6f}[tts_hook]"
        ),
    ]

    command = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(tts_audio_path)]

    sfx_labels = []
    for offset, item in enumerate(intro_sfx, start=2):
        path = Path(item["path"])
        delay_ms = max(0, int(round(float(item.get("delay", 0.0)) * 1000)))
        command.extend(["-i", str(path)])
        label = f"sfx{offset}"
        filters.append(
            f"[{offset}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={float(sfx_volume):.4f},adelay={delay_ms}|{delay_ms},"
            f"apad=pad_dur={hook_duration:.6f},atrim=0:{hook_duration:.6f}[{label}]"
        )
        sfx_labels.append(f"[{label}]")

    if sfx_labels:
        mix_inputs = "[tts_hook]" + "".join(sfx_labels)
        filters.append(
            f"{mix_inputs}amix=inputs={1 + len(sfx_labels)}:duration=first:dropout_transition=0[hooka]"
        )
    else:
        filters.append("[tts_hook]anull[hooka]")

    if has_audio:
        filters.append(
            "[0:a]asetpts=PTS-STARTPTS,aresample=48000,"
            "aformat=channel_layouts=stereo,"
            f"afade=t=in:st=0:d={fade_duration:.6f}[maina]"
        )
    else:
        filters.append(
            f"anullsrc=r=48000:cl=stereo,atrim=duration={video_duration:.6f},"
            "asetpts=PTS-STARTPTS[maina]"
        )

    filters.append("[hooka][maina]concat=n=2:v=0:a=1[aout]")

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "19",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    _run(command)
    return output_path
