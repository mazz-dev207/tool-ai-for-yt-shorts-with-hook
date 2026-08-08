from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Tuple

from src.config import (
    VOICEOVER_DUCKING_VOLUME,
    VOICEOVER_AUDIO_FADE,
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
    duration_result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )

    try:
        duration = max(0.0, float(duration_result.stdout.strip()))
    except ValueError as exc:
        raise RuntimeError(
            f"Nu pot determina durata video pentru {path}"
        ) from exc

    audio_result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )

    return {
        "duration": duration,
        "has_audio": bool(audio_result.stdout.strip()),
    }


def choose_teaser_range(
    clip: dict,
    video_duration: float,
    hook_duration: float,
) -> Tuple[float, float, str]:
    hook_duration = min(
        max(0.1, hook_duration),
        max(0.1, video_duration),
    )

    anchors = clip.get("retention_anchors") or []

    # LLM-urile pot întoarce uneori string-uri în loc de obiecte.
    anchors = [
        item
        for item in anchors
        if isinstance(item, dict)
    ]

    if anchors:
        anchor = max(
            anchors,
            key=lambda item: float(
                item.get("importance", 0) or 0
            ),
        )

        try:
            anchor_start = float(
                anchor.get("start", 0.0)
            )
            anchor_end = float(
                anchor.get("end", anchor_start)
            )
        except (TypeError, ValueError):
            anchor_start = 0.0
            anchor_end = 0.0

        center = max(
            anchor_start,
            (anchor_start + anchor_end) / 2.0,
        )

        start = (
            center
            - hook_duration / 2.0
        )

        start = max(
            0.0,
            min(
                max(
                    0.0,
                    video_duration - hook_duration,
                ),
                start,
            ),
        )

        return (
            start,
            min(
                video_duration,
                start + hook_duration,
            ),
            "retention_anchor",
        )

    return (
        0.0,
        min(
            video_duration,
            hook_duration,
        ),
        "clip_start",
    )


def mix_voiceover_hook(
    video_path: Path,
    tts_audio_path: Path,
    output_path: Path,
    hook_duration: float,
    teaser_start: float,
    teaser_end: float,
    ducking_volume: float = VOICEOVER_DUCKING_VOLUME,
    fade_duration: float = VOICEOVER_AUDIO_FADE,
):
    """
    Audio behavior:

    1. Cât timp rulează hook-ul TTS:
       - se aude NUMAI voice-over-ul;
       - audio-ul original este complet mut.

    2. La sfârșitul hook-ului:
       - voice-over-ul face fade-out;
       - după terminarea lui începe audio-ul original;
       - audio-ul original face fade-in.

    Nu există original audio audibil sub vocea TTS.
    `ducking_volume` este păstrat doar pentru compatibilitate cu API-ul vechi.
    """

    video_path = Path(video_path)
    tts_audio_path = Path(tts_audio_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_info = probe_video(
        video_path
    )

    video_duration = video_info[
        "duration"
    ]
    has_audio = video_info[
        "has_audio"
    ]

    hook_duration = max(
        0.1,
        float(hook_duration),
    )

    teaser_start = max(
        0.0,
        min(
            video_duration,
            float(teaser_start),
        ),
    )

    teaser_end = max(
        teaser_start + 0.01,
        min(
            video_duration,
            float(teaser_end),
        ),
    )

    teaser_actual = max(
        0.01,
        teaser_end - teaser_start,
    )

    teaser_pad = max(
        0.0,
        hook_duration - teaser_actual,
    )

    # Protecție pentru hook-uri foarte scurte.
    fade_duration = max(
        0.0,
        min(
            float(fade_duration),
            hook_duration / 3.0,
        ),
    )

    tts_fade_out_start = max(
        0.0,
        hook_duration - fade_duration,
    )

    filters = [
        # Vizualul teaser-ului rulează cât timp vorbește TTS.
        (
            f"[0:v]"
            f"trim=start={teaser_start:.6f}:end={teaser_end:.6f},"
            f"setpts=PTS-STARTPTS,"
            f"tpad=stop_mode=clone:stop_duration={teaser_pad:.6f}"
            f"[hookv]"
        ),

        # După teaser începe clipul original de la început.
        "[0:v]setpts=PTS-STARTPTS[mainv]",

        "[hookv][mainv]concat=n=2:v=1:a=0[vout]",

        # TTS: fără audio original dedesubt.
        (
            f"[1:a]"
            f"atrim=0:{hook_duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"aresample=48000,"
            f"aformat=channel_layouts=stereo,"
            f"apad=pad_dur={hook_duration:.6f},"
            f"atrim=0:{hook_duration:.6f},"
            f"afade=t=out:"
            f"st={tts_fade_out_start:.6f}:"
            f"d={fade_duration:.6f}"
            f"[hooka]"
        ),
    ]

    if has_audio:
        # IMPORTANT:
        # audio-ul original NU este folosit deloc în partea de hook.
        # El începe doar după ce hook-ul s-a terminat.
        filters.append(
            (
                "[0:a]"
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=channel_layouts=stereo,"
                f"afade=t=in:st=0:d={fade_duration:.6f}"
                "[maina]"
            )
        )
    else:
        filters.append(
            (
                f"anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={video_duration:.6f},"
                "asetpts=PTS-STARTPTS"
                "[maina]"
            )
        )

    # Concatenare, NU amix:
    # hook audio -> original audio.
    # Astfel originalul nu poate fi auzit sub TTS.
    filters.append(
        "[hooka][maina]"
        "concat=n=2:v=0:a=1"
        "[aout]"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(tts_audio_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-cq",
        "19",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    _run(
        command
    )

    return output_path
