from pathlib import Path
import subprocess
import json
import sys

from src.config import (
    INPUT_DIR,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    TRANSCRIPT_DIR,
    SMARTCUT_MINIMUM_SEGMENT_DURATION,
    SMARTCUT_REACTION_SEARCH_WINDOW,
)
from src.logger import info, success, warning
from src.smart_cut import probe_video_duration, refine_edit_plan


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


def _encode_args(output: Path):
    return [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "19",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]


def _cut_continuous(video: Path, start: float, end: float, output: Path):
    duration = end - start
    if duration <= 0:
        raise ValueError("Durată invalidă pentru clip.")

    command = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video),
        "-t", str(duration),
        *_encode_args(output),
    ]
    _run(command)


def _cut_extractively(video: Path, segments: list, output: Path):
    clean_segments = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if end - start >= 0.05:
            clean_segments.append((start, end))

    if not clean_segments:
        raise ValueError("Edit plan fără segmente valide.")

    if len(clean_segments) == 1:
        _cut_continuous(video, clean_segments[0][0], clean_segments[0][1], output)
        return

    command = ["ffmpeg", "-y"]

    for start, end in clean_segments:
        command.extend([
            "-ss", f"{start:.4f}",
            "-t", f"{end - start:.4f}",
            "-i", str(video),
        ])

    filters = []
    concat_inputs = []

    for index in range(len(clean_segments)):
        filters.append(f"[{index}:v:0]setpts=PTS-STARTPTS[v{index}]")
        filters.append(f"[{index}:a:0]asetpts=PTS-STARTPTS[a{index}]")
        concat_inputs.append(f"[v{index}][a{index}]")

    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(clean_segments)}:v=1:a=1[outv][outa]"
    )

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outv]",
        "-map", "[outa]",
        *_encode_args(output),
    ])

    _run(command)


def _load_transcript(video_name: str) -> list[dict]:
    path = TRANSCRIPT_DIR / f"{video_name}.json"
    if not path.exists():
        warning("[SMARTCUT] Transcript lipsă; se va folosi edit plan-ul original.")
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, list) else []
    except Exception as exc:
        warning(f"[SMARTCUT] Nu pot citi transcriptul: {exc}")
        return []


def repair_micro_segments_for_smartcut(segments: list[dict]) -> list[dict]:
    """Repair tiny retention cuts before SmartCut so one micro-cut does not invalidate the whole plan."""
    cleaned = []
    for segment in segments or []:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            cleaned.append({**segment, "start": start, "end": end})

    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    if len(cleaned) < 2:
        return cleaned

    result: list[dict] = []
    index = 0
    bridge_limit = max(0.75, float(SMARTCUT_REACTION_SEARCH_WINDOW))

    while index < len(cleaned):
        current = dict(cleaned[index])
        duration = current["end"] - current["start"]
        if duration >= SMARTCUT_MINIMUM_SEGMENT_DURATION:
            result.append(current)
            index += 1
            continue

        previous_gap = float("inf")
        next_gap = float("inf")
        if result:
            previous_gap = max(0.0, current["start"] - result[-1]["end"])
        if index + 1 < len(cleaned):
            next_gap = max(0.0, cleaned[index + 1]["start"] - current["end"])

        if result and previous_gap <= next_gap and previous_gap <= bridge_limit:
            info(
                f"[SMARTCUT] Micro-segment {duration:.2f}s merged backward "
                f"across {previous_gap:.2f}s gap"
            )
            result[-1]["end"] = max(result[-1]["end"], current["end"])
            index += 1
            continue

        if index + 1 < len(cleaned) and next_gap <= bridge_limit:
            nxt = dict(cleaned[index + 1])
            info(
                f"[SMARTCUT] Micro-segment {duration:.2f}s merged forward "
                f"across {next_gap:.2f}s gap"
            )
            current["end"] = max(current["end"], nxt["end"])
            result.append(current)
            index += 2
            continue

        result.append(current)
        index += 1

    return result


def _refine_clips(video_name: str, video: Path, clips: list[dict]) -> list[dict]:
    transcript = _load_transcript(video_name)
    try:
        video_duration = probe_video_duration(video)
    except Exception as exc:
        warning(f"[SMARTCUT] ffprobe failed: {exc}; păstrez edit plan-ul existent.")
        return clips

    for index, clip in enumerate(clips, start=1):
        had_extract_segments = bool(clip.get("segments"))
        original_segments = clip.get("segments") or [
            {
                "start": float(clip["start"]),
                "end": float(clip["end"]),
            }
        ]

        if had_extract_segments:
            original_segments = repair_micro_segments_for_smartcut(original_segments)

        refined = refine_edit_plan(
            video_name=video_name,
            clip_index=index,
            original_segments=original_segments,
            transcript=transcript,
            video_duration=video_duration,
        )
        if not refined:
            continue

        if had_extract_segments:
            clip["segments"] = refined
            clip["start"] = refined[0]["start"]
            clip["end"] = refined[-1]["end"]
            clip["duration"] = round(
                sum(segment["end"] - segment["start"] for segment in refined),
                3,
            )
        elif len(refined) == 1:
            clip["start"] = refined[0]["start"]
            clip["end"] = refined[0]["end"]
            clip["duration"] = round(refined[0]["end"] - refined[0]["start"], 3)

    return clips


def cut(video_name: str):
    video = INPUT_DIR / f"{video_name}.mp4"
    highlights = HIGHLIGHTS_DIR / f"{video_name}.json"

    if not video.exists():
        raise FileNotFoundError(video)
    if not highlights.exists():
        raise FileNotFoundError(highlights)

    with open(highlights, encoding="utf-8") as file:
        clips = json.load(file)

    if not isinstance(clips, list):
        raise ValueError("Highlights JSON invalid.")

    clips = _refine_clips(video_name, video, clips)
    with open(highlights, "w", encoding="utf-8") as file:
        json.dump(clips, file, indent=2, ensure_ascii=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = 0

    for index, clip in enumerate(clips, start=1):
        output = OUTPUT_DIR / f"clip_{index}.mp4"
        segments = clip.get("segments") or []

        info(
            f"Export clip {index}/{len(clips)} | "
            f"{len(segments) if segments else 1} segment(e)"
        )

        if segments:
            _cut_extractively(video, segments, output)
        else:
            _cut_continuous(
                video,
                float(clip["start"]),
                float(clip["end"]),
                output,
            )

        exported += 1

    success(f"Exportate {exported} clipuri.")
    return OUTPUT_DIR


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/cut.py "video_name"')
        sys.exit(1)

    try:
        cut(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"Eroare: {exc}")
        sys.exit(1)
