from pathlib import Path
import subprocess

from src.config import (
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
)
from src.logger import info, success, warning
from src.smart_crop import write_sendcmd, initial_crop_xy
from src.smart_crop_v2 import (
    analyze_smart_crop_v2,
    initial_gameplay_xy,
    save_smartcrop_debug,
    validate_layout_plan,
    write_gameplay_sendcmd,
)


def escape_filter_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def _run(command: list[str], error_message: str) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{error_message}\n{result.stderr}")


def _encode_args(output: Path) -> list[str]:
    return [
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "19",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output),
    ]


def _render_legacy(video: Path, subtitle: Path, output: Path, plan, clip_name: str) -> None:
    command_file = TEMP_DIR / f"{clip_name}_crop.cmd"
    write_sendcmd(plan, command_file)
    initial_x, initial_y = initial_crop_xy(plan)
    subtitle_path = escape_filter_path(subtitle)
    command_path = escape_filter_path(command_file)

    filter_chain = (
        f"sendcmd=f='{command_path}',"
        f"crop@smart=w={plan.crop_width}:h={plan.crop_height}:x={initial_x}:y={initial_y},"
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
        f"subtitles='{subtitle_path}'"
    )

    command = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", filter_chain,
        *_encode_args(output),
    ]
    _run(command, "FFmpeg a eșuat la SmartCrop legacy.")


def _render_gameplay_webcam(video: Path, subtitle: Path, output: Path, plan, clip_name: str) -> None:
    if not validate_layout_plan(plan) or plan.webcam_region is None:
        raise ValueError("SmartCrop GAMEPLAY_WEBCAM plan invalid")

    command_file = TEMP_DIR / f"{clip_name}_gameplay_crop.cmd"
    write_gameplay_sendcmd(plan, command_file)
    initial_x, initial_y = initial_gameplay_xy(plan)
    subtitle_path = escape_filter_path(subtitle)
    command_path = escape_filter_path(command_file)
    webcam = plan.webcam_region

    filter_complex = (
        "[0:v]split=2[game_src][cam_src];"
        f"[game_src]sendcmd=f='{command_path}',"
        f"crop@gameplay=w={plan.gameplay_crop_width}:h={plan.gameplay_crop_height}:"
        f"x={initial_x}:y={initial_y},"
        f"scale={VIDEO_WIDTH}:{plan.gameplay_output_height}[gameplay];"
        f"[cam_src]crop=w={webcam.w}:h={webcam.h}:x={webcam.x}:y={webcam.y},"
        f"scale={VIDEO_WIDTH}:{plan.webcam_output_height}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{plan.webcam_output_height}:(ow-iw)/2:(oh-ih)/2[webcam];"
        f"[gameplay][webcam]vstack=inputs=2,"
        f"subtitles='{subtitle_path}'[vout]"
    )

    command = [
        "ffmpeg", "-y", "-i", str(video),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        *_encode_args(output),
    ]
    _run(command, "FFmpeg a eșuat la SmartCrop GAMEPLAY_WEBCAM.")


def render(clip_name: str):
    video = OUTPUT_DIR / f"{clip_name}.mp4"
    subtitle = SUBTITLES_DIR / f"{clip_name}.ass"

    if not video.exists():
        raise FileNotFoundError(f"Video lipsă: {video}")
    if not subtitle.exists():
        raise FileNotFoundError(f"Subtitle lipsă: {subtitle}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    output = FINAL_DIR / f"{clip_name}_final.mp4"

    info(f"[SMARTCROP] Analysing visual layout for {clip_name}")
    plan = analyze_smart_crop_v2(video)
    save_smartcrop_debug(clip_name, plan)

    if plan.mode == "GAMEPLAY_WEBCAM":
        try:
            info("[SMARTCROP] Layout: GAMEPLAY_TOP_WEBCAM_BOTTOM")
            if plan.reaction_events:
                info(f"[SMARTCROP] Reaction signals detected: {len(plan.reaction_events)}")
            _render_gameplay_webcam(video, subtitle, output, plan, clip_name)
            success(f"Clip randat: {output.name}")
            return output
        except Exception as exc:
            warning(f"[SMARTCROP] GAMEPLAY_WEBCAM render failed: {exc}")
            warning("[SMARTCROP] Falling back to existing SmartCrop")

    _render_legacy(video, subtitle, output, plan.legacy_plan, clip_name)
    success(f"Clip randat: {output.name}")
    return output
