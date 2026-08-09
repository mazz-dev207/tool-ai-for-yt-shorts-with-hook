from pathlib import Path
import subprocess

from src.config import (
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    SMARTCROP_WEBCAM_PERSISTENCE,
)
from src.logger import info, success, warning
from src.smart_crop import write_sendcmd, initial_crop_xy
from src.smart_crop_profiles import upgrade_plan_for_profile
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


def _webcam_corner(plan) -> str:
    webcam = plan.webcam_region
    if webcam is None:
        return "none"

    center_x = webcam.x + webcam.w / 2
    center_y = webcam.y + webcam.h / 2
    horizontal = "left" if center_x < plan.input_width / 2 else "right"
    vertical = "top" if center_y < plan.input_height / 2 else "bottom"
    return f"{vertical}-{horizontal}"


def _log_smartcrop_decision(plan) -> None:
    info(
        f"[SMARTCROP] Detected mode: {plan.mode} | "
        f"confidence={float(plan.confidence):.2f} | "
        f"source={plan.input_width}x{plan.input_height}"
    )

    reason = getattr(plan, "decision_reason", "")
    if reason:
        info(f"[SMARTCROP] Reason: {reason}")

    if plan.mode == "GAMEPLAY_WEBCAM" and plan.webcam_region is not None:
        webcam = plan.webcam_region
        info(
            f"[SMARTCROP] Webcam: {_webcam_corner(plan)} | "
            f"bbox=x{webcam.x},y{webcam.y},w{webcam.w},h{webcam.h}"
        )
        info(
            f"[SMARTCROP] Layout: GAMEPLAY_TOP_WEBCAM_BOTTOM | "
            f"gameplay={VIDEO_WIDTH}x{plan.gameplay_output_height} | "
            f"webcam={VIDEO_WIDTH}x{plan.webcam_output_height}"
        )
        motion_samples = sum(
            1 for point in plan.focus_points
            if getattr(point, "source", "") == "motion"
        )
        info(
            f"[SMARTCROP] Tracking: focus_samples={len(plan.focus_points)} | "
            f"motion_samples={motion_samples} | "
            f"reaction_signals={len(plan.reaction_events)}"
        )
        return

    if plan.mode == "GAMEPLAY_ONLY":
        motion_samples = sum(
            1 for point in plan.focus_points
            if getattr(point, "source", "") == "motion"
        )
        info(
            f"[SMARTCROP] Gameplay-only dynamic 9:16 crop | "
            f"crop={plan.gameplay_crop_width}x{plan.gameplay_crop_height} | "
            f"focus_samples={len(plan.focus_points)} | motion_samples={motion_samples}"
        )
        return

    if plan.mode == "PODCAST_MULTI_SPEAKER":
        regions = getattr(plan, "speaker_regions", [])
        info(
            f"[SMARTCROP] Podcast stable split layout | speakers={len(regions)} | "
            "layout=SPEAKER_TOP_SPEAKER_BOTTOM"
        )
        return

    if plan.mode == "TALKING_HEAD":
        info("[SMARTCROP] Talking-head face/person tracking enabled")
        return

    info(
        "[SMARTCROP] Using existing SmartCrop fallback | "
        f"persistent_webcam_confidence={float(plan.confidence):.2f} | "
        f"required={SMARTCROP_WEBCAM_PERSISTENCE:.2f}"
    )


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


def _render_gameplay_only(video: Path, subtitle: Path, output: Path, plan, clip_name: str) -> None:
    if plan.gameplay_crop_width <= 0 or plan.gameplay_crop_height <= 0 or not plan.focus_points:
        raise ValueError("SmartCrop GAMEPLAY_ONLY plan invalid")

    command_file = TEMP_DIR / f"{clip_name}_gameplay_only_crop.cmd"
    write_gameplay_sendcmd(plan, command_file)
    initial_x, initial_y = initial_gameplay_xy(plan)
    subtitle_path = escape_filter_path(subtitle)
    command_path = escape_filter_path(command_file)

    filter_chain = (
        f"sendcmd=f='{command_path}',"
        f"crop@gameplay=w={plan.gameplay_crop_width}:h={plan.gameplay_crop_height}:"
        f"x={initial_x}:y={initial_y},"
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT},"
        f"subtitles='{subtitle_path}'"
    )

    command = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", filter_chain,
        *_encode_args(output),
    ]
    _run(command, "FFmpeg a eșuat la SmartCrop GAMEPLAY_ONLY.")


def _render_podcast_multi_speaker(video: Path, subtitle: Path, output: Path, plan) -> None:
    regions = getattr(plan, "speaker_regions", [])
    if len(regions) < 2:
        raise ValueError("SmartCrop PODCAST_MULTI_SPEAKER plan invalid")

    first, second = regions[:2]
    half_height = VIDEO_HEIGHT // 2
    subtitle_path = escape_filter_path(subtitle)

    filter_complex = (
        "[0:v]split=2[s1][s2];"
        f"[s1]crop=w={first.w}:h={first.h}:x={first.x}:y={first.y},"
        f"scale={VIDEO_WIDTH}:{half_height}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{half_height}:(ow-iw)/2:(oh-ih)/2[first];"
        f"[s2]crop=w={second.w}:h={second.h}:x={second.x}:y={second.y},"
        f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT - half_height}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT - half_height}:(ow-iw)/2:(oh-ih)/2[second];"
        f"[first][second]vstack=inputs=2,subtitles='{subtitle_path}'[vout]"
    )

    command = [
        "ffmpeg", "-y", "-i", str(video),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        *_encode_args(output),
    ]
    _run(command, "FFmpeg a eșuat la SmartCrop PODCAST_MULTI_SPEAKER.")


def render(clip_name: str, content_profile: str = "auto"):
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
    try:
        plan = upgrade_plan_for_profile(video, plan, content_profile)
    except Exception as exc:
        warning(f"[SMARTCROP] Profile-aware analysis failed: {exc}; păstrez planul existent.")

    _log_smartcrop_decision(plan)
    save_smartcrop_debug(clip_name, plan)

    if plan.mode == "GAMEPLAY_WEBCAM":
        try:
            _render_gameplay_webcam(video, subtitle, output, plan, clip_name)
            success(f"Clip randat: {output.name}")
            return output
        except Exception as exc:
            warning(f"[SMARTCROP] GAMEPLAY_WEBCAM render failed: {exc}")
            warning("[SMARTCROP] Falling back to existing SmartCrop")

    if plan.mode == "GAMEPLAY_ONLY":
        try:
            _render_gameplay_only(video, subtitle, output, plan, clip_name)
            success(f"Clip randat: {output.name}")
            return output
        except Exception as exc:
            warning(f"[SMARTCROP] GAMEPLAY_ONLY render failed: {exc}")
            warning("[SMARTCROP] Falling back to existing SmartCrop")

    if plan.mode == "PODCAST_MULTI_SPEAKER":
        try:
            _render_podcast_multi_speaker(video, subtitle, output, plan)
            success(f"Clip randat: {output.name}")
            return output
        except Exception as exc:
            warning(f"[SMARTCROP] PODCAST_MULTI_SPEAKER render failed: {exc}")
            warning("[SMARTCROP] Falling back to existing SmartCrop")

    _render_legacy(video, subtitle, output, plan.legacy_plan, clip_name)
    success(f"Clip randat: {output.name}")
    return output
