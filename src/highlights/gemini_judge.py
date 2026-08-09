from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from src.config import (
    GEMINI_API_KEY,
    GEMINI_CACHE_DIR,
    GEMINI_CONTEXT_AFTER,
    GEMINI_CONTEXT_BEFORE,
    GEMINI_MAX_RETRIES,
    GEMINI_MODEL,
    GEMINI_PROMPT_VERSION,
    TEMP_DIR,
)
from src.highlights.profiles import ContentProfile
from src.highlights.ranking import validate_judgement


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "integer"},
        "original_start": {"type": "number"},
        "original_end": {"type": "number"},
        "refined_start": {"type": "number"},
        "refined_end": {"type": "number"},
        "scores": {
            "type": "object",
            "properties": {
                "hook": {"type": "integer"},
                "payoff": {"type": "integer"},
                "emotion": {"type": "integer"},
                "visual_action": {"type": "integer"},
                "surprise": {"type": "integer"},
                "standalone": {"type": "integer"},
                "replayability": {"type": "integer"},
            },
            "required": [
                "hook", "payoff", "emotion", "visual_action",
                "surprise", "standalone", "replayability",
            ],
        },
        "total_score": {"type": "integer"},
        "category": {"type": "string"},
        "reason": {"type": "string"},
        "has_complete_payoff": {"type": "boolean"},
        "requires_previous_context": {"type": "boolean"},
        "recommended": {"type": "boolean"},
    },
    "required": [
        "candidate_id", "original_start", "original_end", "refined_start", "refined_end",
        "scores", "total_score", "category", "reason", "has_complete_payoff",
        "requires_previous_context", "recommended",
    ],
}


class GeminiUnavailable(RuntimeError):
    pass


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "FFmpeg failed")


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return max(0.0, float(result.stdout.strip()))


def build_transcript_context(transcript: list[dict], start: float, end: float) -> str:
    lines: list[str] = []
    for segment in transcript:
        try:
            seg_start = float(segment.get("start", 0.0))
            seg_end = float(segment.get("end", seg_start))
        except (TypeError, ValueError):
            continue
        if seg_end < start or seg_start > end:
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            lines.append(f"[{seg_start:.2f}s-{seg_end:.2f}s] {text}")
    return "\n".join(lines)


def extract_context_video(
    video_path: Path,
    candidate: dict,
    video_duration: float,
    candidate_id: int,
) -> tuple[Path, float, float]:
    start = max(0.0, float(candidate["start"]) - GEMINI_CONTEXT_BEFORE)
    end = min(video_duration, float(candidate["end"]) + GEMINI_CONTEXT_AFTER)
    if end <= start:
        raise ValueError("Invalid Gemini context range")

    output_dir = TEMP_DIR / "gemini_context"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"candidate_{candidate_id:03d}.mp4"

    _run([
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(output),
    ])
    return output, start, end


def _cache_key(video_path: Path, candidate: dict, profile: ContentProfile) -> str:
    stat = video_path.stat()
    payload = {
        "video": str(video_path.resolve()).lower(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "start": round(float(candidate["start"]), 3),
        "end": round(float(candidate["end"]), 3),
        "model": GEMINI_MODEL,
        "profile": profile.name,
        "prompt_version": GEMINI_PROMPT_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cache(key: str) -> dict | None:
    path = GEMINI_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(key: str, value: dict) -> None:
    GEMINI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = GEMINI_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_prompt(
    candidate_id: int,
    candidate: dict,
    transcript_text: str,
    profile: ContentProfile,
    context_start: float,
    context_end: float,
) -> str:
    return f"""
You are an expert short-form video editor and viral content analyst.
Judge ONE pre-generated highlight candidate using BOTH the supplied video/audio and transcript.
The legacy selector is a high-recall candidate generator. Your job is to judge, rerank and refine boundaries.

Do not reward a candidate simply because it contains loud audio, fast movement, laughter, or strong language.
A good short requires a meaningful moment with a clear reason for the viewer to continue watching.
Prefer complete micro-stories over isolated interesting sentences.
A high-scoring clip should usually contain a hook, a development, and a payoff or meaningful reaction.

CONTENT PROFILE: {profile.name}
PROFILE PRIORITIES: {profile.description}
PROFILE WEIGHTS (total 100): {json.dumps(profile.weights, separators=(',', ':'))}

STANDARD COMPONENT LIMITS:
- hook: 0-25
- payoff: 0-20
- emotion: 0-15
- visual_action: 0-15
- surprise: 0-10
- standalone: 0-10
- replayability: 0-5

BOUNDARY RULES:
- Candidate original range: {float(candidate['start']):.3f}s -> {float(candidate['end']):.3f}s.
- Available multimodal context: {context_start:.3f}s -> {context_end:.3f}s.
- refined_start/refined_end MUST stay inside available context.
- Include the minimum context needed for comprehension.
- Do not start in the middle of an essential sentence or action.
- Do not end before the payoff, punchline, result or meaningful reaction.
- Remove dead air and unnecessary setup.
- Duration is content-driven, not fixed.

CANDIDATE ID: {candidate_id}
LEGACY METADATA:
{json.dumps(candidate, ensure_ascii=False, separators=(',', ':'))}

TIMESTAMPED TRANSCRIPT:
{transcript_text}

Return only the requested structured judgement. Use the full scoring range; scores near 100 should be rare.
""".strip()


class GeminiHighlightJudge:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise GeminiUnavailable("GEMINI_API_KEY lipsește")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiUnavailable(
                "Pachetul google-genai lipsește. Instalează: pip install google-genai"
            ) from exc

        self._types = types
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _wait_until_active(self, uploaded: Any, timeout_seconds: float = 180.0) -> Any:
        started = time.time()
        current = uploaded
        while time.time() - started < timeout_seconds:
            state = getattr(getattr(current, "state", None), "name", None)
            if state in {None, "ACTIVE"}:
                return current
            if state == "FAILED":
                raise RuntimeError("Gemini File API processing failed")
            time.sleep(2.0)
            current = self.client.files.get(name=current.name)
        raise TimeoutError("Gemini File API processing timeout")

    def judge(
        self,
        video_path: Path,
        transcript: list[dict],
        candidate: dict,
        candidate_id: int,
        profile: ContentProfile,
        video_duration: float,
    ) -> tuple[dict, bool]:
        cache_key = _cache_key(video_path, candidate, profile)
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached, True

        context_video, context_start, context_end = extract_context_video(
            video_path, candidate, video_duration, candidate_id
        )
        transcript_text = build_transcript_context(transcript, context_start, context_end)
        prompt = build_prompt(
            candidate_id, candidate, transcript_text, profile, context_start, context_end
        )

        last_error: Exception | None = None
        for attempt in range(1, max(1, GEMINI_MAX_RETRIES) + 1):
            uploaded = None
            try:
                uploaded = self.client.files.upload(file=str(context_video))
                uploaded = self._wait_until_active(uploaded)
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[uploaded, prompt],
                    config=self._types.GenerateContentConfig(
                        temperature=0.15,
                        response_mime_type="application/json",
                        response_json_schema=RESPONSE_SCHEMA,
                    ),
                )
                raw = json.loads(str(response.text or "").strip())
                validated = validate_judgement(
                    raw,
                    candidate=candidate,
                    video_duration=video_duration,
                    profile=profile,
                    context_start=context_start,
                    context_end=context_end,
                )
                _save_cache(cache_key, validated)
                return validated, False
            except Exception as exc:
                last_error = exc
                if attempt < max(1, GEMINI_MAX_RETRIES):
                    time.sleep(min(2 ** attempt, 6))
            finally:
                if uploaded is not None:
                    try:
                        self.client.files.delete(name=uploaded.name)
                    except Exception:
                        pass

        raise RuntimeError(f"Gemini judge failed after retries: {last_error}")
