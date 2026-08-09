from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.config import (
    HIGHLIGHTS_DIR,
    SMARTCUT_BOUNDARY_SEARCH_WINDOW,
    SMARTCUT_DEAD_AIR_THRESHOLD,
    SMARTCUT_DEBUG,
    SMARTCUT_ENABLED,
    SMARTCUT_MINIMUM_CUT_SPACING,
    SMARTCUT_MINIMUM_SEGMENT_DURATION,
    SMARTCUT_POST_CONTEXT,
    SMARTCUT_PRE_CONTEXT,
    SMARTCUT_REACTION_PADDING,
    SMARTCUT_REACTION_SEARCH_WINDOW,
    SMARTCUT_SENTENCE_POST_PADDING,
    SMARTCUT_SENTENCE_PRE_PADDING,
)
from src.logger import info, warning


_SENTENCE_END = re.compile(r"[.!?…][\"')\]]*$")
_REACTION_HINT = re.compile(
    r"(!{1,}|what\??|wow|no way|oh my|omg|laugh|laughter|haha|hahaha|"
    r"scream|screaming|yell|yelling|shock|shocked|crazy|insane)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WordStamp:
    word: str
    start: float
    end: float
    segment_index: int


@dataclass(frozen=True)
class BoundaryCandidate:
    time: float
    source: str
    score: float


def probe_video_duration(video_path: Path) -> float:
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


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flatten_words(transcript: list[dict]) -> list[WordStamp]:
    words: list[WordStamp] = []
    for segment_index, segment in enumerate(transcript or []):
        for raw in segment.get("words", []) or []:
            start = _float(raw.get("start"))
            end = _float(raw.get("end"), start)
            token = str(raw.get("word", "")).strip()
            if not token or end <= start:
                continue
            words.append(WordStamp(token, start, end, segment_index))
    words.sort(key=lambda item: (item.start, item.end))
    return words


def validate_input_segments(
    segments: Iterable[dict],
    video_duration: float,
) -> list[dict]:
    cleaned: list[dict] = []
    for raw in segments or []:
        if not isinstance(raw, dict):
            continue
        start = max(0.0, _float(raw.get("start")))
        end = min(max(0.0, video_duration), _float(raw.get("end")))
        if end <= start or end - start < 0.05:
            continue
        cleaned.append({"start": round(start, 4), "end": round(end, 4)})

    cleaned.sort(key=lambda item: (item["start"], item["end"]))
    deduped: list[dict] = []
    for item in cleaned:
        if (
            deduped
            and abs(item["start"] - deduped[-1]["start"]) < 0.001
            and abs(item["end"] - deduped[-1]["end"]) < 0.001
        ):
            continue
        if deduped and item["start"] < deduped[-1]["end"]:
            deduped[-1]["end"] = round(max(deduped[-1]["end"], item["end"]), 4)
        else:
            deduped.append(dict(item))
    return deduped


def extract_transcript_context(
    transcript: list[dict],
    boundary: float,
    before: float = SMARTCUT_PRE_CONTEXT,
    after: float = SMARTCUT_POST_CONTEXT,
) -> list[dict]:
    start = max(0.0, boundary - before)
    end = boundary + after
    return [
        segment
        for segment in transcript or []
        if _float(segment.get("end")) >= start
        and _float(segment.get("start")) <= end
    ]


def detect_silence_regions(
    words: list[WordStamp],
    start: float,
    end: float,
    min_silence: float = SMARTCUT_DEAD_AIR_THRESHOLD,
) -> list[tuple[float, float]]:
    relevant = [word for word in words if word.end >= start and word.start <= end]
    regions: list[tuple[float, float]] = []
    for previous, current in zip(relevant, relevant[1:]):
        if current.start - previous.end >= min_silence:
            regions.append((previous.end, current.start))
    return regions


def analyze_speech_rhythm(
    words: list[WordStamp],
    boundary: float,
    radius: float = 1.5,
) -> dict:
    nearby = [
        word for word in words
        if word.end >= boundary - radius and word.start <= boundary + radius
    ]
    if not nearby:
        return {"word_rate": 0.0, "mean_gap": 0.0}
    duration = max(0.001, nearby[-1].end - nearby[0].start)
    gaps = [
        max(0.0, current.start - previous.end)
        for previous, current in zip(nearby, nearby[1:])
    ]
    return {
        "word_rate": len(nearby) / duration,
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
    }


def _segment_boundary_points(
    transcript: list[dict],
    search_start: float,
    search_end: float,
) -> list[tuple[float, str]]:
    points: list[tuple[float, str]] = []
    for segment in transcript or []:
        seg_start = _float(segment.get("start"))
        seg_end = _float(segment.get("end"), seg_start)
        text = str(segment.get("text", "")).strip()
        if search_start <= seg_start <= search_end:
            points.append((seg_start, "sentence_start"))
        if search_start <= seg_end <= search_end:
            points.append((seg_end, "sentence_end" if _SENTENCE_END.search(text) else "clause_end"))
    return points


def score_boundary_candidate(
    time_value: float,
    rough_time: float,
    source: str,
    kind: str,
    words: list[WordStamp],
) -> float:
    base = {
        "sentence_start": 1.35 if kind == "start" else 0.85,
        "sentence_end": 1.35 if kind == "end" else 0.85,
        "clause_end": 1.00,
        "silence": 1.15,
        "word_start": 0.95 if kind == "start" else 0.60,
        "word_end": 0.95 if kind == "end" else 0.60,
        "original": 0.75,
    }.get(source, 0.5)
    score = base - min(0.8, abs(time_value - rough_time) * 0.35)
    for word in words:
        if word.start + 0.015 < time_value < word.end - 0.015:
            score -= 2.0
            break
    if analyze_speech_rhythm(words, time_value)["mean_gap"] >= 0.18:
        score += 0.12
    return score


def find_boundary_candidates(
    rough_time: float,
    kind: str,
    transcript: list[dict],
    words: list[WordStamp],
    search_window: float = SMARTCUT_BOUNDARY_SEARCH_WINDOW,
) -> list[BoundaryCandidate]:
    search_start = max(0.0, rough_time - search_window)
    search_end = rough_time + search_window
    raw_points: list[tuple[float, str]] = [(rough_time, "original")]

    for word in words:
        if word.end < search_start:
            continue
        if word.start > search_end:
            break
        if search_start <= word.start <= search_end:
            raw_points.append((word.start, "word_start"))
        if search_start <= word.end <= search_end:
            raw_points.append((word.end, "word_end"))

    raw_points.extend(_segment_boundary_points(transcript, search_start, search_end))
    for silence_start, silence_end in detect_silence_regions(words, search_start, search_end):
        raw_points.append(((silence_start + silence_end) / 2.0, "silence"))

    seen: set[int] = set()
    candidates: list[BoundaryCandidate] = []
    for time_value, source in raw_points:
        key = int(round(time_value * 1000))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            BoundaryCandidate(
                time=time_value,
                source=source,
                score=score_boundary_candidate(time_value, rough_time, source, kind, words),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def snap_to_word_boundary(boundary: float, kind: str, words: list[WordStamp]) -> float:
    for word in words:
        if word.start < boundary < word.end:
            return word.start if kind == "start" else word.end
    return boundary


def snap_to_sentence_boundary(
    boundary: float,
    kind: str,
    transcript: list[dict],
    words: list[WordStamp],
) -> float:
    candidates = find_boundary_candidates(boundary, kind, transcript, words)
    return candidates[0].time if candidates else boundary


def trim_dead_air(
    start: float,
    end: float,
    words: list[WordStamp],
) -> tuple[float, float, float]:
    relevant = [word for word in words if word.end >= start and word.start <= end]
    if not relevant:
        return start, end, 0.0
    removed = 0.0
    first = relevant[0]
    last = relevant[-1]
    if first.start - start >= SMARTCUT_DEAD_AIR_THRESHOLD:
        removed += first.start - start
        start = max(start, first.start - SMARTCUT_SENTENCE_PRE_PADDING)
    if end - last.end >= SMARTCUT_DEAD_AIR_THRESHOLD:
        removed += end - last.end
        end = min(end, last.end + SMARTCUT_SENTENCE_POST_PADDING)
    return start, end, removed


def preserve_payoff(
    end: float,
    transcript: list[dict],
    video_duration: float,
) -> float:
    search_end = min(video_duration, end + SMARTCUT_REACTION_SEARCH_WINDOW)
    following = [
        segment for segment in transcript or []
        if _float(segment.get("end")) > end
        and _float(segment.get("start")) <= search_end
    ]
    if not following:
        return end
    first = following[0]
    text = str(first.get("text", "")).strip()
    gap = max(0.0, _float(first.get("start")) - end)
    if gap <= 0.25 and len(text.split()) <= 16:
        return min(video_duration, _float(first.get("end")) + SMARTCUT_SENTENCE_POST_PADDING)
    return end


def preserve_reaction(
    end: float,
    transcript: list[dict],
    video_duration: float,
) -> float:
    search_end = min(video_duration, end + SMARTCUT_REACTION_SEARCH_WINDOW)
    for segment in transcript or []:
        seg_start = _float(segment.get("start"))
        seg_end = _float(segment.get("end"), seg_start)
        if seg_end <= end or seg_start > search_end:
            continue
        if _REACTION_HINT.search(str(segment.get("text", "")).strip()):
            return min(video_duration, seg_end + SMARTCUT_REACTION_PADDING)
    return end


def validate_audio_continuity(
    left_end: float,
    right_start: float,
    words: list[WordStamp],
) -> bool:
    for boundary in (left_end, right_start):
        for word in words:
            if word.start + 0.015 < boundary < word.end - 0.015:
                return False
    return True


def validate_visual_continuity(*_args, **_kwargs) -> bool:
    return True


def merge_nearby_segments(
    segments: list[dict],
    words: list[WordStamp],
    minimum_spacing: float = SMARTCUT_MINIMUM_CUT_SPACING,
) -> list[dict]:
    if not segments:
        return []
    result = [dict(segments[0])]
    for current in segments[1:]:
        previous = result[-1]
        gap = current["start"] - previous["end"]
        if gap < minimum_spacing and validate_audio_continuity(previous["end"], current["start"], words):
            info(f"[SMARTCUT] Merged two cuts separated by {max(0.0, gap):.2f}s")
            previous["end"] = max(previous["end"], current["end"])
        else:
            result.append(dict(current))
    return result


def enforce_minimum_cut_spacing(
    segments: list[dict],
    minimum_spacing: float = SMARTCUT_MINIMUM_CUT_SPACING,
) -> list[dict]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current["start"] - previous["end"]
        if gap < minimum_spacing:
            previous["end"] = max(previous["end"], current["end"])
        else:
            merged.append(dict(current))
    return merged


def validate_final_edit_plan(segments: list[dict], video_duration: float) -> bool:
    if not segments:
        return False
    previous_end = -1.0
    for segment in segments:
        start = _float(segment.get("start"), -1.0)
        end = _float(segment.get("end"), -1.0)
        if start < 0 or end > video_duration + 0.001 or end <= start:
            return False
        if end - start < SMARTCUT_MINIMUM_SEGMENT_DURATION:
            return False
        if start < previous_end - 0.001:
            return False
        previous_end = end
    return True


def fallback_to_original_plan(
    original_segments: list[dict],
    video_duration: float,
) -> list[dict]:
    return validate_input_segments(original_segments, video_duration)


def _save_debug(video_name: str, clip_index: int, payload: dict) -> None:
    if not SMARTCUT_DEBUG:
        return
    output_dir = HIGHLIGHTS_DIR / "debug" / video_name / "smartcut"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"clip_{clip_index}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def refine_edit_plan(
    *,
    video_name: str,
    clip_index: int,
    original_segments: list[dict],
    transcript: list[dict],
    video_duration: float,
) -> list[dict]:
    original = fallback_to_original_plan(original_segments, video_duration)
    if not SMARTCUT_ENABLED or not original:
        return original

    words = flatten_words(transcript)
    if not words:
        warning("[SMARTCUT] Transcript/word timestamps lipsesc; fallback la edit plan-ul original.")
        return original

    info(f"[SMARTCUT] Input segments: {len(original)}")
    refined: list[dict] = []
    changes: list[dict] = []

    try:
        for index, segment in enumerate(original, start=1):
            rough_start = segment["start"]
            rough_end = segment["end"]
            info(f"[SMARTCUT] Segment {index}: {rough_start:.2f} -> {rough_end:.2f}")

            start = snap_to_word_boundary(rough_start, "start", words)
            end = snap_to_word_boundary(rough_end, "end", words)

            start_candidates = find_boundary_candidates(start, "start", transcript, words)
            end_candidates = find_boundary_candidates(end, "end", transcript, words)
            if start_candidates:
                start = start_candidates[0].time
            if end_candidates:
                end = end_candidates[0].time

            start = max(0.0, start - SMARTCUT_SENTENCE_PRE_PADDING)
            end = min(video_duration, end + SMARTCUT_SENTENCE_POST_PADDING)
            start, end, removed_dead_air = trim_dead_air(start, end, words)

            payoff_end = preserve_payoff(end, transcript, video_duration)
            reaction_end = preserve_reaction(payoff_end, transcript, video_duration)
            end = max(end, payoff_end, reaction_end)

            if end - start < SMARTCUT_MINIMUM_SEGMENT_DURATION:
                start, end = rough_start, rough_end

            start = max(0.0, min(start, video_duration))
            end = max(start, min(end, video_duration))

            info(f"[SMARTCUT] Snapped start {rough_start:.2f} -> {start:.2f}")
            info(f"[SMARTCUT] Snapped end {rough_end:.2f} -> {end:.2f}")
            if removed_dead_air > 0:
                info(f"[SMARTCUT] Removed {removed_dead_air:.2f}s dead air")
            if reaction_end > payoff_end + 0.01:
                info(f"[SMARTCUT] Preserved reaction until {reaction_end:.2f}")

            refined.append({"start": round(start, 4), "end": round(end, 4)})
            changes.append(
                {
                    "original": {"start": rough_start, "end": rough_end},
                    "final": {"start": round(start, 4), "end": round(end, 4)},
                    "dead_air_removed": round(removed_dead_air, 4),
                }
            )

        refined = validate_input_segments(refined, video_duration)
        refined = merge_nearby_segments(refined, words)
        refined = enforce_minimum_cut_spacing(refined)

        if not validate_final_edit_plan(refined, video_duration):
            raise ValueError("SmartCut final edit plan failed validation")

        info(f"[SMARTCUT] Final segments: {len(refined)}")
        _save_debug(video_name, clip_index, {"original": original, "final": refined, "changes": changes})
        return refined

    except Exception as exc:
        warning(f"[SMARTCUT] Analysis failed: {exc}")
        warning("[SMARTCUT] Falling back to original edit plan")
        return original
