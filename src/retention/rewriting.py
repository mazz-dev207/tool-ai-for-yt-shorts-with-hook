from __future__ import annotations

from typing import Iterable, List


ROLES = {"hook", "context", "escalation", "payoff", "bridge", "reaction"}


def _all_word_boundaries(transcript: List[dict], start: float, end: float):
    starts = []
    ends = []

    for segment in transcript:
        if float(segment.get("end", 0.0)) < start:
            continue
        if float(segment.get("start", 0.0)) > end:
            continue

        starts.append(float(segment.get("start", start)))
        ends.append(float(segment.get("end", end)))

        for word in segment.get("words", []):
            word_start = float(word.get("start", 0.0))
            word_end = float(word.get("end", word_start))
            if start <= word_start <= end:
                starts.append(word_start)
            if start <= word_end <= end:
                ends.append(word_end)

    return sorted(set(starts)), sorted(set(ends))


def _snap(value: float, boundaries: List[float], max_distance: float = 0.85) -> float:
    if not boundaries:
        return value

    nearest = min(boundaries, key=lambda candidate: abs(candidate - value))
    if abs(nearest - value) <= max_distance:
        return nearest
    return value


def normalize_segments(
    raw_segments: Iterable[dict],
    transcript: List[dict],
    allowed_start: float,
    allowed_end: float,
    max_segments: int,
) -> List[dict]:
    starts, ends = _all_word_boundaries(transcript, allowed_start, allowed_end)
    result = []

    if not isinstance(raw_segments, (list, tuple)):
        raw_segments = []

    for raw in list(raw_segments)[:max_segments]:
        if not isinstance(raw, dict):
            continue

        try:
            start = float(raw.get("start"))
            end = float(raw.get("end"))
        except (TypeError, ValueError):
            continue

        start = max(allowed_start, min(allowed_end, start))
        end = max(allowed_start, min(allowed_end, end))

        start = _snap(start, starts)
        end = _snap(end, ends)

        if end - start < 0.35:
            continue

        role = str(raw.get("role", "bridge")).lower().strip()
        if role not in ROLES:
            role = "bridge"

        item = {
            "start": round(start, 3),
            "end": round(end, 3),
            "role": role,
            "reason": str(raw.get("reason", "")).strip(),
        }

        # Evită duplicate identice, dar păstrează ordinea propusă de editor.
        if result and result[-1]["start"] == item["start"] and result[-1]["end"] == item["end"]:
            continue

        result.append(item)

    # Unește segmente consecutive, în aceeași ordine, când gap-ul este aproape zero.
    merged = []
    for item in result:
        if (
            merged
            and item["start"] >= merged[-1]["end"]
            and item["start"] - merged[-1]["end"] <= 0.08
            and item["role"] == merged[-1]["role"]
        ):
            merged[-1]["end"] = item["end"]
            if item["reason"]:
                merged[-1]["reason"] = "; ".join(
                    part for part in [merged[-1]["reason"], item["reason"]] if part
                )
        else:
            merged.append(item)

    return merged


def segment_duration(segments: Iterable[dict]) -> float:
    return sum(max(0.0, float(item["end"]) - float(item["start"])) for item in segments)


def edit_penalty(segments: List[dict]) -> float:
    if not segments:
        return 100.0

    penalty = max(0, len(segments) - 4) * 1.5

    backward_jumps = 0
    large_forward_jumps = 0

    for previous, current in zip(segments, segments[1:]):
        if current["start"] < previous["start"]:
            backward_jumps += 1
        elif current["start"] - previous["end"] > 20.0:
            large_forward_jumps += 1

    penalty += backward_jumps * 6.0
    penalty += large_forward_jumps * 2.0
    return penalty


def fallback_variant(candidate: dict) -> dict:
    start = float(candidate["start"])
    end = float(candidate["end"])
    return {
        "name": "original_fallback",
        "strategy": "extractive",
        "rationale": "Fallback: păstrează candidatul original.",
        "segments": [
            {
                "start": start,
                "end": end,
                "role": "escalation",
                "reason": "original_candidate",
            }
        ],
        "scores": {},
    }


def normalize_variants(
    raw_variants: Iterable[dict],
    transcript: List[dict],
    allowed_start: float,
    allowed_end: float,
    candidate: dict,
    max_variants: int,
    max_segments: int,
    min_duration: float,
    max_duration: float,
) -> List[dict]:
    result = []

    if not isinstance(raw_variants, (list, tuple)):
        raw_variants = []

    for index, raw in enumerate(list(raw_variants)[:max_variants], start=1):
        if not isinstance(raw, dict):
            continue

        strategy = str(raw.get("strategy", "extractive")).strip().lower()
        if strategy != "extractive":
            # Audio generat/TTS nu există încă în proiect. Nu selectăm variante
            # care nu pot fi produse cu audio-ul original.
            continue

        segments = normalize_segments(
            raw.get("segments", []),
            transcript,
            allowed_start,
            allowed_end,
            max_segments,
        )
        duration = segment_duration(segments)

        if not segments:
            continue
        if duration < min_duration * 0.65:
            continue
        if duration > max_duration * 1.20:
            continue

        result.append(
            {
                "name": str(raw.get("name", f"variant_{index}")),
                "strategy": "extractive",
                "rationale": str(raw.get("rationale", "")).strip(),
                "segments": segments,
                "scores": raw.get("scores", {}),
                "duration": round(duration, 3),
                "edit_penalty": round(edit_penalty(segments), 2),
            }
        )

    # Păstrăm întotdeauna și originalul ca benchmark/fallback.
    original = fallback_variant(candidate)
    original["duration"] = round(segment_duration(original["segments"]), 3)
    original["edit_penalty"] = 0.0
    result.append(original)

    return result
