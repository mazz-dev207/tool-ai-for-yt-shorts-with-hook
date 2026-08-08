from __future__ import annotations

from typing import Iterable, List, Optional


def build_timeline(segments: Iterable[dict]) -> List[dict]:
    timeline = []
    target_cursor = 0.0

    for index, segment in enumerate(segments):
        source_start = float(segment["start"])
        source_end = float(segment["end"])
        duration = max(0.0, source_end - source_start)
        if duration <= 0:
            continue

        timeline.append(
            {
                "index": index,
                "source_start": round(source_start, 3),
                "source_end": round(source_end, 3),
                "target_start": round(target_cursor, 3),
                "target_end": round(target_cursor + duration, 3),
                "role": segment.get("role", "bridge"),
            }
        )
        target_cursor += duration

    return timeline


def remap_source_time(source_time: float, timeline: List[dict]) -> Optional[float]:
    source_time = float(source_time)
    for item in timeline:
        if item["source_start"] <= source_time <= item["source_end"]:
            return item["target_start"] + (source_time - item["source_start"])
    return None


def remap_items(items: Iterable[dict], timeline: List[dict]) -> List[dict]:
    result = []

    if not isinstance(items, (list, tuple)):
        return result

    for item in items:
        if not isinstance(item, dict):
            continue

        try:
            source_start = float(item.get("start"))
            source_end = float(item.get("end", source_start))
        except (TypeError, ValueError):
            continue

        for segment in timeline:
            overlap_start = max(source_start, segment["source_start"])
            overlap_end = min(source_end, segment["source_end"])
            if overlap_end <= overlap_start:
                continue

            target_start = segment["target_start"] + (overlap_start - segment["source_start"])
            target_end = segment["target_start"] + (overlap_end - segment["source_start"])

            mapped = dict(item)
            mapped["source_start"] = round(overlap_start, 3)
            mapped["source_end"] = round(overlap_end, 3)
            mapped["start"] = round(target_start, 3)
            mapped["end"] = round(target_end, 3)
            result.append(mapped)

    return result


def extract_remapped_words(transcript: List[dict], segments: Iterable[dict]) -> List[dict]:
    words = []
    target_offset = 0.0

    for segment in segments:
        source_start = float(segment["start"])
        source_end = float(segment["end"])
        duration = max(0.0, source_end - source_start)
        if duration <= 0:
            continue

        seen = set()

        for transcript_segment in transcript:
            if float(transcript_segment.get("end", 0.0)) < source_start:
                continue
            if float(transcript_segment.get("start", 0.0)) > source_end:
                continue

            for word in transcript_segment.get("words", []):
                word_start = float(word.get("start", 0.0))
                word_end = float(word.get("end", word_start))
                midpoint = (word_start + word_end) / 2.0

                if not (source_start <= midpoint <= source_end):
                    continue

                key = (round(word_start, 4), round(word_end, 4), str(word.get("word", "")))
                if key in seen:
                    continue
                seen.add(key)

                local_start = max(0.0, word_start - source_start)
                local_end = min(duration, max(local_start, word_end - source_start))

                words.append(
                    {
                        "word": str(word.get("word", "")).strip(),
                        "start": round(target_offset + local_start, 4),
                        "end": round(target_offset + local_end, 4),
                    }
                )

        target_offset += duration

    return words


def build_retention_buckets(
    duration: float,
    anchors: Iterable[dict],
    risks: Iterable[dict],
    bucket_size: float = 5.0,
) -> List[dict]:
    duration = max(0.0, float(duration))
    if duration <= 0:
        return []

    buckets = []
    cursor = 0.0

    while cursor < duration - 1e-6:
        end = min(duration, cursor + bucket_size)
        span = max(0.001, end - cursor)
        score = 60.0

        for anchor in anchors or []:
            if not isinstance(anchor, dict):
                continue

            overlap = max(
                0.0,
                min(end, float(anchor.get("end", cursor)))
                - max(cursor, float(anchor.get("start", end))),
            )
            if overlap > 0:
                importance = max(0.0, min(100.0, float(anchor.get("importance", 50))))
                score += (overlap / span) * importance * 0.45

        for risk in risks or []:
            if not isinstance(risk, dict):
                continue

            overlap = max(
                0.0,
                min(end, float(risk.get("end", cursor)))
                - max(cursor, float(risk.get("start", end))),
            )
            if overlap > 0:
                severity = max(0.0, min(100.0, float(risk.get("severity", 50))))
                score -= (overlap / span) * severity * 0.55

        score = max(0, min(100, round(score)))
        if score >= 80:
            label = "very_good"
        elif score >= 65:
            label = "good"
        elif score >= 45:
            label = "medium"
        else:
            label = "retention_hole"

        buckets.append({
            "start": round(cursor, 3),
            "end": round(end, 3),
            "score": score,
            "label": label,
        })
        cursor = end

    return buckets
