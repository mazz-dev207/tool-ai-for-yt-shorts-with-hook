from __future__ import annotations

import re
from typing import Iterable, List


FILLER_WORDS = {
    # Romanian
    "ă", "aaa", "ăă", "ăăă", "mmm", "hmm", "gen", "practic", "deci",
    "cumva", "adică", "efectiv", "literalmente", "știi", "stii", "păi", "pai",
    # English
    "uh", "um", "erm", "hmm", "like", "basically", "literally", "actually",
    "you know", "i mean", "kind of", "sort of",
}


def _norm(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[^\wăâîșț]+|[^\wăâîșț]+$", "", value, flags=re.UNICODE)
    return value


def flatten_words(transcript: List[dict], start: float, end: float) -> List[dict]:
    words = []
    for segment in transcript:
        if float(segment.get("end", 0.0)) < start:
            continue
        if float(segment.get("start", 0.0)) > end:
            continue

        for word in segment.get("words", []):
            word_start = float(word.get("start", 0.0))
            word_end = float(word.get("end", word_start))
            midpoint = (word_start + word_end) / 2.0
            if start <= midpoint <= end:
                words.append(
                    {
                        "word": str(word.get("word", "")).strip(),
                        "start": word_start,
                        "end": word_end,
                    }
                )

    words.sort(key=lambda item: (item["start"], item["end"]))
    return words


def _analyze_word_sequence(words: List[dict], duration: float) -> dict:
    duration = max(0.01, duration)
    word_count = len(words)
    wps = word_count / duration

    pauses = []
    repeated_words = 0
    filler_count = 0

    previous_norm = None

    for index, word in enumerate(words):
        token = _norm(word["word"])
        if token in FILLER_WORDS:
            filler_count += 1

        if token and previous_norm and token == previous_norm:
            repeated_words += 1
        if token:
            previous_norm = token

        if index < len(words) - 1:
            gap = max(0.0, float(words[index + 1]["start"]) - float(word["end"]))
            if gap >= 0.45:
                pauses.append(
                    {
                        "start": float(word["end"]),
                        "end": float(words[index + 1]["start"]),
                        "duration": gap,
                    }
                )

    long_pause_seconds = sum(item["duration"] for item in pauses if item["duration"] >= 0.75)
    filler_ratio = filler_count / max(1, word_count)
    repetition_ratio = repeated_words / max(1, word_count)
    pause_ratio = long_pause_seconds / duration

    # Ritmul ideal diferă în funcție de conținut, deci penalizarea este blândă.
    speed_penalty = 0.0
    if wps < 1.45:
        speed_penalty = min(22.0, (1.45 - wps) * 18.0)
    elif wps > 4.4:
        speed_penalty = min(14.0, (wps - 4.4) * 6.0)

    pacing_score = 100.0
    pacing_score -= min(35.0, pause_ratio * 120.0)
    pacing_score -= min(25.0, filler_ratio * 180.0)
    pacing_score -= min(15.0, repetition_ratio * 120.0)
    pacing_score -= speed_penalty
    pacing_score = max(0.0, min(100.0, pacing_score))

    useful_words = max(0, word_count - filler_count - repeated_words)
    useful_wps = useful_words / duration
    information_density_proxy = max(0.0, min(100.0, 30.0 + useful_wps * 20.0))

    risks = []
    for pause in pauses:
        if pause["duration"] >= 0.75:
            risks.append(
                {
                    "start": round(pause["start"], 3),
                    "end": round(pause["end"], 3),
                    "reason": "long_pause",
                    "severity": min(100, round(45 + pause["duration"] * 25)),
                }
            )

    return {
        "duration": round(duration, 3),
        "word_count": word_count,
        "words_per_second": round(wps, 3),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        "repeated_word_count": repeated_words,
        "long_pause_seconds": round(long_pause_seconds, 3),
        "pacing_score": round(pacing_score),
        "information_density_proxy": round(information_density_proxy),
        "deterministic_risks": risks,
    }


def analyze_pacing(transcript: List[dict], start: float, end: float) -> dict:
    words = flatten_words(transcript, start, end)
    return _analyze_word_sequence(words, max(0.01, end - start))


def analyze_segments_pacing(transcript: List[dict], segments: Iterable[dict]) -> dict:
    joined_words = []
    offset = 0.0
    total_duration = 0.0

    for segment in segments:
        source_start = float(segment["start"])
        source_end = float(segment["end"])
        duration = max(0.0, source_end - source_start)
        if duration <= 0:
            continue

        for word in flatten_words(transcript, source_start, source_end):
            joined_words.append(
                {
                    "word": word["word"],
                    "start": offset + max(0.0, word["start"] - source_start),
                    "end": offset + min(duration, max(0.0, word["end"] - source_start)),
                }
            )

        offset += duration
        total_duration += duration

    return _analyze_word_sequence(joined_words, max(0.01, total_duration))
