from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List


@dataclass
class TTSResult:
    audio_path: Path
    duration: float
    provider: str
    voice: str
    language: str
    word_timings: List[dict] = field(default_factory=list)


class TTSProvider(ABC):
    name = "base"

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: Path,
        language: str = "auto",
        voice: str = "",
    ) -> TTSResult:
        raise NotImplementedError


def split_words(text: str) -> List[str]:
    return [item for item in re.findall(r"\S+", str(text or "").strip()) if item]


def approximate_word_timings(text: str, duration: float) -> List[dict]:
    words = split_words(text)
    duration = max(0.01, float(duration))

    if not words:
        return []

    weights = []
    for word in words:
        clean = re.sub(r"[^\wÀ-ÿĂÂÎȘȚăâîșț]+", "", word, flags=re.UNICODE)
        weights.append(max(1.0, min(8.0, float(len(clean) or 1))))

    total_weight = sum(weights) or float(len(words))
    cursor = 0.0
    result = []

    for index, (word, weight) in enumerate(zip(words, weights)):
        if index == len(words) - 1:
            end = duration
        else:
            end = min(duration, cursor + duration * weight / total_weight)

        result.append(
            {
                "word": word,
                "start": round(cursor, 4),
                "end": round(max(cursor + 0.01, end), 4),
            }
        )
        cursor = end

    if result:
        result[-1]["end"] = round(duration, 4)

    return result
