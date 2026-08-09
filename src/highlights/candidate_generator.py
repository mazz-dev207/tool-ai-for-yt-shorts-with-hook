from __future__ import annotations

from src import highlight_selector as legacy_selector
from src.logger import info


GEMINI_DISCOVERY_MIN_SCORE = 50


def generate_candidates(video_name: str, high_recall: bool = False):
    """
    Refolosește selectorul existent ca Candidate Generator.

    legacy mode păstrează pragul actual testat. În Gemini/compare reducem doar
    temporar pragul de discovery pentru recall mai mare; promptul, timestamp
    snapping, fallback-ul și formatul JSON rămân cele existente.
    """
    if not high_recall:
        return legacy_selector.select_highlights(video_name)

    original_threshold = legacy_selector.MIN_SCORE
    try:
        legacy_selector.MIN_SCORE = min(
            int(original_threshold),
            GEMINI_DISCOVERY_MIN_SCORE,
        )
        info(
            f"[HIGHLIGHTS] High-recall candidate mode: "
            f"score threshold {original_threshold} -> {legacy_selector.MIN_SCORE}."
        )
        return legacy_selector.select_highlights(video_name)
    finally:
        legacy_selector.MIN_SCORE = original_threshold
