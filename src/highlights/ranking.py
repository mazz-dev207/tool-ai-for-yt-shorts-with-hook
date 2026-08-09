from __future__ import annotations

from typing import Iterable

from src.highlights.profiles import STANDARD_SCORE_LIMITS, ContentProfile, profile_total


VALID_CATEGORIES = {
    "funny_reaction", "clutch", "fail", "argument", "surprise", "story_moment",
    "rage", "win", "insight", "confession", "reveal", "emotional", "general",
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def validate_scores(raw: dict | None) -> dict[str, int]:
    raw = raw if isinstance(raw, dict) else {}
    scores: dict[str, int] = {}
    for key, maximum in STANDARD_SCORE_LIMITS.items():
        try:
            value = int(round(float(raw.get(key, 0) or 0)))
        except (TypeError, ValueError):
            value = 0
        scores[key] = int(clamp(value, 0, maximum))
    return scores


def validate_judgement(
    raw: dict,
    candidate: dict,
    video_duration: float,
    profile: ContentProfile,
    context_start: float,
    context_end: float,
) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Gemini judgement must be an object")

    original_start = float(candidate["start"])
    original_end = float(candidate["end"])

    try:
        refined_start = float(raw.get("refined_start", original_start))
        refined_end = float(raw.get("refined_end", original_end))
    except (TypeError, ValueError):
        refined_start, refined_end = original_start, original_end

    safe_start = clamp(refined_start, max(0.0, context_start), min(video_duration, context_end))
    safe_end = clamp(refined_end, max(0.0, context_start), min(video_duration, context_end))
    if safe_end <= safe_start + 0.5:
        safe_start, safe_end = original_start, original_end

    scores = validate_scores(raw.get("scores"))
    calculated_total = profile_total(scores, profile)

    category = str(raw.get("category", "general") or "general").strip().lower()
    if category not in VALID_CATEGORIES:
        category = "general"

    return {
        "candidate_id": int(raw.get("candidate_id", candidate.get("candidate_id", 0)) or 0),
        "original_start": round(original_start, 3),
        "original_end": round(original_end, 3),
        "refined_start": round(safe_start, 3),
        "refined_end": round(safe_end, 3),
        "scores": scores,
        "total_score": calculated_total,
        "category": category,
        "reason": str(raw.get("reason", "")).strip()[:600],
        "has_complete_payoff": bool(raw.get("has_complete_payoff", False)),
        "requires_previous_context": bool(raw.get("requires_previous_context", False)),
        "recommended": bool(raw.get("recommended", calculated_total >= 55)),
    }


def overlap_ratio(first: dict, second: dict) -> float:
    start_a = float(first.get("start", first.get("refined_start", 0.0)))
    end_a = float(first.get("end", first.get("refined_end", 0.0)))
    start_b = float(second.get("start", second.get("refined_start", 0.0)))
    end_b = float(second.get("end", second.get("refined_end", 0.0)))
    intersection = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    shorter = max(0.001, min(end_a - start_a, end_b - start_b))
    return intersection / shorter


def remove_overlaps(items: Iterable[dict], threshold: float = 0.60) -> list[dict]:
    ordered = sorted(items, key=lambda item: int(item.get("gemini", {}).get("total_score", 0)), reverse=True)
    selected: list[dict] = []
    for item in ordered:
        if any(overlap_ratio(item, existing) >= threshold for existing in selected):
            continue
        selected.append(item)
    return selected


def diversity_rerank(items: list[dict], top_k: int, close_score_margin: int = 6) -> list[dict]:
    pool = sorted(items, key=lambda item: int(item.get("gemini", {}).get("total_score", 0)), reverse=True)
    selected: list[dict] = []
    category_counts: dict[str, int] = {}

    while pool and len(selected) < top_k:
        best_base = int(pool[0].get("gemini", {}).get("total_score", 0))
        close = [item for item in pool if best_base - int(item.get("gemini", {}).get("total_score", 0)) <= close_score_margin]

        def key(item: dict):
            gemini = item.get("gemini", {})
            category = str(gemini.get("category", "general"))
            repeat_penalty = category_counts.get(category, 0) * 3
            return int(gemini.get("total_score", 0)) - repeat_penalty

        winner = max(close, key=key)
        pool.remove(winner)
        selected.append(winner)
        category = str(winner.get("gemini", {}).get("category", "general"))
        category_counts[category] = category_counts.get(category, 0) + 1

    return selected
