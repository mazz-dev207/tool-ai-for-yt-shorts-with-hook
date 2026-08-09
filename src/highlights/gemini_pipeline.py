from __future__ import annotations

import json
from pathlib import Path

from src.config import (
    CONTENT_PROFILE,
    GEMINI_ENABLED,
    GEMINI_MAX_CANDIDATES,
    GEMINI_MIN_SCORE,
    GEMINI_OVERLAP_THRESHOLD,
    GEMINI_TOP_HIGHLIGHTS,
    HIGHLIGHT_MODE,
    HIGHLIGHTS_DIR,
    TRANSCRIPT_DIR,
)
from src.highlights.gemini_judge import GeminiHighlightJudge, GeminiUnavailable, probe_duration
from src.highlights.profiles import infer_profile
from src.highlights.ranking import diversity_rerank, remove_overlaps
from src.logger import info, success, warning


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _legacy_score(candidate: dict) -> float:
    try:
        if candidate.get("score") is not None:
            return float(candidate.get("score") or 0)
        scores = candidate.get("scores") or {}
        for key in ("viral_potential", "retention", "hook"):
            if scores.get(key) is not None:
                return float(scores.get(key) or 0)
    except (TypeError, ValueError):
        pass
    return 0.0


def _prepare_candidates(candidates: list[dict]) -> list[dict]:
    ordered = sorted(candidates, key=_legacy_score, reverse=True)
    limited = ordered[: max(1, GEMINI_MAX_CANDIDATES)]
    result = []
    for index, candidate in enumerate(limited, start=1):
        item = dict(candidate)
        item["candidate_id"] = index
        result.append(item)
    return result


def _merge_judgement(candidate: dict, judgement: dict, profile_name: str, model_name: str) -> dict:
    item = dict(candidate)
    item["start"] = float(judgement["refined_start"])
    item["end"] = float(judgement["refined_end"])
    item["duration"] = round(item["end"] - item["start"], 3)
    item["gemini"] = {
        **judgement,
        "content_profile": profile_name,
        "ai_model": model_name,
    }
    return item


def run_gemini_highlight_stage(
    video_name: str,
    video_path: Path,
    mode: str | None = None,
    content_profile: str | None = None,
) -> Path:
    from src.config import GEMINI_MODEL

    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    selected_mode = str(mode or HIGHLIGHT_MODE or "legacy").strip().lower()

    if selected_mode not in {"legacy", "gemini", "compare"}:
        warning(f"Highlight mode necunoscut '{selected_mode}', folosesc legacy.")
        selected_mode = "legacy"

    if selected_mode == "legacy":
        info("[GEMINI] Highlight mode=legacy; păstrez selectorul existent.")
        return highlights_path

    if not GEMINI_ENABLED:
        warning("[GEMINI] Dezactivat; continui cu selectorul legacy.")
        return highlights_path

    legacy_candidates = _load_json(highlights_path)
    transcript = _load_json(transcript_path)
    if not isinstance(legacy_candidates, list) or not legacy_candidates:
        warning("[GEMINI] Nu există candidați legacy de evaluat.")
        return highlights_path
    if not isinstance(transcript, list):
        warning("[GEMINI] Transcript invalid; continui cu selectorul legacy.")
        return highlights_path

    candidates = _prepare_candidates(legacy_candidates)
    info(f"[HIGHLIGHTS] Generated {len(legacy_candidates)} legacy candidates.")
    if len(candidates) < len(legacy_candidates):
        info(f"[GEMINI] Prefilter: {len(legacy_candidates)} -> {len(candidates)} candidates.")

    try:
        judge = GeminiHighlightJudge()
    except GeminiUnavailable as exc:
        warning(f"[GEMINI] {exc}; continui cu selectorul legacy.")
        return highlights_path

    video_duration = probe_duration(video_path)
    judged: list[dict] = []
    failures = 0
    cache_hits = 0

    info("[GEMINI] Evaluating candidates multimodal...")
    for index, candidate in enumerate(candidates, start=1):
        profile = infer_profile(candidate, content_profile or CONTENT_PROFILE)
        info(
            f"[GEMINI {index}/{len(candidates)}] "
            f"{float(candidate['start']):.1f}s -> {float(candidate['end']):.1f}s | profile={profile.name}"
        )
        try:
            judgement, from_cache = judge.judge(
                video_path=video_path,
                transcript=transcript,
                candidate=candidate,
                candidate_id=int(candidate["candidate_id"]),
                profile=profile,
                video_duration=video_duration,
            )
            cache_hits += int(from_cache)
            info(
                f"[GEMINI {index}/{len(candidates)}] score={judgement['total_score']} "
                f"category={judgement['category']} cache={'hit' if from_cache else 'miss'}"
            )
            if judgement["recommended"] and int(judgement["total_score"]) >= GEMINI_MIN_SCORE:
                judged.append(_merge_judgement(candidate, judgement, profile.name, GEMINI_MODEL))
        except Exception as exc:
            failures += 1
            warning(f"[GEMINI {index}/{len(candidates)}] failed: {exc}")

    if not judged:
        warning("[GEMINI] Niciun rezultat valid; fallback complet la selectorul legacy.")
        _save_json(
            HIGHLIGHTS_DIR / f"{video_name}_gemini_metadata.json",
            {
                "video": video_path.name,
                "mode": selected_mode,
                "fallback": "legacy",
                "evaluated": len(candidates),
                "failures": failures,
                "cache_hits": cache_hits,
            },
        )
        return highlights_path

    info("[RANKING] Removing overlapping candidates...")
    deduped = remove_overlaps(judged, threshold=GEMINI_OVERLAP_THRESHOLD)
    final = diversity_rerank(deduped, top_k=GEMINI_TOP_HIGHLIGHTS)

    for rank, item in enumerate(final, start=1):
        item["rank"] = rank

    metadata = {
        "video": video_path.name,
        "mode": selected_mode,
        "ai_model": GEMINI_MODEL,
        "legacy_candidate_count": len(legacy_candidates),
        "gemini_evaluated_count": len(candidates),
        "gemini_qualified_count": len(judged),
        "selected_count": len(final),
        "failures": failures,
        "cache_hits": cache_hits,
        "highlights": [
            {
                "video": video_path.name,
                "start": item["start"],
                "end": item["end"],
                "duration": item["duration"],
                "ai_model": item["gemini"]["ai_model"],
                "content_profile": item["gemini"]["content_profile"],
                "score": item["gemini"]["total_score"],
                "scores": item["gemini"]["scores"],
                "category": item["gemini"]["category"],
                "reason": item["gemini"]["reason"],
                "performance": {
                    "views": None,
                    "viewed_vs_swiped": None,
                    "average_view_duration": None,
                    "average_percentage_viewed": None,
                    "likes": None,
                    "comments": None,
                    "shares": None,
                    "subscribers_gained": None,
                },
            }
            for item in final
        ],
    }
    _save_json(HIGHLIGHTS_DIR / f"{video_name}_gemini_metadata.json", metadata)

    if selected_mode == "compare":
        _save_json(
            HIGHLIGHTS_DIR / f"{video_name}_highlight_compare.json",
            {
                "video": video_path.name,
                "legacy": legacy_candidates,
                "gemini": final,
            },
        )
        success(
            f"[COMPARE] Raport salvat; downstream rămâne pe legacy. "
            f"Gemini selected {len(final)} highlights."
        )
        return highlights_path

    _save_json(highlights_path, final)
    success(f"[RANKING] Selected TOP {len(final)} Gemini highlights.")
    return highlights_path
