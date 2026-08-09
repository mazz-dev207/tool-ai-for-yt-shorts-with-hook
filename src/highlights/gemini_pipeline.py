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


def _rejection_reason(judgement: dict) -> str | None:
    if not bool(judgement.get("recommended", False)):
        return "gemini_not_recommended"
    if int(judgement.get("total_score", 0) or 0) < GEMINI_MIN_SCORE:
        return "below_min_score"
    return None


def _ids(items: list[dict]) -> set[int]:
    result: set[int] = set()
    for item in items:
        try:
            result.add(int(item.get("candidate_id", 0)))
        except (TypeError, ValueError):
            pass
    return result


def _report_entry(item: dict, status: str, rejection_reason: str | None = None) -> dict:
    gemini = item.get("gemini") or {}
    refined_start = gemini.get("refined_start")
    refined_end = gemini.get("refined_end")
    refined_duration = None
    if refined_start is not None and refined_end is not None:
        refined_duration = round(float(refined_end) - float(refined_start), 3)

    return {
        "candidate_id": item.get("candidate_id"),
        "title": item.get("title"),
        "legacy_start": gemini.get("original_start", item.get("start")),
        "legacy_end": gemini.get("original_end", item.get("end")),
        "legacy_score": round(_legacy_score(item), 2),
        "refined_start": refined_start,
        "refined_end": refined_end,
        "refined_duration": refined_duration,
        "gemini_score": gemini.get("total_score"),
        "scores": gemini.get("scores"),
        "category": gemini.get("category"),
        "reason": gemini.get("reason"),
        "has_complete_payoff": gemini.get("has_complete_payoff"),
        "requires_previous_context": gemini.get("requires_previous_context"),
        "recommended": gemini.get("recommended"),
        "content_profile": gemini.get("content_profile"),
        "ai_model": gemini.get("ai_model"),
        "status": status,
        "rejection_reason": rejection_reason,
        "minimum_required_score": GEMINI_MIN_SCORE,
    }


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
    evaluated: list[dict] = []
    qualified: list[dict] = []
    rejected: list[dict] = []
    failure_items: list[dict] = []
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

            merged = _merge_judgement(candidate, judgement, profile.name, GEMINI_MODEL)
            evaluated.append(merged)
            reason = _rejection_reason(judgement)
            if reason is None:
                qualified.append(merged)
            else:
                rejected.append(_report_entry(merged, "rejected", reason))
        except Exception as exc:
            warning(f"[GEMINI {index}/{len(candidates)}] failed: {exc}")
            failure_items.append({
                "candidate_id": candidate.get("candidate_id"),
                "title": candidate.get("title"),
                "start": candidate.get("start"),
                "end": candidate.get("end"),
                "legacy_score": _legacy_score(candidate),
                "status": "failed",
                "error": str(exc),
            })

    info("[RANKING] Removing overlapping candidates...")
    deduped = remove_overlaps(qualified, threshold=GEMINI_OVERLAP_THRESHOLD)
    final = diversity_rerank(deduped, top_k=GEMINI_TOP_HIGHLIGHTS)
    for rank, item in enumerate(final, start=1):
        item["rank"] = rank

    deduped_ids = _ids(deduped)
    final_ids = _ids(final)
    selected_report: list[dict] = []
    removed_overlap: list[dict] = []
    ranked_out: list[dict] = []

    for item in qualified:
        candidate_id = int(item.get("candidate_id", 0))
        if candidate_id in final_ids:
            selected_report.append(_report_entry(item, "selected"))
        elif candidate_id not in deduped_ids:
            removed_overlap.append(
                _report_entry(item, "removed_overlap", "overlap_with_higher_scoring_candidate")
            )
        else:
            ranked_out.append(_report_entry(item, "ranked_out", "outside_final_top_k"))

    selected_ids = _ids(selected_report)
    overlap_ids = _ids(removed_overlap)
    ranked_out_ids = _ids(ranked_out)
    all_evaluated: list[dict] = []
    for item in evaluated:
        candidate_id = int(item.get("candidate_id", 0))
        if candidate_id in selected_ids:
            status, reason = "selected", None
        elif candidate_id in overlap_ids:
            status, reason = "removed_overlap", "overlap_with_higher_scoring_candidate"
        elif candidate_id in ranked_out_ids:
            status, reason = "ranked_out", "outside_final_top_k"
        else:
            status, reason = "rejected", _rejection_reason(item.get("gemini") or {})
        all_evaluated.append(_report_entry(item, status, reason))

    all_evaluated.sort(key=lambda item: int(item.get("gemini_score", 0) or 0), reverse=True)

    metadata = {
        "video": video_path.name,
        "mode": selected_mode,
        "ai_model": GEMINI_MODEL,
        "legacy_candidate_count": len(legacy_candidates),
        "gemini_evaluated_count": len(evaluated),
        "gemini_qualified_count": len(qualified),
        "gemini_rejected_count": len(rejected),
        "removed_overlap_count": len(removed_overlap),
        "ranked_out_count": len(ranked_out),
        "selected_count": len(final),
        "failures": len(failure_items),
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
        "all_evaluated": all_evaluated,
        "rejected": rejected,
        "removed_overlap": removed_overlap,
        "ranked_out": ranked_out,
        "failed": failure_items,
    }
    _save_json(HIGHLIGHTS_DIR / f"{video_name}_gemini_metadata.json", metadata)

    if selected_mode == "compare":
        _save_json(
            HIGHLIGHTS_DIR / f"{video_name}_highlight_compare.json",
            {
                "video": video_path.name,
                "mode": "compare",
                "ai_model": GEMINI_MODEL,
                "legacy": legacy_candidates,
                "gemini": final,
                "gemini_selected": selected_report,
                "gemini_rejected": rejected,
                "gemini_removed_overlap": removed_overlap,
                "gemini_ranked_out": ranked_out,
                "gemini_failed": failure_items,
                "gemini_all_evaluated": all_evaluated,
                "summary": {
                    "legacy_candidates": len(legacy_candidates),
                    "evaluated": len(evaluated),
                    "qualified": len(qualified),
                    "selected": len(final),
                    "rejected": len(rejected),
                    "removed_overlap": len(removed_overlap),
                    "ranked_out": len(ranked_out),
                    "failed": len(failure_items),
                    "cache_hits": cache_hits,
                },
            },
        )
        success(
            f"[COMPARE] Raport complet salvat; downstream rămâne pe legacy. "
            f"Gemini selected {len(final)} din {len(evaluated)} evaluate."
        )
        return highlights_path

    if not final:
        warning("[GEMINI] Niciun rezultat Gemini nu a trecut ranking-ul; fallback complet la selectorul legacy.")
        return highlights_path

    _save_json(highlights_path, final)
    success(f"[RANKING] Selected TOP {len(final)} Gemini highlights.")
    return highlights_path
