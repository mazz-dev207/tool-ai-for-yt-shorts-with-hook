from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.config import (
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    RETENTION_CONTEXT_BEFORE,
    RETENTION_CONTEXT_AFTER,
    RETENTION_MAX_CANDIDATES,
    RETENTION_MAX_VARIANTS,
    RETENTION_MAX_SEGMENTS,
    RETENTION_MIN_CLIP_DURATION,
    RETENTION_MAX_CLIP_DURATION,
    RETENTION_PREFER_ORIGINAL_HOOK,
    VOICEOVER_MAX_HOOK_WORDS,
    VOICEOVER_MIN_HOOK_SCORE,
)
from src.logger import info, success, warning
from src.retention.analyzer import RetentionAnalyzer
from src.retention.context import build_context
from src.retention.hooks import normalize_hooks, select_best_hook, select_voiceover_hook
from src.retention.pacing import analyze_pacing, analyze_segments_pacing
from src.retention.rewriting import normalize_variants
from src.retention.scoring import normalize_scores, retention_score
from src.retention.timeline import build_timeline, remap_items, build_retention_buckets


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _dict_list(value) -> List[dict]:
    """
    Qwen poate întoarce ocazional string-uri sau alte tipuri într-un array
    care ar trebui să conțină obiecte JSON. Le ignorăm în loc să stricăm
    întreg candidatul.
    """
    if not isinstance(value, (list, tuple)):
        return []

    return [
        item
        for item in value
        if isinstance(item, dict)
    ]


def _fallback_clip(candidate: dict, reason: str) -> dict:
    start = float(candidate["start"])
    end = float(candidate["end"])
    return {
        **candidate,
        "strategy": "extractive_fallback",
        "segments": [
            {
                "start": start,
                "end": end,
                "role": "escalation",
                "reason": reason,
            }
        ],
        "duration": round(max(0.0, end - start), 3),
        "retention_score": int(candidate.get("score", 0) or 0),
        "scores": {},
        "hook": None,
        "retention_anchors": [],
        "retention_risks": [],
        "timeline": [
            {
                "index": 0,
                "source_start": start,
                "source_end": end,
                "target_start": 0.0,
                "target_end": round(max(0.0, end - start), 3),
                "role": "escalation",
            }
        ],
    }


def _evaluate_variants(transcript: List[dict], variants: List[dict], base_scores: dict):
    evaluated = []

    for variant in variants:
        pacing = analyze_segments_pacing(transcript, variant["segments"])
        raw_scores = variant.get("scores") or base_scores
        scores = normalize_scores(raw_scores, pacing)
        final_score = retention_score(scores, variant.get("edit_penalty", 0.0))

        item = dict(variant)
        item["pacing_metrics"] = pacing
        item["scores"] = scores
        item["retention_score"] = final_score
        evaluated.append(item)

    return evaluated


def _removed_segments(candidate: dict, selected_segments: List[dict]) -> List[dict]:
    # Debug simplu pentru segmente cronologice din interiorul candidatului.
    candidate_start = float(candidate["start"])
    candidate_end = float(candidate["end"])

    chronological = sorted(
        [
            item
            for item in selected_segments
            if item["end"] > candidate_start and item["start"] < candidate_end
        ],
        key=lambda item: item["start"],
    )

    removed = []
    cursor = candidate_start

    for item in chronological:
        start = max(candidate_start, float(item["start"]))
        end = min(candidate_end, float(item["end"]))
        if start > cursor + 0.15:
            removed.append(
                {
                    "start": round(cursor, 3),
                    "end": round(start, 3),
                    "reason": "not_selected_by_retention_edit",
                }
            )
        cursor = max(cursor, end)

    if cursor < candidate_end - 0.15:
        removed.append(
            {
                "start": round(cursor, 3),
                "end": round(candidate_end, 3),
                "reason": "not_selected_by_retention_edit",
            }
        )

    return removed


def optimize_retention(video_name: str):
    transcript_path = TRANSCRIPT_DIR / f"{video_name}.json"
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"

    transcript = _load_json(transcript_path)
    candidates = _load_json(highlights_path)

    if not candidates:
        warning("Nu există candidați pentru optimizarea retenției.")
        return highlights_path

    # Păstrăm candidații originali pentru debugging/comparații.
    candidate_backup = HIGHLIGHTS_DIR / f"{video_name}_candidates.json"
    _save_json(candidate_backup, candidates)

    debug_dir = HIGHLIGHTS_DIR / "debug" / video_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    analyzer = RetentionAnalyzer()
    optimized = []

    limited_candidates = candidates[:RETENTION_MAX_CANDIDATES]
    info(f"Optimizez retenția pentru {len(limited_candidates)} candidați...")

    for index, candidate in enumerate(limited_candidates, start=1):
        info(f"Retention candidat {index}/{len(limited_candidates)}")

        try:
            context = build_context(
                transcript,
                candidate,
                RETENTION_CONTEXT_BEFORE,
                RETENTION_CONTEXT_AFTER,
            )

            candidate_pacing = analyze_pacing(
                transcript,
                context["candidate_start"],
                context["candidate_end"],
            )

            analysis = analyzer.analyze(
                context,
                candidate_pacing,
                max_variants=RETENTION_MAX_VARIANTS,
            )

            hooks = normalize_hooks(
                _dict_list(
                    analysis.get("hook_variants", [])
                ),
                context["start"],
                context["end"],
            )
            best_hook = select_best_hook(hooks, RETENTION_PREFER_ORIGINAL_HOOK)
            voiceover_hook = select_voiceover_hook(
                hooks,
                max_words=VOICEOVER_MAX_HOOK_WORDS,
                min_score=VOICEOVER_MIN_HOOK_SCORE,
            )

            variants = normalize_variants(
                _dict_list(
                    analysis.get("variants", [])
                ),
                transcript,
                context["start"],
                context["end"],
                candidate,
                RETENTION_MAX_VARIANTS,
                RETENTION_MAX_SEGMENTS,
                RETENTION_MIN_CLIP_DURATION,
                RETENTION_MAX_CLIP_DURATION,
            )

            evaluated = _evaluate_variants(
                transcript,
                variants,
                analysis.get("scores", {}),
            )

            selected = max(evaluated, key=lambda item: item["retention_score"])
            timeline = build_timeline(selected["segments"])

            ai_risks = _dict_list(
                analysis.get("retention_risks", [])
            )

            deterministic_risks = _dict_list(
                candidate_pacing.get("deterministic_risks", [])
            )

            all_risks = (
                ai_risks
                + deterministic_risks
            )

            mapped_anchors = remap_items(
                _dict_list(
                    analysis.get("retention_anchors", [])
                ),
                timeline,
            )

            mapped_risks = remap_items(
                all_risks,
                timeline,
            )

            mapped_open_loops = remap_items(
                _dict_list(
                    analysis.get("open_loops", [])
                ),
                timeline,
            )

            mapped_pattern_interrupts = remap_items(
                _dict_list(
                    analysis.get("pattern_interrupts", [])
                ),
                timeline,
            )
            retention_timeline = build_retention_buckets(
                selected["duration"], mapped_anchors, mapped_risks
            )

            selected_clip = {
                **candidate,
                "start": round(min(item["start"] for item in selected["segments"]), 3),
                "end": round(max(item["end"] for item in selected["segments"]), 3),
                "duration": round(selected["duration"], 3),
                "debug_index": index,
                "content_type": analysis.get("content_type", "general"),
                "strategy": selected.get("strategy", "extractive"),
                "segments": selected["segments"],
                "hook": best_hook,
                "hook_variants": hooks,
                "voiceover_hook": voiceover_hook,
                "scores": selected["scores"],
                "retention_score": selected["retention_score"],
                "timeline": timeline,
                "retention_anchors": mapped_anchors,
                "retention_risks": mapped_risks,
                "open_loops": mapped_open_loops,
                "pattern_interrupts": mapped_pattern_interrupts,
                "retention_timeline": retention_timeline,
                "removed_segments": _removed_segments(candidate, selected["segments"]),
            }

            optimized.append(selected_clip)

            debug_payload = {
                "candidate": candidate,
                "context": {
                    "start": context["start"],
                    "end": context["end"],
                    "candidate_start": context["candidate_start"],
                    "candidate_end": context["candidate_end"],
                },
                "candidate_pacing": candidate_pacing,
                "analysis": analysis,
                "hooks": hooks,
                "variants": evaluated,
                "selected": selected_clip,
            }
            _save_json(debug_dir / f"clip_{index}.json", debug_payload)

            success(
                f"Retention {index}: {selected_clip['retention_score']}/100 | "
                f"{selected_clip['duration']:.1f}s | {selected_clip['content_type']}"
            )

        except Exception as exc:
            warning(f"Retention candidat {index} a eșuat: {exc}")
            fallback = _fallback_clip(candidate, f"retention_error: {exc}")
            fallback["debug_index"] = index
            optimized.append(fallback)
            _save_json(
                debug_dir / f"clip_{index}.json",
                {"candidate": candidate, "error": str(exc), "selected": fallback},
            )

    if len(candidates) > len(limited_candidates):
        warning(
            f"{len(candidates) - len(limited_candidates)} candidați cu scor mai mic au fost "
            f"săriți pentru a limita timpul de procesare. Mărește RETENTION_MAX_CANDIDATES dacă vrei mai mulți."
        )

    # Sortăm după scorul final, dar păstrăm toate variantele selectate.
    optimized.sort(key=lambda item: item.get("retention_score", 0), reverse=True)

    _save_json(highlights_path, optimized)
    success(f"Retention optimization terminat: {len(optimized)} Shorts pregătite.")
    success(f"Debug JSON: {debug_dir}")
    return highlights_path
