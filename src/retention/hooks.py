from __future__ import annotations

import re
from typing import Iterable, List, Optional


WEAK_PREFIXES = (
    "știai că",
    "stiai ca",
    "în acest videoclip",
    "in acest videoclip",
    "astăzi vom vorbi",
    "astazi vom vorbi",
    "uite un clip",
    "uită-te până la final",
    "uita-te pana la final",
    "nu o să-ți vină să crezi",
    "nu o sa-ti vina sa crezi",
    "did you know",
    "in this video",
    "watch until the end",
    "you won't believe",
)

HOOK_WEIGHTS = {
    "curiosity": 0.18,
    "stakes": 0.12,
    "surprise": 0.12,
    "clarity": 0.13,
    "relevance": 0.16,
    "brevity": 0.10,
    "naturalness": 0.07,
    "specificity": 0.05,
    "emotional_impact": 0.03,
    "fit": 0.04,
}

RO_HINTS = {
    "și", "să", "este", "care", "pentru", "de ce", "acest", "asta", "după",
    "pierde", "câștigă", "greșeala", "nimeni", "momentul", "totul",
}
EN_HINTS = {
    "the", "and", "why", "how", "this", "that", "what", "after", "before",
    "lost", "won", "mistake", "nobody", "moment", "everything",
}


def _clamp_score(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def detect_language(text: str) -> str:
    lowered = f" {str(text or '').lower()} "
    if any(char in lowered for char in "ăâîșț"):
        return "ro"

    ro_hits = sum(1 for item in RO_HINTS if f" {item} " in lowered)
    en_hits = sum(1 for item in EN_HINTS if f" {item} " in lowered)
    return "ro" if ro_hits > en_hits else "en"


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text or "").strip()))


def _brevity_score(text: str) -> int:
    count = _word_count(text)
    if count <= 0:
        return 0
    if count <= 8:
        return 100
    if count <= 10:
        return 94
    if count <= 12:
        return 86
    if count <= 14:
        return 70
    return max(20, 70 - (count - 14) * 8)


def _hook_metrics(hook: dict, text: str) -> dict:
    if not isinstance(hook, dict):
        hook = {}

    raw = hook.get("metrics") or hook.get("hook_scores") or {}

    if not isinstance(raw, dict):
        raw = {}
    metrics = {
        "curiosity": _clamp_score(raw.get("curiosity", hook.get("curiosity", 50))),
        "stakes": _clamp_score(raw.get("stakes", hook.get("stakes", 50))),
        "surprise": _clamp_score(raw.get("surprise", hook.get("surprise", 50))),
        "clarity": _clamp_score(raw.get("clarity", hook.get("clarity", 70))),
        "relevance": _clamp_score(raw.get("relevance", hook.get("relevance", 70))),
        "brevity": _brevity_score(text),
        "naturalness": _clamp_score(raw.get("naturalness", hook.get("naturalness", 70))),
        "specificity": _clamp_score(raw.get("specificity", hook.get("specificity", 60))),
        "emotional_impact": _clamp_score(raw.get("emotional_impact", hook.get("emotional_impact", 50))),
        "fit": _clamp_score(raw.get("fit", hook.get("fit", 70))),
    }
    return metrics


def calculate_hook_score(hook: dict, text: str) -> int:
    if not isinstance(hook, dict):
        hook = {}

    metrics = _hook_metrics(hook, text)
    score = sum(metrics[key] * weight for key, weight in HOOK_WEIGHTS.items())

    if str(text or "").lower().startswith(WEAK_PREFIXES):
        score -= 18

    evidence = hook.get("evidence") or []
    if bool(hook.get("generated", False)) and not evidence:
        score -= 12

    raw_score = _clamp_score(hook.get("score"))
    if raw_score:
        score = score * 0.75 + raw_score * 0.25

    return max(0, min(100, int(round(score))))


def normalize_hooks(raw_hooks: Iterable[dict], context_start: float, context_end: float) -> List[dict]:
    result = []

    if not isinstance(raw_hooks, (list, tuple)):
        return result

    for hook in raw_hooks:
        if not isinstance(hook, dict):
            continue

        text = str(hook.get("text", "")).strip()
        if not text:
            continue

        generated = bool(hook.get("generated", False))
        language = str(hook.get("language", "")).strip().lower() or detect_language(text)
        if language not in {"ro", "en"}:
            language = detect_language(text)

        normalized = {
            "text": text,
            "type": str(hook.get("type", "unknown")).strip() or "unknown",
            "generated": generated,
            "language": language,
            "evidence": (
                hook.get("evidence", [])
                if isinstance(hook.get("evidence", []), list)
                else []
            ),
            "metrics": _hook_metrics(hook, text),
        }
        normalized["score"] = calculate_hook_score(hook, text)

        if not generated:
            try:
                source_start = float(hook.get("source_start"))
                source_end = float(hook.get("source_end"))
            except (TypeError, ValueError):
                continue

            source_start = max(context_start, min(context_end, source_start))
            source_end = max(context_start, min(context_end, source_end))
            if source_end <= source_start:
                continue

            normalized["source_start"] = source_start
            normalized["source_end"] = source_end
            normalized["application"] = "extractive_audio"
        else:
            normalized["application"] = "voiceover_tts_candidate"

        result.append(normalized)

    return result


def select_best_hook(hooks: List[dict], prefer_original: bool = True) -> Optional[dict]:
    if not hooks:
        return None

    def rank(hook: dict):
        generated_penalty = 8 if prefer_original and hook.get("generated") else 0
        return int(hook.get("score", 0)) - generated_penalty

    return max(hooks, key=rank)


def select_voiceover_hook(
    hooks: List[dict],
    max_words: int = 12,
    min_score: int = 60,
) -> Optional[dict]:
    candidates = []

    for hook in hooks or []:
        if not hook.get("generated"):
            continue

        text = str(hook.get("text", "")).strip()
        if not text:
            continue
        if text.lower().startswith(WEAK_PREFIXES):
            continue
        if _word_count(text) > max_words:
            continue
        if int(hook.get("score", 0) or 0) < min_score:
            continue
        if not hook.get("evidence"):
            continue

        candidates.append(hook)

    if not candidates:
        return None

    return max(candidates, key=lambda hook: int(hook.get("score", 0) or 0))
