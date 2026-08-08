from __future__ import annotations


WEIGHTS = {
    "hook": 0.22,
    "curiosity": 0.15,
    "emotion": 0.10,
    "conflict": 0.08,
    "payoff": 0.17,
    "information_density": 0.11,
    "pacing": 0.12,
    "standalone": 0.05,
}


def clamp_score(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def normalize_scores(raw_scores: dict | None, pacing_metrics: dict | None = None) -> dict:
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    if pacing_metrics is not None and not isinstance(pacing_metrics, dict):
        pacing_metrics = None

    scores = {
        key: clamp_score(raw_scores.get(key, 50))
        for key in WEIGHTS
    }

    if pacing_metrics:
        deterministic_pacing = clamp_score(pacing_metrics.get("pacing_score", scores["pacing"]))
        deterministic_density = clamp_score(
            pacing_metrics.get("information_density_proxy", scores["information_density"])
        )

        scores["pacing"] = round(scores["pacing"] * 0.60 + deterministic_pacing * 0.40)
        scores["information_density"] = round(
            scores["information_density"] * 0.70 + deterministic_density * 0.30
        )

    return scores


def retention_score(scores: dict, edit_penalty: float = 0.0) -> int:
    value = sum(clamp_score(scores.get(key, 0)) * weight for key, weight in WEIGHTS.items())
    value -= max(0.0, float(edit_penalty))
    return max(0, min(100, int(round(value))))
