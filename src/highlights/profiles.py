from __future__ import annotations

from dataclasses import dataclass


STANDARD_SCORE_LIMITS = {
    "hook": 25,
    "payoff": 20,
    "emotion": 15,
    "visual_action": 15,
    "surprise": 10,
    "standalone": 10,
    "replayability": 5,
}


@dataclass(frozen=True)
class ContentProfile:
    name: str
    description: str
    weights: dict[str, int]


PROFILES = {
    "general": ContentProfile(
        name="general",
        description=(
            "Prioritize complete micro-stories with an immediate hook, clear progression, "
            "payoff or meaningful reaction, visual interest, standalone clarity and replay/share value."
        ),
        weights={
            "hook": 25,
            "payoff": 20,
            "emotion": 15,
            "visual_action": 15,
            "surprise": 10,
            "standalone": 10,
            "replayability": 5,
        },
    ),
    "gaming": ContentProfile(
        name="gaming",
        description=(
            "Prioritize clutch plays, kills, wins, fails, rage, screaming, funny reactions, "
            "unexpected gameplay, tension, rare/high-action events and the reaction after the event."
        ),
        weights={
            "emotion": 20,
            "visual_action": 20,
            "surprise": 20,
            "payoff": 15,
            "hook": 15,
            "standalone": 5,
            "replayability": 5,
        },
    ),
    "entertainment": ContentProfile(
        name="entertainment",
        description=(
            "Prioritize funny moments, reactions, challenge payoff, arguments, reveals, failures, "
            "wins, emotional turns and plot twists. Reward story progression, not noise alone."
        ),
        weights={
            "hook": 20,
            "payoff": 20,
            "emotion": 20,
            "surprise": 15,
            "visual_action": 10,
            "standalone": 10,
            "replayability": 5,
        },
    ),
    "podcast": ContentProfile(
        name="podcast",
        description=(
            "Prioritize strong or controversial statements, useful insights, surprising facts, "
            "confessions, stories, emotional statements and quotable ideas that stand alone."
        ),
        weights={
            "hook": 25,
            "standalone": 20,
            "replayability": 15,
            "emotion": 10,
            "surprise": 10,
            "payoff": 15,
            "visual_action": 5,
        },
    ),
    "reaction": ContentProfile(
        name="reaction",
        description=(
            "Prioritize authentic reaction, surprise, context for what caused it, visual evidence, "
            "and a complete payoff. Do not reward laughter or shouting without a meaningful event."
        ),
        weights={
            "emotion": 20,
            "surprise": 20,
            "hook": 20,
            "payoff": 15,
            "visual_action": 10,
            "standalone": 10,
            "replayability": 5,
        },
    ),
}


def get_profile(name: str | None) -> ContentProfile:
    normalized = str(name or "general").strip().lower()
    return PROFILES.get(normalized, PROFILES["general"])


def infer_profile(candidate: dict, requested: str | None = None) -> ContentProfile:
    if requested and str(requested).lower() not in {"", "auto"}:
        return get_profile(requested)

    content_type = str(candidate.get("content_type", "") or "").strip().lower()
    aliases = {
        "challenge": "entertainment",
        "storytelling": "entertainment",
        "interview": "podcast",
        "livestream": "reaction",
    }
    return get_profile(aliases.get(content_type, content_type))


def profile_total(scores: dict, profile: ContentProfile) -> int:
    total = 0.0
    for key, points in profile.weights.items():
        limit = STANDARD_SCORE_LIMITS[key]
        try:
            value = float(scores.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(float(limit), value))
        total += (value / limit) * points if limit else 0.0
    return max(0, min(100, int(round(total))))
