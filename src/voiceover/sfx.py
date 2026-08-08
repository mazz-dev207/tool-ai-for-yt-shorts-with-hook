from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Iterable

from src.config import (
    INTRO_SFX_DIR,
    INTRO_SFX_ENABLED,
    INTRO_SFX_RANDOM_TOP_K,
)

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}

AGGRESSIVE_TAGS = {"impact", "hit", "hard", "bass", "slam", "boom"}
SOFT_TAGS = {"soft", "subtle", "gentle", "smooth", "light"}

SEMANTIC_GROUPS = {
    "surprise": {"surprise", "reveal", "twist", "unexpected", "shock"},
    "gaming": {"gaming", "game", "clutch", "win", "victory", "fail", "kill"},
    "storytelling": {"story", "storytelling", "narrative", "cinematic"},
    "emotional": {"emotional", "emotion", "sad", "heartfelt", "hope", "touching"},
    "funny": {"funny", "comedy", "meme", "joke", "humor", "reaction"},
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value or "").lower())
        if token
    }


def _sound_tags(path: Path) -> set[str]:
    return _tokens(path.stem)


def discover_sfx(folder: Path = INTRO_SFX_DIR) -> list[Path]:
    folder = Path(folder)
    if not INTRO_SFX_ENABLED or not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_clip_tags(clip: dict, hook: dict | None = None) -> set[str]:
    tags = set()

    for value in [
        clip.get("content_type", ""),
        clip.get("title", ""),
        clip.get("hook", ""),
        clip.get("reason", ""),
    ]:
        if isinstance(value, dict):
            value = value.get("text", "") or value.get("type", "")
        tags.update(_tokens(value))

    if hook:
        tags.update(_tokens(hook.get("type", "")))
        tags.update(_tokens(hook.get("text", "")))

    expanded = set(tags)
    for group, vocabulary in SEMANTIC_GROUPS.items():
        if tags & vocabulary:
            expanded.add(group)

    return expanded


def classify_intro(clip_tags: set[str]) -> str:
    if clip_tags & SEMANTIC_GROUPS["emotional"]:
        return "emotional"
    if clip_tags & SEMANTIC_GROUPS["funny"]:
        return "funny"
    if clip_tags & SEMANTIC_GROUPS["gaming"]:
        return "gaming"
    if clip_tags & SEMANTIC_GROUPS["surprise"]:
        return "surprise"
    if clip_tags & SEMANTIC_GROUPS["storytelling"]:
        return "storytelling"
    return "general"


def _score_sound(path: Path, clip_tags: set[str], category: str) -> int:
    tags = _sound_tags(path)
    score = len(tags & clip_tags) * 5

    if category == "surprise":
        if "whoosh" in tags:
            score += 7
        if tags & {"impact", "hit", "boom"}:
            score += 7
        if tags & {"reveal", "surprise", "twist"}:
            score += 9

    elif category == "gaming":
        if tags & {"impact", "hit", "slam", "boom"}:
            score += 10
        if tags & {"gaming", "clutch", "win", "victory"}:
            score += 8

    elif category == "storytelling":
        if "whoosh" in tags:
            score += 8
        if tags & SOFT_TAGS:
            score += 8
        if tags & AGGRESSIVE_TAGS:
            score -= 8

    elif category == "emotional":
        if tags & AGGRESSIVE_TAGS:
            score -= 20
        if tags & SOFT_TAGS:
            score += 6
        if tags & {"emotional", "heartfelt", "hope", "cinematic"}:
            score += 7

    elif category == "funny":
        if tags & {"pop", "whoosh", "meme", "funny", "comedy"}:
            score += 9

    else:
        if "whoosh" in tags:
            score += 3

    return score


def _choose_ranked(
    sounds: Iterable[Path],
    clip_tags: set[str],
    category: str,
    required_tags: set[str] | None = None,
    forbidden_tags: set[str] | None = None,
) -> Path | None:
    required_tags = required_tags or set()
    forbidden_tags = forbidden_tags or set()

    ranked = []
    for path in sounds:
        tags = _sound_tags(path)
        if required_tags and not (tags & required_tags):
            continue
        if forbidden_tags and (tags & forbidden_tags):
            continue
        ranked.append((_score_sound(path, clip_tags, category), path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_score = ranked[0][0]

    # Dacă nu există nicio potrivire semantică, fallback random dintre sunetele permise.
    if best_score <= 0:
        pool = [path for _, path in ranked]
    else:
        top_k = max(1, int(INTRO_SFX_RANDOM_TOP_K))
        pool = [path for score, path in ranked[:top_k] if score > 0]

    return random.choice(pool) if pool else None


def select_intro_sfx(clip: dict, hook: dict | None = None) -> list[dict]:
    sounds = discover_sfx()
    if not sounds:
        return []

    clip_tags = build_clip_tags(clip, hook)
    category = classify_intro(clip_tags)
    selections: list[dict] = []

    if category == "emotional":
        # Emotional: fără efect agresiv. Dacă nu există un sunet clar soft/emotional,
        # preferăm liniștea în locul unui fallback nepotrivit.
        path = _choose_ranked(
            sounds,
            clip_tags,
            category,
            forbidden_tags=AGGRESSIVE_TAGS,
        )
        if path and _score_sound(path, clip_tags, category) > 0:
            selections.append({"path": path, "delay": 0.0, "category": category})
        return selections

    if category == "surprise":
        whoosh = _choose_ranked(sounds, clip_tags, category, required_tags={"whoosh", "swish", "swoosh"})
        impact = _choose_ranked(sounds, clip_tags, category, required_tags={"impact", "hit", "boom", "slam"})
        if whoosh:
            selections.append({"path": whoosh, "delay": 0.0, "category": category})
        if impact and impact != whoosh:
            selections.append({"path": impact, "delay": 0.12, "category": category})
        if selections:
            return selections

    if category == "gaming":
        path = _choose_ranked(
            sounds,
            clip_tags,
            category,
            required_tags={"impact", "hit", "boom", "slam"},
        )
        if path:
            return [{"path": path, "delay": 0.0, "category": category}]

    if category == "storytelling":
        path = _choose_ranked(
            sounds,
            clip_tags,
            category,
            required_tags={"whoosh", "swish", "swoosh"},
            forbidden_tags={"hard", "slam", "boom"},
        )
        if path:
            return [{"path": path, "delay": 0.0, "category": category}]

    if category == "funny":
        path = _choose_ranked(
            sounds,
            clip_tags,
            category,
            required_tags={"pop", "whoosh", "swish", "swoosh", "meme"},
        )
        if path:
            return [{"path": path, "delay": 0.0, "category": category}]

    # General fallback: random inteligent dintre cele mai bine potrivite.
    path = _choose_ranked(sounds, clip_tags, category)
    return [{"path": path, "delay": 0.0, "category": category}] if path else []
