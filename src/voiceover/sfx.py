from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List

from src.config import (
    INTRO_SFX_DIR,
    INTRO_SFX_ENABLED,
    INTRO_SFX_EXTENSIONS,
    INTRO_SFX_RANDOM_TOP_K,
)


CATEGORY_TAGS = {
    "surprise": {"surprise", "reveal", "twist", "shock", "unexpected"},
    "gaming": {"gaming", "clutch", "win", "victory", "kill", "ace", "fail"},
    "storytelling": {"story", "storytelling", "narrative", "cinematic"},
    "emotional": {"emotional", "emotion", "sad", "heartfelt", "hope", "touching"},
    "funny": {"funny", "comedy", "meme", "joke", "laugh", "lol"},
}

EFFECT_TAGS = {
    "whoosh": {"whoosh", "swoosh", "swish"},
    "impact": {"impact", "hit", "boom", "slam", "bass"},
    "pop": {"pop", "click", "pluck"},
}

AGGRESSIVE_TAGS = {"impact", "hit", "boom", "slam", "bass", "hard", "heavy", "aggressive"}
SOFT_TAGS = {"soft", "subtle", "light", "gentle", "smooth"}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(text or "").lower())
        if token
    }


def _file_tags(path: Path) -> set[str]:
    return _tokens(path.stem)


def _context_tags(clip: dict, hook: dict) -> set[str]:
    pieces = [
        clip.get("content_type", ""),
        clip.get("title", ""),
        clip.get("reason", ""),
        hook.get("type", "") if isinstance(hook, dict) else "",
        hook.get("text", "") if isinstance(hook, dict) else "",
    ]
    return _tokens(" ".join(str(piece or "") for piece in pieces))


def classify_context(clip: dict, hook: dict) -> set[str]:
    tags = _context_tags(clip, hook)
    categories = set()

    for category, keywords in CATEGORY_TAGS.items():
        if tags & keywords:
            categories.add(category)

    content_type = str(clip.get("content_type", "")).lower()
    hook_type = str((hook or {}).get("type", "")).lower() if isinstance(hook, dict) else ""

    if content_type in {"gaming", "storytelling"}:
        categories.add(content_type)
    if hook_type in {"surprise", "stakes", "gaming", "storytelling"}:
        categories.add("surprise" if hook_type == "surprise" else hook_type)

    return categories


def desired_effects(categories: set[str]) -> List[str]:
    # Emotional content intentionally avoids aggressive intro effects.
    if "emotional" in categories and not ({"surprise", "gaming"} & categories):
        return ["whoosh_soft"]

    if "surprise" in categories:
        return ["whoosh", "impact"]
    if "gaming" in categories:
        return ["impact"]
    if "funny" in categories:
        return ["pop_or_whoosh"]
    if "storytelling" in categories:
        return ["whoosh_soft"]

    return ["whoosh"]


def _effect_matches(file_tags: set[str], effect: str) -> bool:
    if effect == "whoosh_soft":
        return bool(file_tags & EFFECT_TAGS["whoosh"]) and bool(file_tags & SOFT_TAGS)
    if effect == "pop_or_whoosh":
        return bool(file_tags & (EFFECT_TAGS["pop"] | EFFECT_TAGS["whoosh"]))
    return bool(file_tags & EFFECT_TAGS.get(effect, {effect}))


def _score_file(path: Path, categories: set[str], effect: str) -> int:
    tags = _file_tags(path)

    if not _effect_matches(tags, effect):
        return -999

    score = 10

    for category in categories:
        score += 5 * len(tags & CATEGORY_TAGS.get(category, set()))
        if category in tags:
            score += 6

    if effect == "whoosh_soft":
        score += 6 * len(tags & SOFT_TAGS)

    if "emotional" in categories and tags & AGGRESSIVE_TAGS:
        score -= 30

    return score


def _available_files() -> List[Path]:
    if not INTRO_SFX_ENABLED:
        return []

    folder = Path(INTRO_SFX_DIR)
    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in INTRO_SFX_EXTENSIONS
    )


def _choose_for_effect(files: List[Path], categories: set[str], effect: str, used: set[Path]):
    ranked = []
    for path in files:
        if path in used:
            continue
        score = _score_file(path, categories, effect)
        if score > -999:
            ranked.append((score, path))

    if not ranked:
        return None

    ranked.sort(key=lambda item: (-item[0], item[1].name.lower()))
    best_score = ranked[0][0]

    # Random only among strong matches close to the best score.
    shortlist = [
        path
        for score, path in ranked
        if score >= best_score - 4
    ][:max(1, int(INTRO_SFX_RANDOM_TOP_K))]

    return random.choice(shortlist)


def select_intro_sfx(clip: dict, hook: dict) -> List[dict]:
    files = _available_files()
    if not files:
        return []

    categories = classify_context(clip, hook)
    effects = desired_effects(categories)

    selected = []
    used = set()

    for effect in effects:
        path = _choose_for_effect(files, categories, effect, used)

        # For a soft storytelling/emotional whoosh, fall back to any whoosh
        # rather than disabling SFX completely when no *_soft_* file exists.
        if path is None and effect == "whoosh_soft":
            path = _choose_for_effect(files, categories, "whoosh", used)

        if path is None:
            continue

        used.add(path)
        selected.append({
            "path": path,
            "name": path.name,
            "effect": effect,
            "categories": sorted(categories),
        })

    return selected
