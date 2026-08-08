from pathlib import Path
import json
import sys
import time

import ollama

from src.config import HIGHLIGHTS_DIR, TEMP_DIR, OLLAMA_MODEL
from src.logger import info, success


MIN_SCORE = 62

CANDIDATE_WEIGHTS = {
    "hook": 0.24,
    "curiosity": 0.18,
    "emotion": 0.12,
    "story": 0.14,
    "payoff_potential": 0.14,
    "standalone": 0.10,
    "information_density": 0.08,
}


def _clamp(value):
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _candidate_score(scores):
    return round(
        sum(_clamp(scores.get(key, 0)) * weight for key, weight in CANDIDATE_WEIGHTS.items())
    )


def analyze_window(window):
    text = ""
    for segment in window["segments"]:
        text += (
            f"[{segment['start']:.1f}s - {segment['end']:.1f}s] "
            f"{segment['text']}\n"
        )

    prompt = f"""
You are a candidate-discovery editor for YouTube Shorts, TikTok and Reels.
This is only a ROUGH discovery pass. A later retention optimizer will do the detailed edit.

Evaluate the window independently. Do not default to the same score for every window.
Return ONLY valid JSON with:
- title: short factual title
- scores: integer 0-100 fields: hook, curiosity, emotion, story, payoff_potential, standalone, information_density
- reason: one short sentence explaining why this window is or is not promising

Scoring guidance:
0-39 weak, 40-59 mediocre, 60-74 usable, 75-89 strong, 90-100 exceptional.
Be conservative. Do not invent facts.

Transcript:\n{text}
""".strip()

    start_time = time.time()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        stream=False,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Score each dimension independently."},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": 0.15, "think": False},
    )

    info(f"Ollama răspuns în {time.time() - start_time:.2f}s")

    try:
        result = json.loads(response["message"]["content"].strip())
    except json.JSONDecodeError:
        result = {"title": "Untitled", "scores": {}, "reason": "invalid_json"}

    scores = result.get("scores", {}) if isinstance(result, dict) else {}
    result = result if isinstance(result, dict) else {}
    result["title"] = str(result.get("title", "Untitled"))
    result["scores"] = {key: _clamp(scores.get(key, 0)) for key in CANDIDATE_WEIGHTS}
    result["score"] = _candidate_score(result["scores"])
    result["start"] = float(window["start"])
    result["end"] = float(window["end"])
    return result


def overlap_ratio(a, b):
    intersection = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    shorter = max(0.001, min(a["end"] - a["start"], b["end"] - b["start"]))
    return intersection / shorter


def select_highlights(video_name):
    chunk_file = TEMP_DIR / f"{video_name}_chunks.json"
    if not chunk_file.exists():
        raise FileNotFoundError(chunk_file)

    with open(chunk_file, encoding="utf-8") as file:
        windows = json.load(file)

    info(f"Analizez {len(windows)} ferestre pentru candidați...")
    results = []

    for index, window in enumerate(windows, start=1):
        info(f"Fereastră {index}/{len(windows)}")
        try:
            result = analyze_window(window)
            info(f"Candidate score: {result['score']} | {result['title']}")
            if result["score"] >= MIN_SCORE:
                results.append(result)
        except Exception as exc:
            info(f"Eroare la fereastra {index}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)

    final = []
    for clip in results:
        # Ferestrele vecine au overlap intenționat. Eliminăm doar candidații
        # aproape duplicat, nu orice intersecție de 15 secunde.
        if not any(overlap_ratio(clip, selected) >= 0.72 for selected in final):
            final.append(clip)

    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output = HIGHLIGHTS_DIR / f"{video_name}.json"

    with open(output, "w", encoding="utf-8") as file:
        json.dump(final, file, indent=2, ensure_ascii=False)

    success(f"Au rămas {len(final)} candidați pentru retention optimizer.")
    success(f"Candidați salvați: {output}")
    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Utilizare: python src/highlight_selector.py "video_name"')
        sys.exit(1)

    try:
        select_highlights(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"Eroare: {exc}")
        sys.exit(1)
