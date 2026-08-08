from pathlib import Path
import json
import sys
import time

import ollama

from src.config import HIGHLIGHTS_DIR, TEMP_DIR, OLLAMA_MODEL
from src.logger import info, success


MIN_SCORE = 58
MIN_DISCOVERY_DURATION = 6.0
TARGET_MIN_DURATION = 12.0
MAX_DISCOVERY_DURATION = 60.0
MAX_BOUNDARY_SNAP_SECONDS = 1.0
DISCOVERY_NUM_CTX = 4096
DISCOVERY_NUM_PREDICT = 550
OLLAMA_KEEP_ALIVE = "30m"

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


def _flatten_words(window):
    words = []
    window_start = float(window["start"])
    window_end = float(window["end"])

    for segment in window.get("segments", []):
        for word in segment.get("words", []):
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end < window_start or start > window_end:
                continue
            words.append({"start": start, "end": end})

    return words


def _snap(value, boundaries):
    if not boundaries:
        return value
    nearest = min(boundaries, key=lambda item: abs(item - value))
    if abs(nearest - value) <= MAX_BOUNDARY_SNAP_SECONDS:
        return nearest
    return value


def _expand_short_candidate(start, end, window_start, window_end):
    """Extinde un nucleu scurt fără să iasă din fereastra analizată."""
    duration = end - start
    if duration >= TARGET_MIN_DURATION:
        return start, end

    missing = TARGET_MIN_DURATION - duration
    before = min(missing / 2.0, start - window_start)
    after = min(missing - before, window_end - end)

    start -= before
    end += after

    # Dacă una dintre laturi nu avea suficient spațiu, completăm din cealaltă.
    remaining = TARGET_MIN_DURATION - (end - start)
    if remaining > 0:
        extra_before = min(remaining, start - window_start)
        start -= extra_before
        remaining -= extra_before
    if remaining > 0:
        extra_after = min(remaining, window_end - end)
        end += extra_after

    return start, end


def _normalize_candidate(result, window):
    if not isinstance(result, dict):
        return None, "candidate_not_object"

    window_start = float(window["start"])
    window_end = float(window["end"])

    try:
        start = float(result.get("start"))
        end = float(result.get("end"))
    except (TypeError, ValueError):
        return None, "invalid_timestamps"

    start = max(window_start, min(window_end, start))
    end = max(window_start, min(window_end, end))

    raw_duration = end - start
    if raw_duration < MIN_DISCOVERY_DURATION:
        return None, f"too_short_core={raw_duration:.1f}s"
    if raw_duration > MAX_DISCOVERY_DURATION:
        return None, f"too_long={raw_duration:.1f}s"

    # Discovery trebuie să aibă recall bun. Dacă modelul găsește un moment
    # concentrat de 6-11s, îi adăugăm context până la ~12s și lăsăm retention
    # să decidă editarea finală.
    start, end = _expand_short_candidate(start, end, window_start, window_end)

    words = _flatten_words(window)
    start = _snap(start, [item["start"] for item in words])
    end = _snap(end, [item["end"] for item in words])

    duration = end - start
    if duration < MIN_DISCOVERY_DURATION or duration > MAX_DISCOVERY_DURATION:
        return None, f"duration_after_snap={duration:.1f}s"

    raw_scores = result.get("scores", {})
    if not isinstance(raw_scores, dict):
        raw_scores = {}

    scores = {key: _clamp(raw_scores.get(key, 0)) for key in CANDIDATE_WEIGHTS}
    score = _candidate_score(scores)

    if score < MIN_SCORE:
        return None, f"score={score}<{MIN_SCORE}"

    return {
        "title": str(result.get("title", "Untitled")).strip() or "Untitled",
        "scores": scores,
        "score": score,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
    }, None


def analyze_window(window):
    transcript_lines = []
    for segment in window.get("segments", []):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        transcript_lines.append(
            f"[{float(segment['start']):.2f}s-{float(segment['end']):.2f}s] {text}"
        )

    prompt = f"""
You are a FAST candidate-discovery editor for short-form video.
A later retention stage will do the detailed edit and generate the voice-over hook.

Find at most ONE promising Short candidate inside this window.
Discovery should favor RECALL: keep a potentially strong moment even if it still needs trimming or context in the later retention stage.
Choose the useful story range, not automatically the whole window.
A candidate core may be as short as 6 seconds if it contains a strong hook, reveal, reaction, emotional beat, useful fact, funny moment, clutch/fail, or payoff.
Prioritize curiosity, emotion, story progression, payoff potential, standalone context and information density.
Reject only obvious filler, fragmented context and moments with no meaningful hook/payoff potential.
Do not invent anything.

Return ONLY valid JSON.
If there is no promising candidate, return: {{"candidate": null}}
Otherwise return:
{{
  "candidate": {{
    "start": 0.0,
    "end": 0.0,
    "title": "short factual title",
    "scores": {{
      "hook": 0,
      "curiosity": 0,
      "emotion": 0,
      "story": 0,
      "payoff_potential": 0,
      "standalone": 0,
      "information_density": 0
    }}
  }}
}}

WINDOW: {float(window['start']):.2f}s -> {float(window['end']):.2f}s
TRANSCRIPT:
{chr(10).join(transcript_lines)}
""".strip()

    start_time = time.time()
    response = ollama.chat(
        model=OLLAMA_MODEL,
        stream=False,
        think=False,
        keep_alive=OLLAMA_KEEP_ALIVE,
        messages=[
            {"role": "system", "content": "Return only compact valid JSON. Favor recall during discovery; retention will rank quality later."},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={
            "temperature": 0.10,
            "num_ctx": DISCOVERY_NUM_CTX,
            "num_predict": DISCOVERY_NUM_PREDICT,
        },
    )

    elapsed = time.time() - start_time
    load_ms = float(response.get("load_duration", 0) or 0) / 1_000_000
    prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
    output_tokens = int(response.get("eval_count", 0) or 0)
    info(
        f"Ollama răspuns în {elapsed:.2f}s | load={load_ms:.0f}ms | "
        f"in={prompt_tokens} tok | out={output_tokens} tok"
    )

    try:
        payload = json.loads(response["message"]["content"].strip())
    except json.JSONDecodeError:
        return None, "invalid_json"

    raw_candidate = payload.get("candidate") if isinstance(payload, dict) else None
    if raw_candidate is None:
        return None, "model_rejected"
    if not isinstance(raw_candidate, dict):
        return None, "candidate_not_object"

    return _normalize_candidate(raw_candidate, window)


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

    info(f"Analizez {len(windows)} ferestre în modul discovery rapid...")
    started = time.time()
    results = []
    reject_counts = {}

    for index, window in enumerate(windows, start=1):
        info(f"Fereastră {index}/{len(windows)}")
        try:
            result, rejection = analyze_window(window)
            if result is None:
                reason = rejection or "unknown"
                reject_counts[reason] = reject_counts.get(reason, 0) + 1
                info(f"Fără candidat | motiv: {reason}")
                continue
            info(
                f"Candidate score: {result['score']} | "
                f"{result['start']:.2f}s-{result['end']:.2f}s | {result['title']}"
            )
            results.append(result)
        except Exception as exc:
            info(f"Eroare la fereastra {index}: {exc}")

    results.sort(key=lambda item: item["score"], reverse=True)

    final = []
    for clip in results:
        if not any(overlap_ratio(clip, selected) >= 0.60 for selected in final):
            final.append(clip)

    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    output = HIGHLIGHTS_DIR / f"{video_name}.json"
    with open(output, "w", encoding="utf-8") as file:
        json.dump(final, file, indent=2, ensure_ascii=False)

    if reject_counts:
        summary = ", ".join(f"{key}={value}" for key, value in sorted(reject_counts.items()))
        info(f"Discovery respingeri: {summary}")

    success(f"Au rămas {len(final)} candidați pentru retention optimizer.")
    success(f"Candidate discovery terminat în {time.time() - started:.1f}s.")
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
