from __future__ import annotations

from typing import Dict, List


def _clip_segments(transcript: List[dict], start: float, end: float) -> List[dict]:
    return [
        segment
        for segment in transcript
        if float(segment.get("end", 0.0)) >= start
        and float(segment.get("start", 0.0)) <= end
    ]


def build_context(
    transcript: List[dict],
    candidate: dict,
    before_seconds: float,
    after_seconds: float,
) -> Dict:
    if not transcript:
        raise ValueError("Transcriptul este gol.")

    video_end = float(transcript[-1].get("end", 0.0))
    candidate_start = max(0.0, float(candidate["start"]))
    candidate_end = min(video_end, float(candidate["end"]))

    context_start = max(0.0, candidate_start - before_seconds)
    context_end = min(video_end, candidate_end + after_seconds)

    return {
        "start": context_start,
        "end": context_end,
        "candidate_start": candidate_start,
        "candidate_end": candidate_end,
        "before": _clip_segments(transcript, context_start, candidate_start),
        "candidate": _clip_segments(transcript, candidate_start, candidate_end),
        "after": _clip_segments(transcript, candidate_end, context_end),
        "all": _clip_segments(transcript, context_start, context_end),
    }


def format_context_for_llm(context: Dict, max_chars: int = 16000) -> str:
    sections = []

    for label, key in [
        ("CONTEXT_BEFORE", "before"),
        ("CANDIDATE", "candidate"),
        ("CONTEXT_AFTER", "after"),
    ]:
        lines = [label]
        for segment in context[key]:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            text = str(segment.get("text", "")).strip()
            if text:
                lines.append(f"[{start:.2f}-{end:.2f}] {text}")
        sections.append("\n".join(lines))

    payload = "\n\n".join(sections)
    if len(payload) <= max_chars:
        return payload

    # Candidate-ul este cel mai important. Dacă depășim bugetul,
    # păstrăm începutul și finalul contextului, plus candidatul complet.
    candidate_text = sections[1]
    remaining = max(0, max_chars - len(candidate_text) - 20)
    half = remaining // 2
    before = sections[0][-half:] if half else ""
    after = sections[2][:half] if half else ""
    return f"{before}\n\n{candidate_text}\n\n{after}"[:max_chars]
