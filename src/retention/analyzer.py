from __future__ import annotations

import json
import re
import time

import ollama

from src.config import OLLAMA_MODEL, HOOK_LANGUAGE, VOICEOVER_MAX_HOOK_WORDS
from src.logger import info, warning
from src.retention.context import format_context_for_llm


CONTENT_TYPES = (
    "podcast, interview, gaming, challenge, entertainment, livestream, commentary, "
    "documentary, educational, reaction, storytelling, general"
)

RETENTION_NUM_CTX = 4096
RETENTION_NUM_PREDICT = 1400
OLLAMA_KEEP_ALIVE = "30m"


def parse_retention_json(content: str) -> dict:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])

    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        attempts = [
            candidate,
            re.sub(r",\s*([}\]])", r"\1", candidate),
        ]
        for attempt in attempts:
            try:
                result = json.loads(attempt)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as exc:
                last_error = exc

            try:
                result, _end = decoder.raw_decode(attempt.lstrip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError as exc:
                last_error = exc

    raise RuntimeError(f"Retention AI a returnat JSON invalid: {last_error}")


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def analyze(self, context: dict, pacing: dict, max_variants: int = 1) -> dict:
        transcript_text = format_context_for_llm(context, max_chars=9000)

        language_rule = (
            "Generate hooks in the SAME language as the source material."
            if HOOK_LANGUAGE == "auto"
            else f"Generate hooks in language code '{HOOK_LANGUAGE}'."
        )

        prompt = f"""
You are the retention editor for a short-form video system with optional TTS hooks.
Build the strongest truthful Short from the supplied transcript and generate concise factual hook options.

RULES
- Never invent people, numbers, quotes, events, stakes or context.
- Every generated hook must be directly supported by transcript evidence.
- Extractive edit segments may only use timestamp ranges present below.
- Prefer original speech/audio for the main story.
- Avoid cuts in the middle of an idea.
- Remove filler/repetition/dead time only when the result stays natural.
- Prefer HOOK -> MINIMAL CONTEXT -> ESCALATION -> PAYOFF when supported.
- Keep the final Short understandable standalone.
- Prefer roughly 15-60 seconds.
- {language_rule}
- Generated hooks should be 1-3 seconds and no more than {VOICEOVER_MAX_HOOK_WORDS} words.
- Avoid generic clickbait such as "Did you know", "Watch until the end" or "You won't believe".
- Be concise. Keep the complete JSON comfortably below the output limit.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

Return ONLY one valid JSON object with exactly these keys:
{{
  "content_type": "...",
  "scores": {{
    "hook": 0,
    "curiosity": 0,
    "emotion": 0,
    "conflict": 0,
    "payoff": 0,
    "information_density": 0,
    "pacing": 0,
    "standalone": 0
  }},
  "hook_variants": [
    {{
      "text": "short factual generated hook",
      "type": "curiosity",
      "generated": true,
      "language": "en",
      "score": 0,
      "evidence": [{{"start": 0.0, "end": 0.0, "reason": "brief evidence"}}]
    }},
    {{
      "text": "second short factual generated hook",
      "type": "stakes",
      "generated": true,
      "language": "en",
      "score": 0,
      "evidence": [{{"start": 0.0, "end": 0.0, "reason": "brief evidence"}}]
    }}
  ],
  "variants": [
    {{
      "name": "best_edit",
      "strategy": "extractive",
      "rationale": "one short sentence",
      "scores": {{
        "hook": 0,
        "curiosity": 0,
        "emotion": 0,
        "conflict": 0,
        "payoff": 0,
        "information_density": 0,
        "pacing": 0,
        "standalone": 0
      }},
      "segments": [
        {{"start": 0.0, "end": 0.0, "role": "hook", "reason": "short reason"}}
      ]
    }}
  ]
}}

Return exactly TWO generated hook candidates and at most ONE extractive edit variant.
Use no more than 6 edit segments.
Do not add summaries, anchors, risks, open loops, pattern interrupts, markdown or extra keys.

Candidate: {context['candidate_start']:.2f}s -> {context['candidate_end']:.2f}s
Available context: {context['start']:.2f}s -> {context['end']:.2f}s

Deterministic pacing metrics:
{json.dumps(pacing, ensure_ascii=False, separators=(',', ':'))}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        started = time.time()
        response = ollama.chat(
            model=self.model,
            stream=False,
            think=False,
            keep_alive=OLLAMA_KEEP_ALIVE,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only compact valid JSON. Ground every factual claim and timestamp in the transcript."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={
                "temperature": 0.08,
                "num_ctx": RETENTION_NUM_CTX,
                "num_predict": RETENTION_NUM_PREDICT,
            },
        )

        elapsed = time.time() - started
        load_ms = float(response.get("load_duration", 0) or 0) / 1_000_000
        prompt_tokens = int(response.get("prompt_eval_count", 0) or 0)
        output_tokens = int(response.get("eval_count", 0) or 0)

        info(
            f"Retention AI răspuns în {elapsed:.2f}s | load={load_ms:.0f}ms | "
            f"in={prompt_tokens} tok | out={output_tokens} tok"
        )
        if output_tokens >= RETENTION_NUM_PREDICT - 5:
            warning(
                "[RETENTION] Răspunsul a atins limita de output; "
                "parser-ul va încerca recuperarea JSON-ului."
            )

        result = parse_retention_json(response["message"]["content"])

        result.setdefault("retention_anchors", [])
        result.setdefault("retention_risks", [])
        result.setdefault("open_loops", [])
        result.setdefault("pattern_interrupts", [])
        return result
