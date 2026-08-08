from __future__ import annotations

import json
import time

import ollama

from src.config import OLLAMA_MODEL, HOOK_LANGUAGE, VOICEOVER_MAX_HOOK_WORDS
from src.logger import info
from src.retention.context import format_context_for_llm


CONTENT_TYPES = (
    "podcast, interview, gaming, challenge, entertainment, livestream, commentary, "
    "documentary, educational, reaction, storytelling, general"
)


class RetentionAnalyzer:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def analyze(self, context: dict, pacing: dict, max_variants: int = 3) -> dict:
        transcript_text = format_context_for_llm(context)

        language_rule = (
            "Generate generated hooks in the SAME language as the source material."
            if HOOK_LANGUAGE == "auto"
            else f"Generate generated hooks in language code '{HOOK_LANGUAGE}'."
        )

        prompt = f"""
You are the retention editor for a short-form video system.
Your job is NOT to invent a story. Your job is to build the strongest truthful Short possible from the supplied transcript.

NON-NEGOTIABLE RULES:
- Never invent people, money, numbers, dates, quotes, events, consequences, stakes, motives or context.
- Every factual claim in a hook must be supported by the supplied transcript/context.
- Every GENERATED hook MUST include evidence with one or more timestamp ranges supporting its factual claim.
- Extractive variants may ONLY use source timestamp ranges that exist in the supplied transcript.
- Prefer original speech/audio for the main story.
- Avoid cuts in the middle of an idea.
- Avoid robotic edits and unnatural sentence combinations.
- The final story should progress: HOOK -> MINIMAL CONTEXT -> ESCALATION -> PAYOFF when the material supports it.
- Do not force a fixed duration. Prefer roughly 15-60 seconds depending on the idea.
- Use timestamps only from the transcript below.
- {language_rule}
- Generated voice-over hooks must generally be 1-3 seconds and no more than {VOICEOVER_MAX_HOOK_WORDS} words.
- Avoid generic hooks such as: "Did you know", "In this video", "Watch until the end", "You won't believe", greetings, or empty clickbait.
- Adapt priorities to content type: podcast/interview -> strong statements, stories, revelations; gaming -> clutch/fail/win/rare events/reactions; challenge/entertainment -> stakes, progress, eliminations, twists, results; educational -> surprising facts, clear explanations, myths, consequences; storytelling -> conflict, mystery, escalation, twist and payoff.

CONTENT TYPE must be one of: {CONTENT_TYPES}.

JSON SHAPE RULES:
- Arrays that are documented as arrays of objects MUST contain JSON objects only.
- Never return a plain string inside hook_variants, variants, retention_anchors,
  retention_risks, open_loops or pattern_interrupts.
- If no valid item exists for one of those fields, return an empty array [].

Return ONE valid JSON object with these keys:
1. content_type: string
2. summary: short factual summary
3. scores: object with integer 0-100 fields: hook, curiosity, emotion, conflict, payoff, information_density, pacing, standalone
4. retention_anchors: array of objects with start, end, type, importance (0-100), reason
5. retention_risks: array of objects with start, end, reason, severity (0-100)
6. hook_variants: 4-6 objects.
   - At least 3 should have generated=true and be suitable for a TTS voice-over intro.
   - Optionally include extractive hooks with generated=false and source_start/source_end.
   - Every item has: text, type, generated, language, evidence, metrics.
   - metrics has integer 0-100 fields: curiosity, clarity, specificity, stakes, surprise, emotional_impact, relevance, naturalness, fit.
   - evidence is an array of objects with start, end, reason.
   - Use hook types such as curiosity, consequence, contrast, surprise, stakes, gaming, interview, storytelling.
7. open_loops: array of truthful open loops already present or naturally implied by the material. Each has start, end, text, strength (0-100). Do not invent artificial promises.
8. pattern_interrupts: array of OPTIONAL visual recommendations. Each has start, end, type (crop_change/zoom/punch_in/speaker_change/broll/text/effect/caption_position), reason, importance (0-100). Only recommend when attention would benefit.
9. variants: up to {max_variants} EXTRACTIVE edit variants. Each object has name, strategy="extractive", rationale, scores (same 8 score fields), and segments.
   Each segment has start, end, role (hook/context/escalation/payoff/bridge/reaction), reason.
   Segments may skip filler, pauses, repetition and tangents. Reordering is allowed only when it sounds natural and preserves meaning.

The candidate starts at {context['candidate_start']:.2f}s and ends at {context['candidate_end']:.2f}s.
The available context starts at {context['start']:.2f}s and ends at {context['end']:.2f}s.

Deterministic pacing metrics for the original candidate:
{json.dumps(pacing, ensure_ascii=False)}

TIMESTAMPED TRANSCRIPT:
{transcript_text}
""".strip()

        started = time.time()
        response = ollama.chat(
            model=self.model,
            stream=False,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only valid JSON. Be conservative. Ground every factual claim in the supplied transcript. "
                        "Do not default all scores to the same value. Generated hooks must be short, truthful and evidence-backed."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            format="json",
            options={"temperature": 0.15, "think": False},
        )

        elapsed = time.time() - started
        info(f"Retention AI răspuns în {elapsed:.2f}s")

        content = response["message"]["content"].strip()
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Retention AI a returnat JSON invalid: {exc}") from exc

        if not isinstance(result, dict):
            raise RuntimeError("Retention AI nu a returnat un obiect JSON.")

        return result
