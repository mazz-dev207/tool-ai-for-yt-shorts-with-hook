from __future__ import annotations

import json
import os
from pathlib import Path

from src.config import (
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    TEMP_DIR,
    VOICEOVER_ENABLED,
    VOICEOVER_TTS_PROVIDER,
    VOICEOVER_VOICE,
    VOICEOVER_MAX_DURATION,
    VOICEOVER_MAX_HOOK_WORDS,
    VOICEOVER_DUCKING_VOLUME,
    VOICEOVER_AUDIO_FADE,
)
from src.logger import info, success, warning
from src.retention.hooks import select_voiceover_hook
from src.voiceover.mixer import probe_video, choose_teaser_range, mix_voiceover_hook
from src.voiceover.sfx import select_intro_sfx
from src.voiceover.tts import create_tts_provider, fit_audio_to_max_duration
from src.voiceover.base import approximate_word_timings


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _update_debug(video_name: str, index: int, voiceover: dict):
    path = HIGHLIGHTS_DIR / "debug" / video_name / f"clip_{index}.json"
    if not path.exists():
        return

    try:
        payload = _load_json(path)
        payload["voiceover"] = voiceover
        if isinstance(payload.get("selected"), dict):
            payload["selected"]["voiceover"] = voiceover
        _save_json(path, payload)
    except Exception:
        return


def apply_voiceover_hooks(video_name: str):
    highlights_path = HIGHLIGHTS_DIR / f"{video_name}.json"
    clips = _load_json(highlights_path)

    if not VOICEOVER_ENABLED:
        info("AI Voice-Over Hooks sunt dezactivate.")
        return highlights_path

    provider = create_tts_provider(VOICEOVER_TTS_PROVIDER)
    voice_dir = TEMP_DIR / "voiceover"
    voice_dir.mkdir(parents=True, exist_ok=True)

    applied = 0

    for index, clip in enumerate(clips, start=1):
        video_path = OUTPUT_DIR / f"clip_{index}.mp4"
        if not video_path.exists():
            warning(f"Voice-over clip {index}: lipsește {video_path.name}")
            continue

        hook = clip.get("voiceover_hook")
        if not hook:
            hook = select_voiceover_hook(
                clip.get("hook_variants", []),
                max_words=VOICEOVER_MAX_HOOK_WORDS,
            )

        if not hook:
            metadata = {
                "enabled": True,
                "applied": False,
                "reason": "no_valid_generated_hook",
            }
            clip["voiceover"] = metadata
            _update_debug(video_name, int(clip.get("debug_index", index)), metadata)
            warning(f"Voice-over clip {index}: nu există hook generat valid.")
            continue

        text = str(hook.get("text", "")).strip()
        language = str(hook.get("language", "auto") or "auto")

        try:
            tts_path = voice_dir / f"clip_{index}_hook.wav"
            info(f"Voice-over {index}: generez TTS ({provider.name})")

            tts = provider.generate(
                text=text,
                output_path=tts_path,
                language=language,
                voice=VOICEOVER_VOICE,
            )

            fitted_duration = fit_audio_to_max_duration(
                tts.audio_path,
                VOICEOVER_MAX_DURATION,
            )
            tts.duration = fitted_duration
            tts.word_timings = approximate_word_timings(text, fitted_duration)

            video_info = probe_video(video_path)
            teaser_start, teaser_end, teaser_source = choose_teaser_range(
                clip,
                video_info["duration"],
                fitted_duration,
            )

            selected_sfx = select_intro_sfx(clip, hook)
            if selected_sfx:
                names = " + ".join(item["name"] for item in selected_sfx)
                info(f"Intro SFX {index}: {names}")
            else:
                info(f"Intro SFX {index}: niciun efect potrivit / folder gol")

            mixed_path = voice_dir / f"clip_{index}_hooked.mp4"

            try:
                mix_voiceover_hook(
                    video_path=video_path,
                    tts_audio_path=tts.audio_path,
                    output_path=mixed_path,
                    hook_duration=fitted_duration,
                    teaser_start=teaser_start,
                    teaser_end=teaser_end,
                    ducking_volume=VOICEOVER_DUCKING_VOLUME,
                    fade_duration=VOICEOVER_AUDIO_FADE,
                    intro_sfx=selected_sfx,
                )
            except Exception as sfx_exc:
                if not selected_sfx:
                    raise
                warning(
                    f"Intro SFX clip {index} a eșuat ({sfx_exc}); "
                    "reîncerc hook-ul fără SFX."
                )
                selected_sfx = []
                mix_voiceover_hook(
                    video_path=video_path,
                    tts_audio_path=tts.audio_path,
                    output_path=mixed_path,
                    hook_duration=fitted_duration,
                    teaser_start=teaser_start,
                    teaser_end=teaser_end,
                    ducking_volume=VOICEOVER_DUCKING_VOLUME,
                    fade_duration=VOICEOVER_AUDIO_FADE,
                    intro_sfx=[],
                )

            os.replace(mixed_path, video_path)

            sfx_metadata = [
                {
                    "name": item["name"],
                    "effect": item["effect"],
                    "categories": item.get("categories", []),
                }
                for item in selected_sfx
            ]

            metadata = {
                "enabled": True,
                "applied": True,
                "text": text,
                "type": hook.get("type", "unknown"),
                "score": int(hook.get("score", 0) or 0),
                "language": language,
                "duration": round(fitted_duration, 3),
                "provider": tts.provider,
                "voice": tts.voice,
                "word_timings": tts.word_timings,
                "intro_sfx": sfx_metadata,
                "teaser": {
                    "start": round(teaser_start, 3),
                    "end": round(teaser_end, 3),
                    "source": teaser_source,
                },
                "transition": {
                    "original_audio_start": round(fitted_duration, 3),
                    "original_audio_muted_during_hook": True,
                    "audio_ducking": False,
                    "ducking_volume": 0.0,
                    "mode": "tts_with_optional_sfx_then_original_fade_in",
                    "fade_duration": VOICEOVER_AUDIO_FADE,
                },
            }

            clip["voiceover"] = metadata
            _update_debug(video_name, int(clip.get("debug_index", index)), metadata)
            applied += 1
            success(
                f"Voice-over {index}: {fitted_duration:.2f}s | "
                f"hook score {metadata['score']}/100 | SFX={len(sfx_metadata)}"
            )

        except Exception as exc:
            metadata = {
                "enabled": True,
                "applied": False,
                "reason": str(exc),
            }
            clip["voiceover"] = metadata
            _update_debug(video_name, int(clip.get("debug_index", index)), metadata)
            warning(f"Voice-over clip {index} a eșuat: {exc}")

    _save_json(highlights_path, clips)
    success(f"AI Voice-Over aplicat pe {applied}/{len(clips)} clipuri.")
    return highlights_path
