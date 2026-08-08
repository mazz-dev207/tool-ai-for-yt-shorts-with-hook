from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict

from src.config import (
    VOICEOVER_SAPI_RATE,
    VOICEOVER_SAPI_VOLUME,
    VOICEOVER_PIPER_EXECUTABLE,
    VOICEOVER_PIPER_MODEL_EN,
    VOICEOVER_PIPER_MODEL_RO,
    VOICEOVER_KOKORO_REPO_ID,
    VOICEOVER_KOKORO_DEVICE,
    VOICEOVER_KOKORO_SPEED,
    VOICEOVER_KOKORO_DEFAULT_LANGUAGE,
    VOICEOVER_KOKORO_FALLBACK_PROVIDER,
    VOICEOVER_KOKORO_VOICE_EN,
    VOICEOVER_KOKORO_VOICE_EN_GB,
    VOICEOVER_KOKORO_VOICE_ES,
    VOICEOVER_KOKORO_VOICE_FR,
    VOICEOVER_KOKORO_VOICE_HI,
    VOICEOVER_KOKORO_VOICE_IT,
    VOICEOVER_KOKORO_VOICE_PT_BR,
    VOICEOVER_KOKORO_VOICE_JA,
    VOICEOVER_KOKORO_VOICE_ZH,
)
from src.voiceover.base import TTSProvider, TTSResult, approximate_word_timings


KOKORO_SAMPLE_RATE = 24000

# Kokoro v1.0 language codes supported by the official pipeline.
KOKORO_LANGUAGE_CODES: Dict[str, str] = {
    "en": "a",
    "en-us": "a",
    "en_us": "a",
    "en-gb": "b",
    "en_gb": "b",
    "es": "e",
    "fr": "f",
    "fr-fr": "f",
    "hi": "h",
    "it": "i",
    "pt": "p",
    "pt-br": "p",
    "pt_br": "p",
    "ja": "j",
    "jp": "j",
    "zh": "z",
    "zh-cn": "z",
}

KOKORO_DEFAULT_VOICES = {
    "a": VOICEOVER_KOKORO_VOICE_EN,
    "b": VOICEOVER_KOKORO_VOICE_EN_GB,
    "e": VOICEOVER_KOKORO_VOICE_ES,
    "f": VOICEOVER_KOKORO_VOICE_FR,
    "h": VOICEOVER_KOKORO_VOICE_HI,
    "i": VOICEOVER_KOKORO_VOICE_IT,
    "p": VOICEOVER_KOKORO_VOICE_PT_BR,
    "j": VOICEOVER_KOKORO_VOICE_JA,
    "z": VOICEOVER_KOKORO_VOICE_ZH,
}


def _run(command, *, input_text: str | None = None, env=None):
    result = subprocess.run(
        command,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Comanda TTS a eșuat.")

    return result


def probe_audio_duration(path: Path) -> float:
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )

    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError as exc:
        raise RuntimeError(f"Nu pot determina durata TTS pentru {path}") from exc


def _atempo_chain(speed: float) -> str:
    speed = max(0.5, float(speed))
    factors = []

    while speed > 2.0:
        factors.append(2.0)
        speed /= 2.0

    while speed < 0.5:
        factors.append(0.5)
        speed /= 0.5

    factors.append(speed)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def fit_audio_to_max_duration(path: Path, max_duration: float) -> float:
    current = probe_audio_duration(path)
    max_duration = max(0.1, float(max_duration))

    if current <= max_duration + 0.02:
        return current

    speed = current / max_duration
    temp = path.with_name(f"{path.stem}_fit{path.suffix}")

    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-filter:a",
            _atempo_chain(speed),
            str(temp),
        ]
    )

    os.replace(temp, path)
    return probe_audio_duration(path)


class WindowsSapiTTS(TTSProvider):
    name = "windows_sapi"

    def generate(
        self,
        text: str,
        output_path: Path,
        language: str = "auto",
        voice: str = "",
    ) -> TTSResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        powershell = (
            shutil.which("powershell")
            or shutil.which("powershell.exe")
            or shutil.which("pwsh")
        )

        if not powershell:
            raise RuntimeError("Windows PowerShell nu este disponibil pentru provider-ul SAPI.")

        script = r'''
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    if ($env:AI_SHORTS_TTS_VOICE) {
        try { $synth.SelectVoice($env:AI_SHORTS_TTS_VOICE) } catch {}
    }
    elseif ($env:AI_SHORTS_TTS_LANGUAGE -and $env:AI_SHORTS_TTS_LANGUAGE -ne "auto") {
        try {
            $match = $synth.GetInstalledVoices() | Where-Object {
                $_.Enabled -and $_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq $env:AI_SHORTS_TTS_LANGUAGE
            } | Select-Object -First 1
            if ($match) { $synth.SelectVoice($match.VoiceInfo.Name) }
        } catch {}
    }
    $synth.Rate = [Math]::Max(-10, [Math]::Min(10, [int]$env:AI_SHORTS_TTS_RATE))
    $synth.Volume = [Math]::Max(0, [Math]::Min(100, [int]$env:AI_SHORTS_TTS_VOLUME))
    $synth.SetOutputToWaveFile($env:AI_SHORTS_TTS_OUT)
    $synth.Speak($env:AI_SHORTS_TTS_TEXT)
}
finally {
    $synth.Dispose()
}
'''.strip()

        env = os.environ.copy()
        env.update(
            {
                "AI_SHORTS_TTS_TEXT": text,
                "AI_SHORTS_TTS_OUT": str(output_path.resolve()),
                "AI_SHORTS_TTS_VOICE": voice or "",
                "AI_SHORTS_TTS_LANGUAGE": language or "auto",
                "AI_SHORTS_TTS_RATE": str(VOICEOVER_SAPI_RATE),
                "AI_SHORTS_TTS_VOLUME": str(VOICEOVER_SAPI_VOLUME),
            }
        )

        _run(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", script],
            env=env,
        )

        duration = probe_audio_duration(output_path)
        return TTSResult(
            audio_path=output_path,
            duration=duration,
            provider=self.name,
            voice=voice or "system_default",
            language=language,
            word_timings=approximate_word_timings(text, duration),
        )


class KokoroTTS(TTSProvider):
    """
    Kokoro-82M local neural TTS.

    Modelul și vocile sunt descărcate/cached de Hugging Face la prima folosire.
    Același KModel este reutilizat pentru toate limbile, iar pipeline-urile G2P
    sunt cache-uite per limbă.
    """

    name = "kokoro"

    def __init__(self):
        self._model = None
        self._pipelines = {}
        self._device = None

    def _imports(self):
        try:
            import numpy as np
            import soundfile as sf
            import torch
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro nu este instalat. Rulează: "
                "pip install \"kokoro>=0.9.4\" soundfile"
            ) from exc

        return np, sf, torch, KModel, KPipeline

    def _select_device(self, torch):
        configured = str(VOICEOVER_KOKORO_DEVICE or "auto").strip().lower()

        if configured in {"", "auto"}:
            return "cuda" if torch.cuda.is_available() else "cpu"

        if configured == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "VOICEOVER_KOKORO_DEVICE='cuda', dar PyTorch nu vede CUDA. "
                "Folosește 'auto'/'cpu' sau repară instalarea PyTorch CUDA."
            )

        if configured not in {"cuda", "cpu"}:
            raise ValueError(
                f"VOICEOVER_KOKORO_DEVICE invalid: {VOICEOVER_KOKORO_DEVICE}"
            )

        return configured

    def _language_code(self, language: str) -> str | None:
        normalized = str(language or "auto").strip().lower()

        if normalized in {"", "auto"}:
            normalized = str(VOICEOVER_KOKORO_DEFAULT_LANGUAGE or "en").strip().lower()

        return KOKORO_LANGUAGE_CODES.get(normalized)

    def _voice_for(self, lang_code: str, requested_voice: str) -> str:
        requested_voice = str(requested_voice or "").strip()
        if requested_voice:
            return requested_voice

        voice = str(KOKORO_DEFAULT_VOICES.get(lang_code, "") or "").strip()
        if not voice:
            raise RuntimeError(f"Nu există voce Kokoro configurată pentru lang_code='{lang_code}'.")
        return voice

    def _fallback(
        self,
        text: str,
        output_path: Path,
        language: str,
        voice: str,
        reason: str,
    ) -> TTSResult:
        fallback_name = str(VOICEOVER_KOKORO_FALLBACK_PROVIDER or "").strip().lower()

        if not fallback_name or fallback_name in {"none", "off", "disabled"}:
            raise RuntimeError(reason)

        if fallback_name == "kokoro":
            raise RuntimeError(reason)

        fallback = create_tts_provider(fallback_name)
        return fallback.generate(
            text=text,
            output_path=output_path,
            language=language,
            voice="",  # o voce Kokoro nu trebuie trimisă către SAPI/Piper
        )

    def _get_pipeline(self, lang_code: str):
        if lang_code in self._pipelines:
            return self._pipelines[lang_code]

        _np, _sf, torch, KModel, KPipeline = self._imports()

        if self._device is None:
            self._device = self._select_device(torch)

        if self._model is None:
            self._model = (
                KModel(repo_id=VOICEOVER_KOKORO_REPO_ID)
                .to(self._device)
                .eval()
            )

        pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=VOICEOVER_KOKORO_REPO_ID,
            model=self._model,
        )
        self._pipelines[lang_code] = pipeline
        return pipeline

    def generate(
        self,
        text: str,
        output_path: Path,
        language: str = "auto",
        voice: str = "",
    ) -> TTSResult:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text = str(text or "").strip()
        if not text:
            raise ValueError("Textul pentru Kokoro este gol.")

        lang_code = self._language_code(language)
        if lang_code is None:
            return self._fallback(
                text,
                output_path,
                language,
                voice,
                reason=(
                    f"Kokoro nu suportă oficial limba '{language}'. "
                    "Pentru această limbă se folosește fallback-ul configurat."
                ),
            )

        try:
            np, sf, _torch, _KModel, _KPipeline = self._imports()
            pipeline = self._get_pipeline(lang_code)
            selected_voice = self._voice_for(lang_code, voice)

            chunks = []
            generator = pipeline(
                text,
                voice=selected_voice,
                speed=float(VOICEOVER_KOKORO_SPEED),
                split_pattern=r"\n+",
            )

            for result in generator:
                audio = result.audio
                if audio is None:
                    continue

                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()

                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    chunks.append(audio)

            if not chunks:
                raise RuntimeError("Kokoro nu a generat niciun sample audio.")

            waveform = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            sf.write(str(output_path), waveform, KOKORO_SAMPLE_RATE)

            duration = probe_audio_duration(output_path)
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                provider=self.name,
                voice=selected_voice,
                language=language,
                word_timings=approximate_word_timings(text, duration),
            )

        except Exception as exc:
            # Pentru o limbă suportată, o eroare de setup/model trebuie expusă clar.
            # Nu ascundem automat o instalare Kokoro stricată prin fallback.
            raise RuntimeError(f"Kokoro TTS a eșuat: {exc}") from exc


class PiperTTS(TTSProvider):
    name = "piper"

    def _model_for_language(self, language: str) -> Path:
        language = (language or "auto").lower()

        if language == "ro":
            model = VOICEOVER_PIPER_MODEL_RO
        else:
            model = VOICEOVER_PIPER_MODEL_EN

        model_path = Path(model) if model else None
        if not model_path or not model_path.exists():
            raise RuntimeError(
                f"Model Piper lipsă pentru limba '{language}'. Configurează VOICEOVER_PIPER_MODEL_RO/EN."
            )

        return model_path

    def generate(
        self,
        text: str,
        output_path: Path,
        language: str = "auto",
        voice: str = "",
    ) -> TTSResult:
        executable = shutil.which(VOICEOVER_PIPER_EXECUTABLE)
        if not executable:
            raise RuntimeError(
                f"Nu găsesc executabilul Piper: {VOICEOVER_PIPER_EXECUTABLE}"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = self._model_for_language(language)

        _run(
            [
                executable,
                "--model",
                str(model_path),
                "--output_file",
                str(output_path),
            ],
            input_text=text,
        )

        duration = probe_audio_duration(output_path)
        return TTSResult(
            audio_path=output_path,
            duration=duration,
            provider=self.name,
            voice=voice or model_path.stem,
            language=language,
            word_timings=approximate_word_timings(text, duration),
        )


def create_tts_provider(name: str) -> TTSProvider:
    normalized = str(name or "").strip().lower()

    if normalized in {"windows_sapi", "sapi", "windows"}:
        return WindowsSapiTTS()

    if normalized in {"kokoro", "kokoro_local"}:
        return KokoroTTS()

    if normalized == "piper":
        return PiperTTS()

    raise ValueError(f"Provider TTS necunoscut: {name}")
