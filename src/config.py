from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

# Foldere proiect
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
FINAL_DIR = BASE_DIR / "final"
TRANSCRIPT_DIR = BASE_DIR / "transcript"
SCENES_DIR = BASE_DIR / "scenes"
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
SUBTITLES_DIR = BASE_DIR / "subtitles"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"

# Whisper
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# Ollama
OLLAMA_MODEL = "qwen3:8b"

# Caption Engine
CAPTION_STYLE = "modern"
CAPTION_ANIMATION = "capcut"
FONT_NAME = "Anton"
FONT_SIZE = 92
FONT_COLOR = "&H00FFFFFF"
HIGHLIGHT_COLOR = "&H0000FFFF"
OUTLINE_COLOR = "&H00000000"
BACKGROUND_COLOR = "&H64000000"
OUTLINE = 4
SHADOW = 1
ALIGNMENT = 2
MARGIN_V = 140
MAX_WORDS = 3
PAUSE_THRESHOLD = 0.35
MIN_WORDS = 1

# Rendering
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920

# --------------------------------------------------
# Retention Engine
# --------------------------------------------------
RETENTION_ENABLED = True
RETENTION_CONTEXT_BEFORE = 25.0
RETENTION_CONTEXT_AFTER = 15.0
RETENTION_MAX_CANDIDATES = 8
RETENTION_MAX_VARIANTS = 1
RETENTION_PREFER_ORIGINAL_HOOK = True
RETENTION_MAX_SEGMENTS = 6
RETENTION_MIN_CLIP_DURATION = 12.0
RETENTION_MAX_CLIP_DURATION = 60.0
RETENTION_MIN_FINAL_SCORE = 55

# --------------------------------------------------
# AI Voice-Over Hooks
# --------------------------------------------------
VOICEOVER_ENABLED = True

# Hook generat în aceeași limbă ca sursa; poți forța "ro" sau "en".
HOOK_LANGUAGE = "auto"
VOICEOVER_MAX_HOOK_WORDS = 20
VOICEOVER_MIN_HOOK_SCORE = 60
VOICEOVER_MAX_DURATION = 4.0

# Provider implicit fără dependențe Python suplimentare pe Windows.
# Alternative: "piper" pentru TTS neural local.
VOICEOVER_TTS_PROVIDER = "kokoro"
VOICEOVER_VOICE = ""  # gol = vocea implicită a sistemului/modelului


# Kokoro-82M (implicit)
# "auto" folosește CUDA dacă PyTorch o vede, altfel CPU.
VOICEOVER_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
VOICEOVER_KOKORO_DEVICE = "auto"
VOICEOVER_KOKORO_SPEED = 1.10
VOICEOVER_KOKORO_DEFAULT_LANGUAGE = "en"

# Kokoro nu are momentan suport oficial pentru română.
# Pentru hook-uri RO folosim automat SAPI ca fallback.
VOICEOVER_KOKORO_FALLBACK_PROVIDER = "windows_sapi"

# Voci implicite per limbă. Poți schimba doar EN dacă majoritatea clipurilor sunt englezești.
VOICEOVER_KOKORO_VOICE_EN = "af_heart"
VOICEOVER_KOKORO_VOICE_EN_GB = "bf_emma"
VOICEOVER_KOKORO_VOICE_ES = "ef_dora"
VOICEOVER_KOKORO_VOICE_FR = "ff_siwis"
VOICEOVER_KOKORO_VOICE_HI = "hf_alpha"
VOICEOVER_KOKORO_VOICE_IT = "if_sara"
VOICEOVER_KOKORO_VOICE_PT_BR = "pf_dora"
VOICEOVER_KOKORO_VOICE_JA = "jf_alpha"
VOICEOVER_KOKORO_VOICE_ZH = "zf_xiaobei"

# Windows SAPI
VOICEOVER_SAPI_RATE = 1       # -10 .. 10
VOICEOVER_SAPI_VOLUME = 100   # 0 .. 100

# Piper (opțional): setează căile către modelele .onnx dacă alegi provider="piper".
VOICEOVER_PIPER_EXECUTABLE = "piper"
VOICEOVER_PIPER_MODEL_EN = ""
VOICEOVER_PIPER_MODEL_RO = ""

# Audio în timpul hook-ului.
VOICEOVER_DUCKING_VOLUME = 0.12
VOICEOVER_AUDIO_FADE = 0.20
