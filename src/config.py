from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent


def _load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


_load_local_env()

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
# Gemini Highlight Judge
# --------------------------------------------------
# legacy = numai selectorul actual
# gemini = selector actual -> Gemini multimodal -> rerank/refine -> downstream
# compare = rulează Gemini și salvează raport, dar downstream rămâne pe legacy
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_ENABLED = _env_bool("GEMINI_ENABLED", True)
HIGHLIGHT_MODE = os.getenv("HIGHLIGHT_MODE", "legacy").strip().lower()
CONTENT_PROFILE = os.getenv("CONTENT_PROFILE", "auto").strip().lower()
GEMINI_CONTEXT_BEFORE = _env_float("GEMINI_CONTEXT_BEFORE", 8.0)
GEMINI_CONTEXT_AFTER = _env_float("GEMINI_CONTEXT_AFTER", 8.0)
GEMINI_MAX_CANDIDATES = _env_int("GEMINI_MAX_CANDIDATES", 60)
GEMINI_TOP_HIGHLIGHTS = _env_int("GEMINI_TOP_HIGHLIGHTS", 10)
GEMINI_MIN_SCORE = _env_int("GEMINI_MIN_SCORE", 55)
GEMINI_OVERLAP_THRESHOLD = _env_float("GEMINI_OVERLAP_THRESHOLD", 0.60)
GEMINI_MAX_RETRIES = _env_int("GEMINI_MAX_RETRIES", 2)
GEMINI_PROMPT_VERSION = "gemini-highlight-v1"
GEMINI_CACHE_DIR = BASE_DIR / "cache" / "gemini_highlights"

# --------------------------------------------------
# AI Voice-Over Hooks
# --------------------------------------------------
VOICEOVER_ENABLED = True
HOOK_LANGUAGE = "auto"
VOICEOVER_MAX_HOOK_WORDS = 20
VOICEOVER_MIN_HOOK_SCORE = 60
VOICEOVER_MAX_DURATION = 4.0
VOICEOVER_TTS_PROVIDER = "kokoro"
VOICEOVER_VOICE = ""

# Kokoro-82M
VOICEOVER_KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
VOICEOVER_KOKORO_DEVICE = "auto"
VOICEOVER_KOKORO_SPEED = 1.10
VOICEOVER_KOKORO_DEFAULT_LANGUAGE = "en"
VOICEOVER_KOKORO_FALLBACK_PROVIDER = "windows_sapi"
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
VOICEOVER_SAPI_RATE = 1
VOICEOVER_SAPI_VOLUME = 100

# Piper
VOICEOVER_PIPER_EXECUTABLE = "piper"
VOICEOVER_PIPER_MODEL_EN = ""
VOICEOVER_PIPER_MODEL_RO = ""

# Audio în timpul hook-ului.
VOICEOVER_DUCKING_VOLUME = 0.12
VOICEOVER_AUDIO_FADE = 0.20

# --------------------------------------------------
# Smart Intro SFX
# --------------------------------------------------
INTRO_SFX_ENABLED = True
INTRO_SFX_DIR = BASE_DIR / "assets" / "sfx"
INTRO_SFX_VOLUME = 0.30
INTRO_SFX_RANDOM_TOP_K = 3
INTRO_SFX_IMPACT_DELAY_MS = 120
INTRO_SFX_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
