from pathlib import Path
import shutil
import sys
import time

from src.config import (
    INPUT_DIR,
    TRANSCRIPT_DIR,
    HIGHLIGHTS_DIR,
    OUTPUT_DIR,
    SUBTITLES_DIR,
    FINAL_DIR,
    TEMP_DIR,
    RETENTION_ENABLED,
    VOICEOVER_ENABLED,
)
from src.logger import info, success
from src.transcribe import transcribe
from src.chunk_transcript import chunk_transcript
from src.highlight_selector import select_highlights
from src.retention.optimizer import optimize_retention
from src.cut import cut
from src.voiceover.manager import apply_voiceover_hooks
from src.caption_engine import CaptionEngine
from src.renderer import render


def clean_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def timed_step(func, *args, **kwargs):
    started = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - started


def main():
    if len(sys.argv) < 2:
        print('Utilizare: python main.py "video_name"')
        sys.exit(1)

    video_name = " ".join(sys.argv[1:])
    video_path = INPUT_DIR / f"{video_name}.mp4"

    if not video_path.exists():
        print(f"Fișierul nu există:\n{video_path}")
        sys.exit(1)

    pipeline_started = time.time()
    timings = {}

    info("Curăț fișierele vechi...")
    clean_started = time.time()
    for folder in [
        TRANSCRIPT_DIR,
        HIGHLIGHTS_DIR,
        OUTPUT_DIR,
        SUBTITLES_DIR,
        FINAL_DIR,
        TEMP_DIR,
    ]:
        clean_folder(folder)
    timings["cleanup"] = time.time() - clean_started

    info("1/8 Transcriere...")
    _, timings["transcription"] = timed_step(transcribe, video_path)

    info("2/8 Creare ferestre analiză...")
    _, timings["chunking"] = timed_step(chunk_transcript, video_name)

    info("3/8 Candidate discovery...")
    _, timings["candidate_discovery"] = timed_step(select_highlights, video_name)

    if RETENTION_ENABLED:
        info("4/8 Optimizare pentru retenție + hook-uri...")
        _, timings["retention"] = timed_step(optimize_retention, video_name)
    else:
        info("4/8 Retention engine dezactivat.")
        timings["retention"] = 0.0

    info("5/8 Tăiere / asamblare clipuri...")
    _, timings["cut"] = timed_step(cut, video_name)

    if VOICEOVER_ENABLED:
        info("6/8 AI Voice-Over Hooks...")
        _, timings["voiceover"] = timed_step(apply_voiceover_hooks, video_name)
    else:
        info("6/8 AI Voice-Over Hooks dezactivate.")
        timings["voiceover"] = 0.0

    info("7/8 Generare subtitrări...")
    caption_engine = CaptionEngine()
    _, timings["captions"] = timed_step(caption_engine.generate, video_name)

    clips = sorted(
        OUTPUT_DIR.glob("clip_*.mp4"),
        key=lambda path: int(path.stem.split("_")[1]),
    )

    if not clips:
        raise RuntimeError("Nu au fost generate clipuri.")

    info(f"8/8 Randare {len(clips)} clipuri...")
    render_started = time.time()
    for clip in clips:
        render(clip.stem)
    timings["render"] = time.time() - render_started

    total_elapsed = time.time() - pipeline_started

    print()
    success("Pipeline terminat cu succes!")
    print()
    print(f"Clipuri generate: {len(clips)}")
    print(f"Rezultate finale: {FINAL_DIR}")
    print()
    print("--- TIMPI PIPELINE ---")
    for key in [
        "cleanup",
        "transcription",
        "chunking",
        "candidate_discovery",
        "retention",
        "cut",
        "voiceover",
        "captions",
        "render",
    ]:
        print(f"{key:20s}: {timings.get(key, 0.0):8.2f}s")
    print(f"{'TOTAL':20s}: {total_elapsed:8.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(f"❌ Eroare: {exc}")
        sys.exit(1)
