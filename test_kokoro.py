from pathlib import Path

from src.voiceover.tts import create_tts_provider


provider = create_tts_provider("kokoro")
output = Path("temp") / "kokoro_test.wav"

result = provider.generate(
    text="Why did he reject a one billion dollar offer?",
    output_path=output,
    language="en",
)

print("OK")
print("Provider:", result.provider)
print("Voice:", result.voice)
print("Duration:", round(result.duration, 2), "s")
print("File:", result.audio_path.resolve())
