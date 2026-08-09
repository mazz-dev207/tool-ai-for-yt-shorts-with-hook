from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.highlights.gemini_pipeline as pipeline
from src.highlights.gemini_judge import GeminiUnavailable


class GeminiFallbackTests(unittest.TestCase):
    def test_missing_or_failed_gemini_preserves_legacy_highlights(self):
        legacy = [
            {
                "start": 10.0,
                "end": 25.0,
                "score": 82,
                "title": "legacy candidate",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            highlights = root / "highlights"
            transcript = root / "transcript"
            highlights.mkdir()
            transcript.mkdir()

            (highlights / "video.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            (transcript / "video.json").write_text(
                json.dumps([{"start": 0, "end": 30, "text": "hello"}]),
                encoding="utf-8",
            )

            with (
                patch.object(pipeline, "HIGHLIGHTS_DIR", highlights),
                patch.object(pipeline, "TRANSCRIPT_DIR", transcript),
                patch.object(pipeline, "GEMINI_ENABLED", True),
                patch.object(
                    pipeline,
                    "GeminiHighlightJudge",
                    side_effect=GeminiUnavailable("simulated failure"),
                ),
            ):
                result_path = pipeline.run_gemini_highlight_stage(
                    "video",
                    root / "video.mp4",
                    mode="gemini",
                    content_profile="general",
                )

            self.assertEqual(result_path, highlights / "video.json")
            persisted = json.loads((highlights / "video.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted, legacy)


if __name__ == "__main__":
    unittest.main()
