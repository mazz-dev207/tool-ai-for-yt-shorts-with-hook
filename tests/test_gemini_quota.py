from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import src.highlights.gemini_pipeline as pipeline
from src.highlights.gemini_judge import (
    GeminiQuotaExhausted,
    _is_daily_quota_exhausted,
    _retry_delay_seconds,
)


class _QuotaJudge:
    def __init__(self):
        self.calls = 0

    def judge(self, **_kwargs):
        self.calls += 1
        raise GeminiQuotaExhausted("Gemini daily quota exhausted for test model")


class GeminiQuotaTests(unittest.TestCase):
    def test_daily_quota_error_is_detected(self):
        exc = RuntimeError(
            "429 RESOURCE_EXHAUSTED quota exceeded "
            "quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier"
        )
        self.assertTrue(_is_daily_quota_exhausted(exc))

    def test_retry_delay_is_extracted_for_temporary_429(self):
        exc = RuntimeError("429 RESOURCE_EXHAUSTED Please retry in 28.448s")
        self.assertAlmostEqual(_retry_delay_seconds(exc, 2.0), 28.448, places=3)

    def test_daily_quota_stops_remaining_candidates(self):
        legacy = [
            {"start": 10.0, "end": 25.0, "score": 90, "title": "one"},
            {"start": 30.0, "end": 45.0, "score": 80, "title": "two"},
            {"start": 50.0, "end": 65.0, "score": 70, "title": "three"},
        ]
        transcript_data = [
            {"start": 0.0, "end": 70.0, "text": "test transcript"}
        ]
        fake_judge = _QuotaJudge()

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
                json.dumps(transcript_data), encoding="utf-8"
            )

            with (
                patch.object(pipeline, "HIGHLIGHTS_DIR", highlights),
                patch.object(pipeline, "TRANSCRIPT_DIR", transcript),
                patch.object(pipeline, "GEMINI_ENABLED", True),
                patch.object(pipeline, "GeminiHighlightJudge", return_value=fake_judge),
                patch.object(pipeline, "probe_duration", return_value=70.0),
            ):
                result_path = pipeline.run_gemini_highlight_stage(
                    "video",
                    root / "video.mp4",
                    mode="compare",
                    content_profile="general",
                )

            self.assertEqual(fake_judge.calls, 1)
            self.assertEqual(result_path, highlights / "video.json")

            persisted = json.loads(
                (highlights / "video.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, legacy)

            report = json.loads(
                (highlights / "video_highlight_compare.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = report["summary"]
            self.assertTrue(summary["quota_exhausted"])
            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["evaluated"], 0)
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["skipped_quota"], 2)
            self.assertEqual(len(report["gemini_skipped_quota"]), 2)


if __name__ == "__main__":
    unittest.main()
