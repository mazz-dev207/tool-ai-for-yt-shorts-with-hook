from __future__ import annotations

import unittest

from src.highlights.profiles import get_profile
from src.highlights.ranking import (
    diversity_rerank,
    remove_overlaps,
    validate_judgement,
    validate_scores,
)


class GeminiHighlightTests(unittest.TestCase):
    def test_score_validation_clamps_ranges(self):
        scores = validate_scores({
            "hook": 99,
            "payoff": -3,
            "emotion": 15,
            "visual_action": 20,
            "surprise": 10,
            "standalone": 11,
            "replayability": 8,
        })
        self.assertEqual(scores["hook"], 25)
        self.assertEqual(scores["payoff"], 0)
        self.assertEqual(scores["visual_action"], 15)
        self.assertEqual(scores["standalone"], 10)
        self.assertEqual(scores["replayability"], 5)

    def test_boundary_refinement_is_clamped(self):
        candidate = {"candidate_id": 1, "start": 10.0, "end": 30.0}
        raw = {
            "candidate_id": 1,
            "refined_start": -100.0,
            "refined_end": 999.0,
            "scores": {
                "hook": 20,
                "payoff": 15,
                "emotion": 10,
                "visual_action": 10,
                "surprise": 8,
                "standalone": 8,
                "replayability": 4,
            },
            "category": "surprise",
            "reason": "test",
            "has_complete_payoff": True,
            "requires_previous_context": False,
            "recommended": True,
        }
        result = validate_judgement(
            raw,
            candidate=candidate,
            video_duration=60.0,
            profile=get_profile("general"),
            context_start=5.0,
            context_end=40.0,
        )
        self.assertEqual(result["refined_start"], 5.0)
        self.assertEqual(result["refined_end"], 40.0)

    def test_invalid_boundary_falls_back_to_original(self):
        candidate = {"candidate_id": 1, "start": 10.0, "end": 30.0}
        raw = {
            "candidate_id": 1,
            "refined_start": 25.0,
            "refined_end": 20.0,
            "scores": {},
            "category": "general",
            "reason": "test",
            "has_complete_payoff": False,
            "requires_previous_context": True,
            "recommended": False,
        }
        result = validate_judgement(
            raw,
            candidate=candidate,
            video_duration=60.0,
            profile=get_profile("general"),
            context_start=5.0,
            context_end=40.0,
        )
        self.assertEqual(result["refined_start"], 10.0)
        self.assertEqual(result["refined_end"], 30.0)

    def test_overlap_removal_keeps_higher_score(self):
        items = [
            {"start": 10.0, "end": 30.0, "gemini": {"total_score": 91, "category": "clutch"}},
            {"start": 12.0, "end": 32.0, "gemini": {"total_score": 84, "category": "clutch"}},
            {"start": 50.0, "end": 70.0, "gemini": {"total_score": 80, "category": "fail"}},
        ]
        result = remove_overlaps(items, threshold=0.60)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["gemini"]["total_score"], 91)

    def test_diversity_only_breaks_close_scores(self):
        items = [
            {"start": 0, "end": 10, "gemini": {"total_score": 90, "category": "funny_reaction"}},
            {"start": 20, "end": 30, "gemini": {"total_score": 89, "category": "funny_reaction"}},
            {"start": 40, "end": 50, "gemini": {"total_score": 88, "category": "clutch"}},
        ]
        result = diversity_rerank(items, top_k=3, close_score_margin=6)
        self.assertEqual(result[0]["gemini"]["total_score"], 90)
        self.assertEqual(result[1]["gemini"]["category"], "clutch")


if __name__ == "__main__":
    unittest.main()
