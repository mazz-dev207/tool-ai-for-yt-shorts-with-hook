import unittest

from src.highlights.profiles import get_profile
from src.highlights.ranking import validate_judgement


class GeminiDurationGuardrailTests(unittest.TestCase):
    def test_too_short_refinement_is_expanded_or_falls_back(self):
        candidate = {"candidate_id": 1, "start": 120.71, "end": 140.0}
        raw = {
            "candidate_id": 1,
            "refined_start": 126.27,
            "refined_end": 135.11,
            "scores": {
                "hook": 15,
                "payoff": 13,
                "emotion": 10,
                "visual_action": 8,
                "surprise": 6,
                "standalone": 7,
                "replayability": 3,
            },
            "category": "general",
            "reason": "test",
            "has_complete_payoff": True,
            "requires_previous_context": False,
            "recommended": True,
        }
        transcript = [
            {"start": 120.7, "end": 125.5, "text": "Setup."},
            {"start": 125.5, "end": 130.5, "text": "Core idea."},
            {"start": 130.5, "end": 136.5, "text": "Payoff."},
            {"start": 136.5, "end": 140.0, "text": "Reaction."},
        ]

        result = validate_judgement(
            raw,
            candidate=candidate,
            video_duration=200.0,
            profile=get_profile("general"),
            context_start=112.71,
            context_end=148.0,
            transcript=transcript,
        )

        duration = result["refined_end"] - result["refined_start"]
        self.assertGreaterEqual(duration, 12.0)
        self.assertLessEqual(duration, 60.0)


if __name__ == "__main__":
    unittest.main()
