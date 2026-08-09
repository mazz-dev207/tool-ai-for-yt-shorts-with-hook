import unittest

from src.smart_cut import (
    WordStamp,
    fallback_to_original_plan,
    merge_nearby_segments,
    refine_edit_plan,
    snap_to_word_boundary,
    validate_final_edit_plan,
    validate_input_segments,
)


class SmartCutTests(unittest.TestCase):
    def test_validation_clamps_sorts_and_merges_overlap(self):
        result = validate_input_segments(
            [
                {"start": 8.0, "end": 12.0},
                {"start": -2.0, "end": 2.0},
                {"start": 11.0, "end": 15.0},
                {"start": 99.0, "end": 120.0},
                {"start": 7.0, "end": 7.0},
            ],
            video_duration=100.0,
        )
        self.assertEqual(result[0], {"start": 0.0, "end": 2.0})
        self.assertEqual(result[1], {"start": 8.0, "end": 15.0})
        self.assertEqual(result[2], {"start": 99.0, "end": 100.0})

    def test_word_boundary_snap_never_cuts_inside_word(self):
        words = [WordStamp("actually", 10.0, 10.6, 0)]
        self.assertEqual(snap_to_word_boundary(10.3, "start", words), 10.0)
        self.assertEqual(snap_to_word_boundary(10.3, "end", words), 10.6)

    def test_nearby_segments_merge(self):
        words = [
            WordStamp("one", 0.1, 0.5, 0),
            WordStamp("two", 2.2, 2.6, 0),
        ]
        result = merge_nearby_segments(
            [
                {"start": 0.0, "end": 2.0},
                {"start": 2.2, "end": 4.0},
            ],
            words,
            minimum_spacing=0.35,
        )
        self.assertEqual(result, [{"start": 0.0, "end": 4.0}])

    def test_missing_transcript_falls_back(self):
        original = [{"start": 5.0, "end": 10.0}]
        result = refine_edit_plan(
            video_name="unit",
            clip_index=1,
            original_segments=original,
            transcript=[],
            video_duration=30.0,
        )
        self.assertEqual(result, original)

    def test_final_plan_validation(self):
        self.assertTrue(
            validate_final_edit_plan(
                [{"start": 1.0, "end": 3.0}, {"start": 4.0, "end": 6.0}],
                10.0,
            )
        )
        self.assertFalse(
            validate_final_edit_plan(
                [{"start": 4.0, "end": 3.0}],
                10.0,
            )
        )

    def test_fallback_repairs_original_plan(self):
        result = fallback_to_original_plan(
            [{"start": -1.0, "end": 2.0}, {"start": 9.0, "end": 15.0}],
            10.0,
        )
        self.assertEqual(result, [{"start": 0.0, "end": 2.0}, {"start": 9.0, "end": 10.0}])


if __name__ == "__main__":
    unittest.main()
