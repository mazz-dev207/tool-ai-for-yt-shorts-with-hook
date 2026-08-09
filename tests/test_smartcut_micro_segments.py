import unittest

from src.cut import repair_micro_segments_for_smartcut


class SmartCutMicroSegmentTests(unittest.TestCase):
    def test_short_segment_merges_with_nearest_neighbor(self):
        result = repair_micro_segments_for_smartcut(
            [
                {"start": 260.06, "end": 261.80},
                {"start": 262.76, "end": 263.24},
                {"start": 268.70, "end": 271.24},
            ]
        )
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["start"], 260.06)
        self.assertAlmostEqual(result[0]["end"], 263.24)


if __name__ == "__main__":
    unittest.main()
