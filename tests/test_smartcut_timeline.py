import unittest

from src.retention.timeline import extract_remapped_words


class SmartCutTimelineTests(unittest.TestCase):
    def test_words_remap_across_concatenated_segments(self):
        transcript = [
            {
                "start": 10.0,
                "end": 12.0,
                "words": [
                    {"word": "first", "start": 10.5, "end": 10.9},
                ],
            },
            {
                "start": 20.0,
                "end": 22.0,
                "words": [
                    {"word": "second", "start": 20.5, "end": 20.9},
                ],
            },
        ]
        segments = [
            {"start": 10.0, "end": 12.0},
            {"start": 20.0, "end": 22.0},
        ]

        words = extract_remapped_words(transcript, segments)

        self.assertEqual(len(words), 2)
        self.assertAlmostEqual(words[0]["start"], 0.5, places=3)
        self.assertAlmostEqual(words[1]["start"], 2.5, places=3)
        self.assertEqual(words[0]["word"], "first")
        self.assertEqual(words[1]["word"], "second")


if __name__ == "__main__":
    unittest.main()
