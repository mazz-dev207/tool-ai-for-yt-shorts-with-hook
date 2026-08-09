import unittest

from src.retention.analyzer import parse_retention_json


class RetentionJsonTests(unittest.TestCase):
    def test_parses_json_code_fence(self):
        result = parse_retention_json('```json\n{"content_type":"gaming"}\n```')
        self.assertEqual(result["content_type"], "gaming")

    def test_repairs_trailing_commas(self):
        result = parse_retention_json(
            '{"content_type":"gaming","scores":{"hook":80,},}'
        )
        self.assertEqual(result["scores"]["hook"], 80)


if __name__ == "__main__":
    unittest.main()
