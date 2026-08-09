import unittest
from unittest.mock import patch

import numpy as np

from src.smart_crop import CropPlan
from src.smart_crop_profiles import _podcast_regions, upgrade_plan_for_profile
from src.smart_crop_v2 import SmartCropV2Plan


class SmartCropProfileTests(unittest.TestCase):
    def test_gaming_profile_without_webcam_becomes_gameplay_only(self):
        legacy = CropPlan(192, 108, 60, 108, 1.0, [])
        plan = SmartCropV2Plan(
            mode="GENERAL",
            input_width=192,
            input_height=108,
            duration=1.0,
            legacy_plan=legacy,
        )
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        samples = [(0.0, frame, []), (0.5, frame.copy(), [])]
        with patch(
            "src.smart_crop_profiles._sample_video",
            return_value=(192, 108, 1.0, samples),
        ):
            result = upgrade_plan_for_profile("dummy.mp4", plan, "gaming")

        self.assertEqual(result.mode, "GAMEPLAY_ONLY")
        self.assertEqual(result.gameplay_crop_width, 60)
        self.assertEqual(result.gameplay_crop_height, 108)
        self.assertTrue(result.focus_points)

    def test_two_persistent_podcast_faces_create_two_regions(self):
        samples = []
        for index in range(10):
            samples.append(
                (
                    index * 0.5,
                    None,
                    [(200, 220, 120, 120), (1500, 220, 120, 120)],
                )
            )

        regions, confidence = _podcast_regions(samples, 1920, 1080)
        self.assertEqual(len(regions), 2)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertLess(regions[0].x, regions[1].x)


if __name__ == "__main__":
    unittest.main()
