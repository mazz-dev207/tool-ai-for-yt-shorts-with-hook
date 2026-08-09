import unittest

from src.smart_crop import CropPlan
from src.smart_crop_v2 import (
    Rect,
    SmartCropV2Plan,
    infer_persistent_webcam,
    validate_crop_bounds,
    validate_layout_plan,
)


class SmartCropV2Tests(unittest.TestCase):
    def test_validate_crop_bounds_clamps(self):
        rect = validate_crop_bounds(Rect(-10, -20, 5000, 5000), 1920, 1080)
        self.assertEqual(rect.x, 0)
        self.assertEqual(rect.y, 0)
        self.assertEqual(rect.w, 1920)
        self.assertEqual(rect.h, 1080)

    def test_persistent_bottom_right_face_becomes_webcam(self):
        detections = []
        for offset in range(10):
            detections.append([(1600 + offset, 760, 160, 120)])
        rect, confidence, corner = infer_persistent_webcam(detections, 1920, 1080)
        self.assertIsNotNone(rect)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertEqual(corner, "bottom-right")

    def test_transient_face_is_not_webcam(self):
        detections = [[], [], [(1600, 760, 160, 120)], [], [], [], [], [], [], []]
        rect, confidence, _corner = infer_persistent_webcam(detections, 1920, 1080)
        self.assertIsNone(rect)
        self.assertLess(confidence, 0.55)

    def test_gameplay_webcam_layout_dimensions_are_valid(self):
        legacy = CropPlan(
            input_width=1920,
            input_height=1080,
            crop_width=608,
            crop_height=1080,
            duration=30.0,
            points=[],
        )
        plan = SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM",
            input_width=1920,
            input_height=1080,
            duration=30.0,
            legacy_plan=legacy,
            webcam_region=Rect(1500, 700, 360, 300),
            gameplay_region=Rect(0, 0, 1920, 1080),
            gameplay_crop_width=844,
            gameplay_crop_height=1080,
            gameplay_output_height=1382,
            webcam_output_height=538,
        )
        self.assertTrue(validate_layout_plan(plan))
        self.assertEqual(plan.gameplay_output_height + plan.webcam_output_height, 1920)

    def test_missing_webcam_invalidates_gameplay_webcam_plan(self):
        legacy = CropPlan(1920, 1080, 608, 1080, 10.0, [])
        plan = SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM",
            input_width=1920,
            input_height=1080,
            duration=10.0,
            legacy_plan=legacy,
            gameplay_crop_width=844,
            gameplay_crop_height=1080,
            gameplay_output_height=1382,
            webcam_output_height=538,
        )
        self.assertFalse(validate_layout_plan(plan))


if __name__ == "__main__":
    unittest.main()
