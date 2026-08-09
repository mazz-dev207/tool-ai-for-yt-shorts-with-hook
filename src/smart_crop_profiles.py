from __future__ import annotations

import statistics
from pathlib import Path

import cv2

from src.config import (
    SMARTCROP_MAX_CROP_VELOCITY,
    SMARTCROP_MOVEMENT_DEAD_ZONE,
    SMARTCROP_SAMPLE_INTERVAL,
    VIDEO_HEIGHT,
)
from src.smart_crop_v2 import FocusPoint, Rect, SmartCropV2Plan, validate_crop_bounds


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _load_face_detector():
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    return None if detector.empty() else detector


_FACE_DETECTOR = _load_face_detector()


def _detect_faces(frame) -> list[tuple[int, int, int, int]]:
    if _FACE_DETECTOR is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    scale = min(1.0, 640.0 / max(1.0, float(width)))
    small = (
        cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else gray
    )
    faces = _FACE_DETECTOR.detectMultiScale(
        small, scaleFactor=1.10, minNeighbors=5, minSize=(32, 32)
    )
    if scale == 1.0:
        return [tuple(map(int, face)) for face in faces]
    return [
        (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
        for x, y, w, h in faces
    ]


def _sample_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 1.0
    samples = []
    position = 0.0
    try:
        while position <= duration + 0.001:
            capture.set(cv2.CAP_PROP_POS_MSEC, position * 1000.0)
            ok, frame = capture.read()
            if not ok:
                break
            samples.append((position, frame, _detect_faces(frame)))
            position += SMARTCROP_SAMPLE_INTERVAL
    finally:
        capture.release()
    return width, height, duration, samples


def _motion_center(previous_gray, current_gray):
    diff = cv2.absdiff(previous_gray, current_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    active = cv2.countNonZero(mask)
    if active < mask.shape[0] * mask.shape[1] * 0.004:
        return None
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _smooth_focus_points(
    points: list[FocusPoint],
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> list[FocusPoint]:
    if not points:
        return []
    result = [points[0]]
    dead_x = crop_width * SMARTCROP_MOVEMENT_DEAD_ZONE
    dead_y = crop_height * SMARTCROP_MOVEMENT_DEAD_ZONE
    max_x = crop_width * SMARTCROP_MAX_CROP_VELOCITY
    max_y = crop_height * SMARTCROP_MAX_CROP_VELOCITY
    for point in points[1:]:
        previous = result[-1]
        dx = point.center_x - previous.center_x
        dy = point.center_y - previous.center_y
        if abs(dx) <= dead_x:
            dx = 0.0
        if abs(dy) <= dead_y:
            dy = 0.0
        dx = _clamp(dx, -max_x, max_x)
        dy = _clamp(dy, -max_y, max_y)
        center_x = previous.center_x + dx * 0.38
        center_y = previous.center_y + dy * 0.38
        center_x = _clamp(center_x, crop_width / 2, frame_width - crop_width / 2)
        center_y = _clamp(center_y, crop_height / 2, frame_height - crop_height / 2)
        result.append(FocusPoint(point.time, center_x, center_y, point.source))
    return result


def build_gameplay_only_plan(video_path: Path, plan: SmartCropV2Plan) -> SmartCropV2Plan:
    width, height, _duration, samples = _sample_video(video_path)
    crop_width = int(plan.legacy_plan.crop_width)
    crop_height = int(plan.legacy_plan.crop_height)
    points: list[FocusPoint] = []
    previous_gray = None
    for time_position, frame, _faces in samples:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = _motion_center(previous_gray, gray) if previous_gray is not None else None
        if motion is None:
            motion = (width / 2, height / 2)
            source = "center"
        else:
            source = "motion"
        points.append(FocusPoint(time_position, motion[0], motion[1], source))
        previous_gray = gray
    if not points:
        points = [FocusPoint(0.0, width / 2, height / 2, "center")]
    plan.mode = "GAMEPLAY_ONLY"
    plan.confidence = max(float(plan.confidence), 0.75)
    plan.gameplay_region = Rect(0, 0, width, height)
    plan.gameplay_crop_width = crop_width
    plan.gameplay_crop_height = crop_height
    plan.gameplay_output_height = VIDEO_HEIGHT
    plan.focus_points = _smooth_focus_points(points, crop_width, crop_height, width, height)
    plan.decision_reason = "gaming profile + landscape source without persistent webcam"
    return plan


def _face_presence(samples) -> float:
    if not samples:
        return 0.0
    return sum(1 for _time, _frame, faces in samples if faces) / len(samples)


def _podcast_regions(samples, width: int, height: int) -> tuple[list[Rect], float]:
    left_boxes = []
    right_boxes = []
    left_hits = 0
    right_hits = 0
    for _time, _frame, faces in samples:
        left_seen = False
        right_seen = False
        for x, y, w, h in faces:
            if x + w / 2 < width / 2:
                left_boxes.append((x, y, w, h))
                left_seen = True
            else:
                right_boxes.append((x, y, w, h))
                right_seen = True
        left_hits += int(left_seen)
        right_hits += int(right_seen)
    total = max(1, len(samples))
    left_persistence = left_hits / total
    right_persistence = right_hits / total
    confidence = min(left_persistence, right_persistence)
    if confidence < 0.30 or not left_boxes or not right_boxes:
        return [], confidence
    left_center = statistics.median([x + w / 2 for x, _y, w, _h in left_boxes])
    right_center = statistics.median([x + w / 2 for x, _y, w, _h in right_boxes])
    split = int(round((left_center + right_center) / 2))
    split = max(int(width * 0.30), min(int(width * 0.70), split))
    left = validate_crop_bounds(Rect(0, 0, split, height), width, height)
    right = validate_crop_bounds(Rect(split, 0, width - split, height), width, height)
    return [left, right], confidence


def upgrade_plan_for_profile(
    video_path: Path,
    plan: SmartCropV2Plan,
    content_profile: str | None,
) -> SmartCropV2Plan:
    profile = str(content_profile or "auto").strip().lower()
    plan.content_profile = profile
    if profile == "gaming":
        if plan.mode == "GAMEPLAY_WEBCAM":
            plan.decision_reason = "persistent webcam detected for gaming profile"
            return plan
        if plan.input_width > plan.input_height:
            return build_gameplay_only_plan(video_path, plan)
        plan.decision_reason = "gaming profile but source is not landscape"
        return plan
    if profile == "podcast":
        width, height, _duration, samples = _sample_video(video_path)
        regions, confidence = _podcast_regions(samples, width, height)
        if len(regions) >= 2:
            plan.mode = "PODCAST_MULTI_SPEAKER"
            plan.confidence = confidence
            plan.speaker_regions = regions
            plan.decision_reason = "podcast profile + two persistent speaker regions"
            return plan
        presence = _face_presence(samples)
        if presence >= 0.45:
            plan.mode = "TALKING_HEAD"
            plan.confidence = max(float(plan.confidence), presence)
            plan.speaker_regions = []
            plan.decision_reason = "podcast profile + one persistent visible speaker"
            return plan
        plan.mode = "GENERAL"
        plan.speaker_regions = []
        plan.decision_reason = "podcast profile but no stable speaker layout"
        return plan
    if profile in {"entertainment", "reaction", "general", "auto"} and plan.mode == "GENERAL":
        try:
            _width, _height, _duration, samples = _sample_video(video_path)
            presence = _face_presence(samples)
            if presence >= 0.60:
                plan.mode = "TALKING_HEAD"
                plan.confidence = max(float(plan.confidence), presence)
                plan.decision_reason = "persistent face detected"
        except Exception:
            pass
    return plan
