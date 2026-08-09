from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from src.config import (
    HIGHLIGHTS_DIR,
    SMARTCROP_DEBUG,
    SMARTCROP_DETECT_GAMEPLAY_WEBCAM,
    SMARTCROP_GAMEPLAY_RATIO,
    SMARTCROP_MAX_CROP_VELOCITY,
    SMARTCROP_MOVEMENT_DEAD_ZONE,
    SMARTCROP_SAMPLE_INTERVAL,
    SMARTCROP_V2_ENABLED,
    SMARTCROP_WEBCAM_CORNER_ZONE,
    SMARTCROP_WEBCAM_MAX_AREA_RATIO,
    SMARTCROP_WEBCAM_MIN_AREA_RATIO,
    SMARTCROP_WEBCAM_PERSISTENCE,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from src.logger import info, warning
from src.smart_crop import CropPlan, analyze_smart_crop


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class FocusPoint:
    time: float
    center_x: float
    center_y: float
    source: str = "motion"


@dataclass
class SmartCropV2Plan:
    mode: str
    input_width: int
    input_height: int
    duration: float
    legacy_plan: CropPlan
    confidence: float = 0.0
    webcam_region: Optional[Rect] = None
    gameplay_region: Optional[Rect] = None
    gameplay_crop_width: int = 0
    gameplay_crop_height: int = 0
    gameplay_output_height: int = 0
    webcam_output_height: int = 0
    focus_points: list[FocusPoint] = field(default_factory=list)
    reaction_events: list[dict] = field(default_factory=list)


def _even(value: float) -> int:
    result = max(2, int(round(value)))
    return result if result % 2 == 0 else result - 1


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def validate_crop_bounds(rect: Rect, width: int, height: int) -> Rect:
    x = max(0, min(int(rect.x), max(0, width - 2)))
    y = max(0, min(int(rect.y), max(0, height - 2)))
    w = max(2, min(int(rect.w), width - x))
    h = max(2, min(int(rect.h), height - y))
    return Rect(x=x, y=y, w=w, h=h)


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
    if scale < 1.0:
        small = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    faces = _FACE_DETECTOR.detectMultiScale(
        small,
        scaleFactor=1.10,
        minNeighbors=5,
        minSize=(32, 32),
    )
    if scale == 1.0:
        return [tuple(map(int, face)) for face in faces]
    return [
        (
            int(x / scale),
            int(y / scale),
            int(w / scale),
            int(h / scale),
        )
        for x, y, w, h in faces
    ]


def _corner_for_box(box: tuple[int, int, int, int], width: int, height: int) -> str | None:
    x, y, w, h = box
    cx = x + w / 2
    cy = y + h / 2
    zone = SMARTCROP_WEBCAM_CORNER_ZONE
    left = cx <= width * zone
    right = cx >= width * (1.0 - zone)
    top = cy <= height * zone
    bottom = cy >= height * (1.0 - zone)
    if top and left:
        return "top-left"
    if top and right:
        return "top-right"
    if bottom and left:
        return "bottom-left"
    if bottom and right:
        return "bottom-right"
    return None


def infer_persistent_webcam(
    detections: list[list[tuple[int, int, int, int]]],
    width: int,
    height: int,
) -> tuple[Rect | None, float, str | None]:
    if not detections:
        return None, 0.0, None

    by_corner: dict[str, list[tuple[int, int, int, int]]] = {}
    frame_hits: dict[str, int] = {}
    frame_area = max(1.0, float(width * height))

    for frame_faces in detections:
        corners_seen: set[str] = set()
        for box in frame_faces:
            x, y, w, h = box
            area_ratio = (w * h) / frame_area
            if not (SMARTCROP_WEBCAM_MIN_AREA_RATIO <= area_ratio <= SMARTCROP_WEBCAM_MAX_AREA_RATIO):
                continue
            corner = _corner_for_box(box, width, height)
            if corner is None:
                continue
            by_corner.setdefault(corner, []).append(box)
            corners_seen.add(corner)
        for corner in corners_seen:
            frame_hits[corner] = frame_hits.get(corner, 0) + 1

    if not frame_hits:
        return None, 0.0, None

    best_corner = max(frame_hits, key=frame_hits.get)
    persistence = frame_hits[best_corner] / max(1, len(detections))
    if persistence < SMARTCROP_WEBCAM_PERSISTENCE:
        return None, persistence, best_corner

    boxes = by_corner.get(best_corner, [])
    if not boxes:
        return None, persistence, best_corner

    centers_x = [x + w / 2 for x, _y, w, _h in boxes]
    centers_y = [y + h / 2 for _x, y, _w, h in boxes]
    if len(centers_x) >= 2:
        normalized_jitter = max(
            statistics.pstdev(centers_x) / max(1.0, width),
            statistics.pstdev(centers_y) / max(1.0, height),
        )
        if normalized_jitter > 0.08:
            return None, persistence * 0.5, best_corner

    median_x = statistics.median([box[0] for box in boxes])
    median_y = statistics.median([box[1] for box in boxes])
    median_w = statistics.median([box[2] for box in boxes])
    median_h = statistics.median([box[3] for box in boxes])

    pad_x = max(8, int(median_w * 0.65))
    pad_y = max(8, int(median_h * 0.85))
    rect = Rect(
        x=int(median_x - pad_x),
        y=int(median_y - pad_y),
        w=int(median_w + pad_x * 2),
        h=int(median_h + pad_y * 2),
    )
    return validate_crop_bounds(rect, width, height), persistence, best_corner


def _calculate_crop_size_for_aspect(width: int, height: int, target_ratio: float) -> tuple[int, int]:
    source_ratio = width / max(1.0, float(height))
    if source_ratio >= target_ratio:
        crop_h = height
        crop_w = _even(height * target_ratio)
    else:
        crop_w = width
        crop_h = _even(width / target_ratio)
    return min(width, crop_w), min(height, crop_h)


def _mask_rect(image, rect: Rect | None):
    if rect is None:
        return image
    x1 = max(0, rect.x)
    y1 = max(0, rect.y)
    x2 = min(image.shape[1], rect.x2)
    y2 = min(image.shape[0], rect.y2)
    if x2 > x1 and y2 > y1:
        image[y1:y2, x1:x2] = 0
    return image


def _motion_center(previous_gray, current_gray, webcam: Rect | None) -> tuple[float, float] | None:
    diff = cv2.absdiff(previous_gray, current_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    mask = _mask_rect(mask, webcam)
    active = cv2.countNonZero(mask)
    if active < mask.shape[0] * mask.shape[1] * 0.004:
        return None
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def smooth_focus_points(
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
        cx = previous.center_x + dx * 0.38
        cy = previous.center_y + dy * 0.38
        cx = _clamp(cx, crop_width / 2, frame_width - crop_width / 2)
        cy = _clamp(cy, crop_height / 2, frame_height - crop_height / 2)
        result.append(FocusPoint(point.time, cx, cy, point.source))
    return result


def _sample_video(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0.0

    samples: list[tuple[float, object, list[tuple[int, int, int, int]]]] = []
    time_position = 0.0
    if duration <= 0:
        duration = 1.0
    try:
        while time_position <= duration + 0.001:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_position * 1000.0)
            ok, frame = capture.read()
            if not ok:
                break
            samples.append((time_position, frame, _detect_faces(frame)))
            time_position += SMARTCROP_SAMPLE_INTERVAL
    finally:
        capture.release()
    return width, height, duration, samples


def detect_content_layout(video_path: Path) -> SmartCropV2Plan:
    legacy = analyze_smart_crop(video_path)
    if not SMARTCROP_V2_ENABLED:
        return SmartCropV2Plan("GENERAL", legacy.input_width, legacy.input_height, legacy.duration, legacy)

    try:
        width, height, duration, samples = _sample_video(video_path)
        if not samples:
            raise RuntimeError("Nu există frame-uri pentru analiza SmartCrop 2.0")

        detections = [faces for _time, _frame, faces in samples]
        webcam, confidence, corner = infer_persistent_webcam(detections, width, height)

        if webcam is None or not SMARTCROP_DETECT_GAMEPLAY_WEBCAM or width <= height:
            return SmartCropV2Plan(
                mode="GENERAL",
                input_width=width,
                input_height=height,
                duration=duration,
                legacy_plan=legacy,
                confidence=confidence,
            )

        gameplay_output_height = _even(VIDEO_HEIGHT * SMARTCROP_GAMEPLAY_RATIO)
        gameplay_output_height = max(2, min(VIDEO_HEIGHT - 2, gameplay_output_height))
        webcam_output_height = VIDEO_HEIGHT - gameplay_output_height
        gameplay_target_ratio = VIDEO_WIDTH / float(gameplay_output_height)
        crop_w, crop_h = _calculate_crop_size_for_aspect(width, height, gameplay_target_ratio)

        focus_points: list[FocusPoint] = []
        previous_gray = None
        previous_webcam_center = None
        reaction_events: list[dict] = []

        for time_position, frame, faces in samples:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion = _motion_center(previous_gray, gray, webcam) if previous_gray is not None else None
            if motion is None:
                motion = (width / 2, height / 2)
                source = "center"
            else:
                source = "motion"
            focus_points.append(FocusPoint(time_position, motion[0], motion[1], source))

            webcam_faces = []
            for face in faces:
                x, y, w, h = face
                cx = x + w / 2
                cy = y + h / 2
                if webcam.x <= cx <= webcam.x2 and webcam.y <= cy <= webcam.y2:
                    webcam_faces.append(face)
            if webcam_faces:
                face = max(webcam_faces, key=lambda box: box[2] * box[3])
                center = (face[0] + face[2] / 2, face[1] + face[3] / 2)
                if previous_webcam_center is not None:
                    movement = ((center[0] - previous_webcam_center[0]) ** 2 + (center[1] - previous_webcam_center[1]) ** 2) ** 0.5
                    if movement > max(face[2], face[3]) * 0.45:
                        reaction_events.append({"time": round(time_position, 3), "type": "face_motion_spike"})
                previous_webcam_center = center
            previous_gray = gray

        focus_points = smooth_focus_points(focus_points, crop_w, crop_h, width, height)
        gameplay_region = Rect(0, 0, width, height)

        info("[SMARTCROP] Mode detected: GAMEPLAY_WEBCAM")
        info(f"[SMARTCROP] Webcam candidate: {corner} confidence={confidence:.2f}")
        info(f"[SMARTCROP] Webcam bbox: x={webcam.x} y={webcam.y} w={webcam.w} h={webcam.h}")

        return SmartCropV2Plan(
            mode="GAMEPLAY_WEBCAM",
            input_width=width,
            input_height=height,
            duration=duration,
            legacy_plan=legacy,
            confidence=confidence,
            webcam_region=webcam,
            gameplay_region=gameplay_region,
            gameplay_crop_width=crop_w,
            gameplay_crop_height=crop_h,
            gameplay_output_height=gameplay_output_height,
            webcam_output_height=webcam_output_height,
            focus_points=focus_points,
            reaction_events=reaction_events,
        )
    except Exception as exc:
        warning(f"[SMARTCROP] V2 analysis failed: {exc}; fallback la SmartCrop existent.")
        return SmartCropV2Plan(
            mode="GENERAL",
            input_width=legacy.input_width,
            input_height=legacy.input_height,
            duration=legacy.duration,
            legacy_plan=legacy,
        )


def analyze_smart_crop_v2(video_path: Path) -> SmartCropV2Plan:
    return detect_content_layout(Path(video_path))


def _focus_xy(point: FocusPoint, plan: SmartCropV2Plan) -> tuple[float, float]:
    max_x = max(0, plan.input_width - plan.gameplay_crop_width)
    max_y = max(0, plan.input_height - plan.gameplay_crop_height)
    x = _clamp(point.center_x - plan.gameplay_crop_width / 2, 0, max_x)
    y = _clamp(point.center_y - plan.gameplay_crop_height / 2, 0, max_y)
    return x, y


def initial_gameplay_xy(plan: SmartCropV2Plan) -> tuple[float, float]:
    if not plan.focus_points:
        return (
            max(0.0, (plan.input_width - plan.gameplay_crop_width) / 2),
            max(0.0, (plan.input_height - plan.gameplay_crop_height) / 2),
        )
    return _focus_xy(plan.focus_points[0], plan)


def write_gameplay_sendcmd(
    plan: SmartCropV2Plan,
    output_file: Path,
    target_name: str = "gameplay",
) -> Path:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    points = plan.focus_points
    lines: list[str] = []

    if len(points) <= 1:
        x, y = initial_gameplay_xy(plan)
        lines.append(f"0.000 crop@{target_name} x {x:.3f};")
        lines.append(f"0.000 crop@{target_name} y {y:.3f};")
    else:
        for current, nxt in zip(points, points[1:]):
            start = current.time
            end = max(nxt.time, start + 0.001)
            x1, y1 = _focus_xy(current, plan)
            x2, y2 = _focus_xy(nxt, plan)
            lines.append(
                f"{start:.3f}-{end:.3f} [expr] crop@{target_name} x 'lerp({x1:.3f},{x2:.3f},TI)';"
            )
            lines.append(
                f"{start:.3f}-{end:.3f} [expr] crop@{target_name} y 'lerp({y1:.3f},{y2:.3f},TI)';"
            )

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file


def validate_layout_plan(plan: SmartCropV2Plan) -> bool:
    if plan.mode != "GAMEPLAY_WEBCAM":
        return True
    if plan.webcam_region is None:
        return False
    if plan.gameplay_crop_width <= 0 or plan.gameplay_crop_height <= 0:
        return False
    if plan.gameplay_output_height + plan.webcam_output_height != VIDEO_HEIGHT:
        return False
    return True


def save_smartcrop_debug(video_name: str, plan: SmartCropV2Plan) -> None:
    if not SMARTCROP_DEBUG:
        return
    output_dir = HIGHLIGHTS_DIR / "debug" / video_name / "smartcrop"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": plan.mode,
        "confidence": plan.confidence,
        "input": [plan.input_width, plan.input_height],
        "webcam": (
            [plan.webcam_region.x, plan.webcam_region.y, plan.webcam_region.w, plan.webcam_region.h]
            if plan.webcam_region else None
        ),
        "gameplay_crop": [plan.gameplay_crop_width, plan.gameplay_crop_height],
        "layout": {
            "gameplay_height": plan.gameplay_output_height,
            "webcam_height": plan.webcam_output_height,
        },
        "reaction_events": plan.reaction_events,
    }
    (output_dir / "plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
