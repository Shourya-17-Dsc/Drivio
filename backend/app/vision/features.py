"""Pure geometry on MediaPipe face-mesh landmarks (pixel coordinates, shape (N, 2))."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

# MediaPipe FaceMesh indices, ordered p1..p6 = outer corner, 2 upper lid, inner corner, 2 lower lid
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)
_POSE_IDX = (1, 152, 33, 263, 61, 291)  # nose, chin, eye corners, mouth corners

_MODEL_POINTS = np.array(
    [(0, 0, 0), (0, -330, -65), (-225, 170, -135), (225, 170, -135), (-150, -150, -125), (150, -150, -125)],
    dtype=np.float64,
)


@dataclass
class FrameFeatures:
    face_present: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze: float = 0.0   # horizontal iris offset, 0 = looking straight ahead
    landmarks: Optional[np.ndarray] = field(default=None, repr=False)
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


def _d(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Roughly 0.3 open, below 0.2 closed."""
    horizontal = _d(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (_d(pts[1], pts[5]) + _d(pts[2], pts[4])) / (2.0 * horizontal)


def mean_ear(lm: np.ndarray) -> float:
    return (eye_aspect_ratio(lm[list(LEFT_EYE)]) + eye_aspect_ratio(lm[list(RIGHT_EYE)])) / 2.0


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    """Inner-lip opening over mouth width, averaged over three vertical pairs."""
    width = _d(lm[78], lm[308])
    if width < 1e-6:
        return 0.0
    vertical = (_d(lm[13], lm[14]) + _d(lm[82], lm[87]) + _d(lm[312], lm[317])) / 3.0
    return vertical / width


def _fold(angle: float) -> float:
    """Fold an angle in degrees into [-90, 90]."""
    angle = (angle + 180.0) % 360.0 - 180.0
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def head_pose(lm: np.ndarray, width: int, height: int) -> Optional[Tuple[float, float, float]]:
    """Return (pitch, yaw, roll) in degrees using solvePnP on six landmarks.

    Absolute values depend on camera placement, so the analyzer subtracts a
    per-driver baseline captured during calibration.
    """
    import cv2  # lazy: keeps the geometry helpers testable without OpenCV

    image_points = lm[list(_POSE_IDX)].astype(np.float64)
    focal = float(width)
    camera = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rmat, _ = cv2.Rodrigues(rvec)
    angles = cv2.RQDecomp3x3(rmat)[0]
    return _fold(angles[0]), _fold(angles[1]), _fold(angles[2])


def _eye_ratio(lm: np.ndarray, outer: int, inner: int, iris: int) -> float:
    """Where the iris sits between the eye corners: 0 = outer corner, 1 = inner corner."""
    axis = lm[inner] - lm[outer]
    n = float(axis @ axis)
    return 0.5 if n < 1e-6 else float((lm[iris] - lm[outer]) @ axis / n)


def gaze_offset(lm: np.ndarray) -> float:
    """Horizontal gaze from iris landmarks (needs refine_landmarks). Positive = looking toward image right."""
    if len(lm) < 478:
        return 0.0
    right = _eye_ratio(lm, 33, 133, 468)
    left = _eye_ratio(lm, 263, 362, 473)
    return (right - left) / 2.0
