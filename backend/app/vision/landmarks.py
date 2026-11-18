"""Face detection, tracking and landmark extraction (MediaPipe FaceMesh)."""
from __future__ import annotations

import numpy as np

from .features import FrameFeatures, gaze_offset, head_pose, mean_ear, mouth_aspect_ratio


class FaceFeatureExtractor:
    def __init__(self) -> None:
        import mediapipe as mp

        # FaceMesh detects the face once, then tracks it frame to frame.
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> FrameFeatures:
        import cv2

        h, w = frame_bgr.shape[:2]
        result = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return FrameFeatures(face_present=False)

        lm = np.array([(p.x * w, p.y * h) for p in result.multi_face_landmarks[0].landmark], dtype=np.float64)
        pose = head_pose(lm, w, h) or (0.0, 0.0, 0.0)
        x1, y1 = lm.min(axis=0).astype(int)
        x2, y2 = lm.max(axis=0).astype(int)
        return FrameFeatures(
            face_present=True,
            ear=mean_ear(lm),
            mar=mouth_aspect_ratio(lm),
            pitch=pose[0],
            yaw=pose[1],
            roll=pose[2],
            gaze=gaze_offset(lm),
            landmarks=lm,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
        )

    def close(self) -> None:
        self._mesh.close()
