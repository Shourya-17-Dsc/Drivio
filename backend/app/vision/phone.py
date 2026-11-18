"""Optional phone detection with YOLO (COCO class 67 = cell phone). Off by default."""
from __future__ import annotations

import os
import time

import numpy as np


class PhoneDetector:
    def __init__(self, enabled: bool, every_n: int = 6, conf: float = 0.45, hold_s: float = 1.0) -> None:
        self.enabled = enabled
        self.error = None
        self.every_n, self.conf, self.hold_s = every_n, conf, hold_s
        self._n = 0
        self._last_seen = 0.0
        self._model = None
        if enabled:
            try:
                from ultralytics import YOLO

                self._model = YOLO(os.getenv("DRIVIO_YOLO", "yolov8n.pt"))
            except Exception as exc:  # missing package or weights: degrade, don't crash
                self.enabled, self.error = False, str(exc)

    def detect(self, frame: np.ndarray) -> bool:
        if not self.enabled or self._model is None:
            return False
        self._n += 1
        if self._n % self.every_n == 0:
            result = self._model.predict(frame, classes=[67], conf=self.conf, imgsz=320, verbose=False)[0]
            if len(result.boxes) > 0:
                self._last_seen = time.monotonic()
        return (time.monotonic() - self._last_seen) < self.hold_s
