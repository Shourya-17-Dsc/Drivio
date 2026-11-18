"""Frame sources: a real webcam, or a scripted synthetic driver for demos and tests."""
from __future__ import annotations

import math
import time

import numpy as np

from .features import FrameFeatures


class CameraSource:
    simulated = False

    def __init__(self, index: int) -> None:
        import cv2

        self._cv2 = cv2
        self.index = index
        self.backend_name = "unavailable"
        self.cap = None
        self._ready = False
        # DirectShow is often more reliable for Windows webcams, while the
        # default backend remains the first choice for virtual cameras.
        backends = [("default", None), ("directshow", cv2.CAP_DSHOW), ("media_foundation", cv2.CAP_MSMF)]
        for name, backend in backends:
            candidate = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not candidate.isOpened():
                candidate.release()
                continue
            candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ready = False
            for _ in range(3):
                ok, _ = candidate.read()
                if ok:
                    ready = True
                    break
            if ready:
                self.cap, self.backend_name, self._ready = candidate, name, True
                break
            candidate.release()

    @property
    def opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    @property
    def ready(self) -> bool:
        return self._ready

    def read(self):
        if self.cap is None:
            return False, None
        ok, frame = self.cap.read()
        return (True, self._cv2.flip(frame, 1)) if ok else (False, None)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SimulatedSource:
    """Repeats a 60 s script: alert, drowsy, micro-sleep, head turn, recovery."""

    simulated = True
    opened = True

    def __init__(self) -> None:
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(7)

    def read(self):
        time.sleep(1 / 20)
        frame = np.full((480, 640, 3), (48, 42, 20), dtype=np.uint8)
        try:
            import cv2
            cv2.putText(frame, "SIMULATED DRIVER", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 220, 215), 2)
        except ImportError:
            pass
        return True, frame

    def features(self) -> FrameFeatures:
        t = (time.monotonic() - self._t0) % 60.0
        n = lambda s: float(self._rng.normal(0, s))  # noqa: E731
        ear, mar, yaw, pitch = 0.30 + n(0.008), 0.08 + n(0.01), n(1.5), n(1.5)

        if t < 20:  # alert, blinking about every 4 s
            if (t % 4.0) < 0.15:
                ear = 0.10
        elif t < 32:  # drowsy: long slow closures, a yawn
            if (t % 3.0) < 0.9:
                ear = 0.12 + n(0.01)
            if 24 <= t < 26.2:
                mar = 0.75 + n(0.03)
            pitch += 8 * math.sin(t)
        elif t < 36:  # micro-sleep
            if 32.5 <= t < 35.2:
                ear, pitch = 0.09, pitch + 22
        elif t < 44:  # looking away
            yaw += 42
        return FrameFeatures(face_present=True, ear=ear, mar=max(mar, 0.0), pitch=pitch, yaw=yaw)

    def release(self) -> None:
        pass
