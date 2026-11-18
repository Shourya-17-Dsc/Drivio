"""Camera -> face tracking -> features -> fatigue analysis -> score -> classification -> alerts."""
from __future__ import annotations

import threading
import time
from typing import Any
from collections import deque
from typing import Callable, Optional

import numpy as np

try:
    import cv2
except ImportError:  # Simulation and analysis tests do not need OpenCV rendering.
    cv2 = None

from .analysis.alerts import LEVEL_NAMES, AlertEvent, AlertManager
from .analysis.fatigue import FatigueAnalyzer
from .analysis.scoring import AlertnessScorer
from .analysis.safety import SafetyAdvisor
from .audio import AudioAlerter
from .config import Settings
from .vision.features import LEFT_EYE, RIGHT_EYE
from .vision.phone import PhoneDetector
from .vision.sources import CameraSource, SimulatedSource

STATE_BGR = {"alert": (165, 209, 79), "fatigued": (63, 183, 244), "drowsy": (92, 88, 240),
             "distracted": (63, 183, 244), "no_face": (170, 170, 170)}


class MonitoringPipeline:
    def __init__(self, cfg: Settings,
                 on_sample: Optional[Callable[[int, dict], None]] = None,
                 on_event: Optional[Callable[[int, dict], None]] = None,
                 on_calibrated: Optional[Callable[[int, dict], None]] = None) -> None:
        self.cfg, self.on_sample, self.on_event, self.on_calibrated = cfg, on_sample, on_event, on_calibrated
        self.analyzer, self.scorer, self.alerts = FatigueAnalyzer(cfg), AlertnessScorer(cfg), AlertManager(cfg)
        self.safety = SafetyAdvisor(cfg)
        self.audio = AudioAlerter(cfg.enable_audio)
        self.phone = PhoneDetector(cfg.enable_phone_detection)
        self.session_id: Optional[int] = None
        self.ground_truth: Optional[str] = None
        self.driver_id: Optional[int] = None
        self.driver_name: Optional[str] = None
        self._calibration_reported = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._jpeg: Optional[bytes] = None
        self._snapshot: dict = {}
        self._events: deque = deque(maxlen=20)
        self._health = {"camera_ok": False, "vision_ok": False, "fps": 0.0, "latency_ms": 0.0, "error": None}
        self._mode = "camera"
        self._started = time.monotonic()

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="drivio-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def set_session(self, session_id: Optional[int]) -> None:
        if session_id is not None and self.session_id != session_id:
            self.safety.start(time.monotonic())
        elif session_id is None and self.session_id is not None:
            self.safety.stop()
        self.session_id = session_id
        if session_id is None:
            self.ground_truth = None

    def set_ground_truth(self, label: Optional[str]) -> None:
        self.ground_truth = label

    def take_break(self) -> None:
        self.safety.take_break(time.monotonic())

    def set_driver(self, driver_id: Optional[int], driver_name: Optional[str] = None,
                   calibration: Optional[dict[str, Any]] = None) -> None:
        self.driver_id, self.driver_name = driver_id, driver_name
        self.analyzer.reset()
        self.alerts = AlertManager(self.cfg)
        self._calibration_reported = False
        if calibration:
            self.analyzer.load_calibration(calibration)
            self._calibration_reported = True

    def recalibrate(self) -> None:
        self.analyzer.reset()
        self.alerts = AlertManager(self.cfg)
        self._calibration_reported = False

    # -- readers --------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snapshot)

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    # -- worker ---------------------------------------------------------
    def _open_source(self):
        if not self.cfg.simulate:
            try:
                cam = CameraSource(self.cfg.camera_index)
                if cam.ready:
                    self._health["camera_backend"] = cam.backend_name
                    return cam
                cam.release()
                self._health["error"] = (
                    f"Camera {self.cfg.camera_index} opened but produced no frames. "
                    "Check Windows camera permissions and close other camera apps. Showing a simulated driver."
                )
            except ImportError:
                self._health["error"] = "OpenCV is unavailable in this Python environment. Install requirements with Python 3.10-3.12. Showing a simulated driver."
            except Exception as exc:
                self._health["error"] = f"Camera startup failed: {exc}. Showing a simulated driver."
        self._mode = "simulation"
        return SimulatedSource()

    def _run(self) -> None:
        source = self._open_source()
        extractor = None
        if not source.simulated:
            try:
                from .vision.landmarks import FaceFeatureExtractor

                extractor = FaceFeatureExtractor()
            except Exception as exc:
                self._health["error"] = f"Face tracking unavailable: {exc}"
                source.release()
                self._mode, source = "simulation", SimulatedSource()
        self._health.update(camera_ok=True, vision_ok=True)

        last_sample, fps = 0.0, 0.0
        while not self._stop.is_set():
            t0 = time.perf_counter()
            ok, frame = source.read()
            if not ok:
                self._health["camera_ok"] = False
                time.sleep(0.2)
                continue
            self._health["camera_ok"] = True

            now = time.monotonic()
            feats = source.features() if source.simulated else extractor.extract(frame)
            phone = self.phone.detect(frame) if not source.simulated else False
            metrics = self.analyzer.update(now, feats, phone)
            if self.driver_id is not None and self.analyzer.calibrated and not self._calibration_reported:
                self._calibration_reported = True
                if self.on_calibrated:
                    self.on_calibrated(self.driver_id, self.analyzer.calibration_state())
            result = self.scorer.evaluate(metrics)
            self.safety.update(now, result, metrics)
            event = self.alerts.update(now, result, metrics)
            self.audio.play(self.alerts.level)

            if event:
                self._record_event(event)
                self.safety.record_event(event.to_dict())
            if result.factors and now - getattr(self, "_last_log", 0.0) > 5.0:
                self._last_log = now
                print(f"[DRIVIO] score {result.score} {result.state}: "
                      + ", ".join(f"{f.name} -{f.penalty:.0f} ({f.detail})" for f in result.factors), flush=True)
            ok_jpg, buf = False, None
            if cv2 is not None:
                self._annotate(frame, feats, result, metrics)
                ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

            dt = time.perf_counter() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / max(dt, 1e-3)
            self._health.update(fps=round(fps, 1), latency_ms=round(dt * 1000, 1))
            snap = {
                "ts": time.time(), "mode": self._mode, "recording": self.session_id is not None,
                "session_id": self.session_id, "driver_id": self.driver_id, "driver_name": self.driver_name,
                "ground_truth": self.ground_truth, **result.to_dict(), "metrics": metrics.to_dict(), "safety": self.safety.snapshot(),
                "level": self.alerts.level, "level_name": LEVEL_NAMES[self.alerts.level],
                "events": list(self._events)[-8:],
                "health": {**self._health, "audio_ok": self.audio.ok, "phone_detection": self.phone.enabled,
                           "uptime_s": round(time.monotonic() - self._started)},
            }
            with self._lock:
                self._snapshot = snap
                if ok_jpg:
                    self._jpeg = buf.tobytes()

            if self.session_id is not None and self.on_sample and now - last_sample >= 1.0:
                last_sample = now
                self.on_sample(self.session_id, snap)

        source.release()

    def _record_event(self, event: AlertEvent) -> None:
        payload = {**event.to_dict(), "ts": time.time()}
        self._events.append(payload)
        if self.session_id is not None and self.on_event:
            self.on_event(self.session_id, payload)

    @staticmethod
    def _annotate(frame: np.ndarray, feats, result, m) -> None:
        if cv2 is None:
            return

        color = STATE_BGR.get(result.state, (200, 200, 200))
        if feats.face_present and feats.bbox:
            x1, y1, x2, y2 = feats.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if feats.landmarks is not None:
                for idx in (LEFT_EYE, RIGHT_EYE):
                    cv2.polylines(frame, [feats.landmarks[list(idx)].astype(np.int32)], True, color, 1)
        cv2.putText(frame, f"{result.state.upper()}  {result.score}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"EAR {m.ear:.2f} (closed < {m.ear_threshold:.2f})  MAR {m.mar:.2f}  yaw {m.yaw:.0f}",
                    (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (233, 241, 239), 1)
