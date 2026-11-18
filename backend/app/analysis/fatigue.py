"""Temporal analysis: blinks, closures, PERCLOS, yawns, head pose, face presence."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional, Tuple

import numpy as np

from ..config import Settings, settings as default_settings
from ..vision.features import FrameFeatures


@dataclass
class FatigueMetrics:
    ear: float = 0.0
    ear_threshold: float = 0.21
    mar: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    eyes_closed: bool = False
    closure_s: float = 0.0
    blinks_total: int = 0
    blink_rate_pm: float = 0.0
    prolonged_closures: int = 0
    perclos: float = 0.0
    yawning: bool = False
    yawns_total: int = 0
    yawns_recent: int = 0
    head_turned_s: float = 0.0
    head_nodding: bool = False
    gaze: float = 0.0
    gaze_off_s: float = 0.0
    face_present: bool = False
    face_missing_s: float = 0.0
    calibrated: bool = False
    observed_s: float = 0.0
    phone: bool = False

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class FatigueAnalyzer:
    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self.cfg = cfg or default_settings
        self.reset()

    def reset(self) -> None:
        self.ear_thr = self.cfg.ear_threshold
        self.yaw_off = 0.0
        self.pitch_off = 0.0
        self.mar_thr = self.cfg.mar_threshold
        self.gaze_off = 0.0
        self._gaze_since: Optional[float] = None
        self.calibrated = False
        self._cal: list[Tuple[float, ...]] = []
        self._prev_t: Optional[float] = None
        self._yaw_s: Optional[float] = None    # smoothed head angles
        self._pitch_s: Optional[float] = None

        self._first_t: Optional[float] = None
        self._last_face_t: Optional[float] = None
        self._closed_since: Optional[float] = None
        self._blinks: Deque[float] = deque()
        self._blinks_total = 0
        self._prolonged = 0
        self._perclos: Deque[Tuple[float, bool]] = deque()
        self._mar_since: Optional[float] = None
        self._yawn_counted = False
        self._yawns: Deque[float] = deque()
        self._yawns_total = 0
        self._turn_since: Optional[float] = None
        self._nod_since: Optional[float] = None

    def calibration_state(self) -> dict:
        """Return the learned personal baseline in a JSON-safe form."""
        return {
            "ear_threshold": round(float(self.ear_thr), 6),
            "yaw_offset": round(float(self.yaw_off), 6),
            "pitch_offset": round(float(self.pitch_off), 6),
            "mar_threshold": round(float(self.mar_thr), 6),
            "gaze_offset": round(float(self.gaze_off), 6),
            "calibrated": bool(self.calibrated),
        }

    def load_calibration(self, calibration: dict) -> None:
        """Load a previously saved baseline for the selected driver."""
        required = ("ear_threshold", "yaw_offset", "pitch_offset", "mar_threshold", "gaze_offset")
        if not all(key in calibration for key in required):
            raise ValueError("Saved calibration is incomplete")
        self.ear_thr = float(calibration["ear_threshold"])
        self.yaw_off = float(calibration["yaw_offset"])
        self.pitch_off = float(calibration["pitch_offset"])
        self.mar_thr = float(calibration["mar_threshold"])
        self.gaze_off = float(calibration["gaze_offset"])
        self.calibrated = True

    # ------------------------------------------------------------------
    def _calibrate(self, f: FrameFeatures) -> None:
        self._cal.append((f.ear, f.yaw, f.pitch, f.mar, f.gaze))
        if len(self._cal) < self.cfg.calibration_frames:
            return
        ears, yaws, pitches, mars, gazes = (np.array(c) for c in zip(*self._cal))
        open_ear = float(np.percentile(ears, 75))  # blinks fall in the bottom of the range
        self.ear_thr = float(np.clip(0.72 * open_ear, 0.15, 0.26))
        self.yaw_off, self.pitch_off = float(np.median(yaws)), float(np.median(pitches))
        # a yawn is a clear jump above this driver's resting mouth opening
        self.mar_thr = max(self.cfg.mar_threshold, float(np.median(mars)) + 0.35)
        self.gaze_off = float(np.median(gazes))
        self.calibrated = True

    def update(self, t: float, f: FrameFeatures, phone: bool = False) -> FatigueMetrics:
        cfg = self.cfg
        if self._first_t is None:
            self._first_t = t
        if self._prev_t is not None and t - self._prev_t > 0.5:
            # lag spike or dropped frames: don't let the gap look like a long eye closure
            self._closed_since = self._mar_since = self._turn_since = self._nod_since = self._gaze_since = None
            self._yawn_counted = False
        self._prev_t = t
        m = FatigueMetrics(ear_threshold=self.ear_thr, calibrated=self.calibrated,
                           observed_s=t - self._first_t, phone=phone)

        if not f.face_present:
            if self._last_face_t is None:
                self._last_face_t = t
            m.face_missing_s = t - self._last_face_t
            self._closed_since = self._mar_since = self._turn_since = self._nod_since = self._gaze_since = None
            self._yawn_counted = False
            self._yaw_s = self._pitch_s = None
            return self._finish(t, m)

        self._last_face_t = t
        m.face_present = True
        if not self.calibrated:
            self._calibrate(f)
            m.ear_threshold, m.calibrated = self.ear_thr, self.calibrated

        # --- head (smoothed: solvePnP angles jitter frame to frame) ------
        a = 0.35
        self._yaw_s = f.yaw if self._yaw_s is None else a * f.yaw + (1 - a) * self._yaw_s
        self._pitch_s = f.pitch if self._pitch_s is None else a * f.pitch + (1 - a) * self._pitch_s
        m.yaw, m.pitch = self._yaw_s - self.yaw_off, self._pitch_s - self.pitch_off
        if abs(m.yaw) > cfg.yaw_limit_deg:
            self._turn_since = self._turn_since if self._turn_since is not None else t
            m.head_turned_s = t - self._turn_since
        else:
            self._turn_since = None
        if abs(m.pitch) > cfg.pitch_limit_deg:
            self._nod_since = self._nod_since if self._nod_since is not None else t
            m.head_nodding = (t - self._nod_since) >= cfg.nod_min_s
        else:
            self._nod_since = None

        # --- eyes -------------------------------------------------------
        m.ear, m.mar = f.ear, f.mar
        if abs(m.yaw) > 45:
            closed = False  # in profile the eye landmarks are unreliable, so don't call it closed
        else:
            # hysteresis: once closed, the eye must clearly reopen, which stops flicker near the threshold
            closed = f.ear < self.ear_thr or (self._closed_since is not None and f.ear < self.ear_thr * 1.1)
        m.eyes_closed = closed
        if closed:
            if self._closed_since is None:
                self._closed_since = t
            m.closure_s = t - self._closed_since
        else:
            if self._closed_since is not None:
                dur = t - self._closed_since
                if cfg.blink_min_s <= dur <= cfg.blink_max_s:
                    self._blinks.append(t)
                    self._blinks_total += 1
                elif dur >= cfg.prolonged_closure_s:
                    self._prolonged += 1
            self._closed_since = None
        self._perclos.append((t, closed))

        # --- gaze: eyes off the road even if the head stays still ---------
        m.gaze = f.gaze - self.gaze_off
        if abs(m.gaze) > cfg.gaze_limit and not closed:
            self._gaze_since = self._gaze_since if self._gaze_since is not None else t
            m.gaze_off_s = t - self._gaze_since
        else:
            self._gaze_since = None

        # --- yawn -------------------------------------------------------
        if f.mar > self.mar_thr:
            if self._mar_since is None:
                self._mar_since = t
            if t - self._mar_since >= cfg.yawn_min_s:
                m.yawning = True
                if not self._yawn_counted:
                    self._yawn_counted = True
                    self._yawns_total += 1
                    self._yawns.append(t)
        else:
            self._mar_since, self._yawn_counted = None, False

        return self._finish(t, m)

    def _finish(self, t: float, m: FatigueMetrics) -> FatigueMetrics:
        cfg = self.cfg
        while self._blinks and self._blinks[0] < t - 60.0:
            self._blinks.popleft()
        while self._perclos and self._perclos[0][0] < t - cfg.perclos_window_s:
            self._perclos.popleft()
        while self._yawns and self._yawns[0] < t - cfg.yawn_window_s:
            self._yawns.popleft()

        window = min(60.0, m.observed_s)
        m.blinks_total, m.prolonged_closures, m.yawns_total = self._blinks_total, self._prolonged, self._yawns_total
        m.blink_rate_pm = len(self._blinks) / window * 60.0 if window >= 15.0 else 0.0
        m.perclos = (sum(1 for _, c in self._perclos if c) / len(self._perclos)) if self._perclos else 0.0
        m.yawns_recent = len(self._yawns)
        return m
