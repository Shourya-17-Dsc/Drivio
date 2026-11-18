"""Trip-level safety score and break recommendation logic."""
from __future__ import annotations

from typing import Optional

from .fatigue import FatigueMetrics
from .scoring import ScoreResult
from ..config import Settings


class SafetyAdvisor:
    """Accumulate trip exposure without replacing the frame-level alert score."""

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.started_at: Optional[float] = None
        self.last_t: Optional[float] = None
        self.drive_duration_s = 0.0
        self.fatigue_duration_s = 0.0
        self.distraction_duration_s = 0.0
        self.critical_alerts = 0
        self.total_alerts = 0
        self.breaks_taken = 0
        self.score = 100

    def start(self, now: float) -> None:
        self.reset()
        self.active = True
        self.started_at = now
        self.last_t = now

    def stop(self) -> None:
        self.active = False
        self.last_t = None

    def update(self, now: float, result: ScoreResult, metrics: FatigueMetrics) -> None:
        if not self.active:
            return
        dt = 0.0 if self.last_t is None else min(max(now - self.last_t, 0.0), 1.0)
        self.last_t = now
        self.drive_duration_s += dt
        if result.state in ("fatigued", "drowsy"):
            self.fatigue_duration_s += dt
        if result.state == "distracted":
            self.distraction_duration_s += dt

        drive = max(self.drive_duration_s, 1.0)
        fatigue_penalty = min(45.0, self.fatigue_duration_s / drive * 100 * 45)
        distraction_penalty = min(25.0, self.distraction_duration_s / drive * 100 * 25)
        critical_penalty = min(20.0, self.critical_alerts * 10.0)
        alert_penalty = min(10.0, max(0, self.total_alerts - self.critical_alerts) * 1.5)
        self.score = int(round(max(0.0, 100.0 - fatigue_penalty - distraction_penalty - critical_penalty - alert_penalty)))

    def record_event(self, event: dict) -> None:
        if not self.active or event.get("kind") != "escalation":
            return
        self.total_alerts += 1
        if int(event.get("level", 0)) >= 3:
            self.critical_alerts += 1

    def take_break(self, now: float) -> None:
        if not self.active:
            return
        self.breaks_taken += 1
        self.drive_duration_s = 0.0
        self.fatigue_duration_s = 0.0
        self.distraction_duration_s = 0.0
        self.critical_alerts = 0
        self.total_alerts = 0
        self.score = 100
        self.last_t = now

    def snapshot(self) -> dict:
        if not self.active:
            return {"active": False, "score": self.score, "drive_duration_s": round(self.drive_duration_s),
                    "fatigue_duration_s": round(self.fatigue_duration_s),
                    "distraction_duration_s": round(self.distraction_duration_s),
                    "critical_alerts": self.critical_alerts, "breaks_taken": self.breaks_taken,
                    "break_due": False, "recommendation": "Start a session to track trip safety."}
        break_due = (self.drive_duration_s >= self.cfg.break_after_s
                     or self.fatigue_duration_s >= self.cfg.break_fatigue_s
                     or self.critical_alerts >= self.cfg.critical_alerts_before_break)
        if self.critical_alerts >= self.cfg.critical_alerts_before_break:
            recommendation = "Multiple critical alerts detected. Stop safely and rest before continuing."
        elif self.fatigue_duration_s >= self.cfg.break_fatigue_s:
            recommendation = "Repeated fatigue detected. Please stop safely and take a break."
        elif self.drive_duration_s >= self.cfg.break_after_s:
            recommendation = "You have been driving for a long period. Consider taking a break."
        else:
            recommendation = "No break needed yet. Continue monitoring your alertness."
        return {"active": True, "score": self.score, "drive_duration_s": round(self.drive_duration_s),
                "fatigue_duration_s": round(self.fatigue_duration_s),
                "distraction_duration_s": round(self.distraction_duration_s),
                "critical_alerts": self.critical_alerts, "breaks_taken": self.breaks_taken,
                "break_due": break_due, "recommendation": recommendation}
