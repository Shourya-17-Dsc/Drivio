"""Explainable alertness score (0-100), driver-state classification and confidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..config import Settings, settings as default_settings
from .fatigue import FatigueMetrics

ALERT, FATIGUED, DROWSY, DISTRACTED, NO_FACE = "alert", "fatigued", "drowsy", "distracted", "no_face"
STATES = (ALERT, FATIGUED, DROWSY, DISTRACTED, NO_FACE)


@dataclass(frozen=True)
class Factor:
    name: str
    penalty: float
    detail: str
    kind: str  # fatigue | distraction | presence


@dataclass
class ScoreResult:
    score: int
    state: str
    confidence: float
    factors: List[Factor] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "state": self.state,
            "confidence": round(self.confidence, 2),
            "factors": [f.__dict__ for f in self.factors],
        }


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class AlertnessScorer:
    """Start from 100 and subtract a penalty per observed risk. Every penalty is reported."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings

    def evaluate(self, m: FatigueMetrics) -> ScoreResult:
        if m.face_present and not m.calibrated:
            return ScoreResult(100, ALERT, 0.2, [])  # still learning this driver's normal eye and mouth values
        cfg, factors = self.cfg, []

        def add(name: str, penalty: float, detail: str, kind: str) -> None:
            if penalty >= 0.5:
                factors.append(Factor(name, round(penalty, 1), detail, kind))

        if not m.face_present:
            add("Face not visible", _clamp(m.face_missing_s / 4.0) * 40, f"no face for {m.face_missing_s:.1f}s", "presence")
        else:
            # ordinary blinking keeps the eyes closed about 3-6% of the time, so only count what is above that
            add("Eyes closed over time", _clamp((m.perclos - 0.08) / 0.30) * 35,
                f"eyes closed {m.perclos * 100:.0f}% of the last minute", "fatigue")
            if m.closure_s >= 0.4:
                add("Long eye closure", _clamp(m.closure_s / cfg.microsleep_s) * 40,
                    f"eyes shut for {m.closure_s:.1f}s", "fatigue")
            add("Yawning", min(3, m.yawns_recent) * 5 + (5 if m.yawning else 0),
                f"{m.yawns_recent} yawn(s) in 10 min", "fatigue")
            if m.observed_s >= 30 and m.blink_rate_pm > 35:
                add("Unusual blink rate", 8, f"{m.blink_rate_pm:.0f} blinks per minute", "fatigue")
            if m.head_nodding:
                add("Head dropping", 10, f"head tilted {abs(m.pitch):.0f} degrees", "fatigue")
            away = max(m.head_turned_s, m.gaze_off_s)
            if away >= cfg.distraction_after_s:
                what = (f"head turned {abs(m.yaw):.0f} degrees" if m.head_turned_s >= m.gaze_off_s
                        else "eyes off the road")
                add("Looking away", _clamp(away / 3.0) * 25, f"{what} for {away:.1f}s", "distraction")
            if m.phone:
                add("Phone in view", 25, "phone detected in the cabin", "distraction")

        total = sum(f.penalty for f in factors)
        score = int(round(max(0.0, 100.0 - total)))
        fatigue = sum(f.penalty for f in factors if f.kind == "fatigue")
        distraction = sum(f.penalty for f in factors if f.kind == "distraction")

        if not m.face_present and m.face_missing_s >= cfg.face_lost_after_s:
            state = NO_FACE
        elif m.closure_s >= cfg.microsleep_s or score < 35:
            state = DROWSY
        elif distraction >= 10 and (distraction >= fatigue or fatigue < 30):
            state = DISTRACTED
        elif fatigue >= 15:
            state = DROWSY if score < 50 else FATIGUED
        else:
            state = ALERT

        factors.sort(key=lambda f: f.penalty, reverse=True)
        return ScoreResult(score, state, self._confidence(m), factors)

    @staticmethod
    def _confidence(m: FatigueMetrics) -> float:
        """How much to trust this reading: calibrated, enough history, face well framed."""
        if not m.face_present:
            return 0.2
        framing = 1.0 if abs(m.yaw) < 30 else _clamp(1.0 - (abs(m.yaw) - 30) / 40)
        history = _clamp(m.observed_s / 30.0)
        return 0.4 * (1.0 if m.calibrated else 0.3) + 0.3 * history + 0.3 * framing
