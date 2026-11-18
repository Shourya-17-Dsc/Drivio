"""Alert levels with escalation, hysteresis and recovery detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from ..config import Settings, settings as default_settings
from .fatigue import FatigueMetrics
from .scoring import ALERT, DROWSY, NO_FACE, ScoreResult

LEVEL_NAMES = ("none", "caution", "warning", "critical")
STATE_LABELS = {"alert": "Alert", "fatigued": "Fatigued", "drowsy": "Drowsy",
                "distracted": "Distracted", "no_face": "Face not visible"}


@dataclass
class AlertEvent:
    t: float
    level: int
    previous: int
    kind: str  # escalation | recovery
    state: str
    message: str

    def to_dict(self) -> dict:
        return {**asdict(self), "level_name": LEVEL_NAMES[self.level]}


class AlertManager:
    """Raises a level only after its condition holds for a short time, and lowers it
    only after the driver has stayed better for `recover_hold_s`."""

    RAISE_HOLD = {1: 1.5, 2: 1.0, 3: 0.5}

    def __init__(self, cfg: Optional[Settings] = None) -> None:
        self.cfg = cfg or default_settings
        self.level = 0
        self._since: dict = {1: None, 2: None, 3: None}
        self._calm_since: Optional[float] = None

    def target_level(self, r: ScoreResult, m: FatigueMetrics) -> int:
        if m.closure_s >= self.cfg.microsleep_s or r.score < 30:
            return 3
        if r.score < 50 or r.state == DROWSY:
            return 2
        if r.score < 65 or r.state != ALERT:
            return 1
        return 0

    def update(self, t: float, r: ScoreResult, m: FatigueMetrics) -> Optional[AlertEvent]:
        if not m.calibrated:
            return None
        target = self.target_level(r, m)
        for lvl in (1, 2, 3):
            self._since[lvl] = (self._since[lvl] if self._since[lvl] is not None else t) if target >= lvl else None

        sustained = max([l for l in (1, 2, 3) if self._since[l] is not None and t - self._since[l] >= self.RAISE_HOLD[l]],
                        default=0)
        reason = r.factors[0].detail if r.factors else STATE_LABELS.get(r.state, r.state)

        if sustained > self.level:
            prev, self.level, self._calm_since = self.level, sustained, None
            return AlertEvent(t, self.level, prev, "escalation", r.state,
                              f"{STATE_LABELS.get(r.state, r.state)}: {reason}")

        if target < self.level:
            self._calm_since = self._calm_since if self._calm_since is not None else t
            if t - self._calm_since >= self.cfg.recover_hold_s:
                prev, self.level, self._calm_since = self.level, target, None
                return AlertEvent(t, self.level, prev, "recovery", r.state,
                                  "Driver recovered" if target == 0 else f"Risk reduced to {LEVEL_NAMES[target]}")
        else:
            self._calm_since = None
        return None
