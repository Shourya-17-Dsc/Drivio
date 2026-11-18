from app.analysis.fatigue import FatigueMetrics
from app.analysis.safety import SafetyAdvisor
from app.analysis.scoring import ScoreResult
from app.config import Settings


def test_safety_advisor_recommends_break_and_resets_after_break():
    advisor = SafetyAdvisor(Settings(break_after_s=20, break_fatigue_s=2, critical_alerts_before_break=2))
    result = ScoreResult(40, "drowsy", 0.9, [])
    metrics = FatigueMetrics(face_present=True)
    advisor.start(0.0)
    advisor.update(1.0, result, metrics)
    advisor.update(2.0, result, metrics)

    state = advisor.snapshot()
    assert state["break_due"] and state["fatigue_duration_s"] == 2
    assert state["score"] < 100

    advisor.take_break(2.5)
    recovered = advisor.snapshot()
    assert not recovered["break_due"]
    assert recovered["score"] == 100
    assert recovered["breaks_taken"] == 1


def test_critical_alerts_trigger_break_recommendation():
    advisor = SafetyAdvisor(Settings(break_after_s=100, break_fatigue_s=100, critical_alerts_before_break=2))
    advisor.start(0.0)
    advisor.record_event({"kind": "escalation", "level": 3})
    advisor.record_event({"kind": "escalation", "level": 3})

    assert advisor.snapshot()["break_due"]
    assert "critical" in advisor.snapshot()["recommendation"].lower()
