from helpers import calibrated_analyzer, run

from app.analysis.alerts import AlertManager
from app.analysis.scoring import ALERT, DISTRACTED, DROWSY, NO_FACE, AlertnessScorer
from app.config import Settings

scorer = AlertnessScorer(Settings())


def test_alert_driver_scores_high():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 5)
    r = scorer.evaluate(m)
    assert r.score >= 90 and r.state == ALERT and not r.factors


def test_microsleep_is_drowsy_and_explained():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 2.5, ear=0.08)
    r = scorer.evaluate(m)
    assert r.state == DROWSY and r.score < 60
    assert r.factors[0].name == "Long eye closure"


def test_head_turn_is_distraction():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 3.0, yaw=45)
    assert scorer.evaluate(m).state == DISTRACTED


def test_missing_face():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 3.0, face=False)
    r = scorer.evaluate(m)
    assert r.state == NO_FACE and r.confidence < 0.3


def test_alert_escalates_then_recovers():
    a, t = calibrated_analyzer()
    mgr, events = AlertManager(Settings()), []
    for secs, kw in ((3.5, dict(ear=0.08)), (75.0, {})):
        for _ in range(int(secs * 30)):
            m = a.update(t, __import__("helpers").feats(**kw))
            ev = mgr.update(t, scorer.evaluate(m), m)
            if ev:
                events.append(ev)
            t += 1 / 30
    assert events[0].kind == "escalation" and events[0].level == 1  # levels step up one at a time
    assert events[-1].kind == "recovery" and mgr.level == 0
    assert max(e.level for e in events) == 3
