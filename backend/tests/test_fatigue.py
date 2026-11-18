from helpers import calibrated_analyzer, run


def test_calibration_sets_threshold():
    a, _ = calibrated_analyzer()
    assert a.calibrated and 0.15 <= a.ear_thr <= 0.26


def test_short_closure_counts_as_blink():
    a, t = calibrated_analyzer()
    t, _ = run(a, t, 0.15, ear=0.08)
    t, m = run(a, t, 0.5)
    assert m.blinks_total == 1 and m.prolonged_closures == 0


def test_prolonged_closure_and_perclos():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 2.5, ear=0.08)
    assert m.closure_s >= 2.0 and m.eyes_closed
    t, m = run(a, t, 0.5)
    assert m.prolonged_closures == 1 and m.perclos > 0.3


def test_yawn_needs_sustained_opening():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 0.5, mar=0.8)
    assert not m.yawning
    t, m = run(a, t, 1.5, mar=0.8)
    assert m.yawning and m.yawns_total == 1
    t, m = run(a, t, 3.0, mar=0.8)
    assert m.yawns_total == 1  # one continuous yawn is counted once


def test_head_turn_timer():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 2.0, yaw=40)
    assert m.head_turned_s >= 1.5
    t, m = run(a, t, 0.5, yaw=0)
    assert m.head_turned_s == 0


def test_face_lost_timer():
    a, t = calibrated_analyzer()
    t, m = run(a, t, 3.0, face=False)
    assert not m.face_present and m.face_missing_s >= 2.5


def test_calibration_can_be_saved_and_loaded():
    a, _ = calibrated_analyzer()
    saved = a.calibration_state()
    b = __import__("app.analysis.fatigue", fromlist=["FatigueAnalyzer"]).FatigueAnalyzer()
    b.load_calibration(saved)
    assert b.calibrated
    assert b.calibration_state() == saved
