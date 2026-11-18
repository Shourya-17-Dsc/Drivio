import numpy as np

from app.health import evaluate_health
from app.vision.features import eye_aspect_ratio, mouth_aspect_ratio


def eye(open_h: float) -> np.ndarray:
    # outer corner, 2 upper lid, inner corner, 2 lower lid; width 10
    return np.array([(0, 0), (3, -open_h / 2), (7, -open_h / 2), (10, 0), (7, open_h / 2), (3, open_h / 2)], float)


def test_ear_open_vs_closed():
    assert eye_aspect_ratio(eye(3.0)) > 0.25
    assert eye_aspect_ratio(eye(0.4)) < 0.1


def test_ear_degenerate_width():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def test_mar_increases_with_opening():
    def mouth(gap: float) -> np.ndarray:
        lm = np.zeros((400, 2))
        lm[78], lm[308] = (0, 0), (10, 0)
        for top, bottom in ((13, 14), (82, 87), (312, 317)):
            lm[top], lm[bottom] = (5, -gap / 2), (5, gap / 2)
        return lm

    assert mouth_aspect_ratio(mouth(8.0)) > 0.6 > mouth_aspect_ratio(mouth(1.0))
def test_health_distinguishes_ready_and_degraded_states():
    snapshot = {"mode": "simulation", "health": {"camera_ok": True}}
    ready = evaluate_health(snapshot, True, False)
    assert ready["status"] == "ok"
    assert ready["ready"] is True
    assert ready["checks"] == {"database": "ok", "pipeline": "ok", "video_storage": "disabled"}

    degraded = evaluate_health(snapshot, False, False)
    assert degraded["status"] == "degraded"
    assert degraded["ready"] is False
