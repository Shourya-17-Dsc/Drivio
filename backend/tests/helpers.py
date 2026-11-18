from app.analysis.fatigue import FatigueAnalyzer
from app.config import Settings
from app.vision.features import FrameFeatures

FPS = 30


def feats(ear=0.30, mar=0.08, yaw=0.0, pitch=0.0, face=True) -> FrameFeatures:
    return FrameFeatures(face_present=face, ear=ear, mar=mar, yaw=yaw, pitch=pitch)


def calibrated_analyzer() -> tuple:
    a = FatigueAnalyzer(Settings())
    t = 0.0
    for _ in range(100):
        a.update(t, feats())
        t += 1 / FPS
    return a, t


def run(a, t, seconds, **kw):
    """Feed constant features for `seconds`; returns (t, last metrics)."""
    m = None
    for _ in range(int(seconds * FPS)):
        m = a.update(t, feats(**kw))
        t += 1 / FPS
    return t, m
