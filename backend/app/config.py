"""Central configuration. Every threshold lives here so it can be tuned in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DRIVIO_DATA_DIR", BASE_DIR / "data"))
IMAGE_DATASET_DIR = Path(os.getenv("DRIVIO_IMAGE_DATASET_DIR", BASE_DIR.parent))


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default) == "1"


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in os.getenv(name, default).split(",") if value.strip())


@dataclass(frozen=True)
class Settings:
    # runtime
    camera_index: int = int(os.getenv("DRIVIO_CAMERA", "0"))
    simulate: bool = _flag("DRIVIO_SIMULATE", "0")        # synthetic driver, no camera needed
    enable_phone_detection: bool = _flag("DRIVIO_PHONE", "0")
    enable_audio: bool = _flag("DRIVIO_AUDIO", "1")
    database_url: str = os.getenv("DRIVIO_DB", f"sqlite:///{DATA_DIR / 'drivio.db'}")
    store_video: bool = False                              # privacy: frames are never written to disk
    cors_origins: tuple[str, ...] = _csv("DRIVIO_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

    # eyes
    ear_threshold: float = 0.21        # replaced by a per-driver value after calibration
    blink_min_s: float = 0.04
    blink_max_s: float = 0.40
    prolonged_closure_s: float = 1.0
    microsleep_s: float = 2.0
    perclos_window_s: float = 60.0

    # mouth
    mar_threshold: float = 0.55
    yawn_min_s: float = 1.2
    yawn_window_s: float = 600.0

    # head
    yaw_limit_deg: float = 20.0
    pitch_limit_deg: float = 18.0
    nod_min_s: float = 1.0
    distraction_after_s: float = 1.0
    gaze_limit: float = 0.15           # iris offset (fraction of eye width) that counts as eyes off the road
    face_lost_after_s: float = 2.0

    # calibration
    calibration_frames: int = 90

    # alerts
    recover_hold_s: float = 5.0

    # trip safety and breaks
    break_after_s: float = 7200.0
    break_fatigue_s: float = 60.0
    critical_alerts_before_break: int = 2


settings = Settings()
