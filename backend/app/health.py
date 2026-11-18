"""Small, dependency-free readiness evaluation used by the health endpoint."""
from __future__ import annotations


def evaluate_health(snapshot: dict, database_ok: bool, store_video: bool) -> dict:
    pipeline_health = snapshot.get("health", {})
    pipeline_started = bool(snapshot.get("mode")) and bool(pipeline_health)
    checks = {
        "database": "ok" if database_ok else "failed",
        "pipeline": "ok" if pipeline_started else "starting",
        "video_storage": "disabled" if not store_video else "enabled",
    }
    return {
        "status": "ok" if database_ok else "degraded",
        "ready": database_ok and pipeline_started,
        "checks": checks,
        "store_video": store_video,
        "health": pipeline_health,
    }
