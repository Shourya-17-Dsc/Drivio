"""Safety gates for promoting trained models to approved candidates."""
from __future__ import annotations


def deployment_decision(metrics: dict, artifact_exists: bool) -> tuple[bool, str]:
    """Return whether a run may be approved for future live inference."""
    if not artifact_exists:
        return False, "The model artifact is missing"
    if not metrics.get("deployment_ready", False):
        return False, metrics.get("quality_note", "The model failed the deployment-quality gate")
    return True, "Passed the deployment-quality gate and artifact check"
