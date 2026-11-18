from app.ml.governance import deployment_decision


def test_governance_requires_quality_gate_and_artifact():
    assert deployment_decision({"deployment_ready": True}, True)[0] is True
    assert deployment_decision({"deployment_ready": True}, False)[0] is False
    assert deployment_decision({"deployment_ready": False, "quality_note": "weak holdout"}, True) == (False, "weak holdout")
