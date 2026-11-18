from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ml.temporal_training import (
    SequenceRecord,
    build_sequences,
    labelled_dataset_status,
    split_sequences,
    train_temporal_model,
)


def _rows():
    rows = []
    for session in range(8):
        label = "drowsy" if session % 2 else "alert"
        for second in range(12):
            rows.append(SimpleNamespace(
                session_id=session, t=second, ground_truth=label,
                ear=0.15 if label == "drowsy" else 0.30, mar=0.2, perclos=0.3 if label == "drowsy" else 0.03,
                closure_s=1.0 if label == "drowsy" else 0.0, blink_rate=8, yawns_recent=1 if label == "drowsy" else 0,
                yaw=0, pitch=10 if label == "drowsy" else 0, head_turned_s=0, safety_score=60 if label == "drowsy" else 100,
            ))
    return rows


def test_sequences_and_splits_keep_sessions_together():
    records = build_sequences(_rows(), window_size=6, stride=3)
    train, validation, test = split_sequences(records)
    assert records
    assert set(r.group for r in train).isdisjoint({r.group for r in validation})
    assert set(r.group for r in train).isdisjoint({r.group for r in test})
    assert set(r.group for r in validation).isdisjoint({r.group for r in test})
    for split in (train, validation, test):
        assert {record.label for record in split} == {"alert", "drowsy"}


def test_temporal_readiness_requires_each_label_in_three_sessions():
    rows = _rows()
    for row in rows:
        if row.session_id == 7:
            row.ground_truth = "fatigued"
    status = labelled_dataset_status(rows)
    assert status["ready_for_temporal_training"] is False
    assert "three labelled sessions" in status["reason"]


def test_temporal_split_rejects_label_in_only_one_session():
    records = [
        SequenceRecord(np.zeros((3, 10), dtype=np.float32), "drowsy" if group == 7 else "alert", str(group))
        for group in range(8)
    ]
    with pytest.raises(ValueError, match="every label"):
        split_sequences(records)


def test_labelled_dataset_status_reports_training_readiness():
    rows = _rows()
    status = labelled_dataset_status(rows)
    assert status["labelled_samples"] == len(rows)
    assert status["labelled_sessions"] == 8
    assert status["ready_for_temporal_training"] is True
    assert status["sessions_by_label"] == {"alert": 4, "drowsy": 4}


def test_temporal_training_saves_checkpoint(tmp_path):
    records = build_sequences(_rows(), window_size=6, stride=3)
    result = train_temporal_model(records, str(tmp_path / "temporal.pt"), epochs=1, batch_size=4)
    assert Path(result["artifact"]).exists()
    assert result["algorithm"] == "gru_temporal"
    assert result["train_rows"] + result["validation_rows"] + result["test_rows"] == len(records)
