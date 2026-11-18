import pytest
from pydantic import ValidationError

from app.main import ImageTrainBody, StartBody, TemporalTrainBody


def test_image_training_rejects_invalid_requests_before_job_creation():
    with pytest.raises(ValidationError):
        ImageTrainBody(epochs=0)
    with pytest.raises(ValidationError):
        ImageTrainBody(image_size=40)


def test_temporal_training_rejects_unsafe_or_expensive_ranges():
    with pytest.raises(ValidationError):
        TemporalTrainBody(window_size=2)
    with pytest.raises(ValidationError):
        TemporalTrainBody(epochs=51)


def test_session_label_has_a_bounded_size():
    with pytest.raises(ValidationError):
        StartBody(label="x" * 121)
