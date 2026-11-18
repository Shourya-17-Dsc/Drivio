from pathlib import Path

from PIL import Image

from app.ml.image_dataset import audit_image_dataset


def _write(path: Path, value: int) -> None:
    image = Image.new("RGB", (24, 16), (value, value, value))
    image.save(path)


def test_audit_reports_classes_dimensions_and_groups(tmp_path):
    (tmp_path / "Drowsy").mkdir()
    (tmp_path / "Non Drowsy").mkdir()
    _write(tmp_path / "Drowsy" / "A0001.png", 10)
    _write(tmp_path / "Drowsy" / "A0002.png", 20)
    _write(tmp_path / "Non Drowsy" / "B0001.png", 30)

    result = audit_image_dataset(tmp_path)

    assert result["ready_for_training"]
    assert result["total_readable"] == 3
    assert result["class_balance_ratio"] == 0.5
    assert result["dimensions"] == {"24x16": 3}
    assert result["group_count"] == 2
    assert result["classes"]["Drowsy"]["groups"] == {"A": 2}


def test_audit_reports_corrupt_and_missing_class(tmp_path):
    (tmp_path / "Drowsy").mkdir()
    (tmp_path / "Drowsy" / "bad.png").write_bytes(b"not an image")

    result = audit_image_dataset(tmp_path)

    assert not result["ready_for_training"]
    assert result["total_corrupt"] == 1
    assert any("unreadable" in issue for issue in result["issues"])
    assert any("Missing class folder" in issue for issue in result["issues"])
