from pathlib import Path

from PIL import Image

from app.ml.image_training import collect_records, split_records, train_image_model


def _write(path: Path, value: int) -> None:
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def _dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for class_name, value in (("Drowsy", 220), ("Non Drowsy", 30)):
        folder = tmp_path / class_name
        folder.mkdir()
        for group in "ABCDEF":
            _write(folder / f"{group}0001.png", value)
            _write(folder / f"{group}0002.png", value)
    return tmp_path


def test_split_keeps_filename_groups_together(tmp_path):
    records = collect_records(_dataset(tmp_path))
    train, validation, test = split_records(records)

    assert set(r.group for r in train).isdisjoint({r.group for r in validation})
    assert set(r.group for r in train).isdisjoint({r.group for r in test})
    assert set(r.group for r in validation).isdisjoint({r.group for r in test})
    assert len(train) + len(validation) + len(test) == len(records)


def test_image_training_saves_checkpoint_and_metrics(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    result = train_image_model(dataset, tmp_path / "models", epochs=1, batch_size=4, image_size=32, focus="eyes")

    assert Path(result["artifact"]).exists()
    assert result["algorithm"] == "image_cnn"
    assert result["focus"] == "eyes"
    assert 0 <= result["accuracy"] <= 1
    assert len(result["confusion_matrix"]) == 2
    assert result["train_rows"] + result["validation_rows"] + result["test_rows"] == 24
    assert isinstance(result["deployment_ready"], bool)
    assert result["quality_note"]
    assert result["test_group_metrics"]
    assert result["generalization_gap"] >= 0
