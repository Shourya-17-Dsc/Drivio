"""Grouped image-model training for the drowsiness dataset.

This module intentionally owns the complete image pipeline: discovery,
group-aware splitting, preprocessing, training, evaluation and checkpointing.
It does not depend on torchvision so the backend remains easy to install in a
college-lab environment.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .image_dataset import CLASS_NAMES, IMAGE_EXTENSIONS, _group_prefix


LABELS = {"Drowsy": 1, "Non Drowsy": 0}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    group: str


def collect_records(root: str | Path, max_images_per_class: int | None = None) -> list[ImageRecord]:
    """Collect image paths with labels and leakage-control groups."""
    root = Path(root)
    records: list[ImageRecord] = []
    for class_name in CLASS_NAMES:
        folder = root / class_name
        if not folder.is_dir():
            raise ValueError(f"Missing class folder: {class_name}")
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if max_images_per_class is not None:
            paths = paths[:max_images_per_class]
        if not paths:
            raise ValueError(f"Class has no images: {class_name}")
        records.extend(ImageRecord(str(p), LABELS[class_name], _group_prefix(p)) for p in paths)
    return records


def split_records(records: list[ImageRecord], seed: int = 42) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
    """Split by group: no filename group can cross train/validation/test."""
    if len({r.group for r in records}) < 6:
        raise ValueError("At least 6 filename groups are required for grouped splitting")
    groups = np.array([r.group for r in records])
    indices = np.arange(len(records))
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, held_idx = next(first.split(indices, groups=groups))
    held_groups = groups[held_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    val_sub, test_sub = next(second.split(held_idx, groups=held_groups))
    val_idx, test_idx = held_idx[val_sub], held_idx[test_sub]
    return ([records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx])


class ImageTensorDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: Iterable[ImageRecord], image_size: int, augment: bool = False, focus: str = "full") -> None:
        self.records = list(records)
        self.image_size = image_size
        self.augment = augment
        self.focus = focus

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        try:
            with Image.open(record.path) as image:
                image = image.convert("RGB")
                if self.focus == "eyes":
                    # The upper face contains the eyes and eyebrows, which are
                    # the most stable drowsiness signal and less identity-heavy
                    # than the full face/background.
                    image = image.crop((0, 0, image.width, max(1, int(image.height * 0.68))))
                if self.augment:
                    if random.random() < 0.5:
                        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if random.random() < 0.7:
                        image = image.rotate(random.uniform(-7.0, 7.0), resample=Image.Resampling.BILINEAR)
                    if random.random() < 0.8:
                        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))
                        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Unable to read training image: {record.path}") from exc
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        tensor = (tensor - 0.5) / 0.5
        return tensor, record.label


class DrowsinessCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _loader(records: list[ImageRecord], image_size: int, batch_size: int, shuffle: bool, augment: bool, focus: str) -> DataLoader:
    return DataLoader(ImageTensorDataset(records, image_size, augment, focus), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truth: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu().tolist()
            truth.extend(labels.tolist())
            predictions.extend(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "precision_macro": round(float(precision.mean()), 4),
        "recall_macro": round(float(recall.mean()), 4),
        "f1_macro": round(float(f1.mean()), 4),
        "labels": ["Non Drowsy", "Drowsy"],
        "precision_by_class": [round(float(x), 4) for x in precision],
        "recall_by_class": [round(float(x), 4) for x in recall],
        "f1_by_class": [round(float(x), 4) for x in f1],
        "support_by_class": [int(x) for x in support],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "rows": len(truth),
    }


def _evaluate_groups(model: nn.Module, records: list[ImageRecord], image_size: int,
                     batch_size: int, device: torch.device, focus: str) -> dict[str, dict]:
    """Evaluate each held-out group separately to expose domain shift."""
    result: dict[str, dict] = {}
    for group in sorted({record.group for record in records}):
        rows = [record for record in records if record.group == group]
        metrics = _evaluate(model, _loader(rows, image_size, batch_size, False, False, focus), device)
        result[group] = {
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "drowsy_recall": metrics["recall_by_class"][1],
        }
    return result


def train_image_model(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 3,
    batch_size: int = 32,
    image_size: int = 64,
    max_images_per_class: int | None = None,
    focus: str = "full",
    seed: int = 42,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Train, evaluate and save a drowsiness CNN; returns JSON-safe metadata."""
    if not 1 <= epochs <= 30:
        raise ValueError("epochs must be between 1 and 30")
    if batch_size < 1 or image_size not in (32, 48, 64, 96, 128) or focus not in ("full", "eyes"):
        raise ValueError("Use a positive batch size and image_size of 32, 48, 64, 96 or 128")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    records = collect_records(dataset_root, max_images_per_class)
    train_rows, val_rows, test_rows = split_records(records, seed)
    if progress:
        progress({"stage": "split", "train_rows": len(train_rows), "validation_rows": len(val_rows), "test_rows": len(test_rows)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrowsinessCNN().to(device)
    train_loader = _loader(train_rows, image_size, batch_size, True, True, focus)
    val_loader = _loader(val_rows, image_size, batch_size, False, False, focus)
    test_loader = _loader(test_rows, image_size, batch_size, False, False, focus)
    train_counts = np.bincount([record.label for record in train_rows], minlength=2).astype(np.float32)
    class_weights = torch.tensor(train_counts.sum() / (2.0 * np.maximum(train_counts, 1.0)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history: list[dict] = []
    best_state = deepcopy(model.state_dict())
    best_val_f1 = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        val_metrics = _evaluate(model, val_loader, device)
        epoch_result = {"epoch": epoch, "loss": round(running_loss / max(seen, 1), 4), **val_metrics}
        history.append(epoch_result)
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        if progress:
            progress({"stage": "training", **epoch_result})

    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, device)
    test_group_metrics = _evaluate_groups(model, test_rows, image_size, batch_size, device, focus)
    drowsy_recall = test_metrics["recall_by_class"][1]
    deployment_ready = test_metrics["f1_macro"] >= 0.70 and drowsy_recall >= 0.80
    generalization_gap = round(max(0.0, best_val_f1 - test_metrics["f1_macro"]), 4)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact = output_path / f"drowsiness_cnn_{focus}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "labels": ["Non Drowsy", "Drowsy"],
        "image_size": image_size,
        "architecture": "DrowsinessCNN",
        "focus": focus,
        "seed": seed,
    }, artifact)
    result = {
        **test_metrics,
        "algorithm": "image_cnn",
        "device": str(device),
        "artifact": str(artifact),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows": len(test_rows),
        "groups": len({r.group for r in records}),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_f1": best_val_f1,
        "generalization_gap": generalization_gap,
        "test_group_metrics": test_group_metrics,
        "drowsy_recall": drowsy_recall,
        "deployment_ready": deployment_ready,
        "quality_note": "Passed hold-out quality gate" if deployment_ready else "Not ready for live alerts; unseen-group performance needs improvement",
        "focus": focus,
        "image_size": image_size,
        "history": history,
    }
    if progress:
        progress({"stage": "complete", **{k: result[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}})
    return result
