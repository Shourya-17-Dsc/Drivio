"""Session-aware GRU training from explicitly labelled metric sequences."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURES = ["ear", "mar", "perclos", "closure_s", "blink_rate", "yawns_recent", "yaw", "pitch", "head_turned_s", "safety_score"]


def labelled_dataset_status(rows: Iterable) -> dict:
    """Summarize whether labelled sessions are sufficient for a grouped GRU split."""
    rows = list(rows)
    session_labels: dict[str, set[str]] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.ground_truth)
        session = str(row.session_id)
        session_labels.setdefault(session, set()).add(label)
        label_counts[label] = label_counts.get(label, 0) + 1
    sessions_by_label = {
        label: sum(label in labels for labels in session_labels.values())
        for label in sorted(label_counts)
    }
    session_count = len(session_labels)
    has_split_coverage = len(label_counts) >= 2 and all(count >= 3 for count in sessions_by_label.values())
    ready = session_count >= 6 and has_split_coverage
    if ready:
        reason = "Ready for grouped temporal training"
    elif not rows:
        reason = "Collect and label samples before training"
    elif len(label_counts) < 2:
        reason = "At least two labelled conditions are required"
    elif not has_split_coverage:
        reason = "Each condition must appear in at least three labelled sessions"
    else:
        reason = f"Collect at least {6 - session_count} more labelled sessions"
    return {
        "labelled_samples": len(rows),
        "labelled_sessions": session_count,
        "labels": label_counts,
        "sessions_by_label": sessions_by_label,
        "minimum_sessions_required": 6,
        "ready_for_temporal_training": ready,
        "reason": reason,
    }


@dataclass(frozen=True)
class SequenceRecord:
    values: np.ndarray
    label: str
    group: str


def build_sequences(rows: Iterable, window_size: int = 10, stride: int = 5) -> list[SequenceRecord]:
    """Build windows from labelled samples grouped by recording session."""
    if not 3 <= window_size <= 120 or stride < 1:
        raise ValueError("window_size must be 3-120 and stride must be positive")
    grouped: dict[str, list] = {}
    for row in rows:
        if row.ground_truth:
            grouped.setdefault(str(row.session_id), []).append(row)
    sequences: list[SequenceRecord] = []
    for group, samples in grouped.items():
        samples.sort(key=lambda sample: sample.t)
        for start in range(0, len(samples) - window_size + 1, stride):
            window = samples[start:start + window_size]
            values = np.array([[float(getattr(sample, feature) or 0.0) for feature in FEATURES] for sample in window], dtype=np.float32)
            # The target is the condition at the end of the temporal window.
            sequences.append(SequenceRecord(values, str(window[-1].ground_truth), group))
    return sequences


def split_sequences(records: list[SequenceRecord], seed: int = 42):
    groups = np.array([record.group for record in records])
    if len(set(groups)) < 6:
        raise ValueError("At least 6 labelled sessions are required for temporal evaluation")
    indices = np.arange(len(records))
    labels = {record.label for record in records}
    for attempt in range(100):
        first = GroupShuffleSplit(n_splits=1, test_size=2 / 3, random_state=seed + attempt)
        train_idx, held_idx = next(first.split(indices, groups=groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + attempt + 1)
        val_sub, test_sub = next(second.split(held_idx, groups=groups[held_idx]))
        train = [records[i] for i in train_idx]
        validation = [records[held_idx[i]] for i in val_sub]
        test = [records[held_idx[i]] for i in test_sub]
        if all({record.label for record in split} == labels for split in (train, validation, test)):
            return train, validation, test
    raise ValueError("Each train, validation, and test split must contain every label; collect more balanced sessions")


class SequenceDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: list[SequenceRecord], labels: dict[str, int], mean: np.ndarray, std: np.ndarray) -> None:
        self.records, self.labels = records, labels
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        values = (record.values - self.mean) / self.std
        return torch.tensor(values, dtype=torch.float32), self.labels[record.label]


class TemporalGRU(nn.Module):
    def __init__(self, input_size: int, class_count: int, hidden: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden, class_count))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(values)
        return self.classifier(output[:, -1, :])


def _evaluate(model, loader, device, labels: list[str]) -> dict:
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for values, target in loader:
            predictions.extend(model(values.to(device)).argmax(dim=1).cpu().tolist())
            truth.extend(target.tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(range(len(labels))), zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(truth, predictions)), 4),
        "precision_macro": round(float(precision.mean()), 4),
        "recall_macro": round(float(recall.mean()), 4),
        "f1_macro": round(float(f1.mean()), 4),
        "labels": labels,
        "precision_by_class": [round(float(x), 4) for x in precision],
        "recall_by_class": [round(float(x), 4) for x in recall],
        "f1_by_class": [round(float(x), 4) for x in f1],
        "support_by_class": [int(x) for x in support],
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(range(len(labels)))).tolist(),
        "rows": len(truth),
    }


def train_temporal_model(records: list[SequenceRecord], output_path: str, *, epochs: int = 10,
                         batch_size: int = 32, seed: int = 42,
                         progress: Callable[[dict], None] | None = None) -> dict:
    if not 1 <= epochs <= 50 or batch_size < 1:
        raise ValueError("epochs must be 1-50 and batch_size must be positive")
    if not records:
        raise ValueError("No labelled sequences found")
    labels = sorted({record.label for record in records})
    if len(labels) < 2:
        raise ValueError("At least two labelled conditions are required")
    train_rows, val_rows, test_rows = split_sequences(records, seed)
    label_ids = {label: index for index, label in enumerate(labels)}
    train_values = np.concatenate([record.values for record in train_rows], axis=0)
    mean = train_values.mean(axis=0).astype(np.float32)
    std = np.maximum(train_values.std(axis=0), 1e-4).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalGRU(len(FEATURES), len(labels)).to(device)
    train_loader = DataLoader(SequenceDataset(train_rows, label_ids, mean, std), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SequenceDataset(val_rows, label_ids, mean, std), batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(SequenceDataset(test_rows, label_ids, mean, std), batch_size=batch_size, num_workers=0)
    counts = np.bincount([label_ids[r.label] for r in train_rows], minlength=len(labels)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state, best_f1, best_epoch, history = deepcopy(model.state_dict()), -1.0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for values, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, val_loader, device, labels)
        result = {"epoch": epoch, **metrics}
        history.append(result)
        if metrics["f1_macro"] > best_f1:
            best_f1, best_epoch, best_state = metrics["f1_macro"], epoch, deepcopy(model.state_dict())
        if progress:
            progress({"stage": "training", **result})
    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, test_loader, device, labels)
    deployment_ready = test_metrics["f1_macro"] >= 0.70
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": labels, "features": FEATURES,
                "mean": mean.tolist(), "std": std.tolist(), "window_size": len(records[0].values)}, output_path)
    return {**test_metrics, "algorithm": "gru_temporal", "artifact": output_path, "device": str(device),
            "train_rows": len(train_rows), "validation_rows": len(val_rows), "test_rows": len(test_rows),
            "groups": len({record.group for record in records}), "epochs": epochs, "best_epoch": best_epoch,
            "best_validation_f1": best_f1, "deployment_ready": deployment_ready,
            "quality_note": "Passed hold-out quality gate" if deployment_ready else "Not ready for live alerts; collect more labelled sessions",
            "history": history}
