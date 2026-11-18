"""Inspection utilities for image datasets used by the drowsiness model.

The audit deliberately does not copy or persist images. It reads image headers
and verifies that the image library can open each file without decoding all
pixel data, keeping a large local dataset scan responsive.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:  # The API can still audit files in a minimal test environment.
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - production requirements include an image reader
    Image = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("Drowsy", "Non Drowsy")
INSPECTION_LIMIT_PER_CLASS = 200


def _group_prefix(path: Path) -> str:
    """Return the case-preserving leading token for grouped splitting.

    The supplied dataset uses uppercase and lowercase prefixes for different
    image series, so normalizing case would incorrectly merge groups.
    """
    token = ""
    for char in path.stem:
        if not char.isalpha():
            break
        token += char
    return token or "UNKNOWN"


def _read_dimensions(path: Path) -> tuple[int, int] | None:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return None if image is None else (int(image.shape[1]), int(image.shape[0]))
    if Image is not None:
        try:
            with Image.open(path) as image:
                return image.width, image.height
        except (OSError, ValueError):
            return None
    return None


def audit_image_dataset(root: str | Path) -> dict[str, Any]:
    """Audit the expected ``Drowsy`` and ``Non Drowsy`` folders.

    A missing class folder is reported as a validation issue rather than
    raising an exception, so the UI can explain what needs to be fixed.
    """
    dataset_root = Path(root)
    classes: dict[str, Any] = {}
    issues: list[str] = []
    total_files = 0
    total_readable = 0
    total_corrupt = 0
    dimensions: Counter[str] = Counter()
    groups: Counter[str] = Counter()

    for class_name in CLASS_NAMES:
        folder = dataset_root / class_name
        files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        image_files = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
        ignored = len(files) - len(image_files)
        if not folder.is_dir():
            issues.append(f"Missing class folder: {class_name}")

        # Count all image files immediately, then inspect only a bounded
        # sample. This keeps the API responsive for large synced datasets.
        readable = len(image_files)
        corrupt = 0
        class_dimensions: Counter[str] = Counter()
        class_groups: Counter[str] = Counter()
        for path in image_files:
            class_groups[_group_prefix(path)] += 1
            groups[_group_prefix(path)] += 1
        inspected = 0
        for path in image_files[:INSPECTION_LIMIT_PER_CLASS]:
            inspected += 1
            dimensions_for_file = _read_dimensions(path)
            if dimensions_for_file is None:
                corrupt += 1
                readable -= 1
                continue
            width, height = dimensions_for_file
            key = f"{width}x{height}"
            class_dimensions[key] += 1
            dimensions[key] += 1

        if corrupt:
            issues.append(f"{class_name} contains {corrupt} unreadable image(s)")
        if ignored:
            issues.append(f"{class_name} contains {ignored} non-image file(s)")

        total_files += len(image_files)
        total_readable += readable
        total_corrupt += corrupt
        classes[class_name] = {
            "files": len(image_files),
            "readable": readable,
            "corrupt": corrupt,
            "inspected": inspected,
            "ignored": ignored,
            "dimensions": dict(class_dimensions.most_common(10)),
            "groups": dict(class_groups.most_common(20)),
        }

    if total_readable == 0:
        issues.append("No readable images found")
    else:
        for class_name in CLASS_NAMES:
            if classes[class_name]["readable"] == 0:
                issues.append(f"Class has no readable images: {class_name}")

    counts = {name: classes[name]["readable"] for name in CLASS_NAMES}
    largest = max(counts.values(), default=0)
    smallest = min(counts.values(), default=0)
    balance_ratio = round(smallest / largest, 4) if largest else 0.0

    return {
        "root": str(dataset_root),
        "available": dataset_root.is_dir(),
        "ready_for_training": total_readable > 0 and not any(
            classes[name]["readable"] == 0 for name in CLASS_NAMES
        ),
        "classes": classes,
        "total_files": total_files,
        "total_readable": total_readable,
        "total_corrupt": total_corrupt,
        "inspection_limit_per_class": INSPECTION_LIMIT_PER_CLASS,
        "verification": "sampled",
        "class_balance_ratio": balance_ratio,
        "dimensions": dict(dimensions.most_common(10)),
        "group_count": len(groups),
        "group_prefixes": dict(groups.most_common(30)),
        "issues": issues,
    }
