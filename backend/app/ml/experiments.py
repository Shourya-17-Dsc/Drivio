"""Dataset management, model training, evaluation and comparison."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import DATA_DIR
from ..database import ModelRun, Sample

FEATURES = ["ear", "mar", "perclos", "closure_s", "blink_rate", "yawns_recent", "yaw", "pitch", "head_turned_s"]
LABEL = "state"
ALGORITHMS = {
    "random_forest": lambda: make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=0)),
    "gradient_boosting": lambda: make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0)),
    "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
}


def _feature_importance(model, algorithm: str) -> list[dict]:
    """Return normalized global importance for estimators that expose it."""
    try:
        estimator = model[-1]
    except (TypeError, IndexError, KeyError):
        estimator = model
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        return []
    total = float(values.sum())
    if total <= 0:
        return []
    values = values / total
    return [{"feature": feature, "importance": round(float(value), 4)}
            for feature, value in sorted(zip(FEATURES, values), key=lambda pair: pair[1], reverse=True)]


def _dirs() -> tuple[Path, Path]:
    ds, models = DATA_DIR / "datasets", DATA_DIR / "models"
    ds.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    return ds, models


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem).strip("_")
    if not name:
        raise ValueError("Dataset name is empty")
    return name


def list_datasets() -> List[dict]:
    out = []
    for p in sorted(_dirs()[0].glob("*.csv")):
        df = pd.read_csv(p)
        out.append({"name": p.stem, "rows": len(df),
                    "classes": df[LABEL].value_counts().to_dict() if LABEL in df else {}})
    return out


def save_upload(name: str, content: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(content))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid CSV: {exc}") from exc
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("CSV must contain at least one data row")
    for feature in FEATURES:
        try:
            values = pd.to_numeric(df[feature], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature column '{feature}' must contain numeric values") from exc
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Feature column '{feature}' must contain finite values")
        df[feature] = values
    labels = df[LABEL].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError("State labels cannot be empty")
    df[LABEL] = labels
    safe = _safe(name)
    df[FEATURES + [LABEL]].to_csv(_dirs()[0] / f"{safe}.csv", index=False)
    return {"name": safe, "rows": len(df)}


def export_sessions(db: Session, name: str) -> dict:
    rows = db.scalars(select(Sample)).all()
    if not rows:
        raise ValueError("No recorded samples yet. Record a session first.")
    df = pd.DataFrame([{**{f: getattr(r, f) for f in FEATURES}, LABEL: r.state} for r in rows])
    safe = _safe(name)
    df.to_csv(_dirs()[0] / f"{safe}.csv", index=False)
    return {"name": safe, "rows": len(df)}


def export_labeled_sessions(db: Session, name: str) -> dict:
    """Export only samples explicitly annotated during data collection."""
    rows = db.scalars(select(Sample).where(Sample.ground_truth.is_not(None))).all()
    if not rows:
        raise ValueError("No labelled samples yet. Start collection and assign a ground-truth label first.")
    df = pd.DataFrame([{**{f: getattr(r, f) for f in FEATURES}, LABEL: r.ground_truth} for r in rows])
    safe = _safe(name)
    df.to_csv(_dirs()[0] / f"{safe}.csv", index=False)
    return {"name": safe, "rows": len(df)}


def train(db: Session, dataset: str, algorithm: str, test_size: float = 0.25) -> dict:
    ds_dir, model_dir = _dirs()
    path = ds_dir / f"{_safe(dataset)}.csv"
    if not path.exists():
        raise ValueError(f"Unknown dataset: {dataset}")
    df = pd.read_csv(path)
    if df[LABEL].nunique() < 2:
        raise ValueError("The dataset needs at least two driver states to train on.")

    X, y = df[FEATURES].to_numpy(dtype=float), df[LABEL].to_numpy()
    counts = pd.Series(y).value_counts()
    stratify = y if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=0, stratify=stratify)

    if algorithm == "mlp_torch":
        from .torch_model import TorchMLPClassifier

        scaler = StandardScaler().fit(X_tr)
        model = make_pipeline(scaler, TorchMLPClassifier())
    elif algorithm in ALGORITHMS:
        model = ALGORITHMS[algorithm]()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    labels = sorted(set(y))
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_te, pred, labels=labels).tolist(),
        "report": classification_report(y_te, pred, output_dict=True, zero_division=0),
        "feature_importance": _feature_importance(model, algorithm),
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
    }
    run = ModelRun(algorithm=algorithm, dataset=path.stem, metrics_json=json.dumps(metrics), model_path="")
    db.add(run)
    db.flush()
    out = model_dir / f"run_{run.id}_{algorithm}.joblib"
    joblib.dump({"model": model, "features": FEATURES}, out)
    run.model_path = str(out)
    db.commit()
    return run_to_dict(run)


def run_to_dict(r: ModelRun) -> dict:
    m = json.loads(r.metrics_json)
    return {"id": r.id, "created_at": r.created_at.isoformat(), "algorithm": r.algorithm, "dataset": r.dataset,
            "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "labels": m["labels"],
            "confusion_matrix": m["confusion_matrix"], "train_rows": m["train_rows"], "test_rows": m["test_rows"],
            "feature_importance": m.get("feature_importance", []),
            "deployment_status": r.deployment_status or "candidate", "deployment_ready": bool(m.get("deployment_ready", False)),
            "quality_note": m.get("quality_note", "No deployment-quality gate was recorded")}


def list_runs(db: Session) -> List[dict]:
    return [run_to_dict(r) for r in db.scalars(select(ModelRun).order_by(ModelRun.id.desc())).all()]


def compare(db: Session, dataset: str | None = None) -> List[dict]:
    runs = [r for r in list_runs(db) if dataset in (None, "", r["dataset"])]
    return sorted(runs, key=lambda r: (r["f1_macro"], r["accuracy"]), reverse=True)


def make_synthetic_dataset(name: str = "synthetic_demo", n_per_class: int = 300, seed: int = 0) -> dict:
    """Gaussian clusters per driver state, so the experiments page works before any real data exists."""
    rng = np.random.default_rng(seed)
    #            ear   mar  perclos closure blink yawns  yaw  pitch turn
    spec = {
        "alert":      [0.30, 0.08, 0.04, 0.0, 17, 0, 0, 0, 0.0],
        "fatigued":   [0.25, 0.20, 0.16, 0.5, 11, 2, 0, 6, 0.0],
        "drowsy":     [0.18, 0.30, 0.34, 1.6, 6, 3, 0, 18, 0.0],
        "distracted": [0.29, 0.10, 0.05, 0.0, 16, 0, 38, 4, 3.0],
    }
    spread = np.array([0.03, 0.05, 0.05, 0.4, 3, 0.8, 6, 5, 0.8])
    frames = []
    for label, mean in spec.items():
        x = rng.normal(mean, spread, size=(n_per_class, len(FEATURES)))
        x = np.clip(x, 0, None)
        x[:, 5] = np.round(x[:, 5])
        df = pd.DataFrame(x, columns=FEATURES)
        df[LABEL] = label
        frames.append(df)
    out = pd.concat(frames).sample(frac=1, random_state=seed)
    safe = _safe(name)
    out.to_csv(_dirs()[0] / f"{safe}.csv", index=False)
    return {"name": safe, "rows": len(out)}
