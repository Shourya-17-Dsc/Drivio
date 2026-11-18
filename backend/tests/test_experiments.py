import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.ml import experiments
from app.ml.experiments import FEATURES, LABEL, _feature_importance


def test_tree_feature_importance_is_ranked_and_normalized():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, len(FEATURES)))
    y = (x[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=20, random_state=2).fit(x, y)
    result = _feature_importance(model, "random_forest")
    assert len(result) == len(FEATURES)
    assert next(item["importance"] for item in result if item["feature"] == FEATURES[0]) > 0.4
    assert round(sum(item["importance"] for item in result), 3) == 1.0


@pytest.mark.parametrize(
    ("feature_values", "label", "message"),
    [
        ("not-a-number", "alert", "must contain numeric values"),
        ("inf", "alert", "must contain finite values"),
        ("0.2", "   ", "State labels cannot be empty"),
    ],
)
def test_csv_upload_rejects_invalid_values(tmp_path, monkeypatch, feature_values, label, message):
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path)
    row = {feature: 0.2 for feature in FEATURES}
    row[FEATURES[0]] = feature_values
    row[LABEL] = label
    content = pd.DataFrame([row]).to_csv(index=False).encode()

    with pytest.raises(ValueError, match=message):
        experiments.save_upload("invalid", content)


def test_csv_upload_rejects_empty_or_malformed_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "DATA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Invalid CSV"):
        experiments.save_upload("empty", b"")
