from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def test_pretrain_data_schema_basic() -> None:
    data_path = Path(os.getenv("DATA_PATH", "tests/fixtures/sample_dataset.csv"))
    assert data_path.exists(), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)
    required_cols = {"review", "sentiment"}
    missing = required_cols - set(df.columns)

    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df["review"].notna().all(), "review contains missing values"
    assert df["sentiment"].notna().all(), "sentiment contains missing values"
    assert df["review"].astype(str).str.strip().ne("").all(), "review contains empty strings"
    assert df["sentiment"].isin(["positive", "negative"]).all(), "unexpected sentiment labels found"
    assert len(df) >= 10, "Too few rows for a learning experiment"


def test_posttrain_artifacts_exist() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    assert (artifact_dir / "model.joblib").exists(), "model.joblib not found"
    assert (artifact_dir / "metrics.json").exists(), "metrics.json not found"
    assert (artifact_dir / "confusion_matrix.png").exists(), "confusion_matrix.png not found"


def test_posttrain_metrics_json_has_required_keys() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"


def test_posttrain_quality_gate_f1() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))

    f1 = float(metrics["f1"])
    assert f1 >= threshold, f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"
