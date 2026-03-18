from __future__ import annotations

import json
import os
from pathlib import Path


def test_artifacts_exist() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    assert (artifact_dir / "model.joblib").exists(), "model.joblib not found"
    assert (artifact_dir / "metrics.json").exists(), "metrics.json not found"
    assert (artifact_dir / "confusion_matrix.png").exists(), "confusion_matrix.png not found"


def test_metrics_json_has_required_keys() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics, f"Missing metric: {key}"


def test_quality_gate_f1() -> None:
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", ".ci/model"))
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    metrics_path = artifact_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    f1 = float(metrics["f1"])
    assert f1 >= threshold, f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"
