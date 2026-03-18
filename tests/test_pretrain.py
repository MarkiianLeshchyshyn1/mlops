from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml


def test_data_schema_basic() -> None:
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
    assert len(df) >= 10, "Too few rows for a smoke training run"


def test_hydra_config_basic() -> None:
    config_path = Path("config/config.yaml")
    hpo_path = Path("config/hpo/optuna.yaml")
    model_path = Path("config/model/logistic_regression.yaml")

    assert config_path.exists(), f"Missing config: {config_path}"
    assert hpo_path.exists(), f"Missing config: {hpo_path}"
    assert model_path.exists(), f"Missing config: {model_path}"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    hpo_config = yaml.safe_load(hpo_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_path.read_text(encoding="utf-8"))

    assert config["data"]["text_column"] == "review"
    assert config["data"]["target_column"] == "sentiment"
    assert hpo_config["n_trials"] >= 1
    assert hpo_config["sampler"] in {"tpe", "random"}
    assert model_config["type"] == "logistic_regression"
