from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class DataConfig:
    data_path: Path
    text_column: str = "review"
    target_column: str = "sentiment"
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    max_features: int
    min_df: int
    ngram_max: int
    c_value: float
    class_weight: str | None


class DatasetLoader:
    """Loads and validates CSV data for text classification."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def load(self) -> pd.DataFrame:
        if not self.config.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.config.data_path}")

        df = pd.read_csv(self.config.data_path, low_memory=False)
        required = {self.config.text_column, self.config.target_column}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        cleaned = df[[self.config.text_column, self.config.target_column]].dropna()
        cleaned = cleaned[
            cleaned[self.config.text_column].astype(str).str.strip() != ""
        ]
        cleaned[self.config.target_column] = (
            cleaned[self.config.target_column].astype(str).str.strip().str.lower()
        )
        return cleaned.reset_index(drop=True)


class TextPreprocessor:
    """Minimal text normalization that keeps signal while removing HTML/noise."""

    _html_pattern = re.compile(r"<[^>]+>")
    _spaces_pattern = re.compile(r"\s+")

    @classmethod
    def normalize(cls, text: str) -> str:
        text = str(text).lower()
        text = cls._html_pattern.sub(" ", text)
        text = cls._spaces_pattern.sub(" ", text)
        return text.strip()


class SentimentExperimentRunner:
    def __init__(self, data_config: DataConfig, experiment_name: str) -> None:
        self.data_config = data_config
        self.experiment_name = experiment_name

    def _build_pipeline(self, cfg: ExperimentConfig) -> Pipeline:
        vectorizer = TfidfVectorizer(
            preprocessor=TextPreprocessor.normalize,
            strip_accents="unicode",
            lowercase=False,
            max_features=cfg.max_features,
            min_df=cfg.min_df,
            ngram_range=(1, cfg.ngram_max),
            sublinear_tf=True,
        )
        classifier = LogisticRegression(
            C=cfg.c_value,
            max_iter=1200,
            solver="liblinear",
            class_weight=cfg.class_weight,
            random_state=self.data_config.random_state,
        )
        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("clf", classifier),
            ]
        )

    @staticmethod
    def _encode_target(y: pd.Series) -> pd.Series:
        mapping = {"negative": 0, "positive": 1}
        encoded = y.map(mapping)
        if encoded.isna().any():
            uniques = sorted(y.unique().tolist())
            raise ValueError(
                "Target contains unsupported labels. "
                f"Expected positive/negative, got: {uniques}"
            )
        return encoded.astype(int)

    @staticmethod
    def _metric_dict(
        y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        }

    @staticmethod
    def _save_json(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self, cfg: ExperimentConfig) -> dict[str, Any]:
        loader = DatasetLoader(self.data_config)
        df = loader.load()
        X = df[self.data_config.text_column].astype(str)
        y = self._encode_target(df[self.data_config.target_column])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.data_config.test_size,
            random_state=self.data_config.random_state,
            stratify=y,
        )

        pipeline = self._build_pipeline(cfg)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = self._metric_dict(y_test.to_numpy(), y_pred, y_prob)

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=cfg.run_name):
            mlflow.log_params(
                {
                    "dataset_path": str(self.data_config.data_path),
                    "text_column": self.data_config.text_column,
                    "target_column": self.data_config.target_column,
                    "test_size": self.data_config.test_size,
                    "random_state": self.data_config.random_state,
                    "max_features": cfg.max_features,
                    "min_df": cfg.min_df,
                    "ngram_max": cfg.ngram_max,
                    "C": cfg.c_value,
                    "class_weight": cfg.class_weight or "none",
                    "rows_total": int(len(df)),
                    "rows_train": int(len(X_train)),
                    "rows_test": int(len(X_test)),
                }
            )
            mlflow.log_metrics(metrics)

            report = classification_report(
                y_test, y_pred, target_names=["negative", "positive"]
            )
            matrix = confusion_matrix(y_test, y_pred).tolist()
            artifacts_payload = {
                "classification_report": report,
                "confusion_matrix": matrix,
            }

            with TemporaryDirectory() as tmp_dir:
                output_dir = Path(tmp_dir)
                report_path = output_dir / "evaluation.json"
                self._save_json(report_path, artifacts_payload)
                mlflow.log_artifact(str(report_path), artifact_path="evaluation")

            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")

        return {"run_name": cfg.run_name, **metrics}


def default_experiments() -> list[ExperimentConfig]:
    # 6 experiments so the assignment requirement of >=5 is always satisfied.
    return [
        ExperimentConfig(
            "lr_uni_baseline",
            max_features=12000,
            min_df=2,
            ngram_max=1,
            c_value=1.0,
            class_weight=None,
        ),
        ExperimentConfig(
            "lr_uni_strong_reg",
            max_features=12000,
            min_df=2,
            ngram_max=1,
            c_value=0.5,
            class_weight=None,
        ),
        ExperimentConfig(
            "lr_uni_weak_reg",
            max_features=12000,
            min_df=2,
            ngram_max=1,
            c_value=2.0,
            class_weight=None,
        ),
        ExperimentConfig(
            "lr_bi_baseline",
            max_features=20000,
            min_df=2,
            ngram_max=2,
            c_value=1.0,
            class_weight=None,
        ),
        ExperimentConfig(
            "lr_bi_min_df_3",
            max_features=20000,
            min_df=3,
            ngram_max=2,
            c_value=1.0,
            class_weight=None,
        ),
        ExperimentConfig(
            "lr_bi_balanced",
            max_features=25000,
            min_df=2,
            ngram_max=2,
            c_value=1.0,
            class_weight="balanced",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IMDB sentiment model and track experiments with MLflow."
    )
    parser.add_argument("--data-path", type=Path, default=Path("data/dataset.csv"))
    parser.add_argument("--text-column", type=str, default="review")
    parser.add_argument("--target-column", type=str, default="sentiment")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--experiment-name", type=str, default="imdb_sentiment_baselines"
    )
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)

    data_config = DataConfig(
        data_path=args.data_path,
        text_column=args.text_column,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    runner = SentimentExperimentRunner(
        data_config=data_config, experiment_name=args.experiment_name
    )

    results = []
    for exp_cfg in default_experiments():
        result = runner.run(exp_cfg)
        results.append(result)
        print(
            f"{result['run_name']}: "
            f"f1={result['f1']:.4f}, accuracy={result['accuracy']:.4f}, roc_auc={result['roc_auc']:.4f}"
        )

    leaderboard = (
        pd.DataFrame(results)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    print("\nLeaderboard (top by f1):")
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()
