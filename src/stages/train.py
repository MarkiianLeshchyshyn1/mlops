from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TextPreprocessor:
    _html_pattern = re.compile(r"<[^>]+>")
    _spaces_pattern = re.compile(r"\s+")

    @classmethod
    def normalize(cls, text: str) -> str:
        text = str(text).lower()
        text = cls._html_pattern.sub(" ", text)
        text = cls._spaces_pattern.sub(" ", text)
        return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train model on prepared data.")
    parser.add_argument(
        "prepared_dir", type=Path, help="Directory with train.csv and test.csv"
    )
    parser.add_argument(
        "model_dir", type=Path, help="Directory where trained model is stored"
    )
    parser.add_argument("--text-column", type=str, default="review")
    parser.add_argument("--target-column", type=str, default="sentiment")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment-name", type=str, default="imdb_dvc_pipeline")
    parser.add_argument("--run-name", type=str, default="lr_tfidf_pipeline")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--c-value", type=float, default=1.0)
    return parser.parse_args()


def encode_target(y: pd.Series) -> pd.Series:
    mapping = {"negative": 0, "positive": 1}
    encoded = y.map(mapping)
    if encoded.isna().any():
        raise ValueError(
            "Target contains unsupported labels. Expected positive/negative."
        )
    return encoded.astype(int)


def main() -> None:
    args = parse_args()
    train_path = args.prepared_dir / "train.csv"
    test_path = args.prepared_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected prepared files: {train_path} and {test_path}"
        )

    args.model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df[args.text_column].astype(str)
    X_test = test_df[args.text_column].astype(str)
    y_train = encode_target(
        train_df[args.target_column].astype(str).str.strip().str.lower()
    )
    y_test = encode_target(
        test_df[args.target_column].astype(str).str.strip().str.lower()
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=TextPreprocessor.normalize,
                    strip_accents="unicode",
                    lowercase=False,
                    max_features=args.max_features,
                    min_df=args.min_df,
                    ngram_range=(1, args.ngram_max),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.c_value,
                    max_iter=1200,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    model_path = args.model_dir / "model.joblib"
    model_pkl_path = args.model_dir / "model.pkl"
    metrics_path = args.model_dir / "metrics.json"
    confusion_matrix_path = args.model_dir / "confusion_matrix.png"
    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, model_pkl_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["negative", "positive"]
    )
    display.plot(cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "text_column": args.text_column,
                "max_features": args.max_features,
                "min_df": args.min_df,
                "ngram_max": args.ngram_max,
                "c_value": args.c_value,
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="evaluation")
        try:
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")
        except PermissionError:
            # Some restricted environments block MLflow temp directories.
            # Local artifacts are already saved in model_dir and remain usable.
            print(
                "Warning: skipped MLflow model logging due to filesystem permissions."
            )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model artifact: {model_path}")
    print(f"Saved model artifact: {model_pkl_path}")


if __name__ == "__main__":
    main()
