from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stages.train import TextPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register trained model in MLflow.")
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory that contains model.pkl or model.joblib.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentiment_model",
        help="Registered model name in MLflow Model Registry.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        help="Target stage for the registered model version.",
    )
    return parser.parse_args()


def resolve_model_path(model_dir: Path) -> Path:
    pkl_path = model_dir / "model.pkl"
    joblib_path = model_dir / "model.joblib"
    if pkl_path.exists():
        return pkl_path
    if joblib_path.exists():
        return joblib_path
    raise FileNotFoundError(
        f"Model artifact not found in {model_dir}. Expected model.pkl or model.joblib."
    )


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model_dir)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)
    experiment_name = "airflow_registration"
    artifact_location = f"{args.tracking_uri.rstrip('/')}/airflow_registration_artifacts"
    existing_experiment = mlflow.get_experiment_by_name(experiment_name)
    if existing_experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(experiment_name)
    model = joblib.load(model_path)

    with mlflow.start_run(run_name="airflow_model_registration") as run:
        with TemporaryDirectory() as tmp_dir:
            local_model_dir = Path(tmp_dir) / "model"
            mlflow.sklearn.save_model(
                sk_model=model,
                path=str(local_model_dir),
            )
            mlflow.log_artifacts(str(local_model_dir), artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=args.model_name,
        )
        client = MlflowClient()
        client.transition_model_version_stage(
            name=args.model_name,
            version=model_version.version,
            stage=args.stage,
        )
        print(
            f"Registered model '{args.model_name}' version {model_version.version} and moved it to {args.stage}."
        )
        print(f"Run id: {run.info.run_id}")


if __name__ == "__main__":
    main()
