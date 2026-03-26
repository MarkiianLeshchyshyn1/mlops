from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator


PROJECT_DIR = Path("/opt/airflow/project")
DATA_FILE = PROJECT_DIR / "tests" / "fixtures" / "sample_dataset.csv"
MODEL_DIR = PROJECT_DIR / "data" / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"
QUALITY_THRESHOLD = 0.70


def choose_next_step() -> str:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    f1_score = float(metrics.get("f1", 0.0))

    if f1_score >= QUALITY_THRESHOLD:
        return "register_model"
    return "stop_pipeline"


def check_data_exists() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input data  file not found: {DATA_FILE}")


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "lab"],
) as dag:
    check_data = PythonOperator(
        task_id="check_data",
        python_callable=check_data_exists,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command="cd /opt/airflow/project && dvc repro prepare",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/project && dvc repro train",
    )

    branch_on_metrics = BranchPythonOperator(
        task_id="branch_on_metrics",
        python_callable=choose_next_step,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=(
            "cd /opt/airflow/project && "
            "python src/stages/register.py data/models "
            "--tracking-uri file:/opt/airflow/project/mlruns "
            "--model-name sentiment_model "
            "--stage Staging"
        ),
    )

    stop_pipeline = EmptyOperator(task_id="stop_pipeline")

    check_data >> prepare_data >> train_model >> branch_on_metrics
    branch_on_metrics >> register_model
    branch_on_metrics >> stop_pipeline
