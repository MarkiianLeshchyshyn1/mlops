from __future__ import annotations

from pathlib import Path

from airflow.models import DagBag


def test_dag_imports_without_errors() -> None:
    dag_folder = Path(__file__).resolve().parents[1] / "dags"
    dag_bag = DagBag(dag_folder=str(dag_folder), include_examples=False)

    assert len(dag_bag.import_errors) == 0, (
        f"DAG import errors: {dag_bag.import_errors}"
    )
    assert "ml_training_pipeline" in dag_bag.dags
