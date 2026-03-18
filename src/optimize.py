from __future__ import annotations

import json
import random
import subprocess
from hashlib import sha256
from pathlib import Path

import hydra
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


class TextPreprocessor:
    @staticmethod
    def normalize(text: str) -> str:
        text = str(text).lower()
        text = text.replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
        return " ".join(text.split())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def encode_target(y: pd.Series) -> pd.Series:
    mapping = {"negative": 0, "positive": 1}
    encoded = y.map(mapping)
    if encoded.isna().any():
        raise ValueError("Target contains unsupported labels. Expected positive/negative.")
    return encoded.astype(int)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def load_data(cfg: DictConfig) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    prepared_dir = Path(cfg.data.prepared_dir)
    train_path = prepared_dir / "train.csv"
    test_path = prepared_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Expected prepared files in {prepared_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df[cfg.data.text_column].astype(str)
    y_train_full = encode_target(train_df[cfg.data.target_column].astype(str).str.strip().str.lower())
    X_test = test_df[cfg.data.text_column].astype(str)
    y_test = encode_target(test_df[cfg.data.target_column].astype(str).str.strip().str.lower())

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=cfg.data.validation_size,
        random_state=cfg.seed,
        stratify=y_train_full,
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def build_pipeline(cfg: DictConfig, params: dict[str, object]) -> Pipeline:
    class_weight = None if params["class_weight"] == "none" else params["class_weight"]

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=TextPreprocessor.normalize,
                    strip_accents="unicode",
                    lowercase=False,
                    max_features=int(params["max_features"]),
                    min_df=int(params["min_df"]),
                    ngram_range=(1, int(params["ngram_max"])),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=float(params["c_value"]),
                    solver=str(params["solver"]),
                    penalty=str(cfg.model.fixed.penalty),
                    class_weight=class_weight,
                    max_iter=int(cfg.model.fixed.max_iter),
                    random_state=int(cfg.seed),
                ),
            ),
        ]
    )


def suggest_params(trial: optuna.Trial, cfg: DictConfig) -> dict[str, object]:
    search_space = cfg.model.search_space
    return {
        "max_features": trial.suggest_int(
            "max_features",
            int(search_space.max_features.low),
            int(search_space.max_features.high),
            step=int(search_space.max_features.step),
        ),
        "min_df": trial.suggest_categorical("min_df", list(search_space.min_df.choices)),
        "ngram_max": trial.suggest_categorical("ngram_max", list(search_space.ngram_max.choices)),
        "c_value": trial.suggest_float(
            "c_value",
            float(search_space.c_value.low),
            float(search_space.c_value.high),
            log=bool(search_space.c_value.log),
        ),
        "solver": trial.suggest_categorical("solver", list(search_space.solver.choices)),
        "class_weight": trial.suggest_categorical("class_weight", list(search_space.class_weight.choices)),
    }


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_holdout(
    model: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_eval: pd.Series,
    y_eval: pd.Series,
) -> dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    return metric_dict(y_eval, y_pred, y_prob)


def evaluate_cv(
    cfg: DictConfig,
    params: dict[str, object],
    X: pd.Series,
    y: pd.Series,
    folds: int,
    seed: int,
) -> dict[str, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[dict[str, float]] = []

    for train_idx, val_idx in splitter.split(X, y):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        fold_model = build_pipeline(cfg, params)
        scores.append(evaluate_holdout(fold_model, X_train, y_train, X_val, y_val))

    return {
        metric_name: float(np.mean([score[metric_name] for score in scores]))
        for metric_name in scores[0]
    }


def make_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    sampler_name = str(cfg.hpo.sampler).lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=int(cfg.seed), n_startup_trials=int(cfg.hpo.startup_trials))
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=int(cfg.seed))
    raise ValueError(f"Unsupported sampler: {cfg.hpo.sampler}")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def register_model_if_needed(run_id: str, cfg: DictConfig) -> None:
    if not cfg.mlflow.register_model:
        return

    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, str(cfg.mlflow.model_name))
    client.transition_model_version_stage(
        name=str(cfg.mlflow.model_name),
        version=model_version.version,
        stage=str(cfg.mlflow.stage),
    )


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    mlflow.set_experiment(f"{cfg.mlflow.experiment_prefix}_{cfg.hpo.sampler}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(cfg)
    prepared_dir = Path(cfg.data.prepared_dir)
    metadata = {
        "git_commit": git_commit_hash(),
        "train_hash_sha256": file_hash(prepared_dir / "train.csv"),
        "test_hash_sha256": file_hash(prepared_dir / "test.csv"),
        "sampler": str(cfg.hpo.sampler),
        "metric": str(cfg.hpo.metric),
        "seed": int(cfg.seed),
    }

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg)
        with mlflow.start_run(run_name=f"trial_{trial.number:03d}", nested=True):
            mlflow.set_tags(
                {
                    "trial_number": trial.number,
                    "sampler": cfg.hpo.sampler,
                    "model_type": cfg.model.type,
                    "seed": cfg.seed,
                }
            )
            mlflow.log_params(params)

            model = build_pipeline(cfg, params)
            if cfg.hpo.use_cv:
                X_cv = pd.concat([X_train, X_val], ignore_index=True)
                y_cv = pd.concat([y_train, y_val], ignore_index=True)
                metrics = evaluate_cv(cfg, params, X_cv, y_cv, int(cfg.hpo.cv_folds), int(cfg.seed))
            else:
                metrics = evaluate_holdout(model, X_train, y_train, X_val, y_val)

            mlflow.log_metrics(metrics)
            return float(metrics[str(cfg.hpo.metric)])

    with mlflow.start_run(run_name=f"{cfg.hpo.study_name_prefix}_{cfg.hpo.sampler}") as parent_run:
        mlflow.set_tags(
            {
                "sampler": cfg.hpo.sampler,
                "model_type": cfg.model.type,
                "seed": cfg.seed,
                "git_commit": metadata["git_commit"],
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")
        mlflow.log_dict(metadata, "run_metadata.json")

        study = optuna.create_study(
            direction=str(cfg.hpo.direction),
            sampler=make_sampler(cfg),
            study_name=f"{cfg.hpo.study_name_prefix}_{cfg.hpo.sampler}",
        )
        study.optimize(objective, n_trials=int(cfg.hpo.n_trials))

        best_params = dict(study.best_trial.params)
        final_model = build_pipeline(cfg, best_params)
        X_final_train = pd.concat([X_train, X_val], ignore_index=True)
        y_final_train = pd.concat([y_train, y_val], ignore_index=True)
        final_metrics = evaluate_holdout(final_model, X_final_train, y_final_train, X_test, y_test)

        model_dir = Path(cfg.artifacts.model_dir) / str(cfg.hpo.sampler)
        report_dir = Path(cfg.artifacts.report_dir) / str(cfg.hpo.sampler)
        model_path = model_dir / "best_model.joblib"
        params_path = report_dir / "best_params.json"
        metrics_path = report_dir / "metrics.json"

        model_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        final_model.fit(X_final_train, y_final_train)
        joblib.dump(final_model, model_path)

        write_json(params_path, best_params)
        write_json(
            metrics_path,
            {
                "best_validation_score": float(study.best_value),
                "validation_metric": str(cfg.hpo.metric),
                "test_metrics": final_metrics,
            },
        )

        mlflow.log_metric(f"best_validation_{cfg.hpo.metric}", float(study.best_value))
        mlflow.log_metrics({f"test_{name}": value for name, value in final_metrics.items()})
        mlflow.log_dict(best_params, "best_params.json")
        mlflow.log_artifact(str(params_path), artifact_path="reports")
        mlflow.log_artifact(str(metrics_path), artifact_path="reports")
        mlflow.log_artifact(str(model_path), artifact_path="models")

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(final_model, artifact_path="model")
            register_model_if_needed(parent_run.info.run_id, cfg)

        print(json.dumps(best_params, indent=2))
        print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
