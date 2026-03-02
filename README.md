# MLOps: Sentiment Training with MLflow

## What is implemented
- Structured training script: `train_mlflow.py`
- Pipeline: data loading -> text preprocessing -> TF-IDF -> Logistic Regression
- MLflow logging for:
  - hyperparameters
  - metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`)
  - artifacts (`evaluation.json`)
  - trained model
- 6 predefined experiments (assignment requires at least 5)

## Install dependencies
```bash
uv sync
```

## Run training experiments
```bash
uv run python train_mlflow.py
```

Optional args:
```bash
uv run python train_mlflow.py --data-path data/dataset.csv --experiment-name imdb_sentiment_baselines --tracking-uri file:./mlruns
```

## Open MLflow UI
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000` and compare all runs in experiment `imdb_sentiment_baselines`.
