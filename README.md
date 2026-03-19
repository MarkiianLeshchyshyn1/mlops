# MLOps: Lab 3

Проєкт уже містить базовий пайплайн з DVC і MLflow. Для ЛР3 додано лише те, що вимагає завдання:
- `Hydra` конфігурації в `config/`
- `Optuna` оптимізацію в `src/optimize.py`
- nested runs у `MLflow`

## Встановлення
```bash
uv sync
```

## Базовий пайплайн
```bash
uv run dvc repro
```

## HPO з TPE
```bash
uv run python src/optimize.py
```

## HPO з Random sampler
```bash
uv run python src/optimize.py hpo=random
```

## Що робить `src/optimize.py`
- бере `data/prepared/train.csv` і `data/prepared/test.csv`
- ділить `train.csv` на train/validation
- запускає `Optuna Study` на 20 trial-ів
- логує study як parent run, а trial-и як child runs
- перетреновує модель на найкращих параметрах
- зберігає фінальну модель і параметри

## Артефакти
- `models/hpo/<sampler>/best_model.joblib`
- `reports/hpo/<sampler>/best_params.json`
- `reports/hpo/<sampler>/metrics.json`

## Корисні override-и Hydra
```bash
uv run python src/optimize.py hpo.n_trials=30
uv run python src/optimize.py hpo.metric=roc_auc
uv run python src/optimize.py hpo.use_cv=true hpo.cv_folds=5
```

## MLflow UI
```bash
uv run mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000` in browser.

## Lab 4: CI/CD
For CI a small reproducible dataset is stored in `tests/fixtures/sample_dataset.csv`.

Main files added for Lab 4:
- `requirements.txt`
- `.github/workflows/cml.yaml`
- `tests/test_model.py`

Local smoke run:
```bash
uv sync
uv run python src/stages/prepare.py tests/fixtures/sample_dataset.csv .ci/prepared
uv run python src/stages/train.py .ci/prepared .ci/model
uv run pytest tests/test_model.py -q
```
