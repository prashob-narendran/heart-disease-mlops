# Heart Disease Risk Prediction: End-to-End MLOps

Predicts the presence of heart disease from 13 clinical attributes (UCI Heart Disease, Cleveland subset) and is being built out as a reproducible, tested, containerised and monitored ML service.

| Stage | Status |
|---|---|
| 1. Data acquisition & EDA | ✅ done |
| 2. Feature engineering & models | ✅ done |
| 3. MLflow experiment tracking | ✅ done |
| 4. Packaging & reproducibility | ⏳ next (partly done: pipeline saved as one object, pinned deps, seeds) |
| 5. CI/CD (GitHub Actions) | ⏳ |
| 6. Docker + FastAPI | ⏳ |
| 7. Kubernetes deployment | ⏳ |
| 8. Monitoring & logging | ⏳ |

## Dataset

- **Source:** UCI Machine Learning Repository, [Heart Disease (id 45)](https://archive.ics.uci.edu/dataset/45/heart+disease), file `processed.cleveland.data`
- **Size:** 303 patients, 13 features + diagnosis `num` (0-4)
- **Target:** `target = 1` if `num > 0` (disease present), else `0`
- **Missing:** `ca` (4 rows) and `thal` (2 rows), marked `?` in the raw file

Why the Cleveland subset and the primary UCI source: see the top of `notebooks/01_eda.ipynb`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_data.py    # -> data/raw/ + manifest.json (sha256)
python -m src.data_processing      # -> data/processed/heart_clean.csv
python -m src.eda                  # -> reports/figures/*.png
python -m src.train                # tune 4 models, log to MLflow, save models/model.joblib
pytest                             # data + model tests
flake8 src scripts tests           # lint
mlflow ui --backend-store-uri mlruns   # open http://127.0.0.1:5000
```

If the UCI site is unreachable, the download script falls back to the official `ucimlrepo` client. As a last resort, download the file manually and run
`python scripts/download_data.py --from-file path/to/processed.cleveland.data`.

## Project layout

```
├── data/
│   ├── raw/                 processed.cleveland.data + manifest.json
│   └── processed/           heart_clean.csv
├── notebooks/01_eda.ipynb   EDA with findings and modelling implications
├── reports/figures/         EDA plots (used in the report)
├── scripts/download_data.py dataset download + checksum
├── src/
│   ├── config.py            paths, schema, category labels
│   ├── data_processing.py   load / clean / save
│   ├── eda.py               EDA tables and plots
│   ├── features.py          preprocessing pipeline (impute, scale, one-hot)
│   ├── models.py            4 candidate models + hyperparameter grids
│   ├── evaluate.py          metrics, confusion matrix, ROC, permutation importance
│   └── train.py             GridSearchCV + MLflow tracking + model registry
├── models/                  model.joblib + metadata.json (selected model)
├── reports/model_comparison.csv
├── tests/test_data.py       data cleaning tests (no network needed)
├── tests/test_model.py      pipeline/model tests on synthetic data
├── requirements.txt         pinned dependencies
└── setup.cfg                flake8 + pytest config
```

## Cleaning decisions

1. Column names assigned from the UCI documentation (raw file has no header).
2. `?` becomes NaN. **No imputation at this stage.** Imputation is part of the model pipeline, fitted on training folds only, which avoids leakage and lets the API accept unknown `ca`/`thal`.
3. Categorical codes outside the documented sets become NaN.
4. `num` (0-4) becomes binary `target`. `num` is dropped.
5. Exact duplicate rows are removed.

## Modelling (Stage 2)

- **Pipeline:** `to_float` → numeric (median impute → StandardScaler) + categorical (most-frequent impute → OneHotEncoder(ignore unknown)) → estimator. One sklearn object, so preprocessing always travels with the model.
- **Candidates:** Logistic Regression (L1/L2), Random Forest, Gradient Boosting, SVM (RBF), each tuned with `GridSearchCV`, stratified 5-fold CV, scored on accuracy, precision, recall, F1 and ROC-AUC.
- **Split:** stratified 80/20, seed 42. The test set is used for reporting only.
- **Selection rule:** highest CV ROC-AUC; models within 0.01 are treated as tied and the tie goes to higher CV recall (a missed disease case costs more than a false alarm).

## Experiment tracking (Stage 3)

Each `python -m src.train` creates one parent run (session) in the MLflow experiment `heart-disease-classification`, with one nested run per model containing:

| Logged | Content |
|---|---|
| Params | model type, best hyperparameters, CV folds, seed, grid size |
| Metrics | CV mean and std for 5 metrics, test metrics, specificity, FN/FP counts, fit time |
| Artifacts | confusion matrix, ROC curve, permutation importance (PNG + CSV), full CV results |
| Model | the fitted pipeline with signature and input example |
| Tags | dataset sha256, git commit, Python and sklearn versions |

The selected model is registered as **`heart-disease-classifier`** with alias **`champion`**. `python -m src.train --quick` (used by CI) logs to a separate `-ci` experiment and never touches the registry.
