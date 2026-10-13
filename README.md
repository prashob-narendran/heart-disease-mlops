# Heart Disease Risk Prediction: End-to-End MLOps

[![CI](https://github.com/prashob-narendran/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/prashob-narendran/heart-disease-mlops/actions/workflows/ci.yml)

Predicts the presence of heart disease from 13 clinical attributes (UCI Heart Disease, Cleveland subset) and is being built out as a reproducible, tested, containerised and monitored ML service.

| Stage | Status |
|---|---|
| 1. Data acquisition & EDA | ✅ done |
| 2. Feature engineering & models | ✅ done |
| 3. MLflow experiment tracking | ✅ done |
| 4. Packaging & reproducibility | ✅ done |
| 5. CI/CD (GitHub Actions) | ✅ done |
| 6. Docker + FastAPI | ✅ done |
| 7. Kubernetes deployment | ⏳ next |
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
python scripts/check_model.py      # model quality gate
python -m src.predict --input samples/sample_patient.json
pytest                             # 64 unit, integration and API tests
flake8 src scripts tests api           # lint
mlflow ui --backend-store-uri mlruns   # open http://127.0.0.1:5000

uvicorn api.main:app --reload      # API on http://127.0.0.1:8000/docs
docker build -t heart-api .        # container image
docker run -d -p 8000:8000 --name heart-api heart-api
python scripts/smoke_test_api.py   # end-to-end check of the running API
```

Or run everything in one go: `.\scripts\reproduce.ps1` (Windows) / `bash scripts/reproduce.sh` (Linux/macOS),
or as an MLflow Project: `mlflow run . --env-manager local`.

If the UCI site is unreachable, the download script falls back to the official `ucimlrepo` client. As a last resort, download the file manually and run
`python scripts/download_data.py --from-file path/to/processed.cleveland.data`.

## Project layout

```
├── data/
│   ├── raw/                 processed.cleveland.data + manifest.json
│   └── processed/           heart_clean.csv
├── notebooks/01_eda.ipynb   EDA with findings and modelling implications
├── reports/figures/         EDA plots (used in the report)
├── .github/workflows/ci.yml GitHub Actions: lint -> test -> train -> docker
├── api/
│   ├── main.py              FastAPI app: /health, /model-info, /predict, /predict/batch
│   └── schemas.py           request/response models with input validation
├── Dockerfile, .dockerignore, requirements-api.txt   container image
├── samples/                 example patient JSON for inference
├── scripts/
│   ├── download_data.py     dataset download + checksum verification
│   ├── check_model.py       model quality gate (fails below thresholds)
│   ├── smoke_test_api.py    end-to-end test of a running API (local/Docker/K8s)
│   └── reproduce.ps1/.sh    whole pipeline in one command
├── MLproject, python_env.yaml  MLflow Project definition
├── src/
│   ├── config.py            paths, schema, category labels
│   ├── data_processing.py   load / clean / save
│   ├── eda.py               EDA tables and plots
│   ├── features.py          preprocessing pipeline (impute, scale, one-hot)
│   ├── models.py            4 candidate models + hyperparameter grids
│   ├── evaluate.py          metrics, confusion matrix, ROC, permutation importance
│   ├── train.py             GridSearchCV + MLflow tracking + model registry
│   └── predict.py           inference (shared by CLI and API)
├── models/                  model.joblib + metadata.json (selected model)
├── reports/model_comparison.csv
├── tests/test_data.py       data cleaning tests (no network needed)
├── tests/test_model.py      pipeline/model tests on synthetic data
├── tests/test_predict.py    inference tests
├── tests/test_integration.py end-to-end: train.py + MLflow + EDA on synthetic data
├── tests/test_api.py        22 API tests (FastAPI TestClient)
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

## Packaging & reproducibility (Stage 4)

| Requirement | How it is met |
|---|---|
| Model saved with preprocessing | One sklearn `Pipeline` (impute, scale, one-hot, model) saved as `models/model.joblib` and logged to MLflow with signature + input example |
| Model metadata | `models/metadata.json`: params, metrics, features, threshold, dataset sha256, git commit, library versions |
| Pinned environment | `requirements.txt` (exact versions), `.python-version` = 3.11, `python_env.yaml` |
| Deterministic results | seed 42 for split, CV folds and every model; identical results across runs |
| Data integrity | SHA-256 in `data/raw/manifest.json`, verified on every run; `.gitattributes` keeps data bytes identical on Windows |
| Registry | `heart-disease-classifier@champion` in the MLflow Model Registry |
| One-command reproduction | `scripts/reproduce.ps1` / `.sh`, or `mlflow run .` |
| Inference | `python -m src.predict --input <json>` (from file or `--source registry`) |

Example response:

```json
{"prediction": 1, "label": "disease", "disease_probability": 0.84,
 "confidence": 0.84, "risk_level": "high"}
```

## CI/CD (Stage 5)

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. **Lint:** flake8 (fails on any style error)
2. **Test:** 64 tests (data, model, inference, end-to-end, API) with coverage; JUnit XML + coverage reports uploaded
3. **Train:** verify dataset checksum → clean → train and track with MLflow → **quality gate** (CV ROC-AUC ≥ 0.85, CV recall ≥ 0.70, test ROC-AUC ≥ 0.80) → sample inference; model, reports and `mlruns/` uploaded as artifacts; results table written to the job summary

4. **Docker:** build the API image with the model trained in step 3 → run the container → smoke-test all endpoints → wait for the Docker `HEALTHCHECK` to report healthy → on pushes to `main`, publish to `ghcr.io/prashob-narendran/heart-disease-api` (`latest` and `sha-<commit>`)

Each job only runs if the previous one passed, so a lint or test failure stops the pipeline.

## Prediction API & container (Stage 6)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | 200 when the model is loaded, 503 otherwise (used by Docker/Kubernetes probes) |
| GET | `/model-info` | served model, version, metrics, parameters, dataset checksum, commit |
| POST | `/predict` | score one patient |
| POST | `/predict/batch` | score 1-100 patients |
| GET | `/docs` | interactive Swagger UI |

**Validation:** categorical fields accept only their documented codes, numeric fields have plausible ranges, unknown fields are rejected, and `ca`/`thal` may be `null` (imputed by the pipeline). Invalid input returns **422** with the offending field.

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/sample_patient.json
```
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -InFile samples/sample_patient.json
```
```json
{"prediction": 1, "label": "disease", "disease_probability": 0.5518, "confidence": 0.5518,
 "risk_level": "moderate", "model_name": "svm_rbf", "model_version": 4}
```

**Image:** `python:3.11-slim`, minimal runtime dependencies (`requirements-api.txt`, versions identical to training), code + model only (`.dockerignore`), non-root user, `HEALTHCHECK` on `/health`, uvicorn as PID 1 for graceful shutdown, one JSON log line per request with request id and latency (also returned as `X-Request-ID` / `X-Process-Time-ms` headers).
