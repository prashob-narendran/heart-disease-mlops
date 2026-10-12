# Heart Disease Risk Prediction: End-to-End MLOps

Predicts the presence of heart disease from 13 clinical attributes (UCI Heart Disease, Cleveland subset) and is being built out as a reproducible, tested, containerised and monitored ML service.

| Stage | Status |
|---|---|
| 1. Data acquisition & EDA | ✅ done |
| 2. Feature engineering & models | ⏳ next |
| 3. MLflow experiment tracking | ⏳ |
| 4. Packaging & reproducibility | ⏳ |
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
pytest                             # data tests
flake8 src scripts tests           # lint
jupyter notebook notebooks/01_eda.ipynb
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
│   └── eda.py               EDA tables and plots
├── tests/test_data.py       data cleaning tests (no network needed)
├── requirements.txt         pinned dependencies
└── setup.cfg                flake8 + pytest config
```

## Cleaning decisions

1. Column names assigned from the UCI documentation (raw file has no header).
2. `?` becomes NaN. **No imputation at this stage.** Imputation is part of the model pipeline, fitted on training folds only, which avoids leakage and lets the API accept unknown `ca`/`thal`.
3. Categorical codes outside the documented sets become NaN.
4. `num` (0-4) becomes binary `target`. `num` is dropped.
5. Exact duplicate rows are removed.
