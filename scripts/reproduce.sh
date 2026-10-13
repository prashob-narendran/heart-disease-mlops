#!/usr/bin/env bash
# Run the whole pipeline end to end and stop at the first failure.
# Usage (venv active, from the project root):  bash scripts/reproduce.sh
set -euo pipefail

step() { echo; echo "=== $* ==="; }

step "1/8 Data: download or verify checksum"; python scripts/download_data.py
step "2/8 Data: clean";                       python -m src.data_processing
step "3/8 EDA figures";                       python -m src.eda
step "4/8 Lint";                              flake8 src scripts tests api
step "5/8 Unit tests";                        pytest -q
step "6/8 Train + MLflow tracking";           python -m src.train
step "7/8 Model quality gate";                python scripts/check_model.py
step "8/8 Sample prediction";                 python -m src.predict --input samples/sample_patient.json

echo; echo "Pipeline reproduced successfully."
