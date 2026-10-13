# Run the whole pipeline end to end and stop at the first failure.
# Usage (venv active, from the project root):  .\scripts\reproduce.ps1
# Exit codes are checked explicitly after each step ($LASTEXITCODE), so
# Python log lines written to stderr are not mistaken for errors.
$ErrorActionPreference = "Continue"

function Step($title, [scriptblock]$cmd) {
    Write-Host ""
    Write-Host "=== $title ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at: $title (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Step "1/8 Data: download or verify checksum" { python scripts/download_data.py }
Step "2/8 Data: clean"                       { python -m src.data_processing }
Step "3/8 EDA figures"                       { python -m src.eda }
Step "4/8 Lint"                              { flake8 src scripts tests api }
Step "5/8 Unit tests"                        { pytest -q }
Step "6/8 Train + MLflow tracking"           { python -m src.train }
Step "7/8 Model quality gate"                { python scripts/check_model.py }
Step "8/8 Sample prediction"                 { python -m src.predict --input samples/sample_patient.json }

Write-Host ""
Write-Host "Pipeline reproduced successfully." -ForegroundColor Green
