#!/usr/bin/env bash
# Clean-room verification of the project on a fresh Linux machine
# (used on the BITS Prayogshala virtual lab, but works on any Linux box).
#
# Proves the three production-readiness requirements of the assignment:
#   1. everything runs from a clean setup using requirements.txt
#   2. the model serves correctly in an isolated Docker container
#   3. the pipeline fails on code / test / data errors with clear logs
#
# Usage (from the cloned repository):
#   bash scripts/lab_verify.sh info        # machine + repo identity
#   bash scripts/lab_verify.sh setup       # fresh Python 3.11 env + requirements.txt
#   bash scripts/lab_verify.sh pipeline    # data -> EDA -> lint -> tests -> train -> gate -> predict
#   bash scripts/lab_verify.sh docker      # build, run (hardened), healthcheck, smoke test
#   bash scripts/lab_verify.sh failproof   # deliberate failures are caught and reported
#   bash scripts/lab_verify.sh k8s         # optional: kind cluster + app + Prometheus/Grafana
#   bash scripts/lab_verify.sh k8s-down    # delete the kind cluster
#   bash scripts/lab_verify.sh summary     # one-page summary of all stage results
#
# Options (environment variables):
#   LAB_ENV_DIR=/path   where to create the Python environment (default ~/envs/heart-mlops)
#   LAB_NO_CONDA=1      use python -m venv instead of conda
#
# Every stage writes a timestamped log to lab_proof/ and ends with a
# PASSED / FAILED banner, so a screenshot of the end of each stage is the
# evidence for the report.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PROOF="$REPO/lab_proof"
ENV_DIR="${LAB_ENV_DIR:-$HOME/envs/heart-mlops}"   # local disk: faster than ~/workspace
IMAGE="heart-api:local"
CONTAINER="heart-api-lab"
KIND_VERSION="v0.31.0"
KUBECTL_VERSION="v1.34.1"
BIN="$HOME/.local/bin"
mkdir -p "$PROOF" "$BIN"
export PATH="$ENV_DIR/bin:$BIN:$PATH"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

STAGE="${1:-}"
[ -z "$STAGE" ] && { sed -n '2,/^set /p' "$0" | grep '^#'; exit 1; }
LOG="$PROOF/$(date +%Y%m%d_%H%M%S)_${STAGE}.log"
exec > >(tee -a "$LOG") 2>&1

# ------------------------------------------------------------------ helpers
C_G=$'\e[1;32m'; C_R=$'\e[1;31m'; C_B=$'\e[1;36m'; C_Y=$'\e[1;33m'; C_0=$'\e[0m'
FAILS=0
step()  { echo; echo "${C_B}=== $* ===${C_0}"; }
ok()    { echo "${C_G}  PASS${C_0}  $*"; }
bad()   { echo "${C_R}  FAIL${C_0}  $*"; FAILS=$((FAILS + 1)); }
note()  { echo "${C_Y}  NOTE${C_0}  $*"; }
run()   { # run "description" cmd...   -> PASS/FAIL line, keeps going
    local d="$1"; shift
    if "$@"; then ok "$d"; else bad "$d (exit $?)"; fi
}

header() {
    local os
    os="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
    echo "${C_B}################################################################${C_0}"
    echo " Heart Disease MLOps - lab verification : stage '${STAGE}'"
    echo " Date (local)  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo " User @ host   : $(whoami) @ $(hostname)"
    echo " OS / kernel   : ${os:-unknown} / $(uname -r) ($(uname -m))"
    echo " CPU / RAM     : $(nproc) cores / $(free -h | awk '/^Mem:/{print $2}')"
    echo " Repository    : $(git remote get-url origin 2>/dev/null)"
    echo " Commit        : $(git log -1 --format='%h %s (%cd)' --date=short 2>/dev/null)"
    echo " Log file      : $LOG"
    echo "${C_B}################################################################${C_0}"
}

footer() {
    echo
    if [ "$FAILS" -eq 0 ]; then
        echo "${C_G}################  STAGE '${STAGE}' PASSED  ################${C_0}"
    else
        echo "${C_R}########  STAGE '${STAGE}' FAILED ($FAILS check(s))  ########${C_0}"
    fi
    echo "  finished $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)"
    echo "$(date '+%F %T')|${STAGE}|$([ "$FAILS" -eq 0 ] && echo PASSED || echo FAILED)|$(basename "$LOG")" \
        >> "$PROOF/results.txt"
    exit $(( FAILS > 0 ))
}

need_env() {
    if [ ! -x "$ENV_DIR/bin/python" ]; then
        echo "${C_R}Python environment not found at $ENV_DIR - run: bash scripts/lab_verify.sh setup${C_0}"
        exit 1
    fi
}

# Docker without sudo. On the disposable lab VM the simplest reliable way is
# to start the daemon and open the socket to the current user.
docker_ready() {
    if ! docker info >/dev/null 2>&1; then
        note "starting Docker daemon (sudo)"
        sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || true
        sleep 3
    fi
    if ! docker info >/dev/null 2>&1; then
        note "granting this user access to the Docker socket (lab VM only)"
        sudo chmod 666 /var/run/docker.sock
    fi
    docker info >/dev/null 2>&1
}

# ------------------------------------------------------------------ stages
stage_info() {
    step "Python / tools available on the machine"
    for t in python3 conda git docker curl; do
        printf '  %-8s %s\n' "$t" "$(command -v $t >/dev/null && ($t --version 2>&1 | head -1) || echo 'not installed')"
    done
    step "Disk"
    df -h "$HOME" "$REPO" | sed 's/^/  /'
    step "Network reachability"
    for u in https://github.com https://pypi.org https://archive.ics.uci.edu https://registry-1.docker.io/v2/; do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$u")
        printf '  %-42s HTTP %s\n' "$u" "$code"
    done
    step "Repository content (clean clone)"
    git status --short | head -5; echo "  tracked files: $(git ls-files | wc -l)"
    ls -1 | sed 's/^/  /'
}

stage_setup() {
    step "1/4 Create a fresh, isolated Python 3.11 environment at $ENV_DIR"
    rm -rf "$ENV_DIR"; mkdir -p "$(dirname "$ENV_DIR")"
    if command -v conda >/dev/null 2>&1 && [ -z "${LAB_NO_CONDA:-}" ]; then
        # conda-forge only: no Anaconda channel terms prompt, fully open-source
        run "conda env (python 3.11, conda-forge)" \
            conda create -y -q -p "$ENV_DIR" -c conda-forge --override-channels python=3.11 pip
    elif command -v python3.11 >/dev/null 2>&1; then
        run "venv (python3.11)" python3.11 -m venv "$ENV_DIR"
    else
        run "venv ($(python3 --version))" python3 -m venv "$ENV_DIR"
    fi
    need_env
    echo "  interpreter: $("$ENV_DIR/bin/python" --version) at $ENV_DIR/bin/python"

    step "2/4 Install the pinned dependencies from requirements.txt"
    "$ENV_DIR/bin/python" -m pip install -q --upgrade pip
    run "pip install -r requirements.txt" "$ENV_DIR/bin/python" -m pip install -q -r requirements.txt

    step "3/4 Dependency consistency"
    run "pip check (no broken requirements)" "$ENV_DIR/bin/python" -m pip check

    step "4/4 Key package versions (identical to CI and the Docker image)"
    "$ENV_DIR/bin/python" - <<'EOF'
import sys
from importlib.metadata import version
print(f"  {'python':<18} {sys.version.split()[0]}")
for d in ["numpy", "pandas", "scikit-learn", "mlflow", "fastapi", "pydantic",
          "uvicorn", "prometheus-client", "pytest"]:
    print(f"  {d:<18} {version(d)}")
EOF
}

stage_pipeline() {
    need_env
    step "1/9 Dataset integrity: committed raw file vs its SHA-256 manifest"
    run "checksum of committed data" python scripts/download_data.py

    step "2/9 Fresh download from UCI on this machine (must be byte-identical)"
    local before after
    before=$(python -c "import json;print(json.load(open('data/raw/manifest.json'))['sha256'])")
    if python scripts/download_data.py --force; then
        after=$(python -c "import json;print(json.load(open('data/raw/manifest.json'))['sha256'])")
        echo "  manifest sha256 : $before"
        echo "  fresh   sha256 : $after"
        if [ "$before" = "$after" ]; then ok "fresh UCI download identical to the versioned dataset"
        else bad "fresh download differs from the versioned dataset"; fi
    else
        note "UCI not reachable from this network - using the committed copy"
    fi
    git checkout -- data/raw/ 2>/dev/null   # keep the committed manifest

    step "3/9 Clean the data";           run "src.data_processing" python -m src.data_processing
    step "4/9 EDA figures";              run "src.eda" python -m src.eda
    step "5/9 Lint (flake8)";            run "flake8 src scripts tests api" flake8 src scripts tests api
    step "6/9 Unit + integration + API tests"
    run "pytest" pytest -q --cov=src --cov=api --cov-report=term:skip-covered
    step "7/9 Train all models with MLflow tracking (takes a few minutes)"
    run "src.train" python -m src.train
    step "8/9 Model quality gate";       run "check_model.py" python scripts/check_model.py
    step "9/9 Inference on sample patients"
    run "src.predict (batch of 3)" python -m src.predict --input samples/sample_batch.json

    step "Result"
    python - <<'EOF'
import json, pandas as pd
m = json.load(open("models/metadata.json"))
print(f"  selected model : {m.get('model_label')}  (registry version {m.get('registered_version')})")
mt = m.get("metrics", {})
for k in ["cv_roc_auc_mean", "cv_recall_mean", "test_roc_auc", "test_recall", "test_accuracy"]:
    if k in mt:
        print(f"  {k:<16} {mt[k]:.3f}")
t = pd.read_csv("reports/model_comparison.csv", index_col=0)
cols = [c for c in ["cv_roc_auc_mean", "cv_recall_mean", "test_roc_auc", "selected"] if c in t]
print(t[cols].round(3).to_string())
EOF
    echo "  MLflow runs: $(find mlruns -name meta.yaml -path '*/[0-9a-f]*' 2>/dev/null | wc -l) metadata files"
    note "optional: view runs with  mlflow ui --port 5000  then open http://localhost:5000 in the VM browser"
}

stage_docker() {
    need_env
    step "1/6 Docker engine"
    if docker_ready; then ok "docker $(docker version --format '{{.Server.Version}}') reachable"
    else bad "Docker daemon not reachable"; footer; fi

    step "2/6 Build the serving image (python:3.11-slim, non-root, HEALTHCHECK)"
    run "docker build -t $IMAGE ." docker build ${DOCKER_BUILD_ARGS:-} -t "$IMAGE" .
    docker image ls "$IMAGE" | sed 's/^/  /'

    step "3/6 Run isolated + hardened (read-only FS, no capabilities, no privilege escalation)"
    docker rm -f "$CONTAINER" >/dev/null 2>&1
    run "docker run" docker run -d --name "$CONTAINER" -p 8000:8000 \
        --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges "$IMAGE"
    echo "  user inside container: $(docker exec "$CONTAINER" whoami 2>/dev/null)"

    step "4/6 Wait for the Docker HEALTHCHECK"
    local status=""
    for _ in $(seq 1 40); do
        status=$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null)
        [ "$status" = "healthy" ] && break
        sleep 3
    done
    [ "$status" = "healthy" ] && ok "container is healthy" || bad "container health: ${status:-unknown}"
    docker ps --filter "name=$CONTAINER" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' | sed 's/^/  /'

    step "5/6 Smoke test every endpoint"
    run "smoke_test_api.py" python scripts/smoke_test_api.py --url http://localhost:8000 --wait 60

    step "6/6 Sample request / invalid request / logs"
    echo "  POST /predict (samples/sample_patient.json):"
    curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
        -d @samples/sample_patient.json | sed 's/^/    /'; echo
    echo "  POST /predict (samples/invalid_patient.json) -> expect HTTP 422:"
    curl -s -o /dev/null -w '    HTTP %{http_code}\n' -X POST http://localhost:8000/predict \
        -H 'Content-Type: application/json' -d @samples/invalid_patient.json
    echo "  container logs (JSON, last 4 lines):"
    docker logs --tail 4 "$CONTAINER" 2>&1 | sed 's/^/    /'
    note "container left running on http://localhost:8000/docs - stop with: docker rm -f $CONTAINER"
}

# Each check here MUST fail; the stage passes when every failure is caught.
expect_fail() {
    local d="$1"; shift
    echo "  \$ $*"
    "$@" 2>&1 | sed 's/^/  | /' | tail -n "${TAIL:-15}"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then ok "$d -> stopped with exit code $rc"
    else bad "$d -> was NOT detected (exit 0)"; fi
}

stage_failproof() {
    need_env
    LAB_TMP="$(mktemp -d)"; local tmp="$LAB_TMP"
    trap 'git -C "$REPO" checkout -- data/raw/ 2>/dev/null; rm -rf "$LAB_TMP"' EXIT

    step "1/5 Code error: lint violation is rejected by flake8"
    printf 'import os\n\n\ndef add(a,b):\n    return a+b  \n' > "$tmp/bad_style.py"
    expect_fail "lint error" flake8 "$tmp/bad_style.py"

    step "2/5 Test error: a regression in model behaviour fails pytest with a clear diff"
    cat > "$tmp/test_regression_demo.py" <<'EOF'
"""Deliberately wrong expectation, to show how a failing test is reported."""
import json
import joblib
from src.predict import predict_records


def test_high_risk_patient_is_flagged():
    model = joblib.load("models/model.joblib")
    patient = json.load(open("samples/sample_batch.json"))[1]   # high-risk patient
    result = predict_records(model, patient)[0]
    assert result["label"] == "no_disease", result   # wrong on purpose
EOF
    TAIL=25 expect_fail "failing unit test" env PYTHONPATH="$REPO" \
        pytest -q -p no:cacheprovider --rootdir "$REPO" "$tmp/test_regression_demo.py"

    step "3/5 Data error: a tampered dataset is caught by the checksum gate"
    sed -i '1s/^63\.0/64.0/' data/raw/processed.cleveland.data
    expect_fail "corrupted data" python scripts/download_data.py

    step "4/5 Pipeline stops at the first failure (reproduce.sh with the tampered data)"
    TAIL=12 expect_fail "end-to-end pipeline" bash scripts/reproduce.sh
    git checkout -- data/raw/
    run "data restored, checksum OK again" python scripts/download_data.py

    step "5/5 Model quality gate rejects a model below the required metric"
    expect_fail "quality gate (min test ROC-AUC 0.99)" python scripts/check_model.py --min-test-auc 0.99

    echo
    echo "  All deliberate errors were detected, reported clearly and stopped the run."
    echo "  The same checks run in GitHub Actions (lint -> test -> train/quality gate),"
    echo "  where any failure turns the job red and blocks the later jobs."
}

stage_k8s() {
    need_env
    docker_ready || { bad "Docker not reachable"; footer; }
    step "1/7 Install kind $KIND_VERSION and kubectl $KUBECTL_VERSION (user-local)"
    if ! command -v kind >/dev/null; then
        curl -sSLo "$BIN/kind" "https://github.com/kubernetes-sigs/kind/releases/download/$KIND_VERSION/kind-linux-amd64" && chmod +x "$BIN/kind"
    fi
    if ! command -v kubectl >/dev/null; then
        curl -sSLo "$BIN/kubectl" "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl" && chmod +x "$BIN/kubectl"
    fi
    run "kind installed" kind version
    run "kubectl installed" kubectl version --client

    step "2/7 Create a local Kubernetes cluster (kind) and load the image"
    docker image inspect "$IMAGE" >/dev/null 2>&1 || run "docker build" docker build -t "$IMAGE" .
    kind get clusters 2>/dev/null | grep -qx heart || run "kind create cluster" kind create cluster --name heart --wait 120s
    run "kind load image" kind load docker-image "$IMAGE" --name heart
    kubectl get nodes -o wide | sed 's/^/  /'

    step "3/7 Deploy the application (same manifests as Docker Desktop and CI)"
    run "kubectl apply -k k8s/local/" kubectl apply -k k8s/local/
    run "rollout" kubectl -n heart-disease rollout status deployment/heart-api --timeout=180s
    kubectl -n heart-disease get deploy,pods,svc,hpa,pdb | sed 's/^/  /'

    step "4/7 Smoke test through the Service"
    pkill -f "port-forward svc/heart-api" 2>/dev/null
    (kubectl -n heart-disease port-forward svc/heart-api 8080:8080 >/dev/null 2>&1 &)
    sleep 4
    run "smoke test via Service" python scripts/smoke_test_api.py --url http://localhost:8080 --wait 60

    step "5/7 Deploy monitoring (Prometheus + Grafana)"
    run "kubectl apply -k k8s/monitoring/" kubectl apply -k k8s/monitoring/
    run "prometheus rollout" kubectl -n monitoring rollout status deployment/prometheus --timeout=300s
    run "grafana rollout" kubectl -n monitoring rollout status deployment/grafana --timeout=300s
    kubectl -n monitoring get pods,svc | sed 's/^/  /'

    step "6/7 Traffic + end-to-end monitoring check"
    pkill -f "port-forward svc/prometheus" 2>/dev/null; pkill -f "port-forward svc/grafana" 2>/dev/null
    (kubectl -n monitoring port-forward svc/prometheus 9090:9090 >/dev/null 2>&1 &)
    (kubectl -n monitoring port-forward svc/grafana 3000:3000 >/dev/null 2>&1 &)
    sleep 5
    python scripts/load_test.py --url http://localhost:8080 -n 300 -c 4
    sleep 15
    run "check_monitoring.py" python scripts/check_monitoring.py --min-targets 2 --wait 120

    step "7/7 Open in the VM browser"
    echo "  API docs   : http://localhost:8080/docs"
    echo "  Prometheus : http://localhost:9090/targets"
    echo "  Grafana    : http://localhost:3000   (dashboard opens as home page)"
    note "port-forwards keep running in the background; remove everything with: bash scripts/lab_verify.sh k8s-down"
}

stage_k8s_down() {
    pkill -f "kubectl.*port-forward" 2>/dev/null
    run "kind delete cluster" kind delete cluster --name heart
}

stage_summary() {
    step "Stage results on this machine"
    if [ -f "$PROOF/results.txt" ]; then
        printf '  %-19s | %-10s | %-7s | %s\n' "finished" "stage" "result" "log"
        grep -v '|summary|' "$PROOF/results.txt" | while IFS='|' read -r ts st res lg; do
            printf '  %-19s | %-10s | %-7s | %s\n' "$ts" "$st" "$res" "$lg"
        done
    else
        echo "  no stages run yet"
    fi
    echo; echo "  logs: $PROOF"
}

# ------------------------------------------------------------------ main
header
case "$STAGE" in
    info)      stage_info ;;
    setup)     stage_setup ;;
    pipeline)  stage_pipeline ;;
    docker)    stage_docker ;;
    failproof) stage_failproof ;;
    k8s)       stage_k8s ;;
    k8s-down)  stage_k8s_down ;;
    summary)   stage_summary ;;
    *) echo "unknown stage '$STAGE'"; sed -n '2,/^set /p' "$0" | grep '^#'; exit 1 ;;
esac
footer
