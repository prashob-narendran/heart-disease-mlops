"""API tests with FastAPI's TestClient.

A small model is trained on synthetic data and served from a temp folder
(via MODEL_PATH / METADATA_PATH), so these tests need neither the dataset
nor the committed model, and run in CI exactly as on a laptop.
"""

import json
import logging

import joblib
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest
from sklearn.base import clone

from api.main import create_app
from api.schemas import EXAMPLE_PATIENT
from src.config import FEATURES
from src.features import build_pipeline
from src.models import get_candidates
from tests.test_model import make_data

RESULT_KEYS = {"prediction", "label", "disease_probability", "confidence",
               "risk_level"}


@pytest.fixture(scope="module")
def model_files(tmp_path_factory):
    root = tmp_path_factory.mktemp("served")
    X, y = make_data()
    model = build_pipeline(clone(get_candidates(quick=True)[0].estimator))
    model.fit(X, y)
    joblib.dump(model, root / "model.joblib")
    (root / "metadata.json").write_text(json.dumps({
        "model_name": "test_model", "registered_version": 7,
        "threshold": 0.5, "features": FEATURES,
        "metrics": {"cv_roc_auc_mean": 0.9}, "best_params": {"C": 1.0},
    }))
    return root


@pytest.fixture(scope="module")
def client(model_files):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(model_files / "model.joblib"))
    mp.setenv("METADATA_PATH", str(model_files / "metadata.json"))
    with TestClient(create_app()) as c:   # runs start-up (model loading)
        yield c
    mp.undo()


def patient(**overrides):
    p = dict(EXAMPLE_PATIENT)
    p.update(overrides)
    return p


# ---------------------------------------------------------------- health
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_name"] == "test_model"


def test_health_503_when_model_missing(tmp_path):
    mp = pytest.MonkeyPatch()
    mp.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    try:
        with TestClient(create_app()) as c:
            assert c.get("/health").status_code == 503
            assert c.post("/predict", json=patient()).status_code == 503
    finally:
        mp.undo()


def test_root_and_model_info(client):
    assert client.get("/").json()["docs"] == "/docs"
    info = client.get("/model-info").json()
    assert info["model_name"] == "test_model"
    assert info["features"] == FEATURES and info["threshold"] == 0.5


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


# ---------------------------------------------------------------- predict
def test_predict_valid_patient(client):
    r = client.post("/predict", json=patient())
    assert r.status_code == 200
    body = r.json()
    assert RESULT_KEYS <= set(body)
    assert body["prediction"] in (0, 1)
    assert 0 <= body["disease_probability"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["model_name"] == "test_model" and body["model_version"] == 7


def test_predict_accepts_unknown_ca_and_thal(client):
    r = client.post("/predict", json=patient(ca=None, thal=None))
    assert r.status_code == 200
    r = client.post("/predict", json={k: v for k, v in patient().items()
                                      if k not in ("ca", "thal")})
    assert r.status_code == 200


def test_response_headers(client):
    r = client.post("/predict", json=patient(),
                    headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert float(r.headers["X-Process-Time-ms"]) >= 0
    assert r.headers["X-Served-By"]          # pod/container hostname


@pytest.mark.parametrize("bad", [
    {"age": -5},              # out of range
    {"age": 150},
    {"sex": 2},               # invalid code
    {"cp": 0},
    {"thal": 5},
    {"ca": 4},                # Kaggle-style fake category is rejected
    {"chol": "high"},         # wrong type
    {"oldpeak": -1},
])
def test_predict_rejects_invalid_values(client, bad):
    r = client.post("/predict", json=patient(**bad))
    assert r.status_code == 422
    field = next(iter(bad))
    assert any(field in err["loc"] for err in r.json()["detail"])


def test_predict_rejects_missing_required_field(client):
    p = patient()
    del p["age"]
    assert client.post("/predict", json=p).status_code == 422


def test_predict_rejects_unknown_field(client):
    assert client.post("/predict",
                       json=patient(cholesterol=200)).status_code == 422


def test_predict_rejects_non_json(client):
    r = client.post("/predict", content="not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ---------------------------------------------------------------- batch
def test_batch_prediction(client):
    pts = [patient(), patient(age=41, sex=0, cp=2, thalach=172),
           patient(ca=None, thal=None)]
    r = client.post("/predict/batch", json={"patients": pts})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3 and len(body["results"]) == 3
    assert all(RESULT_KEYS <= set(x) for x in body["results"])


def test_batch_matches_single(client):
    single = client.post("/predict", json=patient()).json()
    batch = client.post("/predict/batch",
                        json={"patients": [patient()]}).json()["results"][0]
    assert batch["disease_probability"] == single["disease_probability"]


@pytest.mark.parametrize("n", [0, 101])
def test_batch_size_limits(client, n):
    r = client.post("/predict/batch", json={"patients": [patient()] * n})
    assert r.status_code == 422


# ---------------------------------------------------------------- metrics
def sample(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_format(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    for name in ("heart_api_requests_total",
                 "heart_api_request_duration_seconds_bucket",
                 "heart_api_predictions_total", "heart_api_model_loaded"):
        assert name in r.text


def test_metrics_count_requests_and_predictions(client):
    before_req = sample("heart_api_requests_total", method="POST",
                        endpoint="/predict", status="200")
    before_lat = sample("heart_api_request_duration_seconds_count",
                        method="POST", endpoint="/predict")
    body = client.post("/predict", json=patient()).json()
    before_pred = sample("heart_api_predictions_total", label=body["label"],
                         risk_level=body["risk_level"])
    client.post("/predict", json=patient())
    assert sample("heart_api_requests_total", method="POST",
                  endpoint="/predict", status="200") == before_req + 2
    assert sample("heart_api_request_duration_seconds_count", method="POST",
                  endpoint="/predict") == before_lat + 2
    assert sample("heart_api_predictions_total", label=body["label"],
                  risk_level=body["risk_level"]) == before_pred + 1


def test_metrics_label_errors_and_unknown_paths(client):
    before_422 = sample("heart_api_requests_total", method="POST",
                        endpoint="/predict", status="422")
    client.post("/predict", json=patient(sex=5))
    assert sample("heart_api_requests_total", method="POST",
                  endpoint="/predict", status="422") == before_422 + 1
    # raw URLs never become label values (bounded cardinality)
    client.get("/no/such/path/123")
    assert sample("heart_api_requests_total", method="GET",
                  endpoint="unmatched", status="404") >= 1
    assert "/no/such/path/123" not in client.get("/metrics").text


def test_metrics_batch_and_probability(client):
    before_n = sample("heart_api_batch_size_count")
    before_p = sample("heart_api_prediction_probability_count")
    client.post("/predict/batch", json={"patients": [patient()] * 3})
    assert sample("heart_api_batch_size_count") == before_n + 1
    assert sample("heart_api_prediction_probability_count") == before_p + 3


def test_metrics_model_state(model_files, tmp_path):
    # one app per process in production; here each app re-sets the gauges
    mp = pytest.MonkeyPatch()
    try:
        mp.setenv("MODEL_PATH", str(model_files / "model.joblib"))
        mp.setenv("METADATA_PATH", str(model_files / "metadata.json"))
        with TestClient(create_app()):
            assert sample("heart_api_model_loaded") == 1
            assert sample("heart_api_model_info", model_name="test_model",
                          model_version="7") == 1
        mp.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
        with TestClient(create_app()):
            assert sample("heart_api_model_loaded") == 0
            assert "heart_api_model_info{" not in generate_latest().decode()
    finally:
        mp.undo()


def test_probe_requests_are_not_logged_at_info(client, caplog):
    with caplog.at_level(logging.INFO, logger="heart-api"):
        client.get("/health")
        client.get("/metrics")
        client.post("/predict", json=patient())
    paths = [json.loads(r.message).get("path") for r in caplog.records
             if '"event": "request"' in r.message]
    assert paths == ["/predict"]
