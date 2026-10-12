"""Inference-layer tests (shared by the CLI and the API)."""

import json
from pathlib import Path

import pytest
from sklearn.base import clone

from src.features import build_pipeline
from src.models import get_candidates
from src.predict import predict_records, risk_level
from tests.test_model import make_data

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
KEYS = {"prediction", "label", "disease_probability", "confidence",
        "risk_level"}


@pytest.fixture(scope="module")
def model():
    X, y = make_data()
    est = get_candidates(quick=True)[0].estimator
    return build_pipeline(clone(est)).fit(X, y)


@pytest.mark.parametrize("p,expected", [
    (0.0, "low"), (0.29, "low"), (0.30, "moderate"), (0.59, "moderate"),
    (0.60, "low"), (1.0, "high"),
])
def test_risk_bands(p, expected):
    assert risk_level(p) == expected


def test_single_dict_returns_one_result(model):
    patient = json.loads((SAMPLES / "sample_patient.json").read_text())
    out = predict_records(model, patient, threshold=0.5)
    assert len(out) == 1 and set(out[0]) == KEYS


def test_batch_with_missing_values(model):
    batch = json.loads((SAMPLES / "sample_batch.json").read_text())
    out = predict_records(model, batch, threshold=0.5)
    assert len(out) == len(batch)
    for r in out:
        assert r["prediction"] in (0, 1)
        assert r["label"] == ("disease" if r["prediction"] else "no_disease")
        assert 0.0 <= r["disease_probability"] <= 1.0
        assert 0.5 <= r["confidence"] <= 1.0


def test_threshold_controls_decision(model):
    patient = json.loads((SAMPLES / "sample_patient.json").read_text())
    assert predict_records(model, patient, threshold=0.0)[0]["prediction"] == 1
    assert predict_records(model, patient, threshold=1.01)[0]["prediction"] == 0


def test_missing_feature_is_rejected(model):
    with pytest.raises(ValueError, match="Missing feature columns"):
        predict_records(model, {"age": 50}, threshold=0.5)
