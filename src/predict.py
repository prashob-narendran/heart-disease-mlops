"""Inference: load the packaged model and score patients.

This module is shared by the command line (below) and the FastAPI service
(Stage 6), so both return exactly the same response format.

Response per patient
--------------------
    prediction           0 = no disease, 1 = disease (threshold from metadata)
    label                "no_disease" | "disease"
    disease_probability  P(disease), 0-1
    confidence           probability of the predicted class, 0.5-1
    risk_level           "low" (<0.30) | "moderate" (0.30-0.60) | "high" (>=0.60)

The risk band is an extra, human-readable layer on top of the binary
decision: a 0.52 and a 0.97 are both "disease", but mean very different
things to a clinician deciding on follow-up tests.

Usage
-----
    python -m src.predict --input samples/sample_patient.json
    python -m src.predict --input samples/sample_batch.json
    python -m src.predict --input samples/sample_patient.json --source registry
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import METADATA_FILE, MODEL_FILE
from src.features import to_float_frame

RISK_BANDS = [(0.30, "low"), (0.60, "moderate"), (1.01, "high")]
LABELS = {0: "no_disease", 1: "disease"}
DEFAULT_THRESHOLD = 0.5


def risk_level(p: float) -> str:
    for upper, name in RISK_BANDS:
        if p < upper:
            return name
    return "high"


@lru_cache(maxsize=4)
def load_model(path: str | Path = MODEL_FILE):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.train` first")
    return joblib.load(path)


def load_metadata(path: str | Path = METADATA_FILE) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def load_registry_model(uri: str = "models:/heart-disease-classifier@champion"):
    """Load the champion from the MLflow registry as a native sklearn model."""
    import mlflow

    from src.train import tracking_uri

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow.sklearn.load_model(uri)


def predict_records(model, records, threshold: float | None = None) -> list:
    """Score one dict or a list of dicts with the 13 feature keys.

    Missing optional values (e.g. ca or thal unknown) may be None; the
    pipeline imputes them exactly as during training.
    """
    if isinstance(records, dict):
        records = [records]
    if threshold is None:
        threshold = load_metadata().get("threshold", DEFAULT_THRESHOLD)
    X = to_float_frame(records)
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({
            "prediction": pred,
            "label": LABELS[pred],
            "disease_probability": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
            "risk_level": risk_level(float(p)),
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score patients from JSON")
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON file: one patient object or a list")
    parser.add_argument("--source", choices=["file", "registry"],
                        default="file",
                        help="models/model.joblib (default) or MLflow "
                             "registry champion")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)

    records = json.loads(args.input.read_text())
    model = (load_registry_model() if args.source == "registry"
             else load_model())
    results = predict_records(model, records, args.threshold)

    meta = load_metadata()
    print(json.dumps({
        "model": meta.get("model_name", "unknown"),
        "model_version": meta.get("registered_version"),
        "source": args.source,
        "n": len(results),
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
