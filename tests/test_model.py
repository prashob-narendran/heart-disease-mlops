"""Model-layer tests on synthetic data (no dataset or network needed)."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, VALID_CODES
from src.evaluate import compute_metrics
from src.features import build_pipeline, build_preprocessor, to_float_frame
from src.models import get_candidates
from src.train import select_best


def make_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "trestbps": rng.normal(130, 17, n).round(),
        "chol": rng.normal(245, 50, n).round(),
        "thalach": rng.normal(150, 22, n).round(),
        "oldpeak": rng.exponential(1.0, n).round(1),
    })
    for col in CATEGORICAL_FEATURES:
        df[col] = rng.choice(sorted(VALID_CODES[col]), n)
    # make the target learnable: driven by ca, thal and thalach
    score = (df["ca"] + (df["thal"] == 7) * 1.5 - (df["thalach"] - 150) / 25
             + rng.normal(0, 0.8, n))
    y = (score > np.median(score)).astype(int)
    return df[FEATURES], y


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    est = get_candidates(quick=True)[0].estimator  # logistic regression
    return build_pipeline(clone(est)).fit(X, y)


# ---------------------------------------------------------------- features
def test_to_float_frame_orders_and_casts():
    row = {f: 1 for f in reversed(FEATURES)}
    row["extra"] = "ignored"
    out = to_float_frame([row])
    assert list(out.columns) == FEATURES
    assert all(dt == float for dt in out.dtypes)


def test_to_float_frame_missing_column_raises():
    with pytest.raises(ValueError, match="Missing feature columns"):
        to_float_frame([{"age": 50}])


def test_preprocessor_output_shape(data):
    X, _ = data
    Z = build_preprocessor().fit_transform(X)
    n_onehot = sum(len(VALID_CODES[c]) for c in CATEGORICAL_FEATURES)
    assert Z.shape == (len(X), len(NUMERIC_FEATURES) + n_onehot)
    assert not np.isnan(Z).any()


# ---------------------------------------------------------------- pipeline
def test_prediction_shape_and_labels(fitted, data):
    X, _ = data
    pred = fitted.predict(X)
    assert pred.shape == (len(X),)
    assert set(np.unique(pred)) <= {0, 1}


def test_probabilities_valid(fitted, data):
    X, _ = data
    proba = fitted.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_learns_signal(fitted, data):
    X, y = data
    assert compute_metrics(y, fitted.predict(X),
                           fitted.predict_proba(X)[:, 1])["roc_auc"] > 0.8


def test_handles_missing_and_unseen_values(fitted, data):
    X, _ = data
    row = X.iloc[[0]].astype(float).copy()
    row.loc[:, ["ca", "thal"]] = np.nan   # missing, as in the raw data
    row.loc[:, "cp"] = 9                  # category never seen in training
    proba = fitted.predict_proba(row)
    assert proba.shape == (1, 2) and not np.isnan(proba).any()


def test_accepts_list_of_dicts(fitted, data):
    X, _ = data
    records = X.head(2).to_dict(orient="records")
    assert fitted.predict(records).shape == (2,)


def test_save_and_load_roundtrip(fitted, data, tmp_path):
    X, _ = data
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X),
                               fitted.predict_proba(X))


@pytest.mark.parametrize("cand", get_candidates(quick=True),
                         ids=lambda c: c.name)
def test_every_candidate_fits(cand, data):
    X, y = data
    model = build_pipeline(clone(cand.estimator)).fit(X, y)
    assert model.predict_proba(X.head(5)).shape == (5, 2)


def test_quick_grids_are_prefixed():
    for cand in get_candidates(quick=True):
        assert all(k.startswith("model__") for k in cand.grid)


# ---------------------------------------------------------------- selection
def test_select_best_prefers_auc():
    t = pd.DataFrame({"cv_roc_auc_mean": [0.90, 0.95],
                      "cv_recall_mean": [0.99, 0.70]}, index=["a", "b"])
    assert select_best(t) == "b"


def test_select_best_breaks_ties_on_recall():
    t = pd.DataFrame({"cv_roc_auc_mean": [0.915, 0.910, 0.880],
                      "cv_recall_mean": [0.80, 0.85, 0.95]},
                     index=["a", "b", "c"])
    assert select_best(t) == "b"   # c is outside the 0.01 tie window


def test_compute_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0, 1, 1, 0]),
                        np.array([0.1, 0.6, 0.9, 0.4]))
    assert m["accuracy"] == 0.5 and m["recall"] == 0.5
    assert m["false_negatives"] == 1 and m["false_positives"] == 1
    assert m["roc_auc"] == 0.75
