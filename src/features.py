"""Feature engineering: the preprocessing half of the model pipeline.

Design (driven by the EDA findings, see notebooks/01_eda.ipynb):

* Numeric features (age, trestbps, chol, thalach, oldpeak)
    median imputation -> StandardScaler
  Very different scales (oldpeak 0-6 vs chol 126-564), so scaling is needed
  for the linear model and the SVM. Median is robust to the plausible but
  extreme outliers that were deliberately kept.

* Categorical features (sex, cp, fbs, restecg, exang, slope, ca, thal)
    most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
  The codes are nominal (thal = 3/6/7 has no order), so they are one-hot
  encoded rather than scaled. `handle_unknown="ignore"` protects against
  rare levels (restecg = 1 has only 4 patients) that a CV fold or a live
  request may contain but the training fold never saw.

* Imputation is inside the pipeline, so it is fitted on training folds only
  (no leakage) and the deployed model accepts missing `ca` / `thal`.

The whole thing is one sklearn object, so preprocessing always travels
with the model when it is saved, logged to MLflow or served by the API.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


def to_float_frame(X) -> pd.DataFrame:
    """Select the model features in a fixed order and cast to float.

    Accepts a DataFrame (or list of dicts) with nullable Int64 columns,
    plain ints, floats or None, and returns float columns with NaN for
    missing values - the format SimpleImputer expects. Extra columns are
    dropped; a missing column raises a clear error.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    return X[FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False)),
    ])
    columns = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("to_float", FunctionTransformer(to_float_frame,
                                         feature_names_out="one-to-one")),
        ("columns", columns),
    ])


def build_pipeline(estimator) -> Pipeline:
    """Full model: preprocessing + estimator, as one object."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])
