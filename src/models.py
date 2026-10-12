"""Candidate models and their hyperparameter search spaces.

Four model families, chosen to cover different inductive biases:

* Logistic Regression - linear, interpretable baseline; L1/L2 regularisation
  lets the model down-weight weak features (chol, fbs) on its own.
* Random Forest       - bagged trees; captures interactions, robust to
  outliers and scaling.
* Gradient Boosting   - boosted shallow trees; usually strongest on small
  tabular data, but easiest to overfit with 300 rows.
* SVM (RBF kernel)    - non-linear margin-based model; a useful contrast to
  the tree models on standardised features.

Grid sizes are kept small enough that the full search finishes in about a
minute on a laptop CPU. `quick=True` gives a tiny grid used in CI, where
the goal is to prove the pipeline runs, not to find the best model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.config import RANDOM_SEED


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict
    quick_grid: dict = field(default_factory=dict)


def _prefix(grid: dict) -> dict:
    """Grid keys must address the `model` step of the pipeline."""
    return {f"model__{k}": v for k, v in grid.items()}


def get_candidates(quick: bool = False) -> list[Candidate]:
    seed = RANDOM_SEED
    cands = [
        Candidate(
            "logistic_regression",
            LogisticRegression(solver="liblinear", max_iter=2000,
                               random_state=seed),
            grid={"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                  "penalty": ["l1", "l2"],
                  "class_weight": [None, "balanced"]},
            quick_grid={"C": [1.0], "penalty": ["l2"]},
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(random_state=seed, n_jobs=1),
            grid={"n_estimators": [300],
                  "max_depth": [None, 4, 6],
                  "min_samples_leaf": [1, 3, 5],
                  "max_features": ["sqrt", 0.5],
                  "class_weight": [None, "balanced"]},
            quick_grid={"n_estimators": [50], "max_depth": [4]},
        ),
        Candidate(
            "gradient_boosting",
            GradientBoostingClassifier(random_state=seed),
            grid={"n_estimators": [100, 200],
                  "learning_rate": [0.03, 0.1],
                  "max_depth": [2, 3],
                  "subsample": [0.8, 1.0]},
            quick_grid={"n_estimators": [50], "max_depth": [2]},
        ),
        Candidate(
            "svm_rbf",
            SVC(kernel="rbf", probability=True, random_state=seed),
            grid={"C": [0.1, 0.3, 1.0, 3.0, 10.0],
                  "gamma": ["scale", 0.01, 0.03],
                  "class_weight": [None, "balanced"]},
            quick_grid={"C": [1.0], "gamma": ["scale"]},
        ),
    ]
    for c in cands:
        c.grid = _prefix(c.quick_grid if quick else c.grid)
    return cands
