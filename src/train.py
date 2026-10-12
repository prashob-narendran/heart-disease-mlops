"""Train, tune, compare and track heart-disease models with MLflow.

What one run of this script does
--------------------------------
1. Loads data/processed/heart_clean.csv and makes a stratified 80/20
   train/test split (fixed seed). The test set is touched only for final
   reporting, never for model selection.
2. For each candidate model (src/models.py) runs GridSearchCV with
   stratified 5-fold CV over the full preprocessing + model pipeline,
   scoring accuracy, precision, recall, F1 and ROC-AUC (refit on ROC-AUC).
3. Logs every model as a nested MLflow run under one parent "session" run:
   params, CV mean/std metrics, test metrics, plots, CV table and the
   fitted pipeline (with signature + input example).
4. Selects the final model on CV results only:
   highest CV ROC-AUC; models within 0.01 of the best are treated as tied
   and the tie is broken by higher CV recall (a missed disease case costs
   more than a false alarm in screening).
5. Registers the winner in the MLflow Model Registry
   ("heart-disease-classifier", alias "champion") and saves a portable copy
   to models/model.joblib + models/metadata.json for the API.

Usage
-----
    python -m src.train                  # full search (~1-2 min on a laptop)
    python -m src.train --quick          # tiny grid, used in CI (separate
                                         # experiment, no registry update)
    mlflow ui --backend-store-uri mlruns # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.config import (
    CLEAN_FILE,
    FEATURES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_MANIFEST,
    TARGET,
)
from src.data_processing import load_clean
from src.evaluate import (
    MODEL_LABELS,
    SCORING,
    compute_metrics,
    permutation_importance_frame,
    plot_confusion_matrix,
    plot_importance,
    plot_model_comparison,
    plot_roc_curves,
)
from src.features import build_pipeline
from src.models import get_candidates

log = logging.getLogger("train")

EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL = "heart-disease-classifier"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
TEST_SIZE = 0.2
CV_FOLDS = 5
TIE_TOLERANCE = 0.01


# ---------------------------------------------------------------- helpers
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - git missing or not a repo
        return "unknown"


def dataset_version() -> str:
    try:
        return json.loads(RAW_MANIFEST.read_text())["sha256"][:12]
    except Exception:  # noqa: BLE001
        return "unknown"


def tracking_uri() -> str:
    # Default: a local ./mlruns folder. Override with MLFLOW_TRACKING_URI,
    # e.g. http://localhost:5000 when an MLflow server is running.
    return os.environ.get("MLFLOW_TRACKING_URI",
                          (PROJECT_ROOT / "mlruns").as_uri())


def select_best(table: pd.DataFrame) -> str:
    best_auc = table["cv_roc_auc_mean"].max()
    tied = table[table["cv_roc_auc_mean"] >= best_auc - TIE_TOLERANCE]
    return tied.sort_values(["cv_recall_mean", "cv_roc_auc_mean"],
                            ascending=False).index[0]


def load_xy():
    df = load_clean(CLEAN_FILE)
    return df[FEATURES], df[TARGET].astype(int)


# ---------------------------------------------------------------- core
def tune_and_log(cand, X_train, y_train, X_test, y_test, cv, tmp: Path):
    """Grid-search one candidate inside its own nested MLflow run."""
    with mlflow.start_run(run_name=cand.name, nested=True) as run:
        t0 = time.perf_counter()
        search = GridSearchCV(
            build_pipeline(cand.estimator), cand.grid, scoring=SCORING,
            refit="roc_auc", cv=cv, n_jobs=-1, return_train_score=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            search.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - t0

        best = search.best_estimator_
        i = search.best_index_
        cvr = search.cv_results_

        # ---- params
        mlflow.set_tags({"model_family": cand.name, "stage": "candidate"})
        params = {k.replace("model__", ""): v
                  for k, v in search.best_params_.items()}
        mlflow.log_params({
            "model_type": type(cand.estimator).__name__,
            "cv_folds": CV_FOLDS,
            "random_seed": RANDOM_SEED,
            "grid_size": len(cvr["params"]),
            **{f"best_{k}": v for k, v in params.items()},
        })

        # ---- CV metrics (mean and std across folds, best params)
        row = {}
        for m in SCORING:
            row[f"cv_{m}_mean"] = float(cvr[f"mean_test_{m}"][i])
            row[f"cv_{m}_std"] = float(cvr[f"std_test_{m}"][i])
        row["cv_train_roc_auc_mean"] = float(cvr["mean_train_roc_auc"][i])

        # ---- test metrics (reporting only)
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        test = compute_metrics(y_test, y_pred, y_proba)
        row.update({f"test_{k}": v for k, v in test.items()})
        row["fit_seconds"] = fit_seconds
        mlflow.log_metrics(row)

        # ---- artifacts
        label = MODEL_LABELS.get(cand.name, cand.name)
        cm_path = plot_confusion_matrix(
            y_test, y_pred, f"{label} - test confusion matrix",
            tmp / f"confusion_matrix_{cand.name}.png")
        roc_path = plot_roc_curves({cand.name: (y_test, y_proba)},
                                   tmp / f"roc_curve_{cand.name}.png",
                                   title=f"{label} - test ROC curve")
        imp = permutation_importance_frame(best, X_test, y_test)
        imp_path = plot_importance(
            imp, f"{label} - permutation importance (test)",
            tmp / f"feature_importance_{cand.name}.png")
        imp.to_csv(tmp / f"feature_importance_{cand.name}.csv", index=False)
        cv_table = pd.DataFrame(cvr).drop(columns=["params"])
        cv_table.to_csv(tmp / f"cv_results_{cand.name}.csv", index=False)
        for p in [cm_path, roc_path, imp_path,
                  tmp / f"feature_importance_{cand.name}.csv",
                  tmp / f"cv_results_{cand.name}.csv"]:
            mlflow.log_artifact(str(p), artifact_path="evaluation")

        # ---- model (whole pipeline, with signature + example input)
        example = X_train.head(3).astype(float)
        signature = infer_signature(example, best.predict_proba(example))
        mlflow.sklearn.log_model(
            best, artifact_path="model", signature=signature,
            input_example=example,
            code_paths=[str(PROJECT_ROOT / "src")],
            pyfunc_predict_fn="predict_proba",
        )

        log.info("%-20s cv_auc=%.3f±%.3f cv_recall=%.3f test_auc=%.3f "
                 "test_recall=%.3f (%d combos, %.1fs)", cand.name,
                 row["cv_roc_auc_mean"], row["cv_roc_auc_std"],
                 row["cv_recall_mean"], row["test_roc_auc"],
                 row["test_recall"], len(cvr["params"]), fit_seconds)
        return {"run_id": run.info.run_id, "model": best, "params": params,
                "metrics": row, "y_proba": y_proba, "importance": imp}


def register_champion(run_id: str, name: str) -> str | None:
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL, "champion",
                                          mv.version)
        client.update_model_version(
            REGISTERED_MODEL, mv.version,
            description=f"{name} selected by CV ROC-AUC (recall tie-break)")
        return mv.version
    except Exception as exc:  # noqa: BLE001 - registry is optional
        log.warning("Model registry step skipped: %s", exc)
        return None


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Train and track models")
    parser.add_argument("--quick", action="store_true",
                        help="tiny grids (CI smoke run)")
    parser.add_argument("--no-register", action="store_true",
                        help="skip the MLflow model registry step")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME)
    args = parser.parse_args(argv)
    if args.quick:
        # CI smoke runs must never overwrite the real champion model.
        args.no_register = True
        if args.experiment == EXPERIMENT_NAME:
            args.experiment = f"{EXPERIMENT_NAME}-ci"

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    for noisy in ("mlflow", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    X, y = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)
    log.info("Train %d rows, test %d rows, positives %.1f%% / %.1f%%",
             len(X_train), len(X_test), y_train.mean() * 100,
             y_test.mean() * 100)

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(args.experiment)
    log.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

    results = {}
    session = datetime.now().strftime("session-%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=session) as parent, \
            tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        mlflow.set_tags({
            "stage": "session",
            "dataset": "UCI Heart Disease - Cleveland",
            "dataset_sha256": dataset_version(),
            "git_commit": git_commit(),
            "quick_mode": str(args.quick),
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        })
        mlflow.log_params({
            "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
            "random_seed": RANDOM_SEED, "n_train": len(X_train),
            "n_test": len(X_test), "n_features": len(FEATURES),
            "selection_rule": f"max cv_roc_auc, tie<{TIE_TOLERANCE} -> "
                              "max cv_recall",
        })

        for cand in get_candidates(quick=args.quick):
            results[cand.name] = tune_and_log(cand, X_train, y_train,
                                              X_test, y_test, cv, tmp)

        # ---- comparison + selection
        table = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
        best_name = select_best(table)
        best = results[best_name]
        table["selected"] = table.index == best_name

        REPORT_DIR.mkdir(exist_ok=True)
        cols = ([f"cv_{m}_mean" for m in SCORING]
                + [f"cv_{m}_std" for m in SCORING]
                + [f"test_{m}" for m in SCORING]
                + ["test_specificity", "test_false_negatives",
                   "test_false_positives", "cv_train_roc_auc_mean",
                   "fit_seconds", "selected"])
        table = table[cols]
        table.to_csv(REPORT_DIR / "model_comparison.csv",
                     float_format="%.4f")
        cmp_png = plot_model_comparison(table, FIG_DIR /
                                        "10_model_comparison_cv.png")
        roc_png = plot_roc_curves(
            {k: (y_test, v["y_proba"]) for k, v in results.items()},
            FIG_DIR / "11_roc_curves_test.png",
            title="ROC curves - all models, test set")
        cm_png = plot_confusion_matrix(
            y_test, best["model"].predict(X_test),
            f"{MODEL_LABELS[best_name]} (selected) - test set",
            FIG_DIR / "12_confusion_matrix_selected.png")
        imp_png = plot_importance(
            best["importance"],
            f"{MODEL_LABELS[best_name]} (selected) - permutation importance",
            FIG_DIR / "13_feature_importance_selected.png")

        for p in [REPORT_DIR / "model_comparison.csv", cmp_png, roc_png,
                  cm_png, imp_png]:
            mlflow.log_artifact(str(p), artifact_path="comparison")
        mlflow.set_tag("selected_model", best_name)
        mlflow.log_metrics({f"selected_{k}": v
                            for k, v in best["metrics"].items()
                            if k.startswith(("cv_", "test_"))})

        # mark the winning child run
        mlflow.MlflowClient().set_tag(best["run_id"], "stage", "selected")

        # ---- registry + portable copy for the API
        version = None if args.no_register else register_champion(
            best["run_id"], best_name)

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(best["model"], MODEL_DIR / "model.joblib")
        metadata = {
            "model_name": best_name,
            "model_label": MODEL_LABELS[best_name],
            "registered_model": REGISTERED_MODEL,
            "registered_version": version,
            "mlflow_run_id": best["run_id"],
            "mlflow_parent_run_id": parent.info.run_id,
            "best_params": {k: (v if isinstance(v, (int, float, str))
                                or v is None else str(v))
                            for k, v in best["params"].items()},
            "metrics": {k: round(float(v), 4)
                        for k, v in best["metrics"].items()},
            "features": FEATURES,
            "threshold": 0.5,
            "dataset_sha256": dataset_version(),
            "git_commit": git_commit(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "quick_mode": args.quick,
        }
        (MODEL_DIR / "metadata.json").write_text(
            json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(MODEL_DIR / "metadata.json"),
                            artifact_path="comparison")

    pd.set_option("display.width", 160)
    show = table[["cv_roc_auc_mean", "cv_roc_auc_std", "cv_recall_mean",
                  "cv_f1_mean", "test_roc_auc", "test_recall",
                  "test_accuracy", "selected"]]
    log.info("Model comparison:\n%s", show.round(3).to_string())
    log.info("Selected: %s (registered version %s)", best_name, version)
    log.info("Saved %s", MODEL_DIR / "model.joblib")
    return metadata


if __name__ == "__main__":
    main()
