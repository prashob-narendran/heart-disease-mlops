"""End-to-end tests: run EDA and the full training script (quick grids) on
synthetic data in a temporary folder, with a throwaway MLflow store.

Proves that train.py, evaluate.py and eda.py work together and produce
every expected output, without touching the real data, models or mlruns.
"""

import json

import joblib
import pandas as pd
import pytest

import src.train as train
from src import eda
from src.config import TARGET
from src.predict import predict_records
from tests.test_model import make_data


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    X, y = make_data(n=160, seed=1)
    df = X.copy()
    df[TARGET] = y
    clean = root / "heart_clean.csv"
    df.to_csv(clean, index=False)
    return root, clean


@pytest.fixture(scope="module")
def trained(workspace):
    root, clean = workspace
    mp = pytest.MonkeyPatch()
    mp.setenv("MLFLOW_TRACKING_URI", (root / "mlruns").as_uri())
    mp.setattr(train, "CLEAN_FILE", clean)
    mp.setattr(train, "MODEL_DIR", root / "models")
    mp.setattr(train, "MODEL_FILE", root / "models" / "model.joblib")
    mp.setattr(train, "METADATA_FILE", root / "models" / "metadata.json")
    mp.setattr(train, "REPORT_DIR", root / "reports")
    mp.setattr(train, "FIG_DIR", root / "reports" / "figures")
    meta = train.main(["--quick"])
    yield root, meta
    mp.undo()


def test_training_writes_all_outputs(trained):
    root, meta = trained
    # quick runs must write to the quick/ sub-folders only
    assert not (root / "models" / "model.joblib").exists()
    assert (root / "models" / "quick" / "model.joblib").exists()
    saved = json.loads((root / "models" / "quick" / "metadata.json")
                       .read_text())
    assert saved["model_name"] == meta["model_name"]
    assert saved["quick_mode"] is True
    assert saved["registered_version"] is None   # quick never registers
    table = pd.read_csv(root / "reports" / "quick" / "model_comparison.csv",
                        index_col=0)
    assert len(table) == 4 and table["selected"].sum() == 1
    for f in ["10_model_comparison_cv.png", "11_roc_curves_test.png",
              "12_confusion_matrix_selected.png",
              "13_feature_importance_selected.png"]:
        assert (root / "reports" / "quick" / "figures" / f).exists()


def test_training_logs_to_mlflow(trained):
    import mlflow

    root, meta = trained
    client = mlflow.MlflowClient(tracking_uri=(root / "mlruns").as_uri())
    exp = client.get_experiment_by_name(f"{train.EXPERIMENT_NAME}-ci")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 5                     # 1 session + 4 models
    child = client.get_run(meta["mlflow_run_id"])
    assert "cv_roc_auc_mean" in child.data.metrics
    assert child.data.tags["stage"] == "selected"
    artifacts = {a.path for a in client.list_artifacts(child.info.run_id)}
    assert {"model", "evaluation"} <= artifacts


def test_saved_model_serves_predictions(trained):
    root, _ = trained
    model = joblib.load(root / "models" / "quick" / "model.joblib")
    X, _ = make_data(n=5, seed=2)
    out = predict_records(model, X.to_dict(orient="records"), threshold=0.5)
    assert len(out) == 5


def test_metrics_are_sane(trained):
    _, meta = trained
    m = meta["metrics"]
    for k in ["cv_roc_auc_mean", "test_roc_auc", "test_recall"]:
        assert 0.0 <= m[k] <= 1.0
    assert m["cv_roc_auc_mean"] > 0.7         # synthetic signal is learnable


def test_eda_generates_all_figures(workspace, tmp_path):
    _, clean = workspace
    df = pd.read_csv(clean)
    paths = eda.run_all(df, tmp_path)
    assert len(paths) == len(eda.ALL_PLOTS)
