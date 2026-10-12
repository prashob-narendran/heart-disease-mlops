"""Model quality gate.

Reads models/metadata.json (written by `python -m src.train`) and fails
with exit code 1 if the selected model is below the minimum acceptable
performance. CI runs this after training, so a code change that silently
degrades the model breaks the build instead of reaching deployment.

Usage:
    python scripts/check_model.py
    python scripts/check_model.py --min-cv-auc 0.88 --min-cv-recall 0.75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METADATA_FILE, MODEL_FILE  # noqa: E402

DEFAULTS = {
    "cv_roc_auc_mean": 0.85,
    "cv_recall_mean": 0.70,
    "test_roc_auc": 0.80,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-cv-auc", type=float,
                        default=DEFAULTS["cv_roc_auc_mean"])
    parser.add_argument("--min-cv-recall", type=float,
                        default=DEFAULTS["cv_recall_mean"])
    parser.add_argument("--min-test-auc", type=float,
                        default=DEFAULTS["test_roc_auc"])
    args = parser.parse_args()

    if not MODEL_FILE.exists() or not METADATA_FILE.exists():
        print(f"FAIL: {MODEL_FILE} or {METADATA_FILE} missing - "
              "run `python -m src.train` first")
        return 1

    meta = json.loads(METADATA_FILE.read_text())
    metrics = meta["metrics"]
    gates = {
        "cv_roc_auc_mean": args.min_cv_auc,
        "cv_recall_mean": args.min_cv_recall,
        "test_roc_auc": args.min_test_auc,
    }
    print(f"Model: {meta['model_name']}  (trained {meta['trained_at_utc']})")
    failed = False
    for name, minimum in gates.items():
        value = metrics[name]
        ok = value >= minimum
        failed |= not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {value:.3f}  "
              f"(min {minimum:.2f})")
    print("Quality gate:", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
