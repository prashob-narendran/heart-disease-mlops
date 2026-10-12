"""Load, clean and save the Cleveland heart disease data.

Cleaning decisions (documented here and in the report):

1. Column names - the raw file has no header, so names come from
   `config.RAW_COLUMNS` (UCI documentation order).
2. Missing values - the raw file marks them with "?". They occur only in
   `ca` (4 rows) and `thal` (2 rows), i.e. 6 of 303 patients (~2%).
   They are converted to NaN but NOT filled here. Imputation happens inside
   the scikit-learn pipeline (median / most-frequent), fitted on training
   folds only. This avoids test-set leakage and means the deployed API can
   accept a patient whose `ca` or `thal` value is unknown.
3. Invalid codes - any categorical value outside its documented code set is
   treated as missing rather than silently kept.
4. Target - `num` (0-4) is collapsed to binary: 0 = no disease,
   1-4 = disease present. `num` is then dropped so it cannot leak.
5. Duplicates - exact duplicate rows are removed (none expected, but the
   check protects against a corrupted re-download).

Run directly:  python -m src.data_processing
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_FILE,
    FEATURES,
    NUMERIC_FEATURES,
    RAW_COLUMNS,
    RAW_FILE,
    TARGET,
    VALID_CODES,
)

log = logging.getLogger(__name__)


def load_raw(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Read the header-less raw UCI file, turning "?" into NaN."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/download_data.py` first"
        )
    df = pd.read_csv(
        path,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules described in the module docstring."""
    df = df.copy()

    # Everything should be numeric after "?" -> NaN; coerce defensively.
    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Invalid categorical codes -> NaN.
    for col, valid in VALID_CODES.items():
        bad = df[col].notna() & ~df[col].isin(valid)
        if bad.any():
            log.warning("%s: %d invalid codes set to NaN", col, int(bad.sum()))
            df.loc[bad, col] = np.nan

    # Rows with no diagnosis are unusable.
    df = df.dropna(subset=["num"])

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        log.warning("Dropped %d duplicate rows", before - len(df))

    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    # Categorical codes are whole numbers; nullable Int64 keeps NaN support.
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].round().astype("Int64")
    df["age"] = df["age"].round().astype("Int64")
    for col in NUMERIC_FEATURES:
        if col != "age":
            df[col] = df[col].astype(float)

    return df[FEATURES + [TARGET]].reset_index(drop=True)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    counts = counts[counts > 0]
    return pd.DataFrame({
        "missing": counts,
        "percent": (counts / len(df) * 100).round(2),
    })


def load_clean(path: Path | str = CLEAN_FILE) -> pd.DataFrame:
    """Load the processed CSV with the same dtypes `clean()` produces."""
    df = pd.read_csv(path)
    for col in CATEGORICAL_FEATURES + ["age"]:
        df[col] = df[col].astype("Int64")
    return df


def save_clean(df: pd.DataFrame, path: Path | str = CLEAN_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(message)s")
    raw = load_raw()
    log.info("Raw shape: %s", raw.shape)
    df = clean(raw)
    log.info("Clean shape: %s", df.shape)
    log.info("Missing values:\n%s", missing_report(df).to_string())
    log.info("Target balance: %s",
             df[TARGET].value_counts().sort_index().to_dict())
    out = save_clean(df)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
