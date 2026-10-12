"""Data-layer tests. They use a small synthetic raw file, so they run in CI
without network access or the real dataset."""

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    VALID_CODES,
)
from src.data_processing import clean, load_clean, load_raw, save_clean

RAW_ROWS = [
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0",
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2",
    "67.0,1.0,4.0,120.0,229.0,0.0,2.0,129.0,1.0,2.6,2.0,2.0,7.0,1",
    "37.0,1.0,3.0,130.0,250.0,0.0,0.0,187.0,0.0,3.5,3.0,0.0,3.0,0",
    "53.0,0.0,3.0,128.0,216.0,0.0,2.0,115.0,0.0,0.0,1.0,0.0,?,0",
    "38.0,1.0,3.0,138.0,175.0,0.0,0.0,173.0,0.0,0.0,1.0,?,3.0,0",
    "58.0,1.0,4.0,150.0,270.0,0.0,2.0,111.0,1.0,0.8,1.0,0.0,7.0,3",
    "58.0,1.0,4.0,150.0,270.0,0.0,2.0,111.0,1.0,0.8,1.0,0.0,7.0,3",  # dup
    "60.0,1.0,9.0,140.0,293.0,0.0,2.0,170.0,0.0,1.2,2.0,2.0,7.0,4",  # cp=9
]


@pytest.fixture
def raw_file(tmp_path):
    path = tmp_path / "processed.cleveland.data"
    path.write_text("\n".join(RAW_ROWS) + "\n")
    return path


@pytest.fixture
def cleaned(raw_file):
    return clean(load_raw(raw_file))


def test_load_raw_assigns_columns_and_parses_question_marks(raw_file):
    raw = load_raw(raw_file)
    assert raw.shape == (len(RAW_ROWS), 14)
    assert raw["thal"].isna().sum() == 1
    assert raw["ca"].isna().sum() == 1


def test_no_question_marks_remain(cleaned):
    assert not cleaned.astype(str).isin(["?"]).any().any()


def test_expected_columns_in_order(cleaned):
    assert list(cleaned.columns) == FEATURES + [TARGET]
    assert "num" not in cleaned.columns


def test_target_is_binary(cleaned):
    assert set(cleaned[TARGET].unique()) <= {0, 1}
    # num in {1,2,3,4} must all map to 1
    assert cleaned[TARGET].sum() == 4


def test_duplicates_removed(cleaned):
    assert len(cleaned) == len(RAW_ROWS) - 1
    assert not cleaned.duplicated().any()


def test_invalid_codes_become_missing(cleaned):
    for col, valid in VALID_CODES.items():
        observed = set(cleaned[col].dropna().astype(int))
        assert observed <= valid, f"{col} has invalid codes {observed - valid}"
    assert cleaned["cp"].isna().sum() == 1


def test_missing_only_in_expected_columns(cleaned):
    cols_with_nan = set(cleaned.columns[cleaned.isna().any()])
    assert cols_with_nan <= {"ca", "thal", "cp"}
    assert cleaned[NUMERIC_FEATURES].notna().all().all()


def test_dtypes(cleaned):
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(cleaned[col])
    for col in CATEGORICAL_FEATURES:
        assert str(cleaned[col].dtype) == "Int64"


def test_save_and_reload_roundtrip(cleaned, tmp_path):
    path = save_clean(cleaned, tmp_path / "clean.csv")
    reloaded = load_clean(path)
    pd.testing.assert_frame_equal(cleaned, reloaded, check_dtype=False)
    assert np.array_equal(cleaned.isna().values, reloaded.isna().values)


def test_missing_raw_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nope.data")
