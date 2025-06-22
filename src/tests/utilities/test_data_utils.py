"""

test data_utils.py

Tests for the utilities

"""

import logging
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import nlp_pipeline.common.data_utils as du
from nlp_pipeline.common.caching import memory


# --------------------------------------------------------------------------- #
# Fixture: Redirect joblib cache to isolated temp path before each test       #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolate_cache(monkeypatch, tmp_path):
    """
    Isolates the caching layer by redirecting the cache location to a temporary
    directory for each test execution. Ensures that test runs are not polluted
    by existing cache data.
    """
    monkeypatch.setattr(memory, "location", str(tmp_path), raising=True)
    memory.clear(warn=False)
    yield
    shutil.rmtree(str(tmp_path), ignore_errors=True)


# --------------------------------------------------------------------------- #
# Test: Ethnicity mapping function covers null, known, and unknown inputs      #
# --------------------------------------------------------------------------- #
def test_map_ethnicity_null_and_known_and_unknown():
    """
    Validates that the ethnicity mapping function handles null values,
    known ethnicity terms (case-insensitive), and unknown inputs correctly.
    """
    assert du._map_ethnicity(None) == "Not Asked"
    assert du._map_ethnicity("White British") == "White"
    assert du._map_ethnicity("indian") == "Asian"
    assert du._map_ethnicity("MIXED White And BLACK African") == "African"
    assert du._map_ethnicity("Martian") == "Not Asked"


# --------------------------------------------------------------------------- #
# Helper: Write single-sheet Excel file                                       #
# --------------------------------------------------------------------------- #
def make_excel(df: pd.DataFrame, path: str):
    """
    Writes the given DataFrame to a single-sheet Excel file at the specified path.

    Args:
        df (pd.DataFrame): Data to write.
        path (str): Output Excel file path.
    """
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False)


# --------------------------------------------------------------------------- #
# Test: Minimal preprocessing pipeline behavior                               #
# --------------------------------------------------------------------------- #
import logging


def test_load_and_preprocess_minimal(tmp_path, monkeypatch):
    monkeypatch.setattr(du, "COLUMNS_TO_DROP", ["drop_me"], raising=True)
    monkeypatch.setattr(du, "COLWISE_COLUMNS", {}, raising=True)  # nothing to rename
    monkeypatch.setattr(du, "PATIENT_COLUMNS", {"pat1": "LBL2"}, raising=True)

    # Patch logger.warning to avoid pytest capture bug
    monkeypatch.setattr(
        logging.getLogger("nlp_pipeline.common.data_utils"), "warning", lambda msg: None
    )

    df = pd.DataFrame(
        {
            "study_id": [1, 2],
            "drop_me": ["x", "y"],
            "histopathology_report": [1, 0],
            "pat1": ["NaN", "1"],
        }
    )

    file_path = str(tmp_path / "inp.xlsx")
    make_excel(df, file_path)

    out = du.load_and_preprocess(file_path)

    assert "drop_me" not in out.columns
    assert list(out["histopathology_report"]) == [1, 0]


# --------------------------------------------------------------------------- #
# Test: Full preprocessing pipeline with conditional inclusion                #
# --------------------------------------------------------------------------- #
def test_load_and_preprocess_full_pipeline(tmp_path, monkeypatch):
    """
    Evaluates preprocessing behavior on a full-featured input.
    Validates text aggregation, age group creation, IMD binning, and ethnicity mapping.
    """
    monkeypatch.setattr(
        logging.getLogger("nlp_pipeline.common.data_utils"), "warning", lambda msg: None
    )
    monkeypatch.setattr(du, "COLUMNS_TO_DROP", [], raising=True)
    monkeypatch.setattr(
        du,
        "COLWISE_COLUMNS",
        {
            "preceding_clinic_letter": "Preceding_Clinic_IBD",
            "following_clinic_letter": "Following_Clinic_IBD",
        },
        raising=True,
    )
    monkeypatch.setattr(du, "PATIENT_COLUMNS", {}, raising=True)

    # Supply label-mapped columns directly
    df = pd.DataFrame(
        {
            "study_id": [10],
            "Preceding_Clinic_IBD": ["A"],
            "Following_Clinic_IBD": ["B"],
            "age": [25],
            "imd": [7],
            "ethnicity": ["Pakistani"],
        }
    )

    file_path = str(tmp_path / "full.xlsx")
    make_excel(df, file_path)

    out = du.load_and_preprocess(file_path)

    assert out.loc[0, "combined_text"] == "A B"
    assert out.loc[0, "age_group"] == "20-30"
    assert out.loc[0, "imd_group"] == "7-8"
    assert out.loc[0, "ethnicity"] == "Asian"


# --------------------------------------------------------------------------- #
# Test: Verify caching behavior for load_and_preprocess                       #
# --------------------------------------------------------------------------- #
def test_load_and_preprocess_cache_behavior(tmp_path, monkeypatch):
    """
    Confirms that load_and_preprocess utilizes the caching layer. Ensures
    that repeated calls with the same file path return cached results.
    """
    monkeypatch.setattr(du, "COLUMNS_TO_DROP", [], raising=True)
    monkeypatch.setattr(du, "COLWISE_COLUMNS", {}, raising=True)
    monkeypatch.setattr(du, "PATIENT_COLUMNS", {}, raising=True)

    df = pd.DataFrame({"study_id": [1]})
    file_path = str(tmp_path / "c.xlsx")
    df.to_excel(file_path, index=False)

    out1 = du.load_and_preprocess(file_path)
    out2 = du.load_and_preprocess(file_path)
    pd.testing.assert_frame_equal(out1, out2)

    cache_files = list(os.walk(memory.location))
    assert any(cache_files)
