"""

test regex model

Tests for the regex model

"""

import os
import pytest
import numpy as np
import pandas as pd
import matplotlib

# Disable interactive plotting during tests
matplotlib.use("Agg")

from nlp_pipeline.models.regex.model import (
    train_regex_model,
    predict_regex,
    apply_regex_predictions,
    create_combined_predictions,
    create_per_report_patient_level_predictions,
    create_final_prediction
)

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture
def sample_patterns():
    """Return minimal IBD and drug regex patterns for testing."""
    return {
        "ibd": {
            "crohns": r"\bcrohn'?s?\b",
            "ulcerative_colitis": r"\bulcerative\s+colitis\b"
        },
        "drug": {
            "mesalamine": r"\bmesal(a)?mine\b",
            "immuno": r"\bimmunosuppressant\b"
        }
    }

@pytest.fixture
def sample_df():
    """Return a DataFrame of text columns to test regex matching."""
    return pd.DataFrame({
        "study_id": [101, 102, 103],
        "ibd_text": [
            "Patient diagnosed with Crohns.",
            "Ulcerative colitis found here.",
            "No mention of IBD"
        ],
        "drug_text": [
            "Next step is mesalamine.",
            "Consider immunosuppressant therapy.",
            "No relevant medication."
        ],
        "random_notes": [
            "Random info 1",
            "Random info 2",
            "Random info 3"
        ]
    })


def test_train_regex_model_success(sample_patterns):
    """
    Ensure that train_regex_model compiles IBD and Drug patterns
    into a dict of compiled regexes and returns a non-negative elapsed time.
    """
    ibd_patterns = sample_patterns["ibd"]
    drug_patterns = sample_patterns["drug"]

    compiled, elapsed = train_regex_model(
        ibd_patterns=ibd_patterns,
        drug_patterns=drug_patterns,
        analysis_folder=None,
        note="UnitTest"
    )

    assert isinstance(compiled, dict), "Expected compiled patterns to be returned as a dict."
    assert "ibd" in compiled and "drug" in compiled, "Compiled dict must contain both 'ibd' and 'drug' keys."
    assert isinstance(elapsed, (int, float)) and elapsed >= 0, "Elapsed time must be non-negative."
    assert len(compiled["ibd"]) == len(ibd_patterns), "Number of compiled IBD patterns should match input."
    assert len(compiled["drug"]) == len(drug_patterns), "Number of compiled Drug patterns should match input."


@pytest.mark.parametrize("category,column,expected", [
    ("IBD",  "ibd_text",  np.array([1, 1, 0], dtype=int)),
    ("Drug", "drug_text", np.array([1, 1, 0], dtype=int)),
])

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

def test_predict_regex_matches(category, column, expected, sample_df, sample_patterns):
    """
    Verify that predict_regex returns 1 for matching rows and 0 otherwise.
    Tests both IBD and Drug categories.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )

    result = predict_regex(sample_df, column, compiled, category=category)
    np.testing.assert_array_equal(
        result,
        expected,
        err_msg=f"predict_regex failed for category={category}, column={column}"
    )


def test_predict_regex_with_missing_column(sample_df, sample_patterns):
    """
    If the specified column is absent, predict_regex should return zeros
    without raising an exception.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )

    preds = predict_regex(sample_df, "nonexistent_col", compiled, category="IBD")
    assert len(preds) == len(sample_df), "Length of predictions must equal number of rows."
    np.testing.assert_array_equal(
        preds,
        np.zeros(len(sample_df), dtype=int),
        err_msg="Missing-column predictions should be all zeros."
    )


def test_predict_regex_on_empty_dataframe(sample_patterns):
    """
    When given an empty DataFrame, predict_regex should return an empty array.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )
    empty = pd.DataFrame({"text_column": []})
    preds = predict_regex(empty, "text_column", compiled, category="Drug")
    assert preds.shape == (0,), "Predictions for empty DataFrame must be empty."


def test_apply_regex_predictions_appends_columns(sample_df, sample_patterns):
    """
    apply_regex_predictions should add '<col>_Pred_<Category>' columns for each mapping.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )

    ibd_map  = {"ibd_text": "Label_IBD"}
    drug_map = {"drug_text": None}

    df_ibd  = apply_regex_predictions(sample_df.copy(), ibd_map,  compiled, category="IBD",  show_progress=False)
    df_drug = apply_regex_predictions(sample_df.copy(), drug_map, compiled, category="Drug", show_progress=False)

    assert "ibd_text_Pred_IBD" in df_ibd.columns,  "Expected 'ibd_text_Pred_IBD' column for IBD predictions."
    assert "drug_text_Pred_Drug" in df_drug.columns, "Expected 'drug_text_Pred_Drug' column for Drug predictions."

    # Predictions should match direct calls to predict_regex
    np.testing.assert_array_equal(
        df_ibd["ibd_text_Pred_IBD"].values,
        predict_regex(sample_df, "ibd_text", compiled, category="IBD")
    )
    np.testing.assert_array_equal(
        df_drug["drug_text_Pred_Drug"].values,
        predict_regex(sample_df, "drug_text", compiled, category="Drug")
    )


def test_create_combined_predictions_or_logic(sample_df, sample_patterns):
    """
    create_combined_predictions should produce '<col>_Pred_Combined' that is
    the logical OR of '<col>_Pred_IBD' and '<col>_Pred_Drug'.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )

    df = sample_df.copy()
    df = apply_regex_predictions(df, {"ibd_text": None}, compiled, category="IBD", show_progress=False)
    df = apply_regex_predictions(df, {"ibd_text": None}, compiled, category="Drug", show_progress=False)
    df = create_combined_predictions(df, {"ibd_text": "L"}, {"ibd_text": None})

    assert "ibd_text_Pred_Combined" in df.columns, "Combined prediction column missing."

    ibd_vals  = df["ibd_text_Pred_IBD"].values
    drug_vals = df["ibd_text_Pred_Drug"].values
    comb_vals = df["ibd_text_Pred_Combined"].values

    for i, (i_val, d_val, c_val) in enumerate(zip(ibd_vals, drug_vals, comb_vals)):
        expected = 1 if (i_val == 1 or d_val == 1) else 0
        assert c_val == expected, f"Row {i}: Combined={c_val}, expected {expected}."


def test_create_per_report_patient_level_predictions_preserves_rows(sample_df, sample_patterns):
    """
    create_per_report_patient_level_predictions should add patient-level columns
    without altering the number of rows.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )
    df = sample_df.copy()
    df = apply_regex_predictions(df, {"ibd_text": None}, compiled, category="IBD", show_progress=False)
    df = create_combined_predictions(df, {"ibd_text": "L"}, {})
    df_out = create_per_report_patient_level_predictions(df, {"ibd_text": "L"})

    assert df_out.shape[0] == df.shape[0], "Row count should remain unchanged."
    assert "Patient_ibd_text_Pred_IBD" in df_out.columns, "Expected patient-level prediction column."


def test_create_final_prediction_over_multiple_columns(sample_df, sample_patterns):
    """
    create_final_prediction should generate a 'Final_Prediction' column
    by OR-ing all '<col>_Pred_Combined' across each study.
    """
    compiled, _ = train_regex_model(
        ibd_patterns=sample_patterns["ibd"],
        drug_patterns=sample_patterns["drug"]
    )
    df = sample_df.copy()
    df = apply_regex_predictions(df, {"ibd_text": None}, compiled, category="IBD", show_progress=False)
    df = apply_regex_predictions(df, {"drug_text": None}, compiled, category="Drug", show_progress=False)
    df = create_combined_predictions(df, {"ibd_text": "L"}, {"drug_text": None})
    df_final = create_final_prediction(df)

    assert "Final_Prediction" in df_final.columns, "Final prediction column missing."
    assert df_final["Final_Prediction"].dtype == int, "Final_Prediction must be integer type."
    assert df_final["Final_Prediction"].sum() >= 0, "Final_Prediction sum should be non-negative."
