"""

test feature_importance.py

Tests for the feature importance calculation functions

"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

from nlp_pipeline.common.feature_importance import (
    extract_feature_importance,
    plot_feature_importance,
    save_feature_importance,
)


# --------------------------------------------------------------------------- #
# Helper: Dummy model with vectorizer and classifier                          #
# --------------------------------------------------------------------------- #
class DummyVect:
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def get_feature_names_out(self):
        return np.array(self._feature_names)


class DummyClf:
    def __init__(self, coefficients):
        # coefficients should be a list or array
        self.coef_ = np.array([coefficients])


def create_dummy_model(feature_names, coefficients):
    """
    Constructs a dummy pipeline-like model with named_steps 'vect' and 'clf'.
    """
    model = type("DummyModel", (), {})()
    model.named_steps = {
        "vect": DummyVect(feature_names),
        "clf": DummyClf(coefficients),
    }
    return model


# --------------------------------------------------------------------------- #
# Test: extract_feature_importance returns correct top features               #
# --------------------------------------------------------------------------- #
def test_extract_feature_importance_basic():
    """
    Validates top positive and negative feature extraction for a simple model.
    """
    feature_names = ["a", "b", "c", "d"]
    coefficients = [0.1, -0.2, 0.3, -0.4]
    model = create_dummy_model(feature_names, coefficients)

    top_pos, top_neg = extract_feature_importance(model, top_n=2)

    # Top positive should be sorted by descending coefficients
    assert list(top_pos["feature"]) == ["c", "a"]
    assert list(top_pos["coefficient"]) == [0.3, 0.1]

    # Top negative should be the two smallest coefficients
    assert list(top_neg["feature"]) == ["b", "d"]
    assert list(top_neg["coefficient"]) == [-0.2, -0.4]


# --------------------------------------------------------------------------- #
# Test: plot_feature_importance saves plots and logs                          #
# --------------------------------------------------------------------------- #
def test_plot_feature_importance_creates_plots_and_logs(tmp_path, caplog):
    """
    Ensures that plot_feature_importance generates PNG files for positive and negative
    features and logs an info message.
    """
    caplog.set_level(logging.INFO)

    # Minimal feature importance data
    data_pos = pd.DataFrame({"feature": ["x"], "coefficient": [1.2]})
    data_neg = pd.DataFrame({"feature": ["y"], "coefficient": [-0.5]})
    report_col = "col"
    pred_type = "Type"
    analysis_folder = str(tmp_path)

    plot_feature_importance(data_pos, data_neg, report_col, pred_type, analysis_folder)

    fi_folder = os.path.join(analysis_folder, "feature_importance")
    pos_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_positive_features.png"
    )
    neg_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_negative_features.png"
    )

    # Files created
    assert os.path.isfile(pos_path)
    assert os.path.isfile(neg_path)

    # Log message recorded
    assert any(
        "Saved feature importance plots for col (Type)." in rec.message
        for rec in caplog.records
    )


# --------------------------------------------------------------------------- #
# Test: save_feature_importance writes CSV files and logs                     #
# --------------------------------------------------------------------------- #
def test_save_feature_importance_creates_csv_and_logs(tmp_path, caplog):
    """
    Validates that save_feature_importance writes CSV files for positive and negative
    features and logs an info message.
    """
    caplog.set_level(logging.INFO)

    data_pos = pd.DataFrame({"feature": ["x"], "coefficient": [1.2]})
    data_neg = pd.DataFrame({"feature": ["y"], "coefficient": [-0.5]})
    report_col = "col"
    pred_type = "Type"
    analysis_folder = str(tmp_path)

    save_feature_importance(data_pos, data_neg, report_col, pred_type, analysis_folder)

    fi_folder = os.path.join(analysis_folder, "feature_importance")
    pos_csv = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_positive_features.csv"
    )
    neg_csv = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_negative_features.csv"
    )

    # CSV files created
    assert os.path.isfile(pos_csv)
    assert os.path.isfile(neg_csv)

    # Contents match input DataFrames
    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)
    pd.testing.assert_frame_equal(df_pos, data_pos)
    pd.testing.assert_frame_equal(df_neg, data_neg)

    # Log message recorded
    assert any(
        "Saved feature importance data for col (Type)." in rec.message
        for rec in caplog.records
    )
