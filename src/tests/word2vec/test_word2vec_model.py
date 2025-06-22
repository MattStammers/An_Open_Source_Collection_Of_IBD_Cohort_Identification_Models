"""
test_word2vec_model.py

Unit tests for the Word2Vec-based model code. Mirrors test_bow_model.py.
"""

import os

# Disable interactive plotting backends:
import matplotlib
import numpy as np
import pandas as pd
import pytest

from nlp_pipeline.models.word2vec.model import predict_word2vec, train_word2vec_model

matplotlib.use("Agg")

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture
def sample_df():
    """
    Creates a minimal DataFrame with a text column and a label column
    for testing the Word2Vec model training/prediction.
    """
    data = {
        "text_column": [
            "Example about IBD",
            "Some unrelated text",
            "Positive mention of colitis",
        ],
        "label_column": [1, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_analysis_folder(tmp_path):
    """
    Returns a temporary directory to act as the analysis folder.
    """
    folder = tmp_path / "analysis_word2vec"
    folder.mkdir()
    return str(folder)

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

def test_train_word2vec_model_success(sample_df, temp_analysis_folder):
    """
    Tests that training the Word2Vec model returns a Pipeline and a non-zero
    training time. Also checks whether calibration artifacts are saved.
    """
    # Call with positional arguments: df, text_col, label_col, analysis_root
    model, elapsed = train_word2vec_model(
        sample_df,
        "text_column",
        "label_column",
        temp_analysis_folder,
        top_n=5,  # Smaller top_n for test speed
    )

    assert (
        model is not None
    ), "The model pipeline should not be None when valid data is provided."
    assert elapsed > 0, "Elapsed time should be positive after successful training."

    # Check for calibration output in the expected subfolder
    calib_dir = os.path.join(temp_analysis_folder, "calibration")
    assert os.path.isdir(
        calib_dir
    ), "A 'calibration' folder should be created for calibration plots."
    calib_files = os.listdir(calib_dir)
    assert (
        len(calib_files) >= 1
    ), "Expected at least one calibration plot, but found none."


def test_predict_word2vec(sample_df, temp_analysis_folder):
    """
    Tests that prediction returns an array of the correct shape.
    """
    # Train the Word2Vec pipeline
    model, _ = train_word2vec_model(
        sample_df, "text_column", "label_column", temp_analysis_folder
    )
    predictions = predict_word2vec(model, sample_df, "text_column")
    assert isinstance(predictions, np.ndarray), "Predictions should be a NumPy array."
    assert (
        predictions.shape[0] == sample_df.shape[0]
    ), "Number of predictions should match the number of input rows."


def test_predict_word2vec_with_none_model(sample_df):
    """
    Tests that predict_word2vec returns zeros if the model is None.
    """
    preds = predict_word2vec(None, sample_df, "text_column")
    assert (preds == 0).all(), "All predictions should be 0 when the model is None."


def test_train_word2vec_model_no_data(temp_analysis_folder):
    """
    Tests that training returns (None, 0) if the required text_col is missing
    or if there's no usable data.
    """
    empty_df = pd.DataFrame({"some_column": []})
    model, elapsed = train_word2vec_model(
        empty_df, "missing_column", "label_column", temp_analysis_folder
    )
    assert (
        model is None
    ), "Model should be None if 'text_col' doesn't exist or no data is available."
    assert elapsed == 0, "Elapsed time should be 0 if training did not actually occur."
