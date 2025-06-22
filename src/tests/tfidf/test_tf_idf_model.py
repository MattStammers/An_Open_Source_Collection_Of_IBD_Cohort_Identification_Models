"""

test tf_idf model

Tests for the tf_idf model

"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nlp_pipeline.models.tf_idf.model import predict_tfidf, train_tfidf_model

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture
def dummy_df():
    """Tiny DataFrame—enough to fit and predict."""
    return pd.DataFrame(
        {
            "text": ["ibd", "not ibd", "Crohn's", ""],
            "label": [1, 0, 1, 0],
        }
    )

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

# ------------------------------------------------------------------ #
# 1) Training                                                        #
# ------------------------------------------------------------------ #
def test_train_tfidf_model_returns_pipeline_and_positive_time(tmp_path, dummy_df):
    model, elapsed = train_tfidf_model(
        df=dummy_df,
        text_col="text",
        label_col="label",
        analysis_root=tmp_path,
        top_n=3,
        shap_explain=False,
        do_feature_importance=False,
        do_calibration=False,
        cv=None,
        n_jobs=1,
    )

    assert isinstance(model, Pipeline), "train_tfidf_model should return a Pipeline"
    assert elapsed >= 0, "elapsed time should be positive"


# ------------------------------------------------------------------ #
# 2) Inference                                                       #
# ------------------------------------------------------------------ #
def test_predict_tfidf_labels_and_probabilities(tmp_path, dummy_df):
    model, _ = train_tfidf_model(
        df=dummy_df,
        text_col="text",
        label_col="label",
        analysis_root=tmp_path,
        shap_explain=False,
        do_feature_importance=False,
        do_calibration=False,
        cv=None,
    )

    #   2.1 labels ----------------------------------------------------- #
    labels = predict_tfidf(model, dummy_df, text_col="text", proba=False)
    assert labels.shape == (len(dummy_df),)
    assert set(labels).issubset({0, 1})

    #   2.2 probabilities --------------------------------------------- #
    probas = predict_tfidf(model, dummy_df, text_col="text", proba=True)
    assert probas.shape == (len(dummy_df),)
    assert np.all((0.0 <= probas) & (probas <= 1.0)), "probas must be [0, 1]"


# ------------------------------------------------------------------ #
# 3) Graceful degradation                                            #
# ------------------------------------------------------------------ #
def test_predict_tfidf_with_missing_model_or_column(dummy_df):
    # 3.1 model is None
    assert np.all(
        predict_tfidf(None, dummy_df, text_col="text", proba=False) == 0
    ), "None model should yield zero labels"
    assert np.all(
        predict_tfidf(None, dummy_df, text_col="text", proba=True) == 0.0
    ), "None model should yield zero probabilities"

    # 3.2 column missing
    empty_labels = predict_tfidf(
        Pipeline([]), dummy_df.drop(columns=["text"]), text_col="text", proba=False
    )
    assert np.all(empty_labels == 0), "Missing column should yield zero labels"
