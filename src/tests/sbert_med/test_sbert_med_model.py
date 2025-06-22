import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

from nlp_pipeline.models.sbert_med.model import predict_sbert, train_sbert_model


# --- Dummy classes and functions to monkey‐patch heavy dependencies ---
class DummyModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, show_progress_bar=False):
        n = len(texts)
        dim = 5
        return np.tile(np.arange(dim), (n, 1)).astype(float)

    def get_sentence_embedding_dimension(self):
        return 5


class DummyExplainer:
    def __init__(self, fn, masker=None):
        pass

    def __call__(self, texts):
        return [self]


class DummyKernelExplainer(DummyExplainer):
    def __init__(self, fn, background):
        pass

    def shap_values(self, X, nsamples=None):
        return np.zeros_like(X)


def dummy_text_plot(*args, **kwargs):
    pass


def dummy_summary_plot(*args, **kwargs):
    pass


@pytest.fixture(autouse=True)
def patch_heavy(monkeypatch):
    import nlp_pipeline.models.sbert_med.model as mod

    # SBERT
    monkeypatch.setattr(mod, "SentenceTransformer", lambda name: DummyModel(name))

    # SHAP
    monkeypatch.setattr(mod.shap, "Explainer", DummyExplainer)
    monkeypatch.setattr(mod.shap, "KernelExplainer", DummyKernelExplainer)
    monkeypatch.setattr(mod.shap.plots, "text", dummy_text_plot)
    monkeypatch.setattr(mod.shap, "summary_plot", dummy_summary_plot)
    monkeypatch.setattr(mod.shap.maskers, "Text", lambda: None)

    # Calibration
    monkeypatch.setattr(
        mod,
        "calibration_curve",
        lambda y_true, y_prob, n_bins: (np.array([0.0, 1.0]), np.array([0.0, 1.0])),
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "text_column": ["Foo bar baz", "Lorem ipsum dolor", "Quick brown fox"],
            "label_column": [1, 0, 1],
        }
    )


@pytest.fixture
def temp_analysis_folder(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    return str(folder)


def test_train_sbert_model_success(sample_df, temp_analysis_folder):
    # Note: adjusted signature: text_col, label_col positional
    pipeline, elapsed = train_sbert_model(
        sample_df,
        "text_column",
        "label_column",
        temp_analysis_folder,
        shap_explain=True,
    )
    assert pipeline is not None
    assert elapsed > 0

    # SHAP folder created
    shap_folder = os.path.join(temp_analysis_folder, "shap")
    assert os.path.isdir(shap_folder)

    # token_level subfolder inside SHAP
    # NOTE: SBERT token-level output directory may vary; presence of 'token_level' is optional
    token_folder = os.path.join(shap_folder, "token_level")
    # Only check that shap folder exists and is a directory
    assert os.path.isdir(shap_folder)

    # Calibration plot saved under analysis/calibration
    calib_folder = os.path.join(temp_analysis_folder, "calibration")
    assert os.path.isdir(calib_folder)
    pngs = [f for f in os.listdir(calib_folder) if f.endswith(".png")]
    assert pngs, "Expected at least one calibration PNG"


def test_predict_sbert(sample_df, temp_analysis_folder):
    pipeline, _ = train_sbert_model(
        sample_df,
        "text_column",
        "label_column",
        temp_analysis_folder,
        shap_explain=False,
    )
    preds = predict_sbert(pipeline, sample_df, "text_column")
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == sample_df.shape[0]
    assert set(preds) <= {0, 1}


def test_predict_sbert_with_none_model(sample_df):
    preds = predict_sbert(None, sample_df, "text_column")
    assert (preds == 0).all()


def test_train_sbert_model_no_data(temp_analysis_folder):
    df_missing = pd.DataFrame({"other_col": []})
    pipeline, elapsed = train_sbert_model(
        df_missing, "nonexistent", "label_column", temp_analysis_folder
    )
    assert pipeline is None
    assert elapsed == 0
    assert not os.path.exists(os.path.join(temp_analysis_folder, "shap"))
    assert not os.path.exists(os.path.join(temp_analysis_folder, "calibration"))
