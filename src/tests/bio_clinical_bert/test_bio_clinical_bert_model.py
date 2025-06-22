"""

test bio-clinical bert model

Tests for the bioclinical bert model

"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

import nlp_pipeline.models.bio_clinical_bert.model as mod
from nlp_pipeline.models.bio_clinical_bert.model import (
    predict_bio_clinical_bert,
    train_bio_clinical_bert_for_column,
)

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def patch_heavy(monkeypatch):
    """
    Monkey patch all heavy dependencies so the bio_clinical_bert training
    and explainability run instantly and deterministically.
    """

    # --- Force CPU only, no DataParallel path ---
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # --- Dummy Tokeniser ---
    class DummyTokenizer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(
            self,
            texts,
            return_tensors=None,
            padding=None,
            truncation=None,
            max_length=None,
        ):
            bsz = len(texts)
            seq = max_length or 8
            return {
                "input_ids": np.zeros((bsz, seq), dtype=int),
                "attention_mask": np.ones((bsz, seq), dtype=int),
            }

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

        def convert_ids_to_tokens(self, ids):
            return ["tok"] * len(ids)

    # --- Dummy Model ---
    class DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __init__(self, *args, **kwargs):
            pass

        def to(self, device):
            pass

        def train(self):
            pass

        def eval(self):
            pass

        def parameters(self):
            return []

        def save_pretrained(self, output_dir):
            os.makedirs(output_dir, exist_ok=True)

        def __call__(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            output_attentions=False,
        ):
            class O:
                pass

            o = O()
            if labels is not None:
                o.loss = torch.tensor(0.0)
            bsz = input_ids.shape[0]
            o.logits = torch.zeros((bsz, 2))
            if output_attentions:
                o.attentions = []
            return o

    # --- Dummy Dataset & Loader ---
    class DummyDataset:
        def __init__(self, texts, labels, tokenizer):
            self.n = len(texts)

        def __len__(self):
            return self.n

    class DummyLoader:
        def __init__(self, dataset, batch_size=None, shuffle=False):
            self.n = len(dataset)

        def __iter__(self):
            # Yield exactly one batch
            yield {
                "input_ids": torch.zeros((self.n, 1), dtype=torch.long),
                "attention_mask": torch.ones((self.n, 1), dtype=torch.long),
                "labels": torch.zeros((self.n,), dtype=torch.long),
            }

        def __len__(self):
            return 1

    # --- Stub out optimizers, schedulers, explainability ---
    monkeypatch.setattr(mod, "BertTokenizer", DummyTokenizer)
    monkeypatch.setattr(mod, "BertForSequenceClassification", DummyModel)
    monkeypatch.setattr(mod, "TextDataset", DummyDataset)
    monkeypatch.setattr(mod, "DataLoader", DummyLoader)
    monkeypatch.setattr(mod, "AdamW", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "get_scheduler", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "run_shap_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "run_integrated_gradients", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "run_lime_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod, "run_attention_visualization", lambda *args, **kwargs: None
    )

@pytest.fixture
def sample_df():
    """Minimal DataFrame with a text column and binary label."""
    return pd.DataFrame(
        {
            "text_column": ["Positive example", "Negative example", "Neutral text"],
            "label_column": [1, 0, 1],
        }
    )

@pytest.fixture
def out_dir(tmp_path):
    """Temporary output directory for model artifacts."""
    return str(tmp_path / "analysis")

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

def test_train_and_predict(sample_df, out_dir):
    """
    1) Train with zero epochs (skips the training loop entirely).
    2) Assert returned model, tokeniser, preds, and metrics.
    3) Check artifacts are created.
    4) Call predict_bio_clinical_bert for both probabilities and labels.
    """
    # Train with num_epochs=0 to skip the loop
    model, tokenizer, val_preds, metrics = train_bio_clinical_bert_for_column(
        sample_df,
        sample_df,
        "text_column",
        "label_column",
        mod.BertTokenizer(),
        out_dir,
        num_epochs=0,
        perform_shap=True,
        perform_lime=True,
    )

    # Basic assertions
    assert model is not None
    assert tokenizer is not None
    assert isinstance(val_preds, np.ndarray)
    assert val_preds.shape[0] == len(sample_df)
    assert isinstance(metrics, dict)
    for key in ("Accuracy", "Precision", "Recall", "Specificity", "NPV", "F1_Score"):
        assert key in metrics

    # The output directory should now exist
    assert os.path.isdir(out_dir)

    # Predict probabilities
    probs = predict_bio_clinical_bert(
        model, tokenizer, sample_df, "text_column", batch_size=2, proba=True
    )
    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == len(sample_df)
    # DummyModel logits=0 → softmax positive class = 0.5
    assert np.allclose(probs, 0.5, atol=1e-6)

    # Predict labels
    labs = predict_bio_clinical_bert(
        model, tokenizer, sample_df, "text_column", batch_size=2, proba=False
    )
    assert isinstance(labs, np.ndarray)
    assert labs.shape[0] == len(sample_df)
    # Zero logits → argmax = class 0
    assert set(labs) == {0}
