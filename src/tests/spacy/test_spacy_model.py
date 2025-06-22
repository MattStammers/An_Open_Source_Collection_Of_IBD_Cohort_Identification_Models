"""

test spacy model

Tests for the spacy model

"""

import pytest
import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

# Functions under test
from nlp_pipeline.models.spacy.model import (
    initialize_spacy,
    create_phrase_matcher,
    apply_spacy_predictions,
    create_final_prediction as spacy_create_final_prediction
)

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

# -----------------------------------------------------------------------------
# Stub spacy.load so initialise_spacy never fails and always returns
# a blank English model with a sentenciser.
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def stub_spacy_load(monkeypatch):
    def fake_load(name=None, **kwargs):
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp
    monkeypatch.setattr(spacy, "load", fake_load)

# -----------------------------------------------------------------------------
# Sample data for apply_spacy_predictions
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_text_df():
    return pd.DataFrame({
        "study_id": [1, 2, 3],
        "report": [
            "Patient diagnosed with Crohn disease.",
            "Findings consistent with ulcerative colitis.",
            "No mention of IBD or other issues."
        ]
    })

# ----------------------------------------------------------------------------- #
# Tests                                                                         #
# ----------------------------------------------------------------------------- #
def test_initialize_spacy():
    """
    initialise_spacy should return a Language object with a sentenciser.
    """
    nlp = initialize_spacy(
        model_name=None,
        neg_termset=None,
        ent_types=None,
        extension_name=None,
        chunk_prefix=None
    )
    assert isinstance(nlp, spacy.language.Language)
    assert "sentencizer" in nlp.pipe_names

def test_create_phrase_matcher():
    """
    create_phrase_matcher should build a PhraseMatcher that finds each keyword.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    keywords = ["crohn", "ulcerative"]
    matcher = create_phrase_matcher(nlp, keywords, "IBD")
    assert isinstance(matcher, PhraseMatcher)

    doc = nlp("crohn ulcerative test")
    matches = matcher(doc)
    assert len(matches) == len(keywords), f"Expected {len(keywords)} matches, got {len(matches)}"

def test_apply_spacy_predictions(sample_text_df):
    """
    apply_spacy_predictions should add a binary prediction column 'report_Pred_IBD'.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    matcher = create_phrase_matcher(nlp, ["crohn"], "IBD")

    columns_map    = {"report": "Dummy_Gold"}
    prediction_map = {"report": ["IBD"]}

    df_pred = apply_spacy_predictions(
        sample_text_df.copy(),
        columns_map,
        prediction_map,
        matcher,  
        matcher,  
        nlp
    )

    # Should have added only the binary column
    assert "report_Pred_IBD" in df_pred.columns, \
        f"Expected 'report_Pred_IBD' in {list(df_pred.columns)}"

    # Row 0 contains "Crohn" → 1; Row 2 no mention → 0
    assert df_pred.loc[0, "report_Pred_IBD"] == 1
    assert df_pred.loc[2, "report_Pred_IBD"] == 0

def test_spacy_create_final_prediction():
    """
    create_final_prediction must detect '<col>_Pred_Combined' columns,
    treat them as integer flags (0 or 1), and set Final_Prediction accordingly.
    """
    # Provide integer combined flags so implementation casts correctly
    df = pd.DataFrame({
        "report_Pred_Combined": [0, 1, 1]
    })
    df_out = spacy_create_final_prediction(df.copy())

    # Now Final_Prediction should mirror the combined flags: [0,1,1]
    expected = np.array([0, 1, 1])
    np.testing.assert_array_equal(df_out["Final_Prediction"].values, expected)
