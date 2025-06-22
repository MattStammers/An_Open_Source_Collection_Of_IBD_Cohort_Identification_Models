"""
spacy model.py

This module implements the core functions for a SpaCy-based IBD NLP model.
It includes the following features:

  • A caching decorator for function results (using joblib).
  • A function to initialise a SpaCy pipeline enhanced with NegSpacy for negation detection.
  • Functions to handle keyword/phrase-based matching (IBD, Drug, Combined), including optional
    negation checks with NegSpacy.
  • A function to apply phrase matching predictions to DataFrame text columns, returning integer
    “hit” scores.
  • A helper function to create a final binary prediction from one or more '_Pred_Combined' columns.

IMPORTANT: Any UMLS standardisation is assumed to have occurred upstream. This file no longer
contains logic for concept mapping or text replacement. Text input is taken as-is.

Author: Matt Stammers / UHSFT
"""

import logging
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from negspacy.negation import Negex
from spacy.language import Language
from spacy.tokens import Span
from tqdm import tqdm

from nlp_pipeline.config.constants import (
    CACHE_DIR,
    DRUG_KEYWORDS,
    IBD_KEYWORDS,
    NEG_TERMSET,
)

# ---------------------------------------------------------------------------#
# Caching Setup                                                              #
# ---------------------------------------------------------------------------#

logger = logging.getLogger(__name__)
logger.propagate = False
memory = joblib.Memory(location=CACHE_DIR, verbose=0)

def cached(func):
    """
    Decorator to cache the results of functions using joblib Memory.
    Repeated calls with the same arguments will return cached results,
    thereby reducing recomputation.

    Args:
        func (callable): The function whose results are to be cached.

    Returns:
        callable: The decorated function with caching enabled.
    """
    return memory.cache(func)

# ---------------------------------------------------------------------------#
# Utility for Cleaning Regex Markers in Keywords                             #
# ---------------------------------------------------------------------------#
def clean_keyword(keyword):
    """
    Remove regex word-boundary markers from a keyword if present.

    Args:
        keyword (str): A keyword possibly containing literal regex word-boundary markers
                       (e.g. '\\bibd\\b').

    Returns:
        str: The cleaned keyword without the '\\b' markers.
    """
    if keyword.startswith(r"\b") and keyword.endswith(r"\b"):
        return keyword[2:-2]
    return keyword

# ---------------------------------------------------------------------------#
# SpaCy Pipeline Initialisation with NegSpacy                                #
# ---------------------------------------------------------------------------#
def initialize_spacy(
    model_name: str = "en_core_web_md",
    neg_termset: dict = NEG_TERMSET,
    ent_types: list = None,
    extension_name: str = "negex",
    chunk_prefix: list = None,
) -> spacy.Language:
    """
    Initialise and configure a SpaCy pipeline for English text with NegSpacy,
    using your curated negation termset by default.
    """

    # 1. Load model or fall back to blank English
    if model_name is None:
        nlp = spacy.blank("en")
        logger.info("Created blank SpaCy English pipeline.")
    else:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded SpaCy model '{model_name}'.")
        except OSError:
            logger.error(
                f"SpaCy model '{model_name}' not found. Please ensure it is installed."
            )
            raise

    # 2. Ensure sentencizer
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
        logger.info("Added 'sentencizer' to the SpaCy pipeline.")

    # 3. Default None back to your constants
    if neg_termset is None:
        neg_termset = NEG_TERMSET
    if extension_name is None:
        extension_name = "negex"

    # 4. Build config for NegSpacy
    config: dict[str, object] = {
        "neg_termset": neg_termset,
        "extension_name": extension_name,
    }
    if ent_types is not None:
        config["ent_types"] = ent_types
    if chunk_prefix is not None:
        config["chunk_prefix"] = chunk_prefix

    # 5. Add (or skip) the negation component
    if extension_name not in nlp.pipe_names:
        nlp.add_pipe("negex", last=True, config=config)
        logger.info(f"Added NegSpacy component '{extension_name}'.")
    else:
        logger.warning(
            f"NegSpacy ('{extension_name}') already exists; skipping addition."
        )

    return nlp

# ---------------------------------------------------------------------------#
# Phrase Matching and Negation Detection                                     #
# ---------------------------------------------------------------------------#
def create_phrase_matcher(nlp, keywords, label):
    """
    Create a SpaCy PhraseMatcher configured for the provided keywords.
    The matcher is case-insensitive (using the 'LOWER' attribute) and is associated
    with a specified label.

    Because your configuration (constants.py) defines keywords as regex patterns
    (e.g. r"\\binflammatory bowel disease\\b"), this function cleans them to plain
    text (e.g. "inflammatory bowel disease").

    Args:
        nlp (spacy.Language): The SpaCy pipeline to use for tokenisation.
        keywords (list of str): List of phrases or keywords for matching; these may
                                include regex markers.
        label (str): Label to assign to the matcher patterns.

    Returns:
        spacy.matcher.PhraseMatcher: An initialised PhraseMatcher instance.
    """
    from spacy.matcher import PhraseMatcher

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Clean each keyword from any regex word-boundary markers.
    cleaned_keywords = [clean_keyword(kw) for kw in keywords]

    # Create patterns (using SpaCy’s tokenisation) from the cleaned keywords.
    patterns = [nlp.make_doc(text) for text in cleaned_keywords]

    matcher.add(label, patterns)
    logger.info(
        f"Created PhraseMatcher for label '{label}' with {len(patterns)} pattern(s)."
    )

    return matcher


def detect_phrases_with_negation(doc, matcher):
    """
    Identify spans in a SpaCy Doc that match the phrases provided via the matcher,
    excluding those spans which are negated according to the NegSpacy component.

    Args:
        doc (spacy.tokens.Doc): A processed SpaCy Doc.
        matcher (spacy.matcher.PhraseMatcher): A PhraseMatcher instance with target patterns.

    Returns:
        list of spacy.tokens.Span: List of spans that match the patterns and are not negated.
    """
    matches = matcher(doc)
    non_negated_spans = [
        span
        for match_id, start, end in matches
        for span in (doc[start:end],)
        # if there's no ._.negex extension, treat as 'not negated'
        if not getattr(span._, "negex", False)
    ]
    return non_negated_spans

# ---------------------------------------------------------------------------#
# SpaCy Prediction Logic                                                     #
# ---------------------------------------------------------------------------#
def apply_spacy_predictions(
    df, columns_mapping, prediction_types_mapping, ibd_matcher, drug_matcher, nlp
):
    """
    Apply SpaCy-based phrase matching and negation detection across specified DataFrame columns.
    For each column in the columns_mapping, predictions are generated for each prediction type
    (IBD, Drug, Combined) by counting the number of non-negated phrase matches in the text.

    The count is stored in a new column named:
        <report_col>_Pred_<PredictionType>

    A final binary prediction is typically derived afterward based on whether the count
    is at least 1 (see `create_final_prediction`).

    Args:
        df (pd.DataFrame): Input DataFrame containing text columns.
        columns_mapping (dict): Mapping of {report_column: gold_standard_column}.
        prediction_types_mapping (dict): Mapping of {report_column: [list_of_prediction_types]}.
        ibd_matcher (spacy.matcher.PhraseMatcher): PhraseMatcher for IBD keywords.
        drug_matcher (spacy.matcher.PhraseMatcher): PhraseMatcher for drug keywords.
        nlp (spacy.Language): SpaCy pipeline (with NegSpacy integrated).

    Returns:
        pd.DataFrame: The original DataFrame augmented with new prediction score columns.
    """
    for report_col, gold_col in columns_mapping.items():
        ptypes = prediction_types_mapping.get(report_col, [])
        for ptype in ptypes:
            if ptype in ["IBD", "Drug", "Combined"]:
                pred_col = f"{report_col}_Pred_{ptype}"
                logger.info(
                    f"Applying SpaCy prediction for '{ptype}' on column '{report_col}'."
                )

                text_list = df[report_col].fillna("").tolist()
                results = []

                # Process documents in batches for efficiency.
                for doc in tqdm(
                    nlp.pipe(text_list, batch_size=2000),
                    desc=f"Scoring {ptype} for {report_col}",
                    total=len(text_list),
                ):
                    if ptype == "IBD":
                        matches = detect_phrases_with_negation(doc, ibd_matcher)
                        results.append(1 if matches else 0)
                    elif ptype == "Drug":
                        matches = detect_phrases_with_negation(doc, drug_matcher)
                        results.append(1 if matches else 0)
                    elif ptype == "Combined":
                        ibd_spans = detect_phrases_with_negation(doc, ibd_matcher)
                        drug_spans = detect_phrases_with_negation(doc, drug_matcher)
                        results.append(1 if (ibd_spans or drug_spans) else 0)

                df[pred_col] = results
                logger.info(
                    f"Created prediction column '{pred_col}' with total sum = {np.sum(results)}."
                )
    return df

def create_final_prediction(df):
    """
    Create a 'Final_Prediction' column in the DataFrame based on the combined prediction scores.
    The final prediction is binary: if any of the *_Pred_Combined columns have a count >= 1,
    the final prediction is set to 1; otherwise, it is 0.

    Args:
        df (pd.DataFrame): DataFrame containing one or more columns named with the suffix '_Pred_Combined'.

    Returns:
        pd.DataFrame: The DataFrame with an added 'Final_Prediction' column (0 or 1).
    """
    combined_cols = [col for col in df.columns if col.endswith("_Pred_Combined")]
    if not combined_cols:
        df["Final_Prediction"] = np.nan
        logger.warning(
            "No '_Pred_Combined' columns found; 'Final_Prediction' set to NaN."
        )
        return df

    df["Final_Prediction"] = (df[combined_cols].max(axis=1) >= 1).astype(int)
    logger.info(
        f"Created 'Final_Prediction' column using {len(combined_cols)} columns."
    )
    return df