"""
Regex Model Functions with Mandatory UMLS standardisation and setup for the rest of the scripts

This module provides functions to:
  • Load and apply UMLS-based text standardisation to all input text.
  • Compile regex patterns for IBD and Drug detection.
  • Generate predictions based on regex pattern matching.
  • Combine predictions at both the report and patient levels.

All text processed by this module is standardised by mapping phrases to their
corresponding Concept Unique Identifiers (CUIs) from the UMLS MRCONSO.RRF file.
UMLS integration is mandatory; please ensure that a valid UMLS directory is provided.

Please note: This file is legacy still and uses some old handlers with lots of redundant code.
Because UMLS no longer became particularly relevant to the experiment it was not updated but would need to be adjusted if UMLS was going to become important again.

Author: Matt Stammers / UHSFT
"""

import csv
import logging
import os
import re
import time
import unicodedata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.caching import cached

# --------------------------------------------------------------------------- #
# UMLS Standardisation (used originally in legacy script to create UMLS text  #
# --------------------------------------------------------------------------- #
def normalise_text(text: str) -> str:
    """
    normalise Unicode (NFKC) and turn curly apostrophes into plain ones,
    so our regex and lookups always line up.
    """
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u2019", "'")

@cached
def load_mrconso_mapping(umls_dir):
    """
    Build two maps from MRCONSO_filtered.RRF:
      • cui_to_pref:  CUI → its preferred string
      • term_to_pref: any synonym (lowercase) → that preferred string
    """
    import csv
    import os
    import pickle

    mrconso_path = os.path.join(umls_dir, "MRCONSO_filtered.RRF")
    pickle_cache = os.path.join(umls_dir, "mrconso_mapping.pkl")

    # If old cache exists but holds the wrong format, delete it:
    if os.path.exists(pickle_cache):
        with open(pickle_cache, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            os.remove(pickle_cache)
        else:
            return data 

    cui_to_pref = {} 
    term_to_pref_cui = {} 

    # Read the filtered RRF
    with open(mrconso_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="|")
        for row in reader:
            if len(row) < 15:
                continue

            cui, ts, ispref, tty, original = row[0], row[2], row[6], row[12], row[14]
            lower = original.lower()

            # 1) Register the canonical label for each CUI
            if (ts == "P" or ispref == "Y") and tty in ("PT", "PN", "BN", "IN"):
                cui_to_pref.setdefault(cui, original)

            # 2) Remember every alias’s CUI
            term_to_pref_cui[lower] = cui

    # Build alias → canonical term map
    term_to_pref = {}
    for alias, cui in term_to_pref_cui.items():
        pref = cui_to_pref.get(cui)
        term_to_pref[alias] = pref if pref else alias

    # Cache for next time
    with open(pickle_cache, "wb") as f:
        pickle.dump((cui_to_pref, term_to_pref), f)

    return cui_to_pref, term_to_pref

def extract_cuis_from_umls(umls_dir, keywords):
    """
    Extract CUIs from the UMLS MRCONSO.RRF file for a provided list of keywords.

    Args:
        umls_dir (str): Directory containing the MRCONSO.RRF file.
        keywords (list of str): Keywords/phrases to search for.

    Returns:
        dict: Mapping {lowercased_keyword: [list_of_CUIs]}.
    """
    mrconso_path = os.path.join(umls_dir, "MRCONSO.RRF")
    term_to_cuis = {term.lower(): set() for term in keywords}

    if not os.path.exists(mrconso_path):
        logging.error(f"MRCONSO.RRF not found in directory '{umls_dir}'.")
        return {term: [] for term in term_to_cuis}

    logging.info(f"Loading MRCONSO.RRF from '{mrconso_path}'.")
    with open(mrconso_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="|")
        for row in tqdm(reader, desc="Processing MRCONSO.RRF"):
            if len(row) < 15:
                continue
            cui = row[0]
            term = row[14].lower()
            if term in term_to_cuis:
                term_to_cuis[term].add(cui)

    # Convert sets to lists.
    term_to_cuis = {term: list(cuis) for term, cuis in term_to_cuis.items()}
    logging.info("CUIs extracted successfully from MRCONSO.RRF.")
    return term_to_cuis

def define_keywords_and_extract_cuis(umls_dir):
    """
    Define IBD and Drug keywords and extract Drug CUIs from UMLS.
    This function is provided for demonstration and reuse in other modules.

    Args:
        umls_dir (str): Mandatory UMLS directory containing MRCONSO.RRF.

    Returns:
        tuple: (list_of_ibd_keywords, list_of_drug_keywords, dict_of_drug_cuis)
    """
    ibd_keywords = c.IBD_KEYWORDS
    drug_keywords = c.DRUG_KEYWORDS
    drug_cuis = extract_cuis_from_umls(umls_dir, drug_keywords)
    logging.info("IBD and Drug keywords defined; drug CUIs extracted from UMLS.")
    return ibd_keywords, drug_keywords, drug_cuis

class UMLSStandardiser:
    """
    Standardises text by replacing full phrases with their corresponding CUIs
    using the UMLS MRCONSO.RRF data.
    """

    def __init__(self, umls_dir):
        """
        Initialise with the UMLS directory (MRCONSO.RRF must be present).

        Args:
            umls_dir (str): Path to UMLS files.
        """
        self.umls_dir = umls_dir
        self.term_to_cui = None
        self.sorted_terms = [] 

    def fit(self):
        """
        Load term-to-CUI mapping and precompute sorted phrases and a single regex pattern.

        Returns:
            UMLSStandardiser: self, after loading mappings and compiling the pattern.
        """
        self.cui_to_pref, self.term_to_pref = load_mrconso_mapping(self.umls_dir)

        # Sort phrases so that longer (more specific) phrases are replaced first.
        self.sorted_terms = sorted(
            self.term_to_pref.keys(),
            key=lambda phrase: len(phrase.split()),
            reverse=True,
        )
        # Compile one regex pattern that matches any of the phrases.
        self.pattern = re.compile(
            r"\b(?:" + "|".join(map(re.escape, self.sorted_terms)) + r")\b",
            flags=re.IGNORECASE,
        )
        logging.info(
            "UMLSStandardiser fitted with {} phrases.".format(len(self.sorted_terms))
        )
        return self

    def standardise_text(self, text):
        """
        Replace recognised multi-word phrases with "OriginalPhrase (CUI)"
        but preserve the original case.

        Args:
            text (str or None): Original text.

        Returns:
            str: Standardised text with phrases replaced by CUIs.
        """
        if pd.isnull(text):
            return ""
        # first normalise for consistent matching
        text = normalise_text(text)

        def replacer(match):
            orig = match.group(0)
            key = orig.lower()
            pref = self.term_to_pref.get(key)
            return pref if pref else orig

        return self.pattern.sub(replacer, text).replace("?s", "'s")

    def transform(self, texts):
        """
        Standardise an iterable of text documents.

        Args:
            texts (iterable of str): Documents to standardise.

        Returns:
            list of str: Standardised texts.
        """
        return [self.standardise_text(t) for t in texts]


_umls_standardiser_instance = None

def get_umls_standardiser(umls_dir):
    """
    Return the singleton instance of UMLSStandardiser. If it has not been created yet,
    create, fit, and return it. This ensures that the MRCONSO mapping is loaded only once
    per run.

    Args:
        umls_dir (str): Directory containing UMLS files.

    Returns:
        UMLSStandardiser: Fitted standardiser instance.
    """
    global _umls_standardiser_instance
    if _umls_standardiser_instance is None:
        _umls_standardiser_instance = UMLSStandardiser(umls_dir).fit()
    return _umls_standardiser_instance

def apply_umls_standardisation(df, umls_dir, columns):
    """
    Apply mandatory UMLS text standardisation to specified DataFrame columns.
    For each column, the text is transformed using a shared UMLSStandardiser.

    Args:
        df (pd.DataFrame): Input DataFrame.
        umls_dir (str): Mandatory path to UMLS directory containing MRCONSO.RRF.
        columns (list of str): List of DataFrame columns to standardise.

    Returns:
        pd.DataFrame: DataFrame with standardised text in the specified columns.
    """
    logging.info("Starting UMLS standardisation on columns: " + ", ".join(columns))
    standardiser = get_umls_standardiser(umls_dir)
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = [
            standardiser.standardise_text(txt)
            for txt in tqdm(df[col], desc=f"UMLS standardising {col}", total=len(df))
        ]
    return df

def generate_umls_pattern(keyword):
    """
    Generate a regex pattern for a keyword that allows an optional appended UMLS CUI.

    Args:
        keyword (str): The free-text keyword.

    Returns:
        str: A regex pattern that matches the keyword and optionally a CUI in parentheses.
    """
    escaped_keyword = keyword
    pattern = rf"{escaped_keyword}(?:\s*\(C\d{{4,}}\))?"
    return pattern

# --------------------------------------------------------------------------- #
# Regex Prediction Functions                                                  #
# --------------------------------------------------------------------------- #

def train_regex_model(
    ibd_patterns,
    drug_patterns,
    cui_patterns=None,
    analysis_folder=None,
    note="Final",
    use_umls=False,
):
    """
    Compile regex patterns for IBD and Drug detection.

    If use_umls is True, then modify each keyword to allow an optional appended CUI.
    """
    import time

    start_time = time.time()
    compiled = {}
    n_ibd = n_drug = n_cui = 0

    if ibd_patterns:
        if use_umls:
            # Update each IBD keyword with an optional CUI
            updated_ibd = {
                name: generate_umls_pattern(p) for name, p in ibd_patterns.items()
            }
        else:
            updated_ibd = ibd_patterns
        compiled_ibd = {
            name: re.compile(p, flags=re.IGNORECASE) for name, p in updated_ibd.items()
        }
        compiled["ibd"] = compiled_ibd
        n_ibd = len(compiled_ibd)

    if drug_patterns:
        if use_umls:
            updated_drug = {
                name: generate_umls_pattern(p) for name, p in drug_patterns.items()
            }
        else:
            updated_drug = drug_patterns
        compiled_drug = {
            name: re.compile(p, flags=re.IGNORECASE) for name, p in updated_drug.items()
        }
        compiled["drug"] = compiled_drug
        n_drug = len(compiled_drug)

    # Compile CUI patterns as before.
    if cui_patterns:
        compiled_cui = {
            name: re.compile(p, flags=re.IGNORECASE) for name, p in cui_patterns.items()
        }
        compiled["cui"] = compiled_cui
        n_cui = len(compiled_cui)

    elapsed = time.time() - start_time
    logging.info(
        f"Compiled {n_ibd} IBD patterns, {n_drug} Drug patterns, {n_cui} CUI patterns "
        f"for note='{note}' in {elapsed:.2f} seconds."
    )
    return compiled, elapsed

def predict_regex(df, report_col, compiled_regex, category="IBD"):
    """
    Generate binary predictions for a report column using regex matching.
    Each text entry is scanned using the appropriate compiled patterns.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        report_col (str): Column name to predict on.
        compiled_regex (dict): Compiled regex patterns with keys 'ibd', 'drug', and 'cui'.
        category (str, optional): "IBD", "Drug", or "CUI" (default "IBD").

    Returns:
        numpy.ndarray: Array of binary predictions (1 if any match; else 0).
    """
    if report_col not in df.columns:
        logging.warning(f"Column '{report_col}' not found. Returning zeros.")
        return np.zeros(len(df), dtype=int)

    cat = category.lower()
    if cat == "ibd":
        patterns = compiled_regex.get("ibd", {})
    elif cat == "drug":
        patterns = compiled_regex.get("drug", {})
    elif cat == "cui":
        patterns = compiled_regex.get("cui", {})
    else:
        logging.warning(f"Unknown category '{category}'. Returning zeros.")
        return np.zeros(len(df), dtype=int)

    texts = df[report_col].fillna("").astype(str).values
    preds = []
    for text in texts:
        label = 0
        for pattern in patterns.values():
            if pattern.search(text):
                label = 1
                break
        preds.append(label)
    return np.array(preds, dtype=int)

def apply_regex_predictions(
    df, columns_mapping, compiled_regex, category="IBD", show_progress=True
):
    """
    Apply regex-based predictions across specified report columns.
    Generates new columns (e.g. <report_col>_Pred_IBD) containing binary results.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        columns_mapping (dict): Mapping of report column names.
        compiled_regex (dict): Compiled regex patterns.
        category (str, optional): "IBD" or "Drug" (default "IBD").
        show_progress (bool, optional): If True, display a progress bar.

    Returns:
        pandas.DataFrame: DataFrame with added prediction columns.
    """
    predict_desc = f"Predicting {category}"
    for report_col, _ in columns_mapping.items():
        prediction_col = f"{report_col}_Pred_{category}"
        if show_progress:
            tqdm.pandas(desc=f"{predict_desc} for {report_col}")
            df[prediction_col] = df[report_col].progress_apply(
                lambda x: _predict_text(x, compiled_regex, category)
            )
        else:
            df[prediction_col] = df[report_col].apply(
                lambda x: _predict_text(x, compiled_regex, category)
            )
    return df

def _predict_text(text, compiled_regex, category):
    """
    Helper to generate a binary prediction from a single text entry.

    Args:
        text (str): Text to evaluate.
        compiled_regex (dict): Compiled regex patterns.
        category (str): "IBD" or "Drug".

    Returns:
        int: 1 if any pattern matches; otherwise 0.
    """
    if pd.isnull(text):
        return 0
    cat = category.lower()
    if cat == "ibd":
        patterns = compiled_regex["ibd"]
    elif cat == "drug":
        patterns = compiled_regex["drug"]
    elif cat == "cui":
        patterns = compiled_regex["cui"]
    else:
        return 0

    return 1 if any(p.search(str(text)) for p in patterns.values()) else 0

def create_combined_predictions(df, ibd_cols_map, drug_cols_map):
    """
    Combine separate IBD and Drug predictions into a single prediction column.
    The new column <report_col>_Pred_Combined is set to 1 if either prediction is 1.

    Args:
        df (pandas.DataFrame): DataFrame with IBD and Drug prediction columns.
        ibd_cols_map (dict): Mapping of report column names for IBD predictions.
        drug_cols_map (dict): Mapping of report column names for Drug predictions.

    Returns:
        pandas.DataFrame: DataFrame with added combined prediction columns.
    """
    all_cols = set(ibd_cols_map.keys()).union(drug_cols_map.keys())
    for report_col in all_cols:
        ibd_col = f"{report_col}_Pred_IBD"
        drug_col = f"{report_col}_Pred_Drug"
        comb_col = f"{report_col}_Pred_Combined"
        if ibd_col in df.columns and drug_col in df.columns:
            df[comb_col] = ((df[ibd_col] == 1) | (df[drug_col] == 1)).astype(int)
        elif ibd_col in df.columns:
            df[comb_col] = df[ibd_col]
        elif drug_col in df.columns:
            df[comb_col] = df[drug_col]
        else:
            df[comb_col] = np.nan
    return df

def create_per_report_patient_level_predictions(df, ibd_cols_map):
    """
    Generate patient-level predictions from report-level combined predictions.
    For each report column, a new column 'Patient_<report_col>_Pred_IBD' is created
    by grouping by 'study_id' and taking the maximum (logical OR) across the report predictions.

    Args:
        df (pandas.DataFrame): DataFrame with report-level predictions.
        ibd_cols_map (dict): Mapping of report column names for IBD.

    Returns:
        pandas.DataFrame: DataFrame with additional patient-level prediction columns.
    """
    for report_col in ibd_cols_map.keys():
        combined_col = f"{report_col}_Pred_Combined"
        patient_level_col = f"Patient_{report_col}_Pred_IBD"
        if combined_col not in df.columns:
            logging.warning(
                f"Cannot create '{patient_level_col}'. Missing '{combined_col}'; setting as NaN."
            )
            df[patient_level_col] = np.nan
            continue
        patient_predictions = (
            df.groupby("study_id")[combined_col]
            .max()
            .reset_index()
            .rename(columns={combined_col: patient_level_col})
        )
        df = df.merge(patient_predictions, on="study_id", how="left")
    return df

def create_final_prediction(df):
    """
    Aggregate report-level combined predictions into a single patient-level prediction.
    The 'Final_Prediction' column is set to 1 if any of the *_Pred_Combined values is 1.

    Args:
        df (pandas.DataFrame): DataFrame with *_Pred_Combined columns.

    Returns:
        pandas.DataFrame: DataFrame with an added 'Final_Prediction' column.
    """
    combined_cols = [col for col in df.columns if col.endswith("_Pred_Combined")]
    if not combined_cols:
        logging.warning(
            "No *_Pred_Combined columns found. Setting Final_Prediction to NaN."
        )
        df["Final_Prediction"] = np.nan
        return df
    final_pred_series = (
        df.groupby("study_id")[combined_cols].any().any(axis=1).astype(int)
    )
    df = df.merge(
        final_pred_series.rename("Final_Prediction"), on="study_id", how="left"
    )
    return df