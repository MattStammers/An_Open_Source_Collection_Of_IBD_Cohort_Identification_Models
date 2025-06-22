"""
Data Utilities Module

This module provides functions for common data preprocessing tasks including:
    - Formatting patient numbers.
    - Mapping raw ethnicity values to standardised categories.
    - Loading and preprocessing Excel data (with caching for efficiency).
    - Extracting Concept Unique Identifiers (CUIs) from the UMLS MRCONSO.RRF file.

Caching is applied to computationally expensive functions using a caching decorator.
"""

import csv
import logging
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from nlp_pipeline.common.caching import cached
from nlp_pipeline.config.constants import (
    COLUMNS_TO_DROP,
    COLWISE_COLUMNS,
    PATIENT_COLUMNS,
)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------#
# Column De-Duplicator                                                        #
# ----------------------------------------------------------------------------#

def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

# ----------------------------------------------------------------------------#
# Patient Number Reformatter                                                  #
# ----------------------------------------------------------------------------#

def format_patient_number(x: Any) -> str:
    """
    Format a patient number into a 7-digit string.

    The function converts the input to a float, then to an integer,
    and pads the result with zeros (if needed) to ensure it has a length of 7.
    If conversion fails, "NaN" is returned.

    Args:
        x (Any): The input value representing a patient number.

    Returns:
        str: The formatted patient number, or "NaN" on failure.
    """
    try:
        return str(int(float(x))).zfill(7)
    except (ValueError, TypeError):
        return "NaN"

# ----------------------------------------------------------------------------#
# Ethnicity Mapper                                                            #
# ----------------------------------------------------------------------------#

def _map_ethnicity(x: Any) -> str:
    """
    Map a raw ethnicity value to a standardised category.

    If the input is null, the result is 'Not Asked'. The function translates various
    representations (e.g., "white british", "indian") to broader categories (e.g., "White", "Asian").
    Unrecognised values default to 'Not Asked'.

    Args:
        x (Any): The raw ethnicity string.

    Returns:
        str: The mapped ethnicity category.
    """
    if pd.isnull(x):
        return "Not Asked"
    mapping = {
        "white british": "White",
        "white other": "White",
        "white irish": "White",
        "indian": "Asian",
        "pakistani": "Asian",
        "bangladeshi": "Asian",
        "chinese": "Asian",
        "asian other": "Asian",
        "african": "African",
        "black other": "African",
        "caribbean": "African",
        "mixed white and asian": "Asian",
        "mixed white and black african": "African",
        "mixed white and black caribbean": "African",
        "ethnic other": "Not Asked",
        "mixed other": "Not Asked",
        "not asked": "Not Asked",
        "not stated": "Not Asked",
    }
    return mapping.get(x.lower(), "Not Asked")

# ----------------------------------------------------------------------------#
# Data loader                                                                 #
# ----------------------------------------------------------------------------#

@cached
def load_and_preprocess(
    file_path: str, exclude_combined_clinic: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess an Excel file containing patient and clinical data.

    The function performs several steps:
      - Loads the data from an Excel file.
      - Drops unnecessary columns based on configuration.
      - Ensures the presence of the 'study_id' column.
      - Converts string 'NaN' values in label columns to real NaN and ensures numeric conversion.
      - Combines text from multiple columns into a single 'combined_text' column.
      - Processes demographic fields, creating derived columns like 'age_group' and 'imd_group'.
      - Unifies label columns to create 'Patient_Has_IBD' if not already present.

    Args:
        file_path (str): The path to the Excel file.
        exclude_combined_clinic (bool, optional): If True, the 'combined_clinic_letters' column
            is excluded from the combined text. Defaults to False.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    Raises:
        Exception: If there is an error loading the file.
        KeyError: If the required 'study_id' column is missing.
    """

    logger.info(f"Loading data from '{file_path}'.")
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as e:
        logger.error(f"Error loading '{file_path}': {e}")
        raise
    print(f"[DEBUG] Columns in raw Excel file: {df.columns.tolist()}")
    for col in df.columns:
        if "histopathology" in col.lower():
            print(f"[DEBUG] Found histopathology-related column: {col}")

    # Ensure all COLWISE_COLUMNS keys are preserved as-is (canonical), even if aliased/duplicated
    for text_col in COLWISE_COLUMNS.keys():
        candidates = [col for col in df.columns if col.startswith(text_col)]
        if candidates and text_col not in df.columns:
            df[text_col] = df[candidates[0]]
            logger.warning(f"Aliased {candidates[0]} to expected column '{text_col}'")

    missing = [col for col in COLWISE_COLUMNS.keys() if col not in df.columns]
    if missing:
        logger.warning(f"Text columns missing after preprocessing: {missing}")

    # Ensure all columns are uniquely named to avoid ambiguity
    df.columns = deduplicate_columns(df.columns)

    logger.info(f"Original data shape (after rename): {df.shape}")

    # Drop unnecessary columns.
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    logger.info(f"Dropped columns: {drop_cols}")

    # Ensure 'study_id' exists.
    if "study_id" not in df.columns:
        raise KeyError("'study_id' is required but not in the dataframe.")

    # Only PATIENT_COLUMNS represent numeric labels; leave COLWISE_COLUMNS for text
    possible_label_cols = list(PATIENT_COLUMNS.values())
    # Filter out any empty or None entries
    possible_label_cols = [col for col in possible_label_cols if col]
    logger.info(f"Available columns in DataFrame: {list(df.columns)}")
    logger.info(f"Expected label columns: {possible_label_cols}")
    for col in possible_label_cols:
        if col in df.columns:
            df[col] = df[col].replace("NaN", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            logger.warning(f"Column '{col}' not found in DataFrame; skipping.")

    # Combine text fields from configured columns (use the renamed label names).
    text_labels = list(COLWISE_COLUMNS.values())
    if exclude_combined_clinic:
        # drop the label corresponding to the raw 'combined_clinic_letters' field
        label_to_drop = COLWISE_COLUMNS.get("combined_clinic_letters")
        if label_to_drop in text_labels:
            text_labels.remove(label_to_drop)

    existing_text_labels = [lbl for lbl in text_labels if lbl in df.columns]
    if existing_text_labels:
        df["combined_text"] = (
            df[existing_text_labels].fillna("").astype(str).agg(" ".join, axis=1)
        )
    else:
        df["combined_text"] = ""

    # Ensure demographic fields exist; if missing, fill with NaN.
    for col in ["cohort", "age", "gender", "ethnicity", "imd"]:
        if col not in df.columns:
            df[col] = np.nan

    # Standardise ethnicity values.
    df["ethnicity"] = df["ethnicity"].apply(_map_ethnicity)

    # Create an age group column if 'age' exists.
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=range(0, 101, 10),
            labels=[f"{i}-{i+10}" for i in range(0, 100, 10)],
            include_lowest=True,
            right=False,
        )
    # Create an IMD group column if 'imd' exists.
    if "imd" in df.columns:
        bins = [0, 2, 4, 6, 8, 10]
        labels = ["1-2", "3-4", "5-6", "7-8", "9-10"]
        df["imd_group"] = pd.cut(
            df["imd"], bins=bins, labels=labels, include_lowest=True
        )

    # Create a unified 'Patient_Has_IBD' column if missing, as the max across specified label columns.
    if "Patient_Has_IBD" not in df.columns:
        colwise_labels = list(COLWISE_COLUMNS.values())
        existing = [c for c in colwise_labels if c in df.columns]
        if existing:
            df["Patient_Has_IBD"] = df[existing].max(axis=1)
            logger.info("Created 'Patient_Has_IBD' as the max of colwise labels.")

    logger.info(f"Data shape after preprocessing: {df.shape}")
    return df

# ----------------------------------------------------------------------------#
# Cui Extractor                                                               #
# ----------------------------------------------------------------------------#

@cached
def extract_cuis_from_umls(umls_dir: str, terms: List[str]) -> Dict[str, List[str]]:
    """
    Extract unique CUIs (Concept Unique Identifiers) for the provided terms from the UMLS MRCONSO.RRF file.

    The function reads the MRCONSO.RRF file from the specified UMLS directory. It performs
    a case-insensitive search for the provided terms in the specified field (column 15) of the file.
    If the file is not found or if no matches exist, an empty dictionary (or dictionary with partial
    matches) is returned.

    Args:
        umls_dir (str): The directory containing the MRCONSO.RRF file.
        terms (List[str]): A list of terms to search for in the UMLS file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each term to a sorted list of matching CUIs.
    """
    mrconso_path = os.path.join(umls_dir, "MRCONSO.RRF")
    term_cuis: Dict[str, set] = {}

    if not os.path.exists(mrconso_path):
        logger.error(f"MRCONSO.RRF not found in directory '{umls_dir}'.")
        return {}

    logger.info(f"Extracting CUIs from '{mrconso_path}' for provided terms.")
    with open(mrconso_path, "r", encoding="utf-8") as mrconso_file:
        reader = csv.reader(mrconso_file, delimiter="|")
        terms_lower = {term.lower() for term in terms}
        for row in tqdm(reader, desc="Processing MRCONSO.RRF"):
            if len(row) < 15:
                continue
            cui = row[0]
            term_in_file = row[14].lower().strip()
            if term_in_file in terms_lower:
                original_term = next(
                    (t for t in terms if t.lower() == term_in_file), term_in_file
                )
                if original_term not in term_cuis:
                    term_cuis[original_term] = set()
                term_cuis[original_term].add(cui)

    # Convert sets to sorted lists for consistent output.
    term_cuis_final: Dict[str, List[str]] = {
        term: sorted(list(cuis)) for term, cuis in term_cuis.items()
    }
    logger.info("CUI extraction completed.")
    return term_cuis_final
