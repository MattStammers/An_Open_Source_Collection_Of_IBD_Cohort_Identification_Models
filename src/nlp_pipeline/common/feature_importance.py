"""
Feature Importance Module

This module provides functionality to extract, visualise, and save feature importance
information for machine learning models. It is designed to work with pipeline models that
contain a vectoriser (under the key 'vect') and a classifier (under the key 'clf') with
learned coefficients.

Functions:
    - extract_feature_importance: Extracts the top positive and negative features based on model coefficients.
    - plot_feature_importance: Creates bar plots for the top positive and negative features and saves the plots.
    - save_feature_importance: Saves the extracted feature importance data to CSV files.

Dependencies:
    - os, logging: For file path management and logging.
    - matplotlib.pyplot: For plotting visualisations.
    - seaborn: To style the plots.
    - shap: For potential use in further feature importance analysis (currently imported for extensibility).
    - pandas: For DataFrame creation and CSV operations.
"""

import logging
import os
from typing import Any, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Feature Importance Extractor                                                #
# --------------------------------------------------------------------------- #

def extract_feature_importance(
    model: Any, top_n: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts and returns the top positive and negative features based on model coefficients.

    This function assumes that the provided model is a pipeline containing a vectoriser
    under the key 'vect' and a classifier under the key 'clf'. The classifier is assumed to
    have a 'coef_' attribute.

    Args:
        model (Any): A machine learning pipeline with a vectoriser ('vect') and a classifier ('clf').
        top_n (int, optional): Number of top features to extract for both positive and negative sides. Defaults to 20.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - top_positive: DataFrame containing the top_n features with the highest positive coefficients.
            - top_negative: DataFrame containing the top_n features with the most negative coefficients.
    """
    # Retrieve vectoriser and classifier from the pipeline
    vectoriser = model.named_steps["vect"]
    classifier = model.named_steps["clf"]

    # Get feature names from the vectoriser and coefficients from the classifier
    feature_names = vectoriser.get_feature_names_out()
    coefficients = classifier.coef_[0]

    # Create a DataFrame to hold feature names and their corresponding coefficients
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    )

    # Sort the features based on their coefficients in descending order
    feature_importance = feature_importance.sort_values(
        by="coefficient", ascending=False
    )

    # Select the top positive and negative features
    top_positive = feature_importance.head(top_n)
    top_negative = feature_importance.tail(top_n)

    return top_positive, top_negative

# --------------------------------------------------------------------------- #
# Feature Importance Plotting Func                                            #
# --------------------------------------------------------------------------- #

def plot_feature_importance(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    report_col: str,
    pred_type: str,
    analysis_folder: str,
) -> None:
    """
    Plots and saves bar plots for the top positive and negative features.

    The function creates separate plots for positive and negative features,
    saves them to a dedicated subfolder named 'feature_importance' within the provided
    analysis folder, and logs the output location.

    Args:
        top_positive (pd.DataFrame): DataFrame of top positive features.
        top_negative (pd.DataFrame): DataFrame of top negative features.
        report_col (str): Identifier for the report column (used in the plot title and filename).
        pred_type (str): The prediction type (e.g., 'Final'), used in the plot title and filename.
        analysis_folder (str): The directory where the plots should be saved.
    """
    # Create (or ensure) the feature importance output directory exists
    fi_folder = os.path.join(analysis_folder, "feature_importance")
    os.makedirs(fi_folder, exist_ok=True)

    # Plotting the top positive features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="coefficient", y="feature", data=top_positive)
    plt.title(f"Top Positive Features for {report_col} ({pred_type})")
    plt.tight_layout()
    pos_plot_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_positive_features.png"
    )
    plt.savefig(pos_plot_path)
    plt.close()

    # Plotting the top negative features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="coefficient", y="feature", data=top_negative)
    plt.title(f"Top Negative Features for {report_col} ({pred_type})")
    plt.tight_layout()
    neg_plot_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_negative_features.png"
    )
    plt.savefig(neg_plot_path)
    plt.close()

    logging.info(f"Saved feature importance plots for {report_col} ({pred_type}).")

# --------------------------------------------------------------------------- #
# Feature Importance Store Helper                                             #
# --------------------------------------------------------------------------- #

def save_feature_importance(
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    report_col: str,
    pred_type: str,
    analysis_folder: str,
) -> None:
    """
    Saves the top positive and negative features to CSV files.

    The function writes out two CSV files containing the feature importance data
    (for top positive and negative features respectively) to a 'feature_importance'
    folder under the specified analysis folder.

    Args:
        top_positive (pd.DataFrame): DataFrame containing top positive features.
        top_negative (pd.DataFrame): DataFrame containing top negative features.
        report_col (str): Identifier for the report column (used in the filename).
        pred_type (str): The prediction type (e.g., 'Final'), used in the filename.
        analysis_folder (str): The directory where the CSV files should be saved.
    """
    # Create (or ensure) the feature importance output directory exists
    fi_folder = os.path.join(analysis_folder, "feature_importance")
    os.makedirs(fi_folder, exist_ok=True)

    # Save the dataframes as CSV files
    positive_csv_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_positive_features.csv"
    )
    negative_csv_path = os.path.join(
        fi_folder, f"{report_col}_{pred_type}_top_negative_features.csv"
    )

    top_positive.to_csv(positive_csv_path, index=False)
    top_negative.to_csv(negative_csv_path, index=False)

    logging.info(f"Saved feature importance data for {report_col} ({pred_type}).")
