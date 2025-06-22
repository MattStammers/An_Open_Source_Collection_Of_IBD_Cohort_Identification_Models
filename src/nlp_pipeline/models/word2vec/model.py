"""
word2vec model.py

This module provides utilities to train and use a Logistic Regression
model on spaCy Word2Vec embeddings for text classification. It supports:
  - Word2VecTransformer
  - train_word2vec_model
  - predict_word2vec
  - run_word2vec_shap_analysis
  - explain_word2vec_model

Author: Matt Stammers / UHSFT
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from nlp_pipeline.common.feature_importance import (
    extract_feature_importance,
    plot_feature_importance,
    save_feature_importance,
)

# Module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------- #
# Internal Helpers                                                             #
# ---------------------------------------------------------------------------- #


def _ensure_dir(path: Path) -> None:
    """Create *path* if missing (including parents)."""
    path.mkdir(parents=True, exist_ok=True)


def _plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, dest: Path) -> None:
    """Plot and save a calibration curve."""
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], ls="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(dest, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Transformer                                                                 #
# --------------------------------------------------------------------------- #


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """
    Convert raw text into spaCy Word2Vec embeddings.
    Implements scikit-learn Transformer interface.
    """

    def __init__(self, spacy_model: str = "en_core_web_md") -> None:
        self.spacy_model = spacy_model
        self.nlp: Optional[spacy.Language] = None
        self.embedding_dim: Optional[int] = None

    def fit(
        self, X: List[str], y: Optional[np.ndarray] = None
    ) -> "Word2VecTransformer":
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.spacy_model)
                self.embedding_dim = self.nlp.meta["vectors"]["width"]
                logger.info(
                    "Loaded spaCy model '%s' with dim=%d",
                    self.spacy_model,
                    self.embedding_dim,
                )
            except Exception as exc:
                logger.error(
                    "Failed loading spaCy model '%s': %s",
                    self.spacy_model,
                    exc,
                )
                raise
        return self

    def transform(
        self, X: List[Union[str, bytes]], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self.nlp is None or self.embedding_dim is None:
            raise ValueError("Transformer not fitted. Call 'fit' first.")
        embeddings: List[np.ndarray] = []
        for doc in X:
            proc = self.nlp(str(doc))
            if proc.vector_norm:
                embeddings.append(proc.vector)
            else:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
        return np.vstack(embeddings)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        if self.embedding_dim is None:
            raise ValueError("Call 'fit' before 'get_feature_names_out'.")
        return [f"dim_{i}" for i in range(self.embedding_dim)]


# ---------------------------------------------------------------------------- #
# SHAP Analysis & Explainability                                               #
# ---------------------------------------------------------------------------- #


def run_word2vec_shap_analysis(
    pipe: Pipeline,
    X_text: List[str],
    report_col: str,
    analysis_folder: str,
) -> None:
    """
    Generate and save SHAP summary and bar plots for the pipeline.
    """
    shap_dir = Path(analysis_folder) / "shap"
    _ensure_dir(shap_dir)
    embedder = pipe.named_steps["vect"]
    classifier = pipe.named_steps["clf"]
    X_emb = embedder.transform(X_text)
    masker = shap.maskers.Independent(X_emb)
    explainer = shap.LinearExplainer(classifier, masker)
    shap_vals = explainer(X_emb)
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        features=X_emb,
        feature_names=embedder.get_feature_names_out(),
        show=False,
    )
    plt.title(f"SHAP Summary for {report_col} (Word2Vec)")
    plt.savefig(shap_dir / f"{report_col}_summary.png", bbox_inches="tight")
    plt.close()
    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        features=X_emb,
        feature_names=embedder.get_feature_names_out(),
        plot_type="bar",
        show=False,
    )
    plt.title(f"SHAP Feature Importance for {report_col} (Word2Vec)")
    plt.savefig(
        shap_dir / f"{report_col}_feature_importance.png",
        bbox_inches="tight",
    )
    plt.close()
    logger.info(f"Saved SHAP plots for {report_col} in {shap_dir}")


def explain_word2vec_model(
    pipe: Pipeline,
    analysis_folder: str,
    top_n_features: int = 20,
    top_n_words: int = 5,
) -> None:
    """
    Generate a text report mapping embedding dimensions to top tokens.
    """
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]
    top_idx = np.argsort(np.abs(coefs))[::-1][:top_n_features]
    nlp_model = pipe.named_steps["vect"].nlp
    lines: List[str] = [
        "Explainability Report for Word2Vec Model:",
        "-----------------------------------------",
    ]
    for idx in top_idx:
        coef_val = coefs[idx]
        pos, neg = [], []
        data = nlp_model.vocab.vectors.data
        keys = list(nlp_model.vocab.vectors.keys())
        col = data[:, idx]
        for i in np.argsort(col)[-top_n_words:][::-1]:
            token = nlp_model.vocab.strings[keys[i]]
            pos.append(token)
        for i in np.argsort(col)[:top_n_words]:
            token = nlp_model.vocab.strings[keys[i]]
            neg.append(token)
        lines.append(f"Dimension {idx}: Coefficient={coef_val:.4f}")
        lines.append(f"  Top positive: {', '.join(pos)}")
        lines.append(f"  Top negative: {', '.join(neg)}")
        lines.append("")
    report_path = Path(analysis_folder) / "word2vec_explainability_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved explainability report to {report_path}")


# ---------------------------------------------------------------------------- #
# Training & Prediction                                                        #
# ---------------------------------------------------------------------------- #


def train_word2vec_model(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    analysis_root: Union[str, Path],
    *,
    top_n: int = 20,
    shap_explain: bool = True,
    do_feature_importance: bool = True,
    do_calibration: bool = True,
    cv: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[Optional[Pipeline], float]:
    """
    Fit a spaCy Word2Vec + LogisticRegression pipeline and generate analyses.

    Returns:
        Tuple of (trained pipeline or None, training time in seconds)
    """
    start = time.time()
    if text_col not in df.columns or label_col not in df.columns:
        logger.warning("Columns '%s' or '%s' missing - skipping", text_col, label_col)
        return None, 0.0
    data = df.dropna(subset=[text_col, label_col])
    if data.empty:
        logger.warning("No valid data for '%s'", text_col)
        return None, 0.0
    X = data[text_col].astype(str).tolist()
    y = data[label_col].astype(int).values

    pipeline = Pipeline(
        [
            ("vect", Word2VecTransformer()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    if cv and cv > 1:
        grid = GridSearchCV(
            pipeline,
            param_grid={"clf__penalty": ["l1", "l2"], "clf__C": [0.01, 0.1, 1, 10]},
            cv=cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            verbose=1,
        )
        grid.fit(X, y)
        model = grid.best_estimator_
        logger.info("[%s] best params - %s", text_col, grid.best_params_)
    else:
        pipeline.fit(X, y)
        model = pipeline

    elapsed = time.time() - start
    logger.info("Trained Word2Vec model for '%s' in %.2fs", text_col, elapsed)

    root = Path(analysis_root)
    _ensure_dir(root)

    model_file = root / f"{text_col}_word2vec_model.joblib"
    try:
        joblib.dump(model, model_file)
        logger.info("Saved Word2Vec model to %s", model_file)
    except Exception as exc:
        logger.error("Failed to save Word2Vec model for '%s': %s", text_col, exc)

    if do_calibration:
        try:
            probas = model.predict_proba(X)[:, 1]
            calib_dir = root / "calibration"
            _ensure_dir(calib_dir)
            _plot_calibration(y, probas, calib_dir / f"{text_col}_reliability.png")
        except Exception as exc:
            logger.error("Calibration failed for '%s': %s", text_col, exc)

    if do_feature_importance:
        try:
            pos_imp, neg_imp = extract_feature_importance(model, top_n)
            plot_feature_importance(pos_imp, neg_imp, text_col, "Word2Vec", str(root))
            save_feature_importance(pos_imp, neg_imp, text_col, "Word2Vec", str(root))
        except Exception as exc:
            logger

    return model, elapsed


# ---------------------------------------------------------------------------- #
# Prediction                                                                   #
# ---------------------------------------------------------------------------- #


def predict_word2vec(
    model: Optional[Pipeline], df: pd.DataFrame, text_col: str, *, proba: bool = False
) -> np.ndarray:
    """
    Predict labels or probabilities using the trained Word2Vec pipeline.
    """
    if model is None or text_col not in df.columns:
        size = len(df)
        dtype = float if proba else int
        return np.zeros(size, dtype=dtype)

    texts = df[text_col].fillna("").astype(str).tolist()
    return model.predict_proba(texts)[:, 1] if proba else model.predict(texts)
