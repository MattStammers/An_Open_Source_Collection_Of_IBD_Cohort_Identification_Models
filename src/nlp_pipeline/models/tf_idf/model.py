"""
tf-idf model.py

This module provides utilities to train and use a Logistic Regression
model with TF-IDF features for text classification. It now uses GridSearchCV
for better results.

The module logs its progress and handles missing data or errors gracefully,
returning default values when necessary. It also has a series of flags to turn on
or off functions as needed.

Author: Matt Stammers / UHSFT
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Guarded Import
try:
    import shap
except ModuleNotFoundError:
    shap = None

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------- #
# Internal Helpers                                                             #
# ---------------------------------------------------------------------------- #

def _ensure_dir(path: Path) -> None:
    """Create *path* if missing (parents too)."""
    path.mkdir(parents=True, exist_ok=True)


def _plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, dest: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], ls="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability curve")
    plt.tight_layout()
    plt.savefig(dest, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------- #
# Train TF IDF Model                                                           #
# ---------------------------------------------------------------------------- #

def train_tfidf_model(
    df,
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
) -> Tuple[Optional[Pipeline], float]:
    """Fit a TfidfVectorizer + LogReg model.

    Returns (fitted_pipeline | None, training_time_seconds)
    """
    t0 = time.time()

    if text_col not in df.columns:
        logging.warning("Column '%s' missing - skipping model", text_col)
        return None, 0.0

    data = df.dropna(subset=[text_col, label_col])
    if data.empty:
        logging.warning("No rows with both text and label for '%s'", text_col)
        return None, 0.0

    X = data[text_col].astype(str).values
    y = data[label_col].astype(int).values

    base = Pipeline(
        [
            ("vect", TfidfVectorizer()),
            (
                "clf",
                LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
            ),
        ]
    )

    if cv and cv > 1:
        grid = GridSearchCV(
            base,
            param_grid={
                "vect__max_df": [0.75, 1.0],
                "vect__ngram_range": [(1, 1), (1, 2)],
                "clf__penalty": ["l1", "l2"],
                "clf__C": [0.01, 0.1, 1, 10],
            },
            cv=cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            verbose=1,
        )
        grid.fit(X, y)
        model: Pipeline = grid.best_estimator_
        logging.info("[%s] best params - %s", text_col, grid.best_params_)
    else:
        base.fit(X, y)
        model = base

    elapsed = time.time() - t0
    logging.info("Trained TF-IDF model for '%s' in %.2fs", text_col, elapsed)

    # ------------------- Optional analysis outputs --------------------
    root = Path(analysis_root)
    _ensure_dir(root)

    model_file = root / f"{text_col}_model.joblib"
    try:
        joblib.dump(model, model_file)
        logging.info("Saved model for '%s' at '%s'", text_col, model_file)
    except Exception as e:
        logging.error("Could not save model for '%s': %s", text_col, e)

    # 1) Calibration ---------------------------------------------------
    if do_calibration:
        try:
            probs = model.predict_proba(X)[:, 1]
            _ensure_dir(root / "calibration")
            _plot_calibration(
                y, probs, root / "calibration" / f"{text_col}_reliability.png"
            )
        except Exception as exc:
            logging.error("Calibration failed for '%s': %s", text_col, exc)

    # 2) Feature importance  -------------------------------------------
    if do_feature_importance:
        try:
            vect: TfidfVectorizer = model.named_steps["vect"]
            clf: LogisticRegression = model.named_steps["clf"]
            odds = clf.coef_.ravel()
            feats = np.array(vect.get_feature_names_out())
            order = np.argsort(odds)
            top_pos = feats[order][-top_n:][::-1]
            top_neg = feats[order][:top_n]
            fi_path = root / "feature_importance"
            _ensure_dir(fi_path)
            with (fi_path / f"{text_col}_top_features.txt").open(
                "w", encoding="utf-8"
            ) as fh:
                fh.write("Positive\n========\n")
                fh.write("\n".join(top_pos))
                fh.write("\n\nNegative\n========\n")
                fh.write("\n".join(top_neg))
        except Exception as exc:
            logging.error("Feature-importance failed for '%s': %s", text_col, exc)

    # 3) SHAP (heavy) ---------------------------------------------------
    if shap_explain and shap is not None:
        try:
            X_trans = model.named_steps["vect"].transform(X)
            masker = shap.maskers.Independent(X_trans, max_samples=200)
            explainer = shap.LinearExplainer(model.named_steps["clf"], masker)
            shap_vals = explainer(X_trans)
            shap.summary_plot(
                shap_vals,
                features=X_trans,
                feature_names=model.named_steps["vect"].get_feature_names_out(),
                show=False,
            )
            _ensure_dir(root / "shap")
            plt.gcf().savefig(
                root / "shap" / f"{text_col}_shap_summary.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:
            logging.error("SHAP failed for '%s': %s", text_col, exc)

    return model, elapsed

def predict_tfidf(model: Optional[Pipeline], df, text_col: str, *, proba: bool = False):
    """Vectorise *text_col* with *model* and return label / proba."""
    if model is None or text_col not in df.columns:
        # Keep dtype explicit so downstream code behaves predictably ¬
        return np.zeros(len(df), dtype=float if proba else int)

    X = df[text_col].fillna("").astype(str).values
    if proba:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)