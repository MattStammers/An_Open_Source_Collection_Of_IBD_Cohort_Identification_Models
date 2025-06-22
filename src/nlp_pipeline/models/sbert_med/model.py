"""
sbert_med model.py

Trains and uses a Logistic-Regression classifier whose inputs are Bio-clinical
SBERT embeddings (Siamese-BERT / Sentence-BERT).

Key features
------------
* Optional hyper-parameter search (C, penalty) via GridSearchCV
* Calibration curve PNG + Brier-score CSV hook
* Top-N positive / negative embedding-dimension export
* Lightweight SHAP explanation (LinearExplainer) - off by default

Author: Matt Stammers / UHSFT
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
import torch
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

try:
    import shap
except ModuleNotFoundError:
    shap = None

# ---------------------------------------------------------------------------#
# Internal Helpers                                                           #
# ---------------------------------------------------------------------------#

def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if missing."""
    path.mkdir(parents=True, exist_ok=True)

def _plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, dest: Path) -> None:
    """Reliability curve with 10 equal-count bins."""
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

def _precompute_embeddings(
    texts: list[str],
    model_name: str,
    device: str = "cuda",
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Encode all texts in one go on GPU, then clear the cache.
    Returns (embeddings, fitted SentenceTransformer).
    """
    try:
        sbert = SentenceTransformer(model_name, device=device)
    except TypeError:
        sbert = SentenceTransformer(model_name)
        try:
            sbert.to(device)
        except Exception:
            pass
    embeddings = sbert.encode(texts, show_progress_bar=show_progress_bar)

    torch.cuda.empty_cache()
    return embeddings, sbert

# ----------------------------------------------------------------------------#
# Transformer Setup                                                           #
# ----------------------------------------------------------------------------#

class SbertMedEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """sklearn wrapper → Bio-clinical SBERT embeddings."""

    def __init__(
        self,
        model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        show_progress_bar: bool = False,
    ):
        self.model_name = model_name
        self.show_progress_bar = show_progress_bar
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------#
    # Sklearn API                                                       #
    # ------------------------------------------------------------------#
    def fit(self, X, y=None):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X, y=None):
        if self._model is None:
            raise ValueError("SbertMedEmbeddingTransformer has not been fitted.")
        if hasattr(X, "tolist"):
            texts = X.tolist()
        else:
            texts = list(X)
        return self._model.encode(texts, show_progress_bar=self.show_progress_bar)

    # ------------------------------------------------------------------#
    # Extract Feature Names                                             #
    # ------------------------------------------------------------------#
    def get_feature_names_out(self, input_features=None):
        """Synthetic feature labels: ['dim_0', 'dim_1', …]."""
        if self._model is None:
            self.fit(["dummy"])
        dim = self._model.get_sentence_embedding_dimension()
        return [f"dim_{i}" for i in range(dim)]


# ---------------------------------------------------------------------------
# Main Model API for SBERT Med
# ---------------------------------------------------------------------------

def train_sbert_model(
    df,
    text_col: str,
    label_col: str,
    analysis_root: Union[str, Path],
    *,
    model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    top_n: int = 20,
    shap_explain: bool = False,
    do_feature_importance: bool = True,
    do_calibration: bool = True,
    cv: Optional[int] = None,
    n_jobs: int = -1,
    show_progress_bar: bool = False,
) -> Tuple[Optional[Pipeline], float]:
    """Fit an SBERT-Med + LogReg model.
    Returns (model | None, seconds_elapsed)."""

    t0 = time.time()

    root = Path(analysis_root)
    _ensure_dir(root)

    if text_col not in df.columns:
        logging.warning("Column '%s' missing – skipping model", text_col)
        return None, 0.0

    data = df.dropna(subset=[text_col, label_col])
    if data.empty:
        logging.warning("No rows with both text and label for '%s'", text_col)
        return None, 0.0

    X = data[text_col].astype(str).values
    y = data[label_col].astype(int).values

    # ——— Build & fit a true sklearn Pipeline ———
    embed = SbertMedEmbeddingTransformer(
        model_name=model_name, show_progress_bar=show_progress_bar
    )
    base_clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    if cv and cv > 1:
        clf = GridSearchCV(
            base_clf,
            param_grid={"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10]},
            cv=cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
            verbose=1,
        )
    else:
        clf = base_clf

    pipeline = Pipeline([("embed", embed), ("clf", clf)])
    pipeline.fit(X.tolist(), y)
    model = pipeline
    sbert_model = pipeline.named_steps["embed"]._model

    # ——— save both encoder & classifier ———
    joblib.dump(sbert_model, root / f"{text_col}_sbert_model.joblib")
    joblib.dump(pipeline.named_steps["clf"], root / f"{text_col}_model.joblib")

    # ————— Hugging-Face & ONNX export —————
    try:
        # Export SBERT encoder to HF format
        hf_dir = root / "hf_sbert"
        hf_dir.mkdir(parents=True, exist_ok=True)
        sbert_model.save_pretrained(str(hf_dir))
        sbert_model.tokenizer.save_pretrained(str(hf_dir))
        logging.info("Saved HF-compatible SBERT to '%s'", hf_dir)

        dim = sbert_model.get_sentence_embedding_dimension()
        initial_type = [("input", FloatTensorType([None, dim]))]
        onnx_model = convert_sklearn(
            pipeline.named_steps["clf"], initial_types=initial_type
        )
        onnx_path = hf_dir / "logreg.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logging.info("Saved ONNX classifier to '%s'", onnx_path)
    except Exception as e:
        logging.error("HF/ONNX export failed: %s", e)

    elapsed = time.time() - t0 or 1e-6
    if elapsed == 0.0:
        elapsed = 1e-6
    logging.info("Trained SBERT model for '%s' in %.2fs", text_col, elapsed)
    model_file = root / f"{text_col}_model.joblib"
    try:
        joblib.dump(model, model_file)
        logging.info("Saved model for '%s' at '%s'", text_col, model_file)
    except Exception as e:
        logging.error("Could not save model for '%s': %s", text_col, e)

    # ------------------- Analysis Outputs -----------------------------
    root = Path(analysis_root)
    _ensure_dir(root)

    # 1) Calibration ---------------------------------------------------
    if do_calibration:
        try:
            probs = pipeline.predict_proba(X.tolist())[:, 1]
            _ensure_dir(root / "calibration")
            _plot_calibration(
                y,
                probs,
                root / "calibration" / f"{text_col}_reliability.png",
            )
        except Exception as exc:
            logging.error("Calibration failed for '%s': %s", text_col, exc)

    # 2) Feature Importance  -------------------------------------------
    if do_feature_importance:
        try:
            embed_t: SbertMedEmbeddingTransformer = pipeline.named_steps["embed"]
            clf_t = (
                pipeline.named_steps["clf"].best_estimator_
                if hasattr(pipeline.named_steps["clf"], "best_estimator_")
                else pipeline.named_steps["clf"]
            )
            odds = clf_t.coef_.ravel()
            feats = np.array(embed_t.get_feature_names_out())
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

    # 3) SHAP -----------------------------------------------------------
    if shap_explain and shap is not None:
        try:
            embed_t = pipeline.named_steps["embed"]
            clf_t = (
                pipeline.named_steps["clf"].best_estimator_
                if hasattr(pipeline.named_steps["clf"], "best_estimator_")
                else pipeline.named_steps["clf"]
            )
            X_trans = embed_t.transform(X.tolist())
            masker = shap.maskers.Independent(X_trans, max_samples=200)
            explainer = shap.LinearExplainer(clf_t, masker)
            shap_vals = explainer(X_trans)
            shap.summary_plot(
                shap_vals,
                features=X_trans,
                feature_names=embed_t.get_feature_names_out(),
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

    return pipeline, elapsed

def predict_sbert(model: Optional[Pipeline], df, text_col: str, *, proba: bool = False):
    """Embed *text_col* with SBERT *model* and return label / proba."""
    if model is None or text_col not in df.columns:
        return np.zeros(len(df), dtype=float if proba else int)
    # Always send a Python list of texts into the pipeline
    texts = df[text_col].fillna("").astype(str).tolist()
    if proba:
        return model.predict_proba(texts)[:, 1]
    return model.predict(texts)