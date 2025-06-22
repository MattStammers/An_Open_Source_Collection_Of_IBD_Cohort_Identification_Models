"""
TF-IDF pipeline
==========================
A single entry point that trains, evaluates and analyses a suite of TF-IDF
model under the unified evaluation framework. It supports two text variants:

1. **Raw text**   untouched clinical notes.
2. **UMLS-standardised text**   the same notes after concept normalisation.

For each variant the pipeline performs the following steps
--------------------------------------------------------
Setup
  A. Load and pre-process data.
  B. Train a TF-IDF classifier for every free-text column at **document** level and
     at **patient** level.

Evaluation
  1.  Metric set for each document level model.
  2.  Cumulative (multi-column OR) document prediction.
  3.  Metric set for each patient level model.
  4.  Final OR based patient flag.
  5.  Calibration (Brier) scores.
  6.  SHAP feature importance on aggregated patient text.
  7.  Fairness analysis.
  8.  Persist evaluation, fairness and resource metrics.

Author: Matt Stammers / UHSFT
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.data_utils import load_and_preprocess
from nlp_pipeline.common.evaluation import evaluate
from nlp_pipeline.common.fairness import (
    evaluate_fairness_dual,
    plot_f1_scores_disaggregated,
    plot_fairness_metrics_aggregated,
)
from nlp_pipeline.common.logging_setup import configure_logging
from nlp_pipeline.common.resource_monitor import ResourceMonitor

from .model import predict_tfidf, train_tfidf_model

# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)


def add_doc_digest(df: pd.DataFrame) -> pd.DataFrame:
    df["doc_digest"] = df.apply(_row_digest, axis=1)
    return df


def _aggregate_patients(
    df: pd.DataFrame,
    pred_cols: List[str],
    label_cols: List[str],
    *,
    id_col: str = "study_id",
) -> pd.DataFrame:
    """
    Collapse document-level predictions and labels to patient level:
    - OR (max) over all pred_cols and label_cols that actually exist in df
    - first() for any demographic keys
    """
    # 1) Demographics present
    dem_cols = [d for d in c.DEMOGRAPHICS_KEYS if d in df.columns]

    # 2) Only keep columns that actually exist, and dedupe
    unique_preds = list(dict.fromkeys([col for col in pred_cols if col in df.columns]))
    unique_labels = list(
        dict.fromkeys([col for col in label_cols if col in df.columns])
    )
    cols = [id_col] + unique_preds + unique_labels + dem_cols

    sub = df[cols].copy()

    # 3) Fill missing preds/labels with 0
    to_fill = unique_preds + unique_labels
    sub[to_fill] = sub[to_fill].fillna(0)

    # 4) Build agg map: max for preds/labels, first for demos
    agg_map = {col: "max" for col in to_fill}
    agg_map.update({d: "first" for d in dem_cols})

    # 5) Group by patient ID
    return sub.groupby(id_col, as_index=False).agg(agg_map)

# Text columns used throughout the pipeline 
TEXT_COLS: List[str] = list(c.COLWISE_COLUMNS)

def _row_digest(row: pd.Series) -> str:
    """Return an MD5 hash representing *all* free-text columns in *row*."""
    joined = " ".join(str(row[col]) for col in TEXT_COLS if col in row)
    return hashlib.md5(joined.encode()).hexdigest()

def _assert_data_splits_ok(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    id_col: str = "study_id",
    raise_on_error: bool = True,
) -> None:
    """Fail fast if there is any evidence of data leakage.

    The function checks three conditions and either raises *ValueError* or
    logs a warning, depending on *raise_on_error*.
    """

    problems: List[str] = []

    # A) Patient overlap ---------------------------------------------------
    overlap: Set = set(train_df[id_col]) & set(val_df[id_col])
    if overlap:
        problems.append(
            f"{len(overlap)} patient IDs appear in both splits (e.g. {list(overlap)[:5]})"
        )

    # B) Exact document duplicates ----------------------------------------
    train_hash = set(train_df.apply(_row_digest, axis=1))
    val_hash = set(val_df.apply(_row_digest, axis=1))
    dupes = train_hash & val_hash
    if dupes:
        problems.append(f"{len(dupes)} duplicate documents found across splits")

    # C) Gold columns inside feature matrix -------------------------------
    gold_cols = [col for col in train_df.columns if col.endswith("_Gold")]
    forbidden = set(gold_cols) & set(TEXT_COLS)
    if forbidden:
        problems.append(f"Gold label columns leaked into features: {sorted(forbidden)}")

    text_col_set = set(c.COLWISE_COLUMNS.keys())
    label_col_set = set(c.COLWISE_COLUMNS.values())
    collisions = text_col_set & label_col_set
    assert not collisions, f"COLWISE_COLUMNS has overlapping keys/values: {collisions}"

    if problems:
        msg = " | ".join(problems)
        if raise_on_error:
            raise ValueError(msg)
        logging.warning("Data-split diagnostics: %s", msg)
    else:
        logging.info("Data-split diagnostics passed - no leakage detected")

# --------------------------------------------------------------------------- #
# Core workflow for a single doc type mode ('raw' | 'umls')                   #
# --------------------------------------------------------------------------- #

def run_tfidf_workflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    mode: str,
    analysis_root: Path,
    run_fairness: bool = True,
) -> pd.DataFrame:
    """Train, evaluate and analyse the TF-IDF models for a given *mode*.

    Parameters
    ----------
    train_df, val_df
        Pre-processed data frames for training and validation.
    mode
        Either ``"raw"`` or ``"umls"``; used for folder naming.
    analysis_root
        Root directory where all artefacts for this mode are written.
    run_fairness
        If *True*, produce fairness plots and CSVs.

    Returns
    -------
    pd.DataFrame
        The concatenated evaluation summary table for this *mode*.
    """

    logging.info("[%s] TF-IDF workflow start", mode)
    root = analysis_root / mode
    _ensure_dir(root)

    # Safety checks --------------------------------------------------------
    _assert_data_splits_ok(train_df, val_df)

    results: List[pd.DataFrame] = []

    # ------------------------------------------------------------------ #
    # Document‑level models per report column                            #
    # ------------------------------------------------------------------ #
    doc_label_cols = list(c.COLWISE_COLUMNS.values())
    for df in (train_df, val_df):
        df["Cumulative_Gold"] = df[doc_label_cols].any(axis=1).astype(int)

    for report_col, label_col in c.COLWISE_COLUMNS.items():
        logging.info("[%s] training document-level model '%s'", mode, report_col)
        # make sure the model folder exists
        _ensure_dir(root / "models" / "document" / report_col)
        assert df[report_col].dtype == object, f"'{report_col}' is not text!"
        assert pd.api.types.is_numeric_dtype(
            df[label_col]
        ), f"'{label_col}' is not numeric!"
        model, _ = train_tfidf_model(
            train_df,
            report_col,
            label_col,
            root / "models" / "document" / report_col,
            shap_explain=True,
            do_feature_importance=True,
            do_calibration=True,
            cv=5,
        )
        for split_df in (train_df, val_df):
            split_df[f"{report_col}_Pred_TFIDF"] = predict_tfidf(
                model, split_df, report_col
            )
            split_df[f"{report_col}_Prob_TFIDF"] = predict_tfidf(
                model, split_df, report_col, proba=True
            )

    # Evaluate all doc columns in bulk ------------------------------------
    results += [
        evaluate(
            train_df,
            c.COLWISE_COLUMNS,
            dataset_name="Training_Set",
            pred_type="TFIDF",
            total_count=len(train_df),
        ),
        evaluate(
            val_df,
            c.COLWISE_COLUMNS,
            dataset_name="Validation_Set",
            pred_type="TFIDF",
            total_count=len(val_df),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Cumulative OR across document columns                              #
    # ------------------------------------------------------------------ #
    cum_pred_cols = [f"{rc}_Pred_TFIDF" for rc in c.COLWISE_COLUMNS]
    cum_proba_cols = [f"{rc}_Prob_TFIDF" for rc in c.COLWISE_COLUMNS]

    for df in (train_df, val_df):
        df["Cumulative_Pred_TFIDF"] = df[cum_pred_cols].max(axis=1)
        df["Cumulative_Prob_TFIDF"] = df[cum_proba_cols].max(axis=1)

    cum_map = {"Cumulative_Pred_TFIDF": "Cumulative_Gold"}
    results += [
        evaluate(
            train_df,
            cum_map,
            dataset_name="Training_Set",
            pred_type="Cumulative_TFIDF",
            final=True,
            total_count=len(train_df),
        ),
        evaluate(
            val_df,
            cum_map,
            dataset_name="Validation_Set",
            pred_type="Cumulative_TFIDF",
            final=True,
            total_count=len(val_df),
        ),
    ]

    # ------------------------------------------------------------------- #
    # Doc-trained models tested at patient level                          #
    # ------------------------------------------------------------------- #
    # 1) Document-level prediction columns
    doc_pred_cols = [f"{rc}_Pred_TFIDF" for rc in c.COLWISE_COLUMNS]

    # 2) Collapse to patient level (OR across all docs)
    train_doc2pat = _aggregate_patients(
        train_df,
        doc_pred_cols,
        list(c.PATIENT_COLUMNS.values()),
    )
    val_doc2pat = _aggregate_patients(
        val_df,
        doc_pred_cols,
        list(c.PATIENT_COLUMNS.values()),
    )

    # 3) Compute the single “cumulative” pred at patient level
    train_doc2pat["Cumulative_Pred_TFIDF"] = train_doc2pat[doc_pred_cols].max(axis=1)
    val_doc2pat["Cumulative_Pred_TFIDF"] = val_doc2pat[doc_pred_cols].max(axis=1)

    # 4) Final mapping: cumulative pred → patient gold
    final_mapping = {"Cumulative_Pred_TFIDF": "Patient_Has_IBD"}

    # 5) Evaluate exactly once per split with final=True
    train_eval = evaluate(
        train_doc2pat,
        final_mapping,
        dataset_name="Training_Set",
        pred_type="Doc2Patient_TFIDF",
        total_count=len(train_doc2pat),
        final=True,
    )
    val_eval = evaluate(
        val_doc2pat,
        final_mapping,
        dataset_name="Validation_Set",
        pred_type="Doc2Patient_TFIDF",
        total_count=len(val_doc2pat),
        final=True,
    )

    # 6) Append
    results += [train_eval, val_eval]

    # ------------------------------------------------------------------ #
    # Patient‑level models (per report column)                           #
    # ------------------------------------------------------------------ #
    patient_pred_cols: List[str] = []
    for report_col, label_col in c.IBD_COLUMNS.items():
        logging.info("[%s] training patient-level model '%s'", mode, report_col)
        # ensure directory for patient‐level model
        _ensure_dir(root / "models" / "patient" / report_col)
        model, _ = train_tfidf_model(
            train_df,
            report_col,
            label_col,
            root / "models" / "patient" / report_col,
            shap_explain=False,
            cv=5,
        )
        pred_col = f"Patient_{report_col}_Pred_TFIDF"
        patient_pred_cols.append(pred_col)
        for df in (train_df, val_df):
            df[pred_col] = predict_tfidf(model, df, report_col)

    # Aggregate to single-row-per-patient ---------------------------------
    train_grp = _aggregate_patients(
        train_df, patient_pred_cols, list(c.IBD_COLUMNS.values())
    )
    val_grp = _aggregate_patients(
        val_df, patient_pred_cols, list(c.IBD_COLUMNS.values())
    )

    train_grp["Cumulative_Patient_Level_Gold"] = train_grp[
        list(c.IBD_COLUMNS.values())
    ].max(axis=1)
    val_grp["Cumulative_Patient_Level_Gold"] = val_grp[
        list(c.IBD_COLUMNS.values())
    ].max(axis=1)

    results += [
        evaluate(
            train_grp,
            c.IBD_COLUMNS,
            dataset_name="Training_Set",
            pred_type="Patient_TFIDF",
            group_level=True,
            total_count=len(train_grp),
        ),
        evaluate(
            val_grp,
            c.IBD_COLUMNS,
            dataset_name="Validation_Set",
            pred_type="Patient_TFIDF",
            group_level=True,
            total_count=len(val_grp),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Final OR patient flag + calibrated probability                     #
    # ------------------------------------------------------------------ #
    for grp_df, base_df in ((train_grp, train_df), (val_grp, val_df)):
        grp_df["Final_Prediction"] = grp_df[patient_pred_cols].max(axis=1)
        grp_df["Final_Prob_TFIDF"] = (
            base_df.groupby("study_id")["Cumulative_Prob_TFIDF"]
            .max()
            .reindex(grp_df["study_id"])
            .fillna(0.0)
            .values
        )

    final_map = {"Final_Prediction": "Cumulative_Patient_Level_Gold"}
    results += [
        evaluate(
            train_grp,
            final_map,
            dataset_name="Training_Set",
            pred_type="Final_TFIDF",
            final=True,
            total_count=len(train_grp),
        ),
        evaluate(
            val_grp,
            final_map,
            dataset_name="Validation_Set",
            pred_type="Final_TFIDF",
            final=True,
            total_count=len(val_grp),
        ),
    ]

    # ------------------------------------------------------------------ #
    # Calibration summary (validation only)                              #
    # ------------------------------------------------------------------ #
    res_df = pd.concat(results, ignore_index=True)
    (root / "calibration").mkdir(exist_ok=True, parents=True)
    res_df[res_df["Dataset"] == "Validation_Set"][
        ["Report_Column", "Prediction_Type", "Brier_Score"]
    ].to_csv(root / "calibration" / "brier_scores.csv", index=False)

    # ------------------------------------------------------------------ #
    # Feature importance on aggregated patient text                      #
    # ------------------------------------------------------------------ #
    tmp = train_df.copy()
    text_cols = list(c.COLWISE_COLUMNS.keys())  # To access the raw text
    existing_text_cols = [col for col in text_cols if col in tmp.columns]

    if not existing_text_cols:
        logging.warning(
            f"[{mode}] No valid text columns found for doc_text aggregation – skipping."
        )
        tmp["doc_text"] = ""
    else:
        tmp["doc_text"] = (
            tmp[existing_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
    pat_text = (
        tmp.groupby("study_id")["doc_text"]
        .agg(" ".join)
        .reset_index(name="Patient_Text")
    )
    labels = train_df[["study_id", list(c.IBD_COLUMNS.values())[0]]].drop_duplicates(
        "study_id"
    )
    pat_text = pat_text.merge(labels, on="study_id", how="left")

    # ensure directory for the aggregate text model
    _ensure_dir(root / "models" / "patient_text_aggregate")
    logging.info(
        f"Example of patient text (first 3 rows):\n{pat_text['Patient_Text'].head(3)}"
    )
    assert df[report_col].dtype == object, f"'{report_col}' is not text!"
    assert pd.api.types.is_numeric_dtype(
        df[label_col]
    ), f"'{label_col}' is not numeric!"
    if pat_text["Patient_Text"].str.strip().eq("").all():
        logging.warning(
            f"[{mode}] Skipping Patient_Text model — all documents are empty."
        )
    else:
        train_tfidf_model(
            pat_text,
            "Patient_Text",
            list(c.IBD_COLUMNS.values())[0],
            root / "models" / "patient_text_aggregate",
            shap_explain=True,
            do_feature_importance=True,
            do_calibration=True,
            cv=5,
        )

    # ------------------------------------------------------------------ #
    # Fairness                                                           #
    # ------------------------------------------------------------------ #
    if run_fairness:
        fair_root = root / "fairness"
        for split_name, grp_df in (("training", train_grp), ("validation", val_grp)):
            split_dir = fair_root / split_name
            _ensure_dir(split_dir)

            fair = evaluate_fairness_dual(
                df=grp_df,
                final_col="Final_Prediction",
                gold_col="Cumulative_Patient_Level_Gold",
                demographic_attrs=c.DEMOGRAPHICS_KEYS,
                dataset_name=f"{split_name.capitalize()}_Set",
                note=f"Final_TFIDF_{mode}",
            )

            # ---------- CSVs ----------
            pd.DataFrame(fair["aggregated"]).to_csv(
                split_dir / "aggregated.csv", index=False
            )
            pd.DataFrame(fair["disaggregated"]).to_csv(
                split_dir / "disaggregated.csv", index=False
            )

            # ---------- NEW: make sub-dirs & plots ----------
            agg_plots_dir = split_dir / "aggregated_plots"
            dis_plots_dir = split_dir / "disaggregated_plots"
            _ensure_dir(agg_plots_dir)
            _ensure_dir(dis_plots_dir)

            plot_fairness_metrics_aggregated(
                pd.DataFrame(fair["aggregated"]), agg_plots_dir
            )
            plot_f1_scores_disaggregated(
                pd.DataFrame(fair["disaggregated"]), dis_plots_dir
            )

    return res_df

# ---------------------------------------------------------------------------#
# Entry-Point (command‑line)                                                 #
# ---------------------------------------------------------------------------#

def main(disable_umls: bool = False) -> None:
    """CLI entry-point - executes the full two-variant pipeline."""

    analysis_root = Path(c.TFIDF_ANALYSIS_DIR)
    results_dir = Path(c.FINAL_RESULTS_DIR)
    _ensure_dir(analysis_root)
    _ensure_dir(results_dir)

    # UTF‑8 logging – shields against stray non‑ASCII characters -----------
    configure_logging(
        log_dir=str(c.TFIDF_ANALYSIS_DIR),
        custom_logger=logger,
    )

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    tracker = EmissionsTracker(
        project_name="TFIDF_Pipeline",
        output_dir=str(analysis_root),
        allow_multiple_runs=True,
    )
    tracker.start()

    # -------------------- RAW --------------------
    train_raw = load_and_preprocess(c.TRAIN_FILE_PATH)
    val_raw = load_and_preprocess(c.VAL_FILE_PATH)
    train_raw["doc_digest"] = train_raw.apply(_row_digest, axis=1)
    val_raw["doc_digest"] = val_raw.apply(_row_digest, axis=1)
    val_raw = val_raw[~val_raw["doc_digest"].isin(train_raw["doc_digest"])]
    _assert_data_splits_ok(train_raw, val_raw)
    res_raw = run_tfidf_workflow(
        train_raw, val_raw, mode="raw", analysis_root=analysis_root
    )
    res_raw.to_csv(analysis_root / "eval_tfidf_raw.csv", index=False)
    res_raw.to_csv(results_dir / "eval_tfidf_raw.csv", index=False)

    # -------------------- UMLS -------------------
    if not disable_umls:
        train_umls = load_and_preprocess(c.TRAIN_FILE_UMLS_PATH)
        val_umls = load_and_preprocess(c.VAL_FILE_UMLS_PATH)
        _assert_data_splits_ok(train_umls, val_umls)
        res_umls = run_tfidf_workflow(
            train_umls, val_umls, mode="umls", analysis_root=analysis_root
        )
        res_umls.to_csv(analysis_root / "eval_tfidf_umls.csv", index=False)

    # -------------------- Resources --------------
    monitor.stop()
    cpu_pct, mem_mb = monitor.get_metrics()
    energy_kwh = tracker.stop()

    if hasattr(monitor, "elapsed"):
        elapsed = monitor.elapsed
    else:
        elapsed = None

    pd.DataFrame(
        [
            {
                "cpu_pct": cpu_pct,
                "memory_mb": mem_mb,
                "energy_kwh": energy_kwh,
                "time_sec": elapsed,
            }
        ]
    ).to_csv(analysis_root / "resources.csv", index=False)

    logging.info("TF-IDF pipeline complete")

if __name__ == "__main__":
    main()