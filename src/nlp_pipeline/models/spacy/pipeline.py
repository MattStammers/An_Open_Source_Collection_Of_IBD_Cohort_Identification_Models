#!/usr/bin/env python3
"""
SpaCy-based IBD pipeline
========================
A single entry point that runs a unified SpaCy pipeline over two text variants:

1. **Raw text**   untouched clinical notes.
2. **UMLS-standardised text**   the same notes after concept normalisation.

For each variant the pipeline performs the following steps
--------------------------------------------------------
Setup
  A. Load and pre-process data.
  B. Initialise SpaCy + NegSpacy and build phrase matchers.

Predictions & Evaluation
  1.  Apply SpaCy phrase-matching predictions per document column.
  2.  Aggregate document-level predictions per column (Combined match).
  3.  Cumulative OR across columns at document level.
  4.  Collapse to patient-level via OR across documents.
  5.  Final OR-based patient flag and dummy probabilities.
  6.  Calibration: dummy probabilities to satisfy framework.
  7.  Metric evaluation at each stage (column-wise, document, patient, final).
  8.  Fairness analysis.
  9.  Persist evaluation, fairness and resource metrics.

Author: Matt Stammers / UHSFT
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Set

import pandas as pd
from codecarbon import EmissionsTracker

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.data_utils import load_and_preprocess
from nlp_pipeline.common.evaluation import evaluate
from nlp_pipeline.common.fairness import evaluate_fairness_dual
from nlp_pipeline.common.logging_setup import configure_logging
from nlp_pipeline.common.resource_monitor import ResourceMonitor
from nlp_pipeline.models.spacy.model import (
    apply_spacy_predictions,
    create_final_prediction,
    create_phrase_matcher,
    initialize_spacy,
)

# Temporary monkeypatch
pd.DataFrame.name = None

# ---------------------------------------------------------------------------- #
# Helper utilities                                                             #
# ---------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> None:
    """Create *path* (and parents) if it does not yet exist."""
    path.mkdir(parents=True, exist_ok=True)

def add_doc_digest(df: pd.DataFrame) -> pd.DataFrame:
    df["doc_digest"] = df.apply(
        lambda row: hashlib.md5(
            " ".join(str(row[col]) for col in c.COLWISE_COLUMNS if col in row).encode()
        ).hexdigest(),
        axis=1,
    )
    return df

def _assert_data_splits_ok(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    id_col: str = "study_id",
    raise_on_error: bool = True,
) -> None:
    """Fail fast if there is any evidence of data leakage between splits."""
    problems: List[str] = []
    # Patient overlap
    overlap: Set = set(train_df[id_col]) & set(val_df[id_col])
    if overlap:
        problems.append(f"{len(overlap)} patient IDs in both splits")
    # Document duplicates
    train_hash = set(train_df.apply(lambda r: r["doc_digest"], axis=1))
    val_hash = set(val_df.apply(lambda r: r["doc_digest"], axis=1))
    dupes = train_hash & val_hash
    if dupes:
        problems.append(f"{len(dupes)} duplicate documents across splits")
    # Gold columns leakage
    gold_cols = [col for col in train_df.columns if col.endswith("_Gold")]
    forbidden = set(gold_cols) & set(c.COLWISE_COLUMNS.keys())
    if forbidden:
        problems.append(f"Leaked gold columns: {sorted(forbidden)}")
    if problems:
        msg = " | ".join(problems)
        if raise_on_error:
            raise ValueError(msg)
        logger.warning("Data-split diagnostics: %s", msg)
    else:
        logger.info("Data-split diagnostics passed - no leakage detected")

def _aggregate_patients(
    df: pd.DataFrame,
    pred_cols: List[str],
    label_cols: List[str],
    *,
    id_col: str = "study_id",
) -> pd.DataFrame:
    """
    Collapse document-level predictions and labels to patient level:
    - OR (max) over all pred_cols and label_cols present in df
    - first() for any demographic keys
    """
    dem_cols = [d for d in c.DEMOGRAPHICS_KEYS if d in df.columns]
    unique_preds = [col for col in pred_cols if col in df.columns]
    unique_labels = [col for col in label_cols if col in df.columns]
    cols = [id_col] + unique_preds + unique_labels + dem_cols
    sub = df[cols].copy()

    # Fill NA for all flags
    for col in unique_preds + unique_labels:
        sub[col] = sub[col].fillna(0)

    # Aggregation map: max for flags, first for demographics
    agg_map = {col: "max" for col in unique_preds + unique_labels}
    agg_map.update({d: "first" for d in dem_cols})

    return sub.groupby(id_col, as_index=False).agg(agg_map)

# ---------------------------------------------------------------------------- #
# Core workflow per text mode ('raw' | 'umls')                                 #
# ---------------------------------------------------------------------------- #

def run_spacy_workflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    mode: str,
    analysis_root: Path,
    run_fairness: bool = True,
) -> pd.DataFrame:
    logger.info(f"[{mode}] SpaCy workflow start")
    root = analysis_root / mode
    _ensure_dir(root)

    # 0. Safety checks
    add_doc_digest(train_df)
    add_doc_digest(val_df)
    _assert_data_splits_ok(train_df, val_df)

    # figure out which report-cols we actually have
    present_cols = [rc for rc in c.COLWISE_COLUMNS if rc in train_df.columns]
    missing_cols = set(c.COLWISE_COLUMNS) - set(present_cols)
    for rc in missing_cols:
        logger.warning("Skipping '%s' — missing columns", rc)

    # 1. Initialise SpaCy + NegSpacy
    nlp = initialize_spacy(
        model_name="en_core_web_md",
        neg_termset=c.NEG_TERMSET,
        ent_types=None,
        extension_name="negex",
    )
    ibd_matcher = create_phrase_matcher(nlp, c.IBD_KEYWORDS, label="IBD")
    drug_matcher = create_phrase_matcher(nlp, c.DRUG_KEYWORDS, label="DRUG")

    results: list[dict] = []
    doc_cols = list(c.COLWISE_COLUMNS.keys())
    label_cols = list(c.COLWISE_COLUMNS.values())

    # 2. Document-level Combined predictions per column
    for df, split in [(train_df, "Training_Set"), (val_df, "Validation_Set")]:
        # compute Cumulative_Gold for evaluation
        df["Cumulative_Gold"] = df[label_cols].any(axis=1).astype(int)
        for report_col in present_cols:
            gold_col = c.COLWISE_COLUMNS[report_col]
            mapping = {report_col: gold_col}
            ptypes = {report_col: ["Combined"]}
            df = apply_spacy_predictions(
                df, mapping, ptypes, ibd_matcher, drug_matcher, nlp
            )
            df = create_final_prediction(df)
            # dummy probability for calibration
            pred_col = f"{report_col}_Pred_Combined"
            df[f"{report_col}_Prob_Combined"] = df[pred_col].astype(float)
        # bulk evaluation of each column
        results += evaluate(
            df,
            {rc: c.COLWISE_COLUMNS[rc] for rc in doc_cols},
            dataset_name=split,
            pred_type="Spacy",
            total_count=len(df),
        ).to_dict(orient="records")

    # 3. Cumulative OR across document columns
    cum_pred_cols = [f"{rc}_Pred_Combined" for rc in doc_cols]
    cum_proba_cols = [f"{rc}_Prob_Combined" for rc in doc_cols]
    for df, split in [(train_df, "Training_Set"), (val_df, "Validation_Set")]:
        df["Cumulative_Pred_Spacy"] = df[cum_pred_cols].max(axis=1)
        df["Cumulative_Prob_Spacy"] = df[cum_proba_cols].max(axis=1)
        # evaluate cumulative
        results += evaluate(
            df,
            {"Cumulative_Pred_Spacy": "Cumulative_Gold"},
            dataset_name=split,
            pred_type="Cumulative_Spacy",
            final=True,
            total_count=len(df),
        ).to_dict(orient="records")

    # 4. Collapse to patient level
    # --------------------------------------------------------------
    # Compute list of per-document prediction columns
    cum_pred_cols = [f"{rc}_Pred_Combined" for rc in c.COLWISE_COLUMNS.keys()]

    patient_label_cols = list({v for v in c.PATIENT_COLUMNS.values()})

    train_doc2pat = _aggregate_patients(train_df, cum_pred_cols, patient_label_cols)
    val_doc2pat = _aggregate_patients(val_df, cum_pred_cols, patient_label_cols)

    for grp_df, base_df in [
        (train_doc2pat, train_df),
        (val_doc2pat, val_df),
    ]:
        # OR‐across‐documents for the doc‐level cumulative flags
        grp_df["Cumulative_Pred_Spacy"] = (
            base_df.groupby("study_id")["Cumulative_Pred_Spacy"]
            .max()
            .reindex(grp_df["study_id"])
            .fillna(0)
            .astype(int)
            .values
        )
        grp_df["Cumulative_Prob_Spacy"] = (
            base_df.groupby("study_id")["Cumulative_Prob_Spacy"]
            .max()
            .reindex(grp_df["study_id"])
            .fillna(0.0)
            .values
        )

    # 5. Final OR-based patient flag & dummy probabilities
    # here Final_Prediction is same as Cumulative_Pred_Spacy
    for grp_df, base_df in [(train_doc2pat, train_df), (val_doc2pat, val_df)]:
        grp_df["Final_Prediction"] = grp_df["Cumulative_Pred_Spacy"]
        grp_df["Final_Prob_Spacy"] = (
            base_df.groupby("study_id")["Cumulative_Prob_Spacy"]
            .max()
            .reindex(grp_df["study_id"])
            .fillna(0.0)
            .values
        )
        split = "Training_Set" if grp_df is train_doc2pat else "Validation_Set"
        results += evaluate(
            grp_df,
            {"Final_Prediction": "Patient_Has_IBD"},
            dataset_name=split,
            pred_type="Final_Spacy",
            final=True,
            total_count=len(grp_df),
        ).to_dict(orient="records")

    # Build patient‐level tables (with demographics) for fairness
    dem_cols_train = [d for d in c.DEMOGRAPHICS_KEYS if d in train_doc2pat.columns]
    dem_cols_val = [d for d in c.DEMOGRAPHICS_KEYS if d in val_doc2pat.columns]
    final_train = train_doc2pat.loc[
        :,
        ["study_id", "Patient_Has_IBD", "Final_Prediction", "Final_Prob_Spacy"]
        + dem_cols_train,
    ]
    final_val = val_doc2pat.loc[
        :,
        ["study_id", "Patient_Has_IBD", "Final_Prediction", "Final_Prob_Spacy"]
        + dem_cols_val,
    ]

    # 6. Calibration summary (validation only)
    res_df = pd.DataFrame(results)
    print("== columns:", res_df.columns.tolist())
    print("== sample rows:\n", res_df.head())
    calib_dir = root / "calibration"
    _ensure_dir(calib_dir)
    res_df[res_df["Dataset"] == "Validation_Set"][
        ["Report_Column", "Prediction_Type", "Brier_Score"]
    ].to_csv(calib_dir / "brier_scores.csv", index=False)

    # 7. Fairness analysis
    if run_fairness:
        fair_root = root / "fairness"
        _ensure_dir(fair_root)
        for patient_df, split in (
            (final_train, "Training_Set"),
            (final_val, "Validation_Set"),
        ):
            # pick only demographics present AND with at least 2 unique values
            valid_attrs = [
                attr
                for attr in c.DEMOGRAPHICS_KEYS
                if attr in patient_df.columns
                and patient_df[attr].nunique(dropna=True) >= 2
            ]
            if not valid_attrs:
                logger.warning(
                    "No demographic attribute with ≥2 groups for %s — skipping.", split
                )
                continue

            split_dir = fair_root / split.lower()
            _ensure_dir(split_dir)

            fair = evaluate_fairness_dual(
                df=patient_df,
                final_col="Final_Prediction",
                gold_col="Patient_Has_IBD",
                demographic_attrs=valid_attrs,
                dataset_name=split,
                note=f"Final_Regex_{mode}",
            )

            # Persist CSVs exactly like BOW pipeline
            pd.DataFrame(fair["aggregated"]).to_csv(
                split_dir / "aggregated.csv", index=False
            )
            pd.DataFrame(fair["disaggregated"]).to_csv(
                split_dir / "disaggregated.csv", index=False
            )

            # Finally append to results
            if hasattr(fair["aggregated"], "to_dict"):
                results += fair["aggregated"].to_dict(orient="records")
                results += fair["disaggregated"].to_dict(orient="records")
            else:
                results += fair["aggregated"] + fair["disaggregated"]

    logger.info(f"[{mode}] about to return {len(results)} result rows")
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------- #
# Entry-Point (command-line)                                                   #
# ---------------------------------------------------------------------------- #

def main(disable_umls: bool = False) -> None:
    analysis_root = Path(c.SPACY_ANALYSIS_DIR)
    results_dir = Path(c.FINAL_RESULTS_DIR)
    _ensure_dir(analysis_root)
    _ensure_dir(results_dir)

    configure_logging(log_dir=str(c.SPACY_ANALYSIS_DIR), custom_logger=logger)
    logger.info("Starting SpaCy pipeline")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    tracker = EmissionsTracker(
        project_name="SpaCy_Pipeline",
        output_dir=str(analysis_root),
        allow_multiple_runs=True,
    )
    tracker.start()

    # RAW variant
    train_raw = load_and_preprocess(c.TRAIN_FILE_PATH)
    val_raw = load_and_preprocess(c.VAL_FILE_PATH)
    train_raw = add_doc_digest(train_raw)
    val_raw = add_doc_digest(val_raw)
    val_raw = val_raw[~val_raw["doc_digest"].isin(train_raw["doc_digest"])]
    _assert_data_splits_ok(train_raw, val_raw)
    res_raw = run_spacy_workflow(
        train_raw, val_raw, mode="raw", analysis_root=analysis_root
    )
    res_raw.to_csv(analysis_root / "eval_spacy_raw.csv", index=False)
    res_raw.to_excel(results_dir / "eval_spacy_raw.xlsx", index=False)

    # UMLS variant
    if not disable_umls:
        train_umls = load_and_preprocess(c.TRAIN_FILE_UMLS_PATH)
        val_umls = load_and_preprocess(c.VAL_FILE_UMLS_PATH)
        train_umls = add_doc_digest(train_umls)
        val_umls = add_doc_digest(val_umls)
        _assert_data_splits_ok(train_umls, val_umls)
        res_umls = run_spacy_workflow(
            train_umls, val_umls, mode="umls", analysis_root=analysis_root
        )
        res_umls.to_csv(analysis_root / "eval_spacy_umls.csv", index=False)
        res_umls.to_excel(results_dir / "eval_spacy_umls.xlsx", index=False)

    # Resources
    monitor.stop()
    cpu, mem = monitor.get_metrics()
    energy = tracker.stop()
    elapsed = getattr(monitor, "elapsed", None)
    pd.DataFrame(
        [{"cpu_pct": cpu, "memory_mb": mem, "energy_kwh": energy, "time_sec": elapsed}]
    ).to_csv(analysis_root / "resources.csv", index=False)

    logger.info("SpaCy pipeline complete")

if __name__ == "__main__":
    main()