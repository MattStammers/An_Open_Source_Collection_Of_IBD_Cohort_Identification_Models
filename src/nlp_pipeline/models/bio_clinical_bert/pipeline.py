"""
bio_clinical_bert pipeline
==============================
A single entry point that trains, evaluates and analyses a suite of
bio_clinical_bert text-classification models under the unified evaluation
framework. Two text variants are supported:

1. **Raw text**                 untouched clinical notes.
2. **UMLS-standardised text**   the same notes after concept normalisation.

For each variant the pipeline performs the following steps
----------------------------------------------------------
Setup
  A.  Load and pre-process data.
  B.  Train a bioclinicalBERT classifier for every free-text column at **document**
      level and at **patient** level.

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
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from transformers import BertTokenizer

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
from nlp_pipeline.models.bio_clinical_bert.model import (
    predict_bio_clinical_bert,
    train_bio_clinical_bert_for_column,
)

# --------------------------------------------------------------------------- #
# Helper Utilities                                                            #
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

# To prevent internet contact
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def _ensure_dir(path: Path) -> None:  # create parent folders if missing
    path.mkdir(parents=True, exist_ok=True)

# ── reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── MD5 digest helpers (duplicate detection) ───────────────────────────────
TEXT_COLS: List[str] = list(c.COLWISE_COLUMNS)

def _row_digest(row: pd.Series) -> str:
    joined = " ".join(str(row[col]) for col in TEXT_COLS if col in row)
    return hashlib.md5(joined.encode()).hexdigest()


def add_doc_digest(df: pd.DataFrame) -> pd.DataFrame:
    df["doc_digest"] = df.apply(_row_digest, axis=1)
    return df

# ── patient-level aggregation ───────────────────────────────────────────────
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
    # 1) pick only the columns we aggregate
    dem_cols = [d for d in c.DEMOGRAPHICS_KEYS if d in df.columns]
    cols = [id_col] + [c for c in pred_cols + label_cols if c in df.columns] + dem_cols
    sub = df[cols].copy()

    # 2) fill any missing preds/labels with zero
    to_fill = [c for c in pred_cols + label_cols if c in sub.columns]
    sub[to_fill] = sub[to_fill].fillna(0)

    # 3) build the agg-map: OR (max) for preds&labels, first() for demos
    agg_map: Dict[str, str] = {c: "max" for c in to_fill}
    agg_map.update({d: "first" for d in dem_cols})

    # 4) group in one go
    return sub.groupby(id_col, as_index=False).agg(agg_map)

# ── split-integrity checks ──────────────────────────────────────────────────
def _assert_data_splits_ok(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    id_col: str = "study_id",
    raise_on_error: bool = True,
) -> None:
    """Fail fast if *any* leakage is detected between train/validation."""
    problems: List[str] = []

    # 1) patient overlap ---------------------------------------------------
    overlap: Set = set(train_df[id_col]) & set(val_df[id_col])
    if overlap:
        problems.append(
            f"{len(overlap)} patient IDs appear in both splits "
            f"(e.g. {list(overlap)[:5]})"
        )

    # 2) exact document duplicates ----------------------------------------
    train_hash = set(train_df["doc_digest"])
    val_hash = set(val_df["doc_digest"])
    dupes = train_hash & val_hash
    if dupes:
        problems.append(f"{len(dupes)} duplicate documents found across splits")

    # 3) gold labels accidentally in feature matrix -----------------------
    leaked = set(c.COLWISE_COLUMNS.values()) & set(c.COLWISE_COLUMNS.keys())
    if leaked:
        problems.append(f"COLWISE_COLUMNS keys/values overlap: {sorted(leaked)}")

    if problems:
        msg = " | ".join(problems)
        if raise_on_error:
            raise ValueError(msg)
        logging.warning("Data-split diagnostics: %s", msg)
    else:
        logging.info("Data-split diagnostics passed – no leakage detected")

# --------------------------------------------------------------------------- #
# Core workflow for a single variant ('raw' | 'umls')                         #
# --------------------------------------------------------------------------- #

def run_bio_clinical_bert_workflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    mode: str,
    analysis_root: Path,
    run_fairness: bool = True,
    tokenizer: BertTokenizer,
) -> pd.DataFrame:
    """
    Train, evaluate and analyse bio_clinical_bert models for one text variant.

    Returns
    -------
    pd.DataFrame - concatenated evaluation summary for this *mode*.
    """
    logging.info("[%s] bio_clinical_bert workflow start", mode)
    root = analysis_root / mode
    _ensure_dir(root)

    # ── safety checks ────────────────────────────────────────────────────
    _assert_data_splits_ok(train_df, val_df)

    results: List[pd.DataFrame] = []

    # ----------------------------------------------------------------- #
    # 1. Document-level models (per free-text column)                   #
    # ----------------------------------------------------------------- #
    doc_label_cols = list(c.COLWISE_COLUMNS.values())
    for df in (train_df, val_df):
        df["Cumulative_Gold"] = df[doc_label_cols].any(axis=1).astype(int)

    for report_col, label_col in c.COLWISE_COLUMNS.items():
        logging.info("[%s] training document model '%s'", mode, report_col)
        model_dir = root / "models" / "document" / report_col
        _ensure_dir(model_dir)

        model, _, _, _ = train_bio_clinical_bert_for_column(
            train_df,
            train_df,  # no separate dev set
            report_col=report_col,
            label_col=label_col,
            tokenizer=tokenizer,
            output_dir=str(model_dir),
            perform_shap=True,
            perform_lime=True,
        )

        for split_df in (train_df, val_df):
            split_df[
                f"{report_col}_Pred_bio_clinical_bert"
            ] = predict_bio_clinical_bert(model, tokenizer, split_df, report_col)
            # probability output (if supported)
            split_df[
                f"{report_col}_Prob_bio_clinical_bert"
            ] = predict_bio_clinical_bert(
                model, tokenizer, split_df, report_col, proba=True
            )

    # ---- evaluate all doc columns in bulk ------------------------------
    results += [
        evaluate(
            train_df,
            c.COLWISE_COLUMNS,
            dataset_name="Training_Set",
            pred_type="bio_clinical_bert",
            total_count=len(train_df),
        ),
        evaluate(
            val_df,
            c.COLWISE_COLUMNS,
            dataset_name="Validation_Set",
            pred_type="bio_clinical_bert",
            total_count=len(val_df),
        ),
    ]

    # ----------------------------------------------------------------- #
    # 2. Cumulative OR across document columns                          #
    # ----------------------------------------------------------------- #
    cum_pred_cols = [f"{rc}_Pred_bio_clinical_bert" for rc in c.COLWISE_COLUMNS]
    cum_proba_cols = [f"{rc}_Prob_bio_clinical_bert" for rc in c.COLWISE_COLUMNS]

    for df in (train_df, val_df):
        df["Cumulative_Pred_bio_clinical_bert"] = df[cum_pred_cols].max(axis=1)
        df["Cumulative_Prob_bio_clinical_bert"] = df[cum_proba_cols].max(axis=1)

    cum_map = {"Cumulative_Pred_bio_clinical_bert": "Cumulative_Gold"}
    results += [
        evaluate(
            train_df,
            cum_map,
            dataset_name="Training_Set",
            pred_type="Cumulative_bio_clinical_bert",
            final=True,
            total_count=len(train_df),
        ),
        evaluate(
            val_df,
            cum_map,
            dataset_name="Validation_Set",
            pred_type="Cumulative_bio_clinical_bert",
            final=True,
            total_count=len(val_df),
        ),
    ]

    # ----------------------------------------------------------------- #
    # 2.5 Evaluate doc‐level model at patient level                     #
    # ----------------------------------------------------------------- #
    # helper: aggregate Cumulative_Pred_bio_clinical_bert up to patient
    def _aggregate_doc_preds_to_patient(df: pd.DataFrame) -> pd.DataFrame:
        # 1) pick only the cols we need
        label_cols = list(c.IBD_COLUMNS.values())
        cols = [
            "study_id",
            "Cumulative_Pred_bio_clinical_bert",
            "Cumulative_Prob_bio_clinical_bert",
        ] + label_cols
        sub = df[cols].copy()

        # 2) explicit fill for any missing labels or probs
        sub[label_cols] = sub[label_cols].fillna(0)
        sub["Cumulative_Prob_bio_clinical_bert"] = sub[
            "Cumulative_Prob_bio_clinical_bert"
        ].fillna(0.0)

        # 3) group by patient, aggregating OR‐style via max()
        agg = sub.groupby("study_id", as_index=False).agg(
            {
                "Cumulative_Pred_bio_clinical_bert": "max",
                "Cumulative_Prob_bio_clinical_bert": "max",
                **{col: "max" for col in label_cols},
            }
        )

        # 4) compute the patient‐level gold as OR across all IBD labels
        agg["Cumulative_Patient_Gold"] = agg[label_cols].max(axis=1)
        return agg

    for split_name, split_df in (
        ("Training_Set", train_df),
        ("Validation_Set", val_df),
    ):
        pat_from_doc = _aggregate_doc_preds_to_patient(split_df)

        eval_df = evaluate(
            pat_from_doc,
            {"Cumulative_Pred_bio_clinical_bert": "Cumulative_Patient_Gold"},
            dataset_name=split_name,
            pred_type="bio_clinical_bert",
            total_count=len(pat_from_doc),
            final=True,
        )

        # relabel so your results all say "Doc_to_Patient"
        eval_df["Prediction_Type"] = "Doc_to_Patient"
        results.append(eval_df)

    # ----------------------------------------------------------------- #
    # 3. Patient-level models (per clinical concept)                    #
    # ----------------------------------------------------------------- #
    patient_pred_cols: List[str] = []
    for report_col, label_col in c.IBD_COLUMNS.items():
        logging.info("[%s] training patient model '%s'", mode, report_col)
        model_dir = root / "models" / "patient" / report_col
        _ensure_dir(model_dir)

        model, _, _, _ = train_bio_clinical_bert_for_column(
            train_df,
            train_df,
            report_col=report_col,
            label_col=label_col,
            tokenizer=tokenizer,
            output_dir=str(model_dir),
            perform_shap=False,
            perform_lime=False,
        )

        pred_col = f"Patient_{report_col}_Pred_bio_clinical_bert"
        prob_col = f"{pred_col}_Prob"
        patient_pred_cols.append(pred_col)

        for df in (train_df, val_df):
            df[pred_col] = predict_bio_clinical_bert(model, tokenizer, df, report_col)
            df[prob_col] = predict_bio_clinical_bert(
                model, tokenizer, df, report_col, proba=True
            )

    # ---- aggregate to one-row-per-patient -----------------------------
    train_grp = _aggregate_patients(
        train_df, patient_pred_cols, list(c.IBD_COLUMNS.values())
    )
    val_grp = _aggregate_patients(
        val_df, patient_pred_cols, list(c.IBD_COLUMNS.values())
    )

    train_grp["Cumulative_Patient_Gold"] = train_grp[list(c.IBD_COLUMNS.values())].max(
        axis=1
    )
    val_grp["Cumulative_Patient_Gold"] = val_grp[list(c.IBD_COLUMNS.values())].max(
        axis=1
    )

    results += [
        evaluate(
            train_grp,
            c.IBD_COLUMNS,
            dataset_name="Training_Set",
            pred_type="bio_clinical_bert_Patient",
            group_level=True,
            total_count=len(train_grp),
        ),
        evaluate(
            val_grp,
            c.IBD_COLUMNS,
            dataset_name="Validation_Set",
            pred_type="bio_clinical_bert_Patient",
            group_level=True,
            total_count=len(val_grp),
        ),
    ]

    # ----------------------------------------------------------------- #
    # 4. Final OR patient flag + calibrated probability                 #
    # ----------------------------------------------------------------- #
    for grp_df, base_df in ((train_grp, train_df), (val_grp, val_df)):
        missing = set(patient_pred_cols) - set(grp_df.columns)
        if missing:
            raise RuntimeError(
                f"Pipeline bug: missing patient-level prediction columns: {missing}"
            )
        grp_df["Final_Prediction"] = grp_df[patient_pred_cols].max(axis=1)

        grp_df["Cumulative_Prob_bio_clinical_bert"] = (
            base_df.groupby("study_id")["Cumulative_Prob_bio_clinical_bert"]
            .max()
            .reindex(grp_df["study_id"])
            .fillna(0.0)
            .values
        )

    final_map = {"Final_Prediction": "Cumulative_Patient_Gold"}
    results += [
        evaluate(
            train_grp,
            final_map,
            dataset_name="Training_Set",
            pred_type="Final_bio_clinical_bert",
            final=True,
            total_count=len(train_grp),
        ),
        evaluate(
            val_grp,
            final_map,
            dataset_name="Validation_Set",
            pred_type="Final_bio_clinical_bert",
            final=True,
            total_count=len(val_grp),
        ),
    ]

    # ----------------------------------------------------------------- #
    # 5. Calibration summary (validation only)                          #
    # ----------------------------------------------------------------- #
    res_df = pd.concat(results, ignore_index=True)
    (root / "calibration").mkdir(exist_ok=True, parents=True)
    res_df[res_df["Dataset"] == "Validation_Set"][
        ["Report_Column", "Prediction_Type", "Brier_Score"]
    ].to_csv(root / "calibration" / "brier_scores.csv", index=False)

    # ----------------------------------------------------------------- #
    # 6. SHAP feature importance on aggregated patient text             #
    # ----------------------------------------------------------------- #
    tmp = train_df.copy()
    text_input_cols = list(c.COLWISE_COLUMNS.keys())
    existing_cols = [col for col in text_input_cols if col in tmp.columns]
    if not existing_cols:
        logging.warning(
            "[%s] No text columns to aggregate – skipping Patient_Text", mode
        )
    else:
        tmp["doc_text"] = (
            tmp[existing_cols].fillna("").astype(str).agg(" ".join, axis=1)
        )
    pat_text = (
        tmp.groupby("study_id")["doc_text"]
        .agg(" ".join)
        .reset_index(name="Patient_Text")
    )
    labels = (
        train_df[["study_id", list(c.IBD_COLUMNS.values())[0]]]
        .drop_duplicates("study_id")
        .rename(columns={list(c.IBD_COLUMNS.values())[0]: "Label"})
    )
    pat_text = pat_text.merge(labels, on="study_id", how="left")

    model_dir = root / "models" / "patient_text_aggregate"
    _ensure_dir(model_dir)

    if not pat_text["Patient_Text"].str.strip().eq("").all():
        train_bio_clinical_bert_for_column(
            pat_text,
            pat_text,
            report_col="Patient_Text",
            label_col="Label",
            tokenizer=tokenizer,
            output_dir=str(model_dir),
            perform_shap=True,
            perform_lime=True,
        )
    else:
        logging.warning("[%s] Skipping Patient_Text model - all docs empty", mode)

    # ----------------------------------------------------------------- #
    # 7. Fairness                                                       #
    # ----------------------------------------------------------------- #
    if run_fairness:
        fair_root = root / "fairness"
        for split_name, grp_df in (("training", train_grp), ("validation", val_grp)):
            split_dir = fair_root / split_name
            _ensure_dir(split_dir)

            fair = evaluate_fairness_dual(
                df=grp_df,
                final_col="Final_Prediction",
                gold_col="Cumulative_Patient_Gold",
                demographic_attrs=c.DEMOGRAPHICS_KEYS,
                dataset_name=f"{split_name.capitalize()}_Set",
                note=f"Final_bio_clinical_bert_{mode}",
            )

            # save CSVs + plots
            pd.DataFrame(fair["aggregated"]).to_csv(
                split_dir / "aggregated.csv", index=False
            )
            pd.DataFrame(fair["disaggregated"]).to_csv(
                split_dir / "disaggregated.csv", index=False
            )

            plot_fairness_metrics_aggregated(
                pd.DataFrame(fair["aggregated"]), split_dir / "aggregated_plots"
            )
            plot_f1_scores_disaggregated(
                pd.DataFrame(fair["disaggregated"]),
                split_dir / "disaggregated_plots",
            )

    return res_df

# --------------------------------------------------------------------------- #
#  CLI entry-point – run both variants                                        #
# --------------------------------------------------------------------------- #

def main(disable_umls: bool = False) -> None:
    """Run the full two-variant bio_clinical_bert pipeline (SBERT workflow)."""
    analysis_root = Path(c.BIO_CLINICAL_BERT_ANALYSIS_DIR)
    results_dir = Path(c.FINAL_RESULTS_DIR)
    _ensure_dir(analysis_root)
    _ensure_dir(results_dir)

    # Unicode-safe logging -------------------------------------------------
    configure_logging(
        log_dir=str(c.BIO_CLINICAL_BERT_ANALYSIS_DIR),
        custom_logger=logger,
    )
    for h in logging.getLogger().handlers:
        if (
            isinstance(h, logging.StreamHandler)
            and getattr(h.stream, "encoding", "") != "utf-8"
        ):
            try:
                h.setStream(
                    io.TextIOWrapper(
                        sys.stdout.buffer, encoding="utf-8", write_through=True
                    )
                )
            except Exception:
                pass

    logging.info("Starting bio_clinical_bert pipeline")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    tracker = EmissionsTracker(
        project_name="bio_clinical_bert_Workflow",
        output_dir=str(analysis_root),
        allow_multiple_runs=True,
    )
    tracker.start()

    tokenizer = BertTokenizer.from_pretrained(
        "emilyalsentzer/Bio_ClinicalBERT", local_files_only=True
    )

    # ───────── RAW variant ───────────────────────────────────────────────
    train_raw = add_doc_digest(load_and_preprocess(c.TRAIN_FILE_PATH))
    val_raw = add_doc_digest(load_and_preprocess(c.VAL_FILE_PATH))
    val_raw = val_raw[~val_raw["doc_digest"].isin(train_raw["doc_digest"])]
    _assert_data_splits_ok(train_raw, val_raw)
    res_raw = run_bio_clinical_bert_workflow(
        train_raw,
        val_raw,
        mode="raw",
        analysis_root=analysis_root,
        tokenizer=tokenizer,
    )
    res_raw.to_csv(analysis_root / "eval_bio_clinical_bert_raw.csv", index=False)
    res_raw.to_csv(results_dir / "eval_bio_clinical_bert_raw.csv", index=False)

    # ───────── UMLS variant ──────────────────────────────────────────────
    if not disable_umls:
        train_umls = add_doc_digest(load_and_preprocess(c.TRAIN_FILE_UMLS_PATH))
        val_umls = add_doc_digest(load_and_preprocess(c.VAL_FILE_UMLS_PATH))
        _assert_data_splits_ok(train_umls, val_umls)
        res_umls = run_bio_clinical_bert_workflow(
            train_umls,
            val_umls,
            mode="umls",
            analysis_root=analysis_root,
            tokenizer=tokenizer,
        )
        res_umls.to_csv(analysis_root / "eval_bio_clinical_bert_umls.csv", index=False)

    # ───────── resources summary ─────────────────────────────────────────
    monitor.stop()
    cpu_pct, mem_mb = monitor.get_metrics()
    energy_kwh = tracker.stop()
    elapsed = getattr(monitor, "elapsed", None)

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

    logging.info("bio_clinical_bert SBERT-style pipeline complete")


if __name__ == "__main__":
    main()