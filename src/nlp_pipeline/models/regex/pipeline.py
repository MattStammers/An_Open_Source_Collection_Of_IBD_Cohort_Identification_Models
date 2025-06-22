"""
Regex-based NLP pipeline, unified for both “free-text” (raw) and UMLS-standardised text.

For each mode (“raw” or “umls”) we do exactly the same steps:
  1. Compile IBD & Drug patterns (with or without UMLS normalisation).
  2. Document-level predictions (IBD + Drug) → <col>_Pred_IBD, <col>_Pred_Drug, plus dummy <col>_Prob_*
  3. Combined per-column (OR of IBD+Drug) → <col>_Pred_Combined + dummy <col>_Prob_Combined
  4. Column-wise evaluation (Combined)
  5. Document-level Cumulative_OR and evaluate → Cumulative_Pred_Regex vs Cumulative_Gold
  6. Collapse documents → patient level, build Patient_Has_IBD (+ Final_Prediction/Prob)
  7. Patient-level evaluation
  8. Calibration CSV (validation only)
  9. Fairness CSVs (if demographics present)
 10. Resource tracking / logs

Author: Matt Stammers / UHSFT (unified-regex version)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from codecarbon import EmissionsTracker

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.data_utils import load_and_preprocess
from nlp_pipeline.common.evaluation import evaluate
from nlp_pipeline.common.fairness import evaluate_fairness_dual
from nlp_pipeline.common.logging_setup import configure_logging
from nlp_pipeline.common.resource_monitor import ResourceMonitor
from nlp_pipeline.models.regex.model import (
    apply_regex_predictions,
    create_combined_predictions,
    train_regex_model,
)

# --------------------------------------------------------------------------- #
# Setup Utilities                                                             #
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

def _ensure_dir(path: Path) -> None:
    """Make sure that `path` exists on disk."""
    path.mkdir(parents=True, exist_ok=True)

def _add_dummy_probs(
    df: pd.DataFrame, rep_cols: Dict[str, str], tag: str
) -> pd.DataFrame:
    """
    For each report‐column in `rep_cols`, if "<col>_Pred_<tag>" exists but
    "<col>_Prob_<tag>" doesn’t, create "<col>_Prob_<tag>" as a float copy
    of the corresponding "<col>_Pred_<tag>".
    """
    for rc in rep_cols:
        pred_col = f"{rc}_Pred_{tag}"
        prob_col = f"{rc}_Prob_{tag}"
        if pred_col in df.columns and prob_col not in df.columns:
            df[prob_col] = df[pred_col].astype(float)
    return df

def _aggregate_patients(
    df: pd.DataFrame,
    pred_cols: List[str],
    *,
    id_col: str = "study_id",
) -> pd.DataFrame:
    """
    Collapse document‐level predictions to patient level:
      • max() (OR) over each column in pred_cols that actually exists in df
      • first() for any demographics in c.DEMOGRAPHICS_KEYS
    """
    dem_cols = [d for d in c.DEMOGRAPHICS_KEYS if d in df.columns]
    agg_map: dict[str, str] = {col: "max" for col in pred_cols if col in df.columns}
    agg_map.update({d: "first" for d in dem_cols})
    return df.groupby(id_col, as_index=False).agg(agg_map)

def run_regex_workflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    mode: str,
    analysis_root: Path,
    run_fairness: bool = True,
) -> pd.DataFrame:
    """
    Run one mode (“raw” or “umls”) of the unified regex pipeline.

    The only difference between “raw” and “umls” is that `use_umls=(mode=="umls")`
    is passed to `train_regex_model(...)`. All subsequent steps (predictions,
    evaluations, fairness, etc.) are identical.
    """
    logger.info(f"[{mode}] Regex workflow start")
    root = analysis_root / mode
    _ensure_dir(root)

    # --------------------------------------------------------------------------- #
    # 1. Compile IBD & Drug regex patterns                                        #
    # --------------------------------------------------------------------------- #
    ibd_patterns = {f"ibd_{i}": p for i, p in enumerate(c.IBD_KEYWORDS)}
    drug_patterns = {f"drug_{i}": p for i, p in enumerate(c.DRUG_KEYWORDS)}

    compiled, elapsed = train_regex_model(
        ibd_patterns=ibd_patterns,
        drug_patterns=drug_patterns,
        cui_patterns=None,
        note=f"Regex_Compile_{mode}",
        use_umls=(mode == "umls"),
    )
    logger.info(
        "[%s] compiled %d IBD + %d Drug patterns in %.2fs",
        mode,
        len(ibd_patterns),
        len(drug_patterns),
        elapsed,
    )

    # --------------------------------------------------------------------------- #
    # 2. Document‐level predictions (IBD + Drug) for both train & val             #
    # --------------------------------------------------------------------------- #
    for df in (train_df, val_df):
        # IBD → "<col>_Pred_IBD" + dummy "<col>_Prob_IBD"
        df = apply_regex_predictions(df, c.IBD_COLUMNS, compiled, category="IBD")
        df = _add_dummy_probs(df, c.IBD_COLUMNS, "IBD")

        # Drug → "<col>_Pred_Drug" + dummy "<col>_Prob_Drug"
        df = apply_regex_predictions(df, c.DRUG_COLUMNS, compiled, category="Drug")
        df = _add_dummy_probs(df, c.DRUG_COLUMNS, "Drug")

    # --------------------------------------------------------------------------- #
    # 3. Combined per‐column (OR of IBD + Drug) → "<col>_Pred_Combined"           #
    # --------------------------------------------------------------------------- #
    train_df = create_combined_predictions(train_df, c.IBD_COLUMNS, c.DRUG_COLUMNS)
    val_df = create_combined_predictions(val_df, c.IBD_COLUMNS, c.DRUG_COLUMNS)

    # --------------------------------------------------------------------------- #
    # 4. Combined per‐column (OR of IBD + Drug) → "<col>_Pred_Combined"           #
    # --------------------------------------------------------------------------- #
    for rc in c.COLWISE_COLUMNS:
        pred_c = f"{rc}_Pred_Combined"
        prob_c = f"{rc}_Prob_Combined"
        if pred_c in train_df.columns and prob_c not in train_df.columns:
            train_df[prob_c] = train_df[pred_c].astype(float)
        if pred_c in val_df.columns and prob_c not in val_df.columns:
            val_df[prob_c] = val_df[pred_c].astype(float)

    results: list[dict] = []

    # --------------------------------------------------------------------------- #
    # 5. Column‐wise evaluation (Combined)                                        #
    # --------------------------------------------------------------------------- #
    for df_split, split_name in (
        (train_df, "Training_Set"),
        (val_df, "Validation_Set"),
    ):
        res = evaluate(
            df_split,
            c.COLWISE_COLUMNS,
            dataset_name=split_name,
            pred_type="Combined_Regex",
            total_count=len(df_split),
        )
        results += res.to_dict(orient="records")

    # --------------------------------------------------------------------------- #
    # 6. Document‐level Cumulative OR across columns                              #
    # --------------------------------------------------------------------------- #
    cum_pred_cols = [f"{rc}_Pred_Combined" for rc in c.COLWISE_COLUMNS]
    cum_prob_cols = [f"{rc}_Prob_Combined" for rc in c.COLWISE_COLUMNS]

    for df_split, split_name in (
        (train_df, "Training_Set"),
        (val_df, "Validation_Set"),
    ):
        # --------------------------------------------------------------------------- #
        # 6a. Build "Cumulative_Gold" = OR across all gold columns that exist         #
        # --------------------------------------------------------------------------- #
        gold_cols = [v for v in c.COLWISE_COLUMNS.values() if v in df_split.columns]
        if gold_cols:
            df_split["Cumulative_Gold"] = df_split[gold_cols].any(axis=1).astype(int)
        else:
            df_split["Cumulative_Gold"] = 0

        # --------------------------------------------------------------------------- #
        # 6b. Build "Cumulative_Pred_Regex" = OR across all "<col>_Pred_Combined"     #
        # --------------------------------------------------------------------------- #
        existing_pred_cols = [col for col in cum_pred_cols if col in df_split.columns]
        if existing_pred_cols:
            df_split["Cumulative_Pred_Regex"] = (
                df_split[existing_pred_cols].max(axis=1).fillna(0).astype(int)
            )
        else:
            df_split["Cumulative_Pred_Regex"] = 0

        # --------------------------------------------------------------------------- #
        # 6c. Build "Cumulative_Prob_Regex" = OR across all "<col>_Prob_Combined"     #
        # --------------------------------------------------------------------------- #
        existing_prob_cols = [col for col in cum_prob_cols if col in df_split.columns]
        if existing_prob_cols:
            df_split["Cumulative_Prob_Regex"] = (
                df_split[existing_prob_cols].max(axis=1).fillna(0.0)
            )
        else:
            df_split["Cumulative_Prob_Regex"] = 0.0

        # --------------------------------------------------------------------------- #
        # 6d. Evaluate document‐level cumulative                                      #
        # --------------------------------------------------------------------------- #
        res = evaluate(
            df_split,
            {"Cumulative_Pred_Regex": "Cumulative_Gold"},
            dataset_name=split_name,
            pred_type="Cumulative_Regex",
            final=True,
            total_count=len(df_split),
        )
        results += res.to_dict(orient="records")

    # --------------------------------------------------------------------------- #
    # 7. Collapse to patient level (OR across documents)                          #
    # --------------------------------------------------------------------------- #
    train_doc2pat = _aggregate_patients(train_df, cum_pred_cols)
    val_doc2pat = _aggregate_patients(val_df, cum_pred_cols)

    # --------------------------------------------------------------------------- #
    # 7a. OR‐across‐documents for "Cumulative_Pred_Regex"+"Cumulative_Prob_Regex" #
    # --------------------------------------------------------------------------- #
    for grp_df, base_df in ((train_doc2pat, train_df), (val_doc2pat, val_df)):
        # Build patient‐level "Cumulative_Pred_Regex"
        if "Cumulative_Pred_Regex" in base_df.columns:
            grouped_pred = base_df.groupby("study_id")["Cumulative_Pred_Regex"].max()
            grp_df["Cumulative_Pred_Regex"] = (
                grp_df["study_id"].map(grouped_pred).fillna(0).astype(int)
            )
        else:
            grp_df["Cumulative_Pred_Regex"] = 0

        # Build patient‐level "Cumulative_Prob_Regex"
        if "Cumulative_Prob_Regex" in base_df.columns:
            grouped_prob = base_df.groupby("study_id")["Cumulative_Prob_Regex"].max()
            grp_df["Cumulative_Prob_Regex"] = (
                grp_df["study_id"].map(grouped_prob).fillna(0.0)
            )
        else:
            grp_df["Cumulative_Prob_Regex"] = 0.0

    # --------------------------------------------------------------------------- #
    # 7b. Create “Patient_Has_IBD” by OR‐ing gold‐columns at document level       #
    # --------------------------------------------------------------------------- #
    gold_cols_doc_train = [
        v for v in c.COLWISE_COLUMNS.values() if v in train_df.columns
    ]
    if gold_cols_doc_train:
        grouped_gold_train = train_df.groupby("study_id")[gold_cols_doc_train].max()
        # Now map the “max across all gold‐cols” into train_doc2pat["Patient_Has_IBD"]
        train_doc2pat["Patient_Has_IBD"] = (
            train_doc2pat["study_id"]
            .map(grouped_gold_train.max(axis=1))
            .fillna(0)
            .astype(int)
        )
    else:
        train_doc2pat["Patient_Has_IBD"] = 0

    gold_cols_doc_val = [v for v in c.COLWISE_COLUMNS.values() if v in val_df.columns]
    if gold_cols_doc_val:
        grouped_gold_val = val_df.groupby("study_id")[gold_cols_doc_val].max()
        val_doc2pat["Patient_Has_IBD"] = (
            val_doc2pat["study_id"]
            .map(grouped_gold_val.max(axis=1))
            .fillna(0)
            .astype(int)
        )
    else:
        val_doc2pat["Patient_Has_IBD"] = 0

    # --------------------------------------------------------------------------- #
    # 8. Final OR‐based patient flag & dummy probability                          #
    # --------------------------------------------------------------------------- #
    for grp_df, split_name in [
        (train_doc2pat, "Training_Set"),
        (val_doc2pat, "Validation_Set"),
    ]:
        grp_df["Final_Prediction"] = grp_df["Cumulative_Pred_Regex"]
        grp_df["Final_Prob_Regex"] = grp_df["Cumulative_Prob_Regex"]

        res = evaluate(
            grp_df,
            {"Final_Prediction": "Patient_Has_IBD"},
            dataset_name=split_name,
            pred_type="Final_Regex",
            final=True,
            total_count=len(grp_df),
        )
        results += res.to_dict(orient="records")

    # --------------------------------------------------------------------------- #
    # 9. Build patient‐level DataFrames (with demographics) for fairness          #
    # --------------------------------------------------------------------------- #
    dem_cols_train = [d for d in c.DEMOGRAPHICS_KEYS if d in train_doc2pat.columns]
    dem_cols_val = [d for d in c.DEMOGRAPHICS_KEYS if d in val_doc2pat.columns]

    final_train = train_doc2pat.loc[
        :,
        ["study_id", "Patient_Has_IBD", "Final_Prediction", "Final_Prob_Regex"]
        + dem_cols_train,
    ]
    final_val = val_doc2pat.loc[
        :,
        ["study_id", "Patient_Has_IBD", "Final_Prediction", "Final_Prob_Regex"]
        + dem_cols_val,
    ]

    # --------------------------------------------------------------------------- #
    # 10. Calibration summary (validation only)                                   #
    # --------------------------------------------------------------------------- #
    res_df = pd.DataFrame(results)
    calib_dir = root / "calibration"
    _ensure_dir(calib_dir)
    if not res_df.empty and "Validation_Set" in res_df["Dataset"].unique():
        res_df.loc[
            res_df["Dataset"] == "Validation_Set",
            ["Report_Column", "Prediction_Type", "Brier_Score"],
        ].to_csv(calib_dir / "brier_scores.csv", index=False)

    # --------------------------------------------------------------------------- #
    # 11. Fairness analysis                                                       #
    # --------------------------------------------------------------------------- #
    if run_fairness:
        fair_root = root / "fairness"
        _ensure_dir(fair_root)

        for patient_df, split_name in (
            (final_train, "Training_Set"),
            (final_val, "Validation_Set"),
        ):
            # Only keep demographic attributes that actually exist and have ≥2 groups
            valid_attrs = [
                attr
                for attr in c.DEMOGRAPHICS_KEYS
                if attr in patient_df.columns
                and patient_df[attr].nunique(dropna=True) >= 2
            ]
            if not valid_attrs:
                logger.warning(
                    "Demographic attribute(s) %s missing or ≥2 groups not found for %s – skipped.",
                    c.DEMOGRAPHICS_KEYS,
                    split_name,
                )
                continue

            split_dir = fair_root / split_name.lower()
            _ensure_dir(split_dir)

            fair: Union[pd.DataFrame, dict] = evaluate_fairness_dual(
                df=patient_df,
                final_col="Final_Prediction",
                gold_col="Patient_Has_IBD",
                demographic_attrs=valid_attrs,
                dataset_name=split_name,
                note=f"Final_Regex_{mode}",
            )

            # Persist aggregated/disaggregated CSVs
            if isinstance(fair, pd.DataFrame):
                agg_df = fair["aggregated"]
                dis_df = fair["disaggregated"]
            else:
                from pandas import DataFrame

                agg_df = DataFrame(fair["aggregated"])
                dis_df = DataFrame(fair["disaggregated"])

            agg_df.to_csv(split_dir / "aggregated.csv", index=False)
            dis_df.to_csv(split_dir / "disaggregated.csv", index=False)

            # Append fairness results to “results” so that run_regex_workflow returns them
            results += agg_df.to_dict(orient="records")
            results += dis_df.to_dict(orient="records")

    logger.info(f"[{mode}] about to return {len(results)} result rows")
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- #
# Main Entrypoint                                                             #
# --------------------------------------------------------------------------- #

def main(disable_umls: bool = False) -> None:
    """
    Entry point for the unified regex pipeline:

      • Load + preprocess both train/val (keeping combined_clinic if present)
      • Run run_regex_workflow(...) for “raw” and “umls”
      • Write out eval CSV/XLSX for each
      • Stop resource + emissions tracking and write resource CSV
    """
    analysis_root = Path(c.REGEX_ANALYSIS_DIR)
    results_dir = Path(c.FINAL_RESULTS_DIR)
    _ensure_dir(analysis_root)
    _ensure_dir(results_dir)

    configure_logging(log_dir=str(analysis_root), custom_logger=logger)
    logger.info("Starting unified Regex pipeline")

    # Start resource + emissions tracking
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()
    tracker = EmissionsTracker(
        project_name="Regex_Pipeline",
        output_dir=str(analysis_root),
        allow_multiple_runs=True,
    )
    tracker.start()

    # === 1) RAW variant (“free-text”) ===
    train_raw = load_and_preprocess(c.TRAIN_FILE_PATH, exclude_combined_clinic=False)
    val_raw = load_and_preprocess(c.VAL_FILE_PATH, exclude_combined_clinic=False)

    res_raw = run_regex_workflow(
        train_raw, val_raw, mode="raw", analysis_root=analysis_root
    )
    res_raw.to_csv(analysis_root / "eval_regex_raw.csv", index=False)
    res_raw.to_excel(results_dir / "eval_regex_raw.xlsx", index=False)

    # === 2) UMLS variant ===
    train_umls_path = Path(c.TRAIN_FILE_UMLS_PATH)
    val_umls_path = Path(c.VAL_FILE_UMLS_PATH)
    if not (train_umls_path.exists() and val_umls_path.exists()):
        # If user hasn’t already built UMLS copies, do it here:
        from nlp_pipeline.models.regex.model import apply_umls_standardisation

        text_cols = list(set(c.IBD_COLUMNS) | set(c.DRUG_COLUMNS))
        train_tmp = apply_umls_standardisation(train_raw.copy(), c.UMLS_DIR, text_cols)
        val_tmp = apply_umls_standardisation(val_raw.copy(), c.UMLS_DIR, text_cols)
        train_tmp.to_excel(train_umls_path, index=False)
        val_tmp.to_excel(val_umls_path, index=False)

    train_umls = load_and_preprocess(
        str(train_umls_path), exclude_combined_clinic=False
    )
    val_umls = load_and_preprocess(str(val_umls_path), exclude_combined_clinic=False)

    res_umls = run_regex_workflow(
        train_umls, val_umls, mode="umls", analysis_root=analysis_root
    )
    res_umls.to_csv(analysis_root / "eval_regex_umls.csv", index=False)
    res_umls.to_excel(results_dir / "eval_regex_umls.xlsx", index=False)

    # === Stop resource + emissions tracking ===
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

    logger.info("Unified Regex pipeline complete")

if __name__ == "__main__":
    main()