
"""
Combined LLM Batch Processor and Analytics Pipeline

Steps:
1. Combine all batch_*.xls* files into a single Excel file per LLM/sequence folder
2. Extract and repair JSON responses
3. Parse relevant fields from JSON
4. Add IBD prediction column (Likelihood ≥ 5)
5. Save enriched Excel file
6. Print summary statistics
7. Merge LLM predictions with human-annotated data for evaluation
8. Perform evaluation and fairness analysis

Author: Matt Stammers (UHSFT)
"""

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

import logging
import sys
from pathlib import Path

# Set project root for import access
default_project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(default_project_root / "src"))
print(default_project_root)

# Logging config
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

import json
import re

import numpy as np
import pandas as pd

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.data_utils import load_and_preprocess
from nlp_pipeline.common.evaluation import evaluate
from nlp_pipeline.common.fairness import (
    evaluate_fairness_dual,
    plot_f1_scores_disaggregated,
    plot_fairness_metrics_aggregated,
)

# --------------------------------------------------------------------------- #
# JSON Handlers                                                               #
# --------------------------------------------------------------------------- #

def extract_and_repair_json(text):
    """Attempts to repair and parse a JSON blob from a string."""
    try:
        match = re.search(r"\{.*\}", str(text), re.DOTALL)
        if not match:
            return "No processable JSON response from API"
        j = match.group()
        j = re.sub(r"//.*", "", j)
        j = re.sub(r",\s*}", "}", j)
        j = re.sub(r",\s*]", "]", j)
        return json.loads(j)
    except (json.JSONDecodeError, TypeError):
        return "No processable JSON response from API"

def parse_json_response(js, response_num=1):
    """Extracts structured fields from JSON dict."""
    if isinstance(js, dict):
        return {
            f"Title_{response_num}": js.get("Title", np.nan),
            f"Features_{response_num}": ", ".join(js.get("Features", [])),
            f"Likelihood_of_IBD_{response_num}": js.get("Likelihood of IBD", np.nan),
            f"Certainty_Level_{response_num}": js.get("Certainty Level", np.nan),
            f"Complexity_of_Case_{response_num}": js.get("Complexity of Case", np.nan),
        }
    else:
        return {
            k: np.nan
            for k in [
                f"Title_{response_num}",
                f"Features_{response_num}",
                f"Likelihood_of_IBD_{response_num}",
                f"Certainty_Level_{response_num}",
                f"Complexity_of_Case_{response_num}",
            ]
        }

def is_valid_json(txt):
    try:
        json.loads(str(txt))
        return True
    except Exception:
        return False

# --------------------------------------------------------------------------- #
# Project Root Finder                                                         #
# --------------------------------------------------------------------------- #

def find_project_root():
    """Locates the project root by checking for data/llms folder."""
    root = default_project_root
    for p in [root, root.parent]:
        if (p / "data" / "llms").exists():
            return p
    raise FileNotFoundError(
        "Could not locate data/llms in current or parent directory."
    )

# --------------------------------------------------------------------------- #
# Batch Processing                                                            #
# --------------------------------------------------------------------------- #

def combine_and_process_llm_batches(
    project_root: Path = None, output_suffix="_parsed_results.xlsx"
):
    """
    Full LLM processing pipeline:
    - Combines batch files
    - Repairs and extracts JSON
    - Parses fields
    - Adds prediction
    - Writes output
    """
    project_root = project_root or default_project_root
    llm_root = project_root / "data" / "llms"
    results = {}

    if not llm_root.exists():
        logging.warning(f"LLM root not found: {llm_root}, creating it.")
        llm_root.mkdir(parents=True, exist_ok=True)

    for llm_folder in sorted(llm_root.iterdir()):
        if not llm_folder.is_dir():
            continue

        # Find the first matching sequence folder
        seq_dirs = [
            p
            for p in llm_folder.iterdir()
            if p.is_dir() and "sequence" in p.name.lower()
        ]
        if not seq_dirs:
            logging.warning(f"No sequence folder in {llm_folder.name}")
            continue
        sequence_dir = seq_dirs[0]

        # Find batch files
        batch_files = sorted(
            sequence_dir.glob("batch_*.xls*"), key=lambda p: p.name.lower()
        )
        if not batch_files:
            logging.warning(f"No batch files in {sequence_dir.name}")
            continue

        # Combine batch files
        logging.info(f"Combining {len(batch_files)} batch files from {sequence_dir}...")
        combined_df = pd.concat(
            [pd.read_excel(fp) for fp in batch_files], ignore_index=True
        )

        # Write intermediate file
        concat_filename = (
            f"{llm_folder.name}_{sequence_dir.name}_concatenated_results.xlsx"
        )
        concat_path = sequence_dir / concat_filename
        if concat_path.exists():
            concat_path.unlink()
        combined_df.to_excel(concat_path, index=False)
        logging.info(f"[OK] Wrote concatenated file: {concat_path.name}")

        # Identify JSON and full response columns
        df = combined_df.copy()
        json_cols = [c for c in df.columns if "json_response" in c.lower()]
        full_cols = [c for c in df.columns if "full_response" in c.lower()]
        if not json_cols or not full_cols:
            logging.warning(
                f"Missing JSON or Full response columns in {concat_path.name}, skipping."
            )
            continue
        json_col, full_col = json_cols[0], full_cols[0]

        # Extract and repair JSON
        no_repair_mask = df[json_col].apply(is_valid_json)
        df["Final_JSON"] = [
            json.loads(str(orig)) if valid else extract_and_repair_json(full)
            for valid, orig, full in zip(no_repair_mask, df[json_col], df[full_col])
        ]

        # Parse JSON into structured columns
        parsed_df = pd.DataFrame(
            df["Final_JSON"].apply(lambda x: parse_json_response(x, 1)).tolist()
        )
        df_out = pd.concat([df, parsed_df], axis=1)

        # Add binary prediction
        df_out["IBD_Predicted"] = df_out["Likelihood_of_IBD_1"].apply(
            lambda x: 1 if pd.to_numeric(x, errors="coerce") >= 5 else 0
        )

        # Save final parsed file
        parsed_filename = f"{llm_folder.name}_{sequence_dir.name}{output_suffix}"
        parsed_path = sequence_dir / parsed_filename
        parsed_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_excel(parsed_path, index=False)
        logging.info(f"[OK] Wrote parsed/enriched file: {parsed_path.name}")

        # Stats
        total = len(df_out)
        unique_patients = (
            df_out["study_id_old"].nunique() if "study_id_old" in df_out else "n/a"
        )
        no_response_count = int(
            df_out["Final_JSON"].apply(lambda x: not isinstance(x, dict)).sum()
        )
        no_repair_count = int(no_repair_mask.sum())
        repaired_count = total - no_repair_count - no_response_count
        ibd_predicted_count = df_out["IBD_Predicted"].sum()

        def fmt(count):
            return f"{count}/{total} ({(count / total * 100):.1f}%)"

        print(f"\n[{llm_folder.name}/{sequence_dir.name}] Summary:")
        print(f" • Unique patients:            {unique_patients}")
        print(f" • No repair needed:           {fmt(no_repair_count)}")
        print(f" • JSON repaired:              {fmt(repaired_count)}")
        print(f" • No response:                {fmt(no_response_count)}")
        print(f" • Total successful responses: {fmt(total - no_response_count)}")
        print(f" • IBD predicted (≥5):         {fmt(ibd_predicted_count)}")

        results[f"{llm_folder.name}_{sequence_dir.name}"] = df_out

    return results

# --------------------------------------------------------------------------- #
# Human Validated Gold Standard Data Loading                                  #
# --------------------------------------------------------------------------- #

def load_human_base():
    logging.info("Loading human TRAIN and VALIDATION data.")
    train_df = load_and_preprocess(c.TRAIN_FILE_PATH, exclude_combined_clinic=False)
    val_df = load_and_preprocess(c.VAL_FILE_PATH, exclude_combined_clinic=False)
    df = pd.concat([train_df, val_df], ignore_index=True)

    if "study_id_old" in df.columns:
        df.rename(columns={"study_id_old": "study_id"}, inplace=True)

    df["endoscopy_date"] = pd.to_datetime(df["endoscopy_date"])
    logging.info(f"Combined human base: {len(df)} rows.")
    return df

# --------------------------------------------------------------------------- #
# Merge LLM and Human Validated Data                                          #
# --------------------------------------------------------------------------- #

def merge_and_save_llm():
    project_root = find_project_root()
    human_df = load_human_base()

    if "IBD_Predicted" in human_df.columns:
        human_df.drop(columns=["IBD_Predicted"], inplace=True)
        logging.info("Removed existing IBD_Predicted column from human base.")

    llm_models = [
        "deepseek32b_t0_60",
        "deepseek70b_t0_60",
        "m42_8b_t0_75",
        "mixtral_7b_t0_75",
        "qwen32b_t0_60",
    ]

    drop_cols = [
        "result_report",
        "combined_text",
        "Combined_Content",
        "preceding_clinic_letter_llm",
        "following_clinic_letter_llm",
        "histopathology_report",
        "endoscopy_report",
        "preceding_clinic_letter_human",
        "following_clinic_letter_human",
        "combined_clinic_letters",
    ]

    for model_name in llm_models:
        seq_folder = "zero_shot_all_docs_in_sequence"
        parsed_dir = project_root / "data" / "llms" / model_name / seq_folder
        parsed_file = parsed_dir / f"{model_name}_{seq_folder}_parsed_results.xlsx"
        merged_file = parsed_dir / f"{model_name}_{seq_folder}_merged_human_llm.xlsx"
        merged_file.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n--- {model_name} ---")
        if not parsed_file.exists():
            logging.warning(f"Parsed file not found: {parsed_file}")
            continue

        llm_df = pd.read_excel(parsed_file)
        logging.info(f"LLM rows before dedup: {len(llm_df)}")

        if "study_id_old" in llm_df.columns:
            llm_df.rename(columns={"study_id_old": "study_id"}, inplace=True)
        llm_df["procedure_date"] = pd.to_datetime(llm_df["procedure_date"])

        llm_df = llm_df.drop_duplicates(subset=["study_id", "procedure_date"])
        logging.info(f"LLM rows after dedup: {len(llm_df)}")

        merged = human_df.merge(
            llm_df,
            left_on=["study_id", "endoscopy_date"],
            right_on=["study_id", "procedure_date"],
            how="inner",
            suffixes=("_human", "_llm"),
        )
        logging.info(f"Merged rows before dedup: {len(merged)}")

        merged = merged.drop_duplicates(subset=["study_id", "endoscopy_date"])
        logging.info(f"Merged rows after dedup: {len(merged)}")

        merged_clean = merged.drop(columns=drop_cols, errors="ignore")
        dropped = [col for col in drop_cols if col in merged.columns]
        if dropped:
            logging.info(f"Dropped columns: {', '.join(dropped)}")

        if {"Final_JSON", "IBD_Predicted"}.issubset(merged_clean.columns):
            mask_fail = (
                merged_clean["Final_JSON"] == "No processable JSON response from API"
            )
            merged_clean.loc[mask_fail, "IBD_Predicted"] = np.nan
            logging.info(
                f"Set IBD_Predicted=NaN for {mask_fail.sum()} failed API responses."
            )

        merged_clean.to_excel(merged_file, index=False)
        logging.info(f"Saved merged file: {merged_file}")

# --------------------------------------------------------------------------- #
# Analytics & Fairness                                                        #
# --------------------------------------------------------------------------- #

def analyze_llm(model: str, root: Path) -> None:
    seq = "zero_shot_all_docs_in_sequence"
    merged = (
        root / "data" / "llms" / model / seq / f"{model}_{seq}_merged_human_llm.xlsx"
    )
    if not merged.exists():
        logging.error(f"[{model}] Missing file: {merged}")
        return

    logging.info(f"[{model}] Loading {merged.name}")
    df = pd.read_excel(merged)

    df["Final_Prediction"] = df["IBD_Predicted"].astype(float)

    # Ensure required demographic columns exist as strings
    for col in ("age_group", "imd_group", "gender", "ethnicity"):
        if col not in df.columns:
            logging.warning(f"[{model}] Column '{col}' absent – fairness will skip it.")
        else:
            df[col] = df[col].astype(str)

    # ---------------- Evaluation ---------------- #
    gold_col = getattr(c, "GOLD_STANDARD_COLUMN", "Patient_Has_IBD")

    # The new evaluate() expects a mapping: pred_col → gold_col
    eval_df = evaluate(
        df=df,
        columns_map={"Final_Prediction": gold_col},
        dataset_name="LLM_Validation",
        pred_type="",
        final=True,
        total_count=len(df),
    )

    # ---------------- Fairness ------------------ #
    demo_cols = ["age_group", "ethnicity", "gender", "imd_group"]
    fair = evaluate_fairness_dual(
        df=df,
        final_col="Final_Prediction",
        gold_col=gold_col,
        demographic_attrs=demo_cols,
        dataset_name="LLM_Validation",
        note="Final_Prediction",
    )
    agg_df = pd.DataFrame(fair["aggregated"])
    disagg_df = pd.DataFrame(fair["disaggregated"])

    # ---------------- Save Outputs ---------------- #
    out_root = root / c.LLM_ANALYSIS_DIR / model
    fair_root = out_root / "fairness"
    plots_agg = fair_root / "aggregated_plots"
    plots_dis = fair_root / "disaggregated_plots"
    for p in (out_root, fair_root, plots_agg, plots_dis):
        p.mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(out_root / "llm_final_evaluation.csv", index=False)
    eval_df.to_excel(out_root / "llm_final_evaluation.xlsx", index=False)

    agg_df.to_csv(fair_root / "fairness_aggregated.csv", index=False)
    disagg_df.to_csv(fair_root / "fairness_disaggregated.csv", index=False)
    agg_df.to_excel(fair_root / "fairness_aggregated.xlsx", index=False)
    disagg_df.to_excel(fair_root / "fairness_disaggregated.xlsx", index=False)

    # ---------------- Plots ----------------
    plot_fairness_metrics_aggregated(agg_df, str(plots_agg))
    plot_f1_scores_disaggregated(disagg_df, str(plots_dis))

    logging.info(f"[{model}] Metrics & plots saved to {out_root}")

# --------------------------------------------------------------------------- #
# Entry Point                                                                 #
# --------------------------------------------------------------------------- #

def main() -> None:
    logging.info("Starting combined LLM pipeline")
    # Batch processing
    combine_and_process_llm_batches()
    # Merge with human data
    merge_and_save_llm()
    # Analytics and fairness evaluation
    root = find_project_root()
    for mdl in (
        "deepseek32b_t0_60",
        "deepseek70b_t0_60",
        "m42_8b_t0_75",
        "mixtral_7b_t0_75",
        "qwen32b_t0_60",
    ):
        analyze_llm(mdl, root)
    logging.info("Combined pipeline complete.")


if __name__ == "__main__":
    main()
