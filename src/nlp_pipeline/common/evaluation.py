"""
Evaluation utilities for binary classification, providing a single, flexible
entry point for column-wise, patient-level, and final OR-based evaluations.

Core components:
  - wilson_confint: Compute Wilson score intervals for binomial proportions.
  - bootstrap_confint: Compute bootstrap confidence intervals for arbitrary metrics.
  - MetricEngine: Central engine to compute metrics, calibration, Brier score,
    and confidence intervals via Wilson or bootstrap methods.
  - evaluate_model: Compute and format a standardised metrics dictionary for
    a single prediction column.
  - evaluate: Unified function to apply evaluate_model across multiple columns
    at row-level (column-wise), patient-level, or final prediction levels.
"""
import io
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import nlp_pipeline.config.constants as c
from nlp_pipeline.common.logging_setup import configure_logging

# --------------------------------------------------------------------------- #
#  Setup Custom Logger                                                        #
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Core modules to make statistical computations                              #
# --------------------------------------------------------------------------- #


def wilson_confint(k: int, n: int, alpha: float = 0.05):
    """
    Compute the Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of positive events (successes).
    n : int
        Total number of trials.
    alpha : float, optional
        Significance level (default 0.05 for 95% confidence interval).

    Returns
    -------
    p : float
        Proportion estimate (k / n).
    lo : float
        Lower bound of the confidence interval.
    hi : float
        Upper bound of the confidence interval.

    Notes
    -----
    If n == 0, returns (nan, nan, nan).
    """
    if n <= 0:
        return np.nan, np.nan, np.nan
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return p, lo, hi


# --------------------------------------------------------------------------- #
# 1.  Stratified / Robust bootstrap CI                                        #
# --------------------------------------------------------------------------- #
def bootstrap_confint(
    y_true: np.ndarray,
    y_input: np.ndarray,
    metric_fn,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
    stratify: bool = True,
    max_redraw: int = 10,
):
    """
    Percentile bootstrap CI with *optional* class-stratified resampling.

    Parameters
    ----------
    y_true, y_input
        1-D numpy arrays of identical length.
    metric_fn
        Callable returning a float metric.
    n_boot
        Number of bootstrap replicates.
    alpha
        1 - confidence level.
    random_state
        Seed for reproducibility.
    stratify
        If true, resample within each class to preserve prevalence;
        falls back to plain bootstrap when a class is absent.
    max_redraw
        Maximum redraws when a resample collapses to a single class.
    """
    rng = np.random.default_rng(random_state)
    n = y_true.size
    if n == 0:
        return np.nan, np.nan, np.nan
    # --------------------------------------------------------------
    # helper: draw indices
    # --------------------------------------------------------------
    if stratify and y_true.min() == 0 and y_true.max() == 1:
        pos_idx = np.flatnonzero(y_true == 1)
        neg_idx = np.flatnonzero(y_true == 0)
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        def draw():
            return np.concatenate(
                [
                    rng.choice(pos_idx, n_pos, replace=True),
                    rng.choice(neg_idx, n_neg, replace=True),
                ]
            )

    else:

        def draw():
            return rng.integers(0, n, n)

    # --------------------------------------------------------------
    # generate bootstrap statistics
    # --------------------------------------------------------------
    estimates = []
    for _ in range(n_boot):
        for __ in range(max_redraw):
            idx = draw()
            if stratify and len(np.unique(y_true[idx])) == 1:
                continue
            val = metric_fn(y_true[idx], y_input[idx])
            if not np.isnan(val):
                estimates.append(val)
                break

    est = metric_fn(y_true, y_input)
    if not estimates:
        return est, np.nan, np.nan

    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, lo, hi

# ----------------------------------------------------------------------------#
# 2.  MetricEngine – Full Metrics Class Object                                #
# ----------------------------------------------------------------------------#

class MetricEngine:
    """
    Primary object allowing computation of all other method calls.
    
    PR-AUC now 'requires' probabilities to prevent inappropriate triggering.

    Also formats the outputs as required.
    """

    def __init__(
        self,
        *,
        ci_method: str = "wilson",
        boot_samples: int = 1000,
        random_state: int | None = None,
        calib_bins: int = 10,
        allow_hard_pr_auc: bool = False,
    ):
        self.ci_method = ci_method
        self.boot_samples = boot_samples
        self.random_state = random_state
        self.calib_bins = calib_bins
        self.allow_hard_pr_auc = allow_hard_pr_auc

    # --------------------------------------------------------------
    def compute(self, y_true, y_pred, y_score=None):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        unique = np.unique(y_true)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "npv": tn / (tn + fn) if (tn + fn) else np.nan,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        # ---------- PR-AUC & Brier ----------
        if y_score is not None and len(unique) == 2:
            metrics["pr_auc"] = average_precision_score(y_true, y_score)
            metrics["brier"] = brier_score_loss(y_true, y_score)
        else:
            metrics["pr_auc"] = np.nan
            metrics["brier"] = np.nan

        # ---------- Calibration ----------
        if y_score is not None and len(unique) == 2:
            bins = np.linspace(0.0, 1.0, self.calib_bins + 1)
            frac, mean_pred = calibration_curve(
                y_true, y_score, n_bins=self.calib_bins, strategy="uniform"
            )
            counts, _ = np.histogram(y_score, bins=bins)
            valid = counts > 0
            metrics["ece"] = np.abs(frac - mean_pred).dot(counts[valid]) / len(y_score)
        else:
            metrics["ece"] = np.nan

        # ---------- Confidence intervals ----------
        n = len(y_true)
        cis = {
            "accuracy": wilson_confint((y_true == y_pred).sum(), n),
            "precision": wilson_confint(tp, tp + fp) if (tp + fp) else (np.nan,) * 3,
            "recall": wilson_confint(tp, tp + fn) if (tp + fn) else (np.nan,) * 3,
            "specificity": wilson_confint(tn, tn + fp) if (tn + fp) else (np.nan,) * 3,
            "npv": wilson_confint(tn, tn + fn) if (tn + fn) else (np.nan,) * 3,
            "f1": bootstrap_confint(
                y_true,
                y_pred,
                lambda a, b: f1_score(a, b, zero_division=0),
                n_boot=self.boot_samples,
                alpha=0.05,
                random_state=self.random_state,
            ),
            "mcc": bootstrap_confint(
                y_true,
                y_pred,
                matthews_corrcoef,
                n_boot=self.boot_samples,
                alpha=0.05,
                random_state=self.random_state,
            ),
        }

        # PR-AUC CI only when score-based AND two classes
        if y_score is not None and len(unique) == 2:
            cis["pr_auc"] = bootstrap_confint(
                y_true,
                y_score,
                average_precision_score,
                n_boot=self.boot_samples,
                alpha=0.05,
                random_state=self.random_state,
                stratify=True,
            )
        else:
            cis["pr_auc"] = (np.nan, np.nan, np.nan)

        return metrics, cis

    def fmt(self, est: float, lo: float, hi: float) -> str:
        """
        Format an estimate and CI as a percentage string.

        Parameters
        ----------
        est : float
            Point estimate (0.0-1.0).
        lo : float
            Lower CI bound.
        hi : float
            Upper CI bound.

        Returns
        -------
        str
            Formatted string, e.g.
            '51.23% (CI: 48.00% - 54.00%)' or '82.35% (CI: N/A)' if bounds missing.
        """
        if np.isnan(lo) or np.isnan(hi):
            return f"{est*100:.2f}% (CI: N/A)"
        return f"{est*100:.2f}% (CI: {lo*100:.2f}% - {hi*100:.2f}%)"

# --------------------------------------------------------------------------- #
# 3. Evaluate Model Handler                                                   #
# --------------------------------------------------------------------------- #

def evaluate_model(
    df: pd.DataFrame,
    label_col: str,
    pred_col: str,
    dataset_name: str = "",
    report_col: str = "",
    pred_type: str = "",
    pred_proba: str = None,
    total_count: int = None,
) -> dict:
    """
    Compute and format evaluation metrics for a single prediction column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing true labels and predictions.
    label_col : str
        Column name for true binary labels (0/1).
    pred_col : str
        Column name for predicted labels (0/1).
    dataset_name : str, optional
        Identifier (e.g. 'Training_Set').
    report_col : str, optional
        Report column name for context.
    pred_type : str, optional
        Suffix indicating prediction type (e.g. 'BOW').
    pred_proba : str or None, optional
        Column name for predicted probabilities.
    total_count : int or None, optional
        Total number of examples for coverage percentage.

    Returns
    -------
    dict
        Contains keys:
          - Dataset, Report_Column, Gold_Standard_Column, Prediction_Type,
            Demographic_Attribute, Group,
          - Accuracy, Precision, Recall, Specificity, NPV, F1_Score,
            MCC, PR_AUC, Brier_Score, ECE,
          - Coverage_Count, Coverage_Pct
    """

    if pred_proba is None or pred_proba not in df.columns:
        logger.warning(f"No probability column for '{report_col}' — disabling PR‐AUC")
        pred_proba = None

    y_true = df[label_col].astype(int).to_numpy()
    y_pred = df[pred_col].astype(int).to_numpy()
    y_score = df[pred_proba].astype(float).to_numpy() if pred_proba else None

    engine = MetricEngine(
        allow_hard_pr_auc=(pred_proba is None)  # True only when scores absent
    )
    metrics, cis = engine.compute(y_true, y_pred, y_score)

    # unpack MCC & PR-AUC CIs
    mcc_est, mcc_lo, mcc_hi = cis["mcc"]
    pr_est, pr_lo, pr_hi = cis["pr_auc"]

    result = {
        "Dataset": dataset_name,
        "Report_Column": report_col,
        "Gold_Standard_Column": label_col,
        "Prediction_Type": pred_type,
        "Demographic_Attribute": "",
        "Group": "",
        "Accuracy": engine.fmt(*cis["accuracy"]),
        "Precision": engine.fmt(*cis["precision"]),
        "Recall": engine.fmt(*cis["recall"]),
        "Specificity": engine.fmt(*cis["specificity"]),
        "NPV": engine.fmt(*cis["npv"]),
        "F1_Score": engine.fmt(*cis["f1"]),
        "MCC": f"{metrics['mcc']:.4f} (CI: {mcc_lo:.4f} - {mcc_hi:.4f})",
        "PR_AUC": f"{pr_est*100:.2f}% (CI: {pr_lo*100:.2f}% - {pr_hi*100:.2f}%)",
        "Brier_Score": f"{metrics['brier']:.4f}",
        "ECE": f"{metrics['ece']:.4f}",
        "Coverage_Count": len(y_true),
        "Coverage_Pct": (f"{len(y_true)/total_count*100:.2f}%" if total_count else ""),
    }
    return result

# --------------------------------------------------------------------------- #
# 4. Primary Model Evaluation Function                                        #
# --------------------------------------------------------------------------- #

def evaluate(
    df: pd.DataFrame,
    columns_map: dict,
    dataset_name: str,
    pred_type: str,
    total_count: int = None,
    group_level: bool = False,
    final: bool = False,
) -> pd.DataFrame:
    """
    Unified evaluation across columns, patient-level, or final predictions.

    Column-wise (default):
      - Expects columns '{col}_Pred_{pred_type}' and optional
        '{col}_Prob_{pred_type}'.

    Patient-level (group_level=True):
      - Expects columns 'Patient_{col}_Pred_{pred_type}'.

    Final (final=True):
      - Keys of columns_map serve as direct prediction-column names; values
        indicate gold-label columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing all necessary prediction and label columns.
    columns_map : dict
        Mapping from report_col -> gold_label_col (or final_pred_col if final).
    dataset_name : str
        Name of dataset for labeling (e.g. 'Training_Set').
    pred_type : str
        Prediction-type suffix (e.g. 'BOW').
    total_count : int or None, optional
        Denominator for coverage percentage.
    group_level : bool, optional
        If True, evaluate patient-level predictions.
    final : bool, optional
        If True, evaluate final OR-based predictions; keys of columns_map
        are treated as prediction-column names.

    Returns
    -------
    pd.DataFrame
        DataFrame of one row per report_col, with all standard metrics.
    """
    # To prevent test failures
    configure_logging(
        log_dir=str(c.FINAL_RESULTS_DIR),
        custom_logger=logger,
    )

    for handler in logging.root.handlers:
        stream = getattr(handler, "stream", None)
        # pytest’s live‐log uses StringIO without .section()
        if isinstance(stream, io.StringIO):
            handler.setLevel(logging.WARNING)
            if not hasattr(stream, "section"):
                stream.section = lambda *args, **kwargs: None

    # Final result calculation
    results = []
    for report_col, label_col in columns_map.items():
        # Determine column names
        if final:
            pred_col = report_col
            # look for a same-name probability column
            prob_cand = report_col.replace("_Pred_", "_Prob_")
            proba_col = prob_cand if prob_cand in df.columns else None

        elif group_level:
            pred_col = f"Patient_{report_col}_Pred_{pred_type}"
            prob_cand = f"Patient_{report_col}_Prob_{pred_type}"
            proba_col = prob_cand if prob_cand in df.columns else None
        else:
            pred_col = f"{report_col}_Pred_{pred_type}"
            proba_col = f"{report_col}_Prob_{pred_type}"

        # FALLBACK: try other common suffixes
        if proba_col is None:
            for alt in ("Score", "Probability"):
                cand = f"{report_col}_{alt}_{pred_type}"
                if cand in df.columns:
                    proba_col = cand
                    logger.warning(
                        f"Using alternate probability column '{cand}' for '{report_col}'"
                    )
                    break

        # Skip if necessary columns are missing
        if label_col not in df.columns or pred_col not in df.columns:
            logger.warning(f"Skipping '{report_col}'—missing columns")
            continue

        # Filter out rows with NaNs in required columns
        req_cols = [label_col, pred_col]
        if proba_col and proba_col in df.columns:
            req_cols.append(proba_col)
        subset = df.dropna(subset=req_cols)
        if subset.empty:
            logger.warning(f"No data for '{report_col}' in '{dataset_name}'")
            continue

        # Perform evaluation
        row = evaluate_model(
            subset,
            label_col=label_col,
            pred_col=pred_col,
            dataset_name=dataset_name,
            report_col=report_col,
            pred_type=pred_type,
            pred_proba=proba_col,
            total_count=total_count,
        )
        results.append(row)

    return pd.DataFrame(results)
