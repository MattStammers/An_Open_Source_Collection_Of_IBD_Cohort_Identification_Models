"""
Fairness-Aware Evaluation Utilities
===================================

This module computes both *performance* and *fairness* metrics for binary-
classification tasks.  It is designed to sit on top of the core evaluation
functions provided in ``nlp_pipeline.common.evaluation``

Main capabilities
-----------------
1.  **Group-fairness metrics**
    • *Demographic Parity* (rate ratios)
    • *Equal Opportunity* (TPR gaps)
    • *Disparate Impact* (rate ratios, 80 % rule)

2.  **Performance metrics with confidence intervals** for every group and
    for the full population, including Accuracy, Precision, Recall,
    Specificity, NPV, F1, and MCC.

3.  **Dual-level reporting** - aggregated (whole attribute) *and*
    disaggregated (per subgroup).

4.  **Turn-key plotting helpers** for aggregated/disaggregated fairness
    metrics and subgroup F1 scores.

The only global constant is :pydata:`REFERENCE_GROUPS`, defining the
“privileged” (reference) group for each protected attribute.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from nlp_pipeline.common.evaluation import bootstrap_confint, wilson_confint

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
REFERENCE_GROUPS: Dict[str, str] = {
    "age_group": "20-30",
    "gender": "F",
    "ethnicity": "Black",
    "imd_group": "1-2",
}
"""Default reference (privileged) group for each protected attribute."""

# -------------------------------------------------------------------- #
# Local helpers to keep everything self-contained for testing          #
# -------------------------------------------------------------------- #

# Stub pytest’s live-log StringIO streams
for _handler in logging.root.handlers:
    _stream = getattr(_handler, "stream", None)
    if isinstance(_stream, io.StringIO) and not hasattr(_stream, "section"):
        _handler.setLevel(logging.WARNING)
        _stream.section = lambda *args, **kwargs: None


def _fmt_pct_ci(est: float, lo: float, hi: float) -> str:
    """Return '12.34 % (CI: 10.00 % – 15.00 %)' or '… (CI: N/A)'."""
    return (
        f"{est*100:.2f}% (CI: {lo*100:.2f}% - {hi*100:.2f}%)"
        if not (np.isnan(lo) or np.isnan(hi))
        else f"{est*100:.2f}% (CI: N/A)"
    )

def _specificity_npv(tn: int, fp: int, fn: int) -> tuple[float, float]:
    """Compute specificity and NPV from confusion-matrix cells."""
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return spec, npv


def _bootstrap_f1_ci(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for F1 score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return bootstrap_confint(
        y_true,
        y_pred,
        lambda a, b: f1_score(a, b, zero_division=0),
        n_boot=n_boot,
        alpha=alpha,
        random_state=random_state,
    )

# --------------------------------------------------------------------------- #
#  Fairness metrics                                                           #
# --------------------------------------------------------------------------- #

def demographic_parity(
    df: pd.DataFrame,
    protected_attr: str,
    prediction_col: str,
    reference_group: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute *Demographic Parity* rate ratios.

    Notes
    -----
    If a *reference group* is not supplied, the first group with a **non-zero**
    positive-prediction rate is selected; if **all** groups have zero events,
    an empty dict is returned.

    Parameters
    ----------
    df :
        Dataset with predictions and demographic column.
    protected_attr :
        Column holding the protected attribute (e.g. ``"gender"``).
    prediction_col :
        Binary prediction column (0/1).
    reference_group :
        Group to use as denominator.  If absent or rate == 0, a fallback is
        chosen automatically.

    Returns
    -------
    dict
        Keys follow the pattern
        ``Demographic_Parity_<attr>_Group_<grp>_vs_<ref>`` with *rate ratios*
        as values.
    """
    rates = df.groupby(protected_attr, observed=False)[prediction_col].mean()
    groups = rates.index.tolist()

    if len(groups) < 2:
        logging.warning(
            "Protected attribute '%s' has <2 groups - cannot compute "
            "Demographic Parity.",
            protected_attr,
        )
        return {}

    # pick reference
    ref = (
        reference_group
        if reference_group in rates.index and rates.loc[reference_group] > 0
        else next((g for g in groups if rates.loc[g] > 0), None)
    )
    if ref is None:
        logging.warning(
            "All groups in '%s' have zero positives - Demographic Parity " "undefined.",
            protected_attr,
        )
        return {}

    parity: Dict[str, float] = {}
    for grp in groups:
        if grp == ref:
            continue
        key = f"Demographic_Parity_{protected_attr}_Group_{grp}_vs_{ref}"
        parity[key] = rates.loc[grp] - rates.loc[ref]
    return parity


def equal_opportunity(
    df: pd.DataFrame,
    protected_attr: str,
    prediction_col: str,
    true_col: str,
    reference_group: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute *Equal Opportunity* gaps (absolute TPR differences).

    Parameters
    ----------
    df :
        Dataset with ground truth and predictions.
    protected_attr :
        Demographic attribute.
    prediction_col :
        Binary prediction column.
    true_col :
        Ground-truth label column.
    reference_group :
        Group taken as baseline.  If ``None`` the first (sorted) group is used.

    Returns
    -------
    dict
        Keys are
        ``Equal_Opportunity_<attr>_Group_<grp>_vs_<ref>``; values are
        ``|TPR_grp - TPR_ref|``.
    """
    groups = df[protected_attr].unique()
    tprs: Dict[Any, float] = {}

    for grp in groups:
        df_grp = df[(df[protected_attr] == grp) & (df[true_col] == 1)]
        if df_grp.empty:
            logging.warning(
                "No positive instances for group '%s' in Equal Opportunity "
                "calculation.",
                grp,
            )
            tprs[grp] = np.nan
            continue

        try:
            tn, fp, fn, tp = confusion_matrix(
                df_grp[true_col], df_grp[prediction_col], labels=[0, 1]
            ).ravel()
            tprs[grp] = tp / (tp + fn) if (tp + fn) else 0.0
        except ValueError:  # single-class collapse
            tprs[grp] = 0.0

    ref = reference_group if reference_group in tprs else sorted(tprs)[0]

    results: Dict[str, float] = {}
    for grp, tpr in tprs.items():
        if grp == ref:
            continue
        key = f"Equal_Opportunity_{protected_attr}_Group_{grp}_vs_{ref}"
        results[key] = abs(tpr - tprs[ref])
    return results


def disparate_impact(
    df: pd.DataFrame,
    protected_attr: str,
    prediction_col: str,
    reference_group: Optional[str] = None,
) -> Dict[str, Union[float, np.floating]]:
    """
    Compute *Disparate Impact* rate ratios.

    The metric is commonly compared against the “80 % rule”.

    Parameters
    ----------
    df, protected_attr, prediction_col :
        See :func:`demographic_parity`.
    reference_group :
        Mandatory - a disparate-impact ratio is undefined without an explicit
        reference.

    Returns
    -------
    dict
        Keys are
        ``Disparate_Impact_<attr>_Group_<grp>_vs_<ref>``; values are rate
        ratios (NaN if the reference rate is 0).
    """
    rates = df.groupby(protected_attr, observed=False)[prediction_col].mean()
    if reference_group not in rates.index:
        logging.warning(
            "Reference group '%s' not found in '%s' - Disparate Impact skipped.",
            reference_group,
            protected_attr,
        )
        return {}

    ref_rate = rates.loc[reference_group]
    results: Dict[str, Union[float, np.floating]] = {}
    for grp, rate in rates.items():
        if grp == reference_group:
            continue
        key = f"Disparate_Impact_{protected_attr}_Group_{grp}_vs_{reference_group}"
        results[key] = rate / ref_rate if ref_rate > 0 else np.nan
    if ref_rate == 0:
        logging.warning(
            "Reference group '%s' has zero positive predictions - "
            "Disparate Impact ratios set to NaN.",
            reference_group,
        )
    return results

# --------------------------------------------------------------------------- #
#  Core helper – joint perf/fairness computation                              #
# --------------------------------------------------------------------------- #

def _compute_fairness_metrics(
    df_subset: pd.DataFrame,
    demo_attr: str,
    final_col: str,
    gold_col: str,
    dataset_name: str,
    note: str,
    group_label: Any,
) -> Dict[str, Any]:
    """
    Compute **all** performance + fairness metrics for *one* dataset slice.

    The slice can be the entire dataset (``group_label == "all"``) or a single
    subgroup.

    Returns
    -------
    dict
        Keys match the schema used in the original pipeline.
    """
    # --- basic performance -------------------------------------------------- #
    y_true = df_subset[gold_col].astype(int)
    y_pred = df_subset[final_col].astype(int)
    n = len(y_true)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:  # single-class slice
        if (y_true == 1).all():
            tn, fp, fn, tp = 0, 0, 0, n
        else:
            tn, fp, fn, tp = n, 0, 0, 0

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    spec, npv = _specificity_npv(tn, fp, fn)

    # --- confidence intervals (Wilson + bootstrap) ------------------------- #
    acc_p, acc_lo, acc_hi = wilson_confint((y_true == y_pred).sum(), n)
    prec_p, prec_lo, prec_hi = (
        wilson_confint(tp, tp + fp) if (tp + fp) else (np.nan, np.nan, np.nan)
    )
    rec_p, rec_lo, rec_hi = (
        wilson_confint(tp, tp + fn) if (tp + fn) else (np.nan, np.nan, np.nan)
    )
    spec_p, spec_lo, spec_hi = (
        wilson_confint(tn, tn + fp) if (tn + fp) else (np.nan, np.nan, np.nan)
    )
    npv_p, npv_lo, npv_hi = (
        wilson_confint(tn, tn + fn) if (tn + fn) else (np.nan, np.nan, np.nan)
    )

    f1_obs, f1_lo, f1_hi = _bootstrap_f1_ci(y_true, y_pred, alpha=0.05)
    f1_str = (
        f"{f1_obs*100:.2f}% (CI: {f1_lo*100:.2f}% - {f1_hi*100:.2f}%)"
        if not np.isnan(f1_lo) and not np.isnan(f1_hi)
        else f"{f1_obs*100:.2f}% (CI: N/A)"
    )

    # --- fairness metrics --------------------------------------------------- #
    ref_value = REFERENCE_GROUPS.get(demo_attr)
    dp = demographic_parity(df_subset, demo_attr, final_col, reference_group=ref_value)
    eo = equal_opportunity(
        df_subset, demo_attr, final_col, gold_col, reference_group=ref_value
    )
    di = disparate_impact(df_subset, demo_attr, final_col, reference_group=ref_value)

    return {
        "Dataset": dataset_name,
        "Report_Column": note,
        "Gold_Standard_Column": gold_col,
        "Prediction_Type": "Final",
        "Demographic_Attribute": demo_attr,
        "Group": group_label,
        "Accuracy": _fmt_pct_ci(acc_p, acc_lo, acc_hi),
        "Precision": _fmt_pct_ci(prec_p, prec_lo, prec_hi),
        "Recall": _fmt_pct_ci(rec_p, rec_lo, rec_hi),
        "Specificity": _fmt_pct_ci(spec_p, spec_lo, spec_hi),
        "NPV": _fmt_pct_ci(npv_p, npv_lo, npv_hi),
        "F1_Score": f1_str,
        "MCC": f"{mcc:.4f}",
        "PR_AUC": "N/A",  # not available in this context
        "Coverage_Count": n,
        "Coverage_Pct": "",  # can be filled upstream if desired
        "Demographic_Parity": json.dumps(dp),
        "Equal_Opportunity": json.dumps(eo),
        "Disparate_Impact": json.dumps(di),
    }

# --------------------------------------------------------------------------- #
#  Public API – dual-level fairness evaluation                                #
# --------------------------------------------------------------------------- #

def evaluate_fairness_dual(
    df: pd.DataFrame,
    final_col: str,
    gold_col: str,
    demographic_attrs: List[str],
    dataset_name: str,
    note: str = "Final_Prediction",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Evaluate fairness on *aggregated* and *disaggregated* levels.

    Parameters
    ----------
    df :
        Input DataFrame with predictions, labels and demographic attributes.
    final_col :
        Column holding **final** binary predictions (0/1).
    gold_col :
        Column holding ground-truth labels (0/1).
    demographic_attrs :
        List of attribute names to evaluate (e.g. ``["gender", "ethnicity"]``).
    dataset_name :
        Tag identifying the dataset split (``"Train"``, ``"Test"``, ...).
    note :
        Report column name - kept for backward compatibility.

    Returns
    -------
    dict
        ``{"aggregated": [...], "disaggregated": [...]}``, where each value is
        a list of metric dictionaries ready to be turned into a DataFrame.
    """
    aggregated, disaggregated = [], []

    for attr in demographic_attrs:
        if attr not in df.columns:
            logging.warning("Demographic attribute '%s' missing - skipped.", attr)
            continue

        df_attr = df.dropna(subset=[attr, final_col, gold_col])
        if df_attr.empty:
            logging.warning(
                "No data for attribute '%s' in dataset '%s'.", attr, dataset_name
            )
            continue

        # aggregated
        aggregated.append(
            _compute_fairness_metrics(
                df_attr,
                attr,
                final_col,
                gold_col,
                dataset_name,
                note,
                group_label="all",
            )
        )

        # disaggregated
        for grp in df_attr[attr].unique():
            df_grp = df_attr[df_attr[attr] == grp]
            if not df_grp.empty:
                disaggregated.append(
                    _compute_fairness_metrics(
                        df_grp,
                        attr,
                        final_col,
                        gold_col,
                        dataset_name,
                        note,
                        group_label=grp,
                    )
                )

    return {"aggregated": aggregated, "disaggregated": disaggregated}

# --------------------------------------------------------------------------- #
#  Plotting Helpers                                                           #
# --------------------------------------------------------------------------- #

def plot_fairness_metrics_aggregated(
    results_df: pd.DataFrame,
    analysis_folder: str,
    report_filter: str = "Final_Prediction",
) -> None:
    """
    Bar-plot aggregated fairness metrics (``Group == "all"``)."""
    _plot_fairness(
        results_df[results_df["Group"] == "all"],
        analysis_folder,
        suffix="aggregated",
        report_filter=report_filter,
    )

def plot_fairness_metrics_disaggregated(
    results_df: pd.DataFrame,
    analysis_folder: str,
    report_filter: str = "Final_Prediction",
) -> None:
    """
    Bar-plot disaggregated fairness metrics (subgroup level)."""
    _plot_fairness(
        results_df[results_df["Group"] != "all"],
        analysis_folder,
        suffix="disaggregated",
        report_filter=report_filter,
    )

def _plot_fairness(
    results_df: pd.DataFrame,
    analysis_folder: str,
    suffix: str,
    report_filter: str = "Final_Prediction",
) -> None:
    """
    Internal helper to visualise fairness metrics.

    The function looks for JSON-encoded columns *Demographic_Parity*,
    *Equal_Opportunity*, *Disparate_Impact*, expands them and produces one bar
    chart per attribute & metric.
    """
    metrics = ["Demographic_Parity", "Equal_Opportunity", "Disparate_Impact"]

    for metric in metrics:
        if metric not in results_df:
            continue

        df_metric = results_df[results_df["Report_Column"] == report_filter].copy()
        logging.info(
            "[%s] %s rows after Report_Column filter.", suffix, df_metric.shape[0]
        )
        if df_metric.empty:
            continue

        # explode JSON into wide columns
        df_metric.dropna(subset=[metric], inplace=True)
        expanded = df_metric[metric].apply(json.loads).apply(pd.Series)
        fairness_data = df_metric[["Demographic_Attribute", "Group"]].join(expanded)

        for attr in fairness_data["Demographic_Attribute"].unique():
            melted = fairness_data[fairness_data["Demographic_Attribute"] == attr].melt(
                id_vars=["Demographic_Attribute", "Group"],
                var_name="Comparison",
                value_name="Value",
            )
            if melted.empty:
                continue

            def _simplify(key: str) -> str:
                """Shorten verbose JSON keys for x-axis labels."""
                try:
                    parts = key.split("_Group_")
                    return parts[1].replace("_vs_", " vs ")  # e.g. "M vs F"
                except Exception:
                    return key

            melted["Label"] = melted["Comparison"].apply(_simplify)

            plt.figure(figsize=(12, 8))
            ax = sns.barplot(data=melted, x="Label", y="Value", hue="Group")
            ax.set(
                title=f"{metric.replace('_', ' ')} – {attr} ({suffix})",
                xlabel="Group comparison",
                ylabel=metric.replace("_", " "),
            )
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            for container in ax.containers:
                ax.bar_label(container, fmt="%.2f")

            path = os.path.join(
                analysis_folder, f"{metric}_{attr}_{suffix}_fairness_plot.png"
            )
            plt.savefig(path)
            plt.close()
            logging.info("[%s] Saved plot → %s", suffix, path)

def plot_f1_scores_disaggregated(
    results_df: pd.DataFrame,
    output_folder: str,
) -> None:
    """
    Plot subgroup F1 scores extracted from *disaggregated* results.

    Parameters
    ----------
    results_df :
        Disaggregated result DataFrame returned by
        :func:`evaluate_fairness_dual` (``results["disaggregated"]`` →
        ``pd.DataFrame``).
    output_folder :
        Directory to save figures.  Created if absent.
    """
    os.makedirs(output_folder, exist_ok=True)

    def _extract_pct(text: str) -> float:
        try:
            return float(text.split("%")[0])
        except Exception:
            return np.nan

    for attr in results_df["Demographic_Attribute"].unique():
        sub = results_df[results_df["Demographic_Attribute"] == attr].copy()
        sub["F1_numeric"] = sub["F1_Score"].apply(_extract_pct)

        # natural (numeric) sort if possible
        sub["Group_num"] = pd.to_numeric(sub["Group"], errors="coerce")
        sub.sort_values(["Group_num", "Group"], inplace=True)

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=sub, x="Group", y="F1_numeric")
        ax.set(title=f"F1 scores – {attr}", xlabel="Subgroup", ylabel="F1 (%)")
        plt.ylim(0, 100)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f")
        path = os.path.join(output_folder, f"F1_scores_{attr}_disaggregated.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logging.info("Saved F1-score plot -> %s", path)