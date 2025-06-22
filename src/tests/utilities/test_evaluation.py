"""

test evaluation.py

Tests for the evaluation functions

"""

import io
import logging
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from nlp_pipeline.common.evaluation import (
    MetricEngine,
    bootstrap_confint,
    evaluate,
    evaluate_model,
    wilson_confint,
)

# Stub StringIO streams to support pytest live log stream.section
for handler in logging.root.handlers:
    stream = getattr(handler, "stream", None)
    if isinstance(stream, io.StringIO) and not hasattr(stream, "section"):
        stream.section = lambda *args, **kwargs: None

# Enable propagation so caplog can capture warnings
logger_eval = logging.getLogger("nlp_pipeline.common.evaluation")
logger_eval.propagate = True

# --------------------------------------------------------------------------- #
# Test: Wilson confidence interval behavior                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "k,n,alpha,expected_p",
    [
        (0, 0, 0.05, np.nan),  # no trials
        (5, 10, 0.05, 0.5),  # simple proportion
        (0, 10, 0.05, 0.0),  # zero successes
    ],
)
def test_wilson_confint_basic(k, n, alpha, expected_p):
    """
    Validates that wilson_confint returns correct proportion p=k/n and handles n=0.
    """
    p, lo, hi = wilson_confint(k, n, alpha)
    if math.isnan(expected_p):
        assert math.isnan(p)
        assert math.isnan(lo)
        assert math.isnan(hi)
    else:
        assert p == pytest.approx(expected_p)
        assert 0.0 <= lo <= p <= hi <= 1.0


# --------------------------------------------------------------------------- #
# Test: Bootstrap confidence interval                                         #
# --------------------------------------------------------------------------- #
def test_bootstrap_confint_perfect_accuracy():
    """
    For perfect predictions, bootstrap CI should be (1.0, 1.0, 1.0).
    """
    y_true = np.array([1, 0, 1, 0])
    y_pred = y_true.copy()
    est, lo, hi = bootstrap_confint(
        y_true,
        y_pred,
        lambda a, b: accuracy_score(a, b),
        n_boot=100,
        alpha=0.05,
        random_state=42,
    )
    assert est == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_confint_empty():
    """
    Empty input arrays should yield nan intervals.
    """
    y_true = np.array([])
    y_pred = np.array([])
    est, lo, hi = bootstrap_confint(y_true, y_pred, lambda a, b: 0.0)
    assert math.isnan(est)
    assert math.isnan(lo)
    assert math.isnan(hi)


# --------------------------------------------------------------------------- #
# Test: MetricEngine compute and fmt                                          #
# --------------------------------------------------------------------------- #
def test_metric_engine_compute_and_fmt():
    """
    Verifies metrics and confidence intervals computation, and formatting.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = y_true.copy()
    y_score = y_pred.astype(float)

    engine = MetricEngine(ci_method="wilson", boot_samples=10, random_state=0)
    metrics, cis = engine.compute(y_true, y_pred, y_score)

    # Basic metric
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert "pr_auc" in metrics and metrics["pr_auc"] > 0.0
    assert "ece" in metrics and 0.0 <= metrics["ece"] <= 1.0

    # Check that confidence intervals dict has expected keys
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "pr_auc"]:
        assert key in cis and isinstance(cis[key], tuple) and len(cis[key]) == 3

    # Test formatting with valid bounds
    formatted = engine.fmt(0.5, 0.4, 0.6)
    assert "50.00% (CI: 40.00% - 60.00%)" == formatted

    # Test formatting with missing bounds
    formatted_na = engine.fmt(0.5, np.nan, 0.6)
    assert "50.00% (CI: N/A)" == formatted_na


# --------------------------------------------------------------------------- #
# Test: evaluate_model output structure                                       #
# --------------------------------------------------------------------------- #
def test_evaluate_model_structure_and_values():
    """
    Validates evaluate_model returns a dict with correctly formatted metrics.
    """
    df = pd.DataFrame(
        {
            "label": [0, 1, 1, 0],
            "pred": [0, 1, 1, 0],
            "prob": [0.1, 0.9, 0.8, 0.2],
        }
    )
    result = evaluate_model(
        df,
        label_col="label",
        pred_col="pred",
        dataset_name="DS",
        report_col="R",
        pred_type="T",
        pred_proba="prob",
        total_count=4,
    )

    # Check key presence and coverage
    expected_keys = [
        "Dataset",
        "Report_Column",
        "Gold_Standard_Column",
        "Prediction_Type",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "NPV",
        "F1_Score",
        "MCC",
        "PR_AUC",
        "Brier_Score",
        "ECE",
        "Coverage_Count",
        "Coverage_Pct",
    ]
    for key in expected_keys:
        assert key in result
    assert result["Coverage_Count"] == 4
    assert result["Coverage_Pct"] == "100.00%"


# --------------------------------------------------------------------------- #
# Test: evaluate skips missing and handles modes                              #
# --------------------------------------------------------------------------- #
def test_evaluate_skips_missing_and_handles_modes(tmp_path, caplog):
    """
    Ensures evaluate skips missing columns, warns appropriately, and
    correctly evaluates group_level and final modes.
    """
    caplog.set_level(logging.WARNING, logger="nlp_pipeline.common.evaluation")

    df = pd.DataFrame(
        {
            "col1_Pred_X": [0, 1],
            "col1_Prob_X": [0.2, 0.8],
            "label1": [0, 1],
        }
    )
    cols_map = {"col1": "label1", "col2": "label2"}

    # Default mode
    df_eval = evaluate(df, cols_map, dataset_name="D", pred_type="X", total_count=2)
    assert list(df_eval["Report_Column"]) == ["col1"]
    assert any("Skipping 'col2'" in rec.getMessage() for rec in caplog.records)

    # Group-level mode
    df_group = df.rename(columns={"col1_Pred_X": "Patient_col1_Pred_X"})
    caplog.clear()
    df_eval_group = evaluate(df_group, {"col1": "label1"}, "D", "X", group_level=True)
    assert list(df_eval_group["Report_Column"]) == ["col1"]

    # Final mode uses keys directly
    df_final = pd.DataFrame({"col1": [0, 1], "label": [0, 1]})
    caplog.clear()
    df_eval_final = evaluate(df_final, {"col1": "label"}, "D", "X", final=True)
    assert list(df_eval_final["Report_Column"]) == ["col1"]
