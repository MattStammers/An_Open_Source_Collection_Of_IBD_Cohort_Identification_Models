"""

test fairness.py

Tests for the fairness functions

"""

import logging

import numpy as np
import pandas as pd
import pytest

from nlp_pipeline.common.fairness import (
    REFERENCE_GROUPS,
    demographic_parity,
    disparate_impact,
    equal_opportunity,
)


# --------------------------------------------------------------------------- #
# Helpers for building simple DataFrames                                      #
# --------------------------------------------------------------------------- #
def make_df(protected, preds, truths=None):
    data = {"grp": protected, "pred": preds}
    if truths is not None:
        data["true"] = truths
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Test demographic_parity                                                     #
# --------------------------------------------------------------------------- #
def test_demographic_parity_basic():
    # Group A: 2/4 positives = 0.5; Group B: 3/5 = 0.6
    df = make_df(["A"] * 4 + ["B"] * 5, [1, 0, 1, 0] + [1, 1, 0, 1, 0])
    # Default reference: first group with >0 rate = "A"
    out = demographic_parity(df, "grp", "pred")
    # Only B vs A
    assert pytest.approx(out["Demographic_Parity_grp_Group_B_vs_A"]) == 0.6 - 0.5


def test_demographic_parity_with_reference_arg():
    # Swap rates so B has lower; explicitly use B as ref
    df = make_df(["A"] * 4 + ["B"] * 4, [1, 1, 0, 1] + [0, 0, 1, 0])
    # A: 0.75, B: 0.25 -> parity A vs B = 0.75/0.25 = 3.0
    out = demographic_parity(df, "grp", "pred", reference_group="B")
    assert list(out) == ["Demographic_Parity_grp_Group_A_vs_B"]
    assert out["Demographic_Parity_grp_Group_A_vs_B"] == pytest.approx(0.5)


def test_demographic_parity_all_zero_rates(caplog):
    df = make_df(["X", "Y", "Z"], [0, 0, 0])
    caplog.set_level(logging.WARNING)
    out = demographic_parity(df, "grp", "pred")
    assert out == {}
    assert "All groups in 'grp' have zero positives" in caplog.text


def test_demographic_parity_single_group(caplog):
    df = make_df(["only"] * 5, [1, 0, 1, 1, 0])
    caplog.set_level(logging.WARNING)
    out = demographic_parity(df, "grp", "pred")
    assert out == {}
    assert "has <2 groups" in caplog.text


# --------------------------------------------------------------------------- #
# Test equal_opportunity                                                      #
# --------------------------------------------------------------------------- #
def test_equal_opportunity_basic():
    # group A: positives [1,0,1] -> TPR=2/3; group B: [1,1] -> TPR=1.0
    df = make_df(["A", "A", "A", "B", "B"], [1, 0, 1, 1, 0], truths=[1, 1, 1, 1, 1])
    out = equal_opportunity(df, "grp", "pred", "true", reference_group="A")
    # Only B vs A
    assert list(out) == ["Equal_Opportunity_grp_Group_B_vs_A"]
    # TPR_A = 2/3, TPR_B = 1/2 → gap = |1/2 - 2/3| = 1/6
    assert out["Equal_Opportunity_grp_Group_B_vs_A"] == pytest.approx(abs(0.5 - 2 / 3))


def test_equal_opportunity_missing_positives(caplog):
    # group A has no positives in truth
    df = make_df(["A", "B", "B"], [1, 0, 1], truths=[0, 1, 1])
    caplog.set_level(logging.WARNING)
    out = equal_opportunity(df, "grp", "pred", "true", reference_group="B")
    # A->nan TPR; difference nan
    assert np.isnan(out["Equal_Opportunity_grp_Group_A_vs_B"])
    assert "No positive instances for group 'A'" in caplog.text


# --------------------------------------------------------------------------- #
# Test disparate_impact                                                       #
# --------------------------------------------------------------------------- #
def test_disparate_impact_basic():
    # A: 2/4=0.5, B: 1/3≈0.333
    df = make_df(["A"] * 4 + ["B"] * 3, [1, 0, 1, 0] + [1, 0, 0])
    out = disparate_impact(df, "grp", "pred", reference_group="A")
    assert list(out) == ["Disparate_Impact_grp_Group_B_vs_A"]
    assert out["Disparate_Impact_grp_Group_B_vs_A"] == pytest.approx((1 / 3) / 0.5)


def test_disparate_impact_missing_reference(caplog):
    df = make_df(["X", "Y"], [1, 0])
    caplog.set_level(logging.WARNING)
    out = disparate_impact(df, "grp", "pred", reference_group="Z")
    assert out == {}
    assert "Reference group 'Z' not found" in caplog.text


def test_disparate_impact_zero_reference_rate(caplog):
    df = make_df(["A", "A", "B"], [0, 0, 1])
    caplog.set_level(logging.WARNING)
    out = disparate_impact(df, "grp", "pred", reference_group="A")
    # A rate = 0 → B vs A = nan
    assert np.isnan(out["Disparate_Impact_grp_Group_B_vs_A"])
    assert "zero positive predictions" in caplog.text
