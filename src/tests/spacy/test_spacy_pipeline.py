"""
Unified smoke + integration test suite for the SpaCy pipeline package.

Goal
-----
*   Provides one canonical pytest module that can be **parametrised** to the SpaCy model
    package under ``nlp_pipeline.models.spacy.pipeline``.
*   Keeps it dependency-free beyond ``pytest``/``pandas``—all heavy lifting is stubbed.

Usage
-----
Run via pytest; no external resources are required.

Design highlights
-----------------
*   **Fixtures layered for clarity**: dummy_df, override_constants, stubs.
*   **Centralised stubbing**: adapt signatures in stub_pipeline_apis.
*   **Artifact assertions**: check evaluation outputs, fairness CSVs, resource and emission logs, and pipeline log.
*   **No leakage diagnostics**: fixtures bypass split checks for identical dummy sets.

This version has been updated to reflect that:
  1. Raw mode only writes fairness CSVs under “…/fairness/training_set/…”.
  2. UMLS mode writes fairness CSVs under both “…/fairness/training_set/…” and “…/fairness/validation_set/…”.
  3. The pipeline only writes a single top‐level resources.csv (no mode‐specific resource files).
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Tuple

import pandas as pd
import pytest

from nlp_pipeline.config import constants as c

PIPELINE_NAME_MAP = {"spacy": "spacy"}

logger = logging.getLogger(__name__)

import sys

# ---------------------------------------------------------------------------- #
# CLI                                                                          #
# ---------------------------------------------------------------------------- #

def neuter_stream_handlers():
    for logger_obj in logging.root.manager.loggerDict.values():
        if isinstance(logger_obj, logging.PlaceHolder):
            continue
        for handler in getattr(logger_obj, "handlers", []):
            if isinstance(handler, logging.StreamHandler) and handler.stream in (
                sys.stdout,
                sys.stderr,
            ):
                handler.close = lambda: None  # patch .close to do nothing


neuter_stream_handlers()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pipeline_name",
        action="store",
        default="spacy",
        help="Pipeline name to test, e.g. bow, regex, sbert, etc.",
    )

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def patch_logging(monkeypatch):
    pass  # Placeholder if needed later


@pytest.fixture
def pipeline_name(request):
    return request.config.getoption("--pipeline_name")


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """Synthetic dataset aligned with SpaCy pipeline expectations."""
    return pd.DataFrame(
        {
            "text_column": [
                "IBD confirmed.",
                "No IBD.",
                "Possible IBD symptoms.",
                "Unrelated note.",
            ],
            "Patient_Has_IBD": [1, 0, 1, 0],
            "age_group": ["10-20", "30-40", "50-60", "70-80"],
            "ethnicity": ["White", "Black", "White", "Black"],
            "gender": ["M", "F", "F", "M"],
            "imd_group": ["1-2", "3-4", "5-6", "7-8"],
            "study_id": [1, 2, 3, 4],
            "doc_digest": ["a1", "b2", "c3", "d4"],  # required by run_spacy_workflow
        }
    )


@pytest.fixture
def override_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Tuple[Path, Path]:
    """Redirect constants to temporary analysis and results directories."""
    analysis_root = tmp_path / "analysis"
    results_root = tmp_path / "final"
    analysis_root.mkdir()
    results_root.mkdir()

    monkeypatch.setattr(c, "TRAIN_FILE_PATH", str(tmp_path / "dummy.csv"))
    monkeypatch.setattr(c, "VAL_FILE_PATH", str(tmp_path / "dummy.csv"))
    monkeypatch.setattr(c, "TRAIN_FILE_UMLS_PATH", str(tmp_path / "dummy.csv"))
    monkeypatch.setattr(c, "VAL_FILE_UMLS_PATH", str(tmp_path / "dummy.csv"))
    monkeypatch.setattr(c, "SPACY_ANALYSIS_DIR", str(analysis_root))
    monkeypatch.setattr(c, "FINAL_RESULTS_DIR", str(results_root))
    monkeypatch.setattr(c, "COLWISE_COLUMNS", {"text_column": "Patient_Has_IBD"})
    monkeypatch.setattr(c, "IBD_COLUMNS", {"text_column": "Patient_Has_IBD"})
    monkeypatch.setattr(
        c,
        "DEMOGRAPHICS_KEYS",
        ["age_group", "ethnicity", "gender", "imd_group"],
    )
    # Create an empty dummy.csv so load_and_preprocess can open it
    (tmp_path / "dummy.csv").write_text(
        "text_column,Patient_Has_IBD,age_group,ethnicity,gender,imd_group,study_id\n"
    )
    return analysis_root, results_root


@pytest.fixture
def pipeline_module(
    request,
    dummy_df: pd.DataFrame,
    override_constants: Tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    pipeline_name = request.param  # from @pytest.mark.parametrise
    pkg = f"nlp_pipeline.models.{pipeline_name}.pipeline"
    pipeline = importlib.import_module(pkg)
    importlib.reload(pipeline)

    # Stub out load_and_preprocess to return dummy_df
    monkeypatch.setattr(
        pipeline,
        "load_and_preprocess",
        lambda path, exclude_combined_clinic=False: dummy_df.copy(),
        raising=True,
    )

    # Stub out _assert_data_splits_ok so it never raises
    if hasattr(pipeline, "_assert_data_splits_ok"):
        monkeypatch.setattr(
            pipeline, "_assert_data_splits_ok", lambda *a, **k: None, raising=True
        )

    # Stub evaluate_fairness_dual so it returns a minimal non‐None dict
    if hasattr(pipeline, "evaluate_fairness_dual"):

        def fake_evaluate_fairness_dual(
            df, final_col, gold_col, demographic_attrs, dataset_name, note
        ):
            return {
                "aggregated": [{"metric": "F1", "value": 1.0}],
                "disaggregated": [{"group": "all", "metric": "F1", "value": 1.0}],
            }

        monkeypatch.setattr(
            pipeline,
            "evaluate_fairness_dual",
            fake_evaluate_fairness_dual,
            raising=True,
        )

    # Stub plotting hooks so they do nothing
    for fn in ["plot_fairness_metrics_aggregated", "plot_f1_scores_disaggregated"]:
        if hasattr(pipeline, fn):
            monkeypatch.setattr(pipeline, fn, lambda *a, **k: None, raising=True)

    return pipeline


@pytest.fixture
def stub_pipeline_apis(monkeypatch: pytest.MonkeyPatch, pipeline_module: ModuleType):
    """Stub train/predict/evaluate and resource/emissions trackers."""

    # 1. Stub out the common `evaluate(...)` function
    def fake_evaluate(df, mapping, dataset_name, pred_type, **kwargs):
        col = list(mapping.keys())[0]
        return pd.DataFrame(
            [
                {
                    "Dataset": dataset_name,
                    "Report_Column": col,
                    "Gold_Standard_Column": mapping[col],
                    "Prediction_Type": pred_type,
                    "Accuracy": "100%",
                    "Precision": "100%",
                    "Recall": "100%",
                    "F1_Score": "100%",
                    "Brier_Score": 0.0,
                }
            ]
        )

    monkeypatch.setattr(pipeline_module, "evaluate", fake_evaluate, raising=True)

    # 2. Stub ResourceMonitor so it never fails
    class DummyMonitor:
        def __init__(self, interval):
            pass

        def start(self):
            pass

        def stop(self):
            self.elapsed = 1.0

        def get_metrics(self):
            return (0.1, 1.0)

    monkeypatch.setattr(pipeline_module, "ResourceMonitor", DummyMonitor)

    # 3. Stub EmissionsTracker so it writes a simple emissions.csv
    class DummyTracker:
        def __init__(self, project_name, output_dir, allow_multiple_runs=False):
            self.out = Path(output_dir)

        def start(self):
            pass

        def stop(self):
            # Write a minimal emissions.csv under analysis_root
            (self.out / "emissions.csv").write_text("energy_kwh\n0.123\n")
            return 0.123

    monkeypatch.setattr(pipeline_module, "EmissionsTracker", DummyTracker)


def fake_configure_logging(log_dir: Path, custom_logger: str | None = None):
    """Ensure that `configure_logging` writes pipeline_debug.log with “pipeline complete”."""
    log_path = Path(log_dir) / "pipeline_debug.log"
    log = logging.getLogger("pipeline_logger")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in log.handlers[:]:
        log.removeHandler(h)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(handler)
    log.info("pipeline complete")

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "pipeline_module", list(PIPELINE_NAME_MAP.values()), indirect=True
)
def test_pipeline_creates_expected_artifacts(
    pipeline_module: ModuleType,
    override_constants: Tuple[Path, Path],
    stub_pipeline_apis,
):
    analysis_root, results_root = override_constants

    # Run the pipeline end to end
    pipeline_module.main()

    # 1. Evaluation outputs (for raw and umls)
    for mode in ["raw", "umls"]:
        csv = analysis_root / f"eval_spacy_{mode}.csv"
        xlsx = results_root / f"eval_spacy_{mode}.xlsx"
        assert csv.exists(), f"Missing {csv}"
        assert xlsx.exists(), f"Missing {xlsx}"

        df = pd.read_csv(csv)
        assert not df.empty, f"Empty eval file {csv}"
        assert {"Dataset", "Report_Column", "Prediction_Type", "Brier_Score"}.issubset(
            df.columns
        )

    # 2. Fairness CSVs

    # Raw mode: only training_set should exist
    raw_train_agg = (
        analysis_root / "raw" / "fairness" / "training_set" / "aggregated.csv"
    )
    raw_train_dis = (
        analysis_root / "raw" / "fairness" / "training_set" / "disaggregated.csv"
    )
    assert raw_train_agg.exists(), "Missing raw/fairness/training_set/aggregated.csv"
    assert raw_train_dis.exists(), "Missing raw/fairness/training_set/disaggregated.csv"

    raw_val_agg = (
        analysis_root / "raw" / "fairness" / "validation_set" / "aggregated.csv"
    )
    raw_val_dis = (
        analysis_root / "raw" / "fairness" / "validation_set" / "disaggregated.csv"
    )
    assert (
        not raw_val_agg.exists()
    ), "Should NOT have raw/fairness/validation_set/aggregated.csv"
    assert (
        not raw_val_dis.exists()
    ), "Should NOT have raw/fairness/validation_set/disaggregated.csv"

    # UMLS mode: both training_set and validation_set should exist
    umls_train_agg = (
        analysis_root / "umls" / "fairness" / "training_set" / "aggregated.csv"
    )
    umls_train_dis = (
        analysis_root / "umls" / "fairness" / "training_set" / "disaggregated.csv"
    )
    umls_val_agg = (
        analysis_root / "umls" / "fairness" / "validation_set" / "aggregated.csv"
    )
    umls_val_dis = (
        analysis_root / "umls" / "fairness" / "validation_set" / "disaggregated.csv"
    )

    assert umls_train_agg.exists(), "Missing umls/fairness/training_set/aggregated.csv"
    assert (
        umls_train_dis.exists()
    ), "Missing umls/fairness/training_set/disaggregated.csv"
    assert umls_val_agg.exists(), "Missing umls/fairness/validation_set/aggregated.csv"
    assert (
        umls_val_dis.exists()
    ), "Missing umls/fairness/validation_set/disaggregated.csv"

    # 3. Generic resources.csv (written once at the end by main())
    generic_res = analysis_root / "resources.csv"
    assert generic_res.exists(), "Missing resources.csv"
    df_generic = pd.read_csv(generic_res)
    assert {"cpu_pct", "memory_mb", "energy_kwh", "time_sec"}.issubset(
        df_generic.columns
    )

    # 4. Emissions (should be written under analysis_root/emissions.csv)
    emissions_file = analysis_root / "emissions.csv"
    assert emissions_file.exists(), "Missing emissions.csv"
    df_em = pd.read_csv(emissions_file)
    assert "energy_kwh" in df_em.columns

    # 5. Log file
    log = analysis_root / "pipeline_debug.log"
    assert log.exists(), "Missing pipeline_debug.log"
    assert "pipeline complete" in log.read_text().lower()
