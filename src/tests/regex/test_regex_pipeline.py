"""
Unified smoke + integration test suite for the Regex pipeline package.

Goal
-----
*   Provides one canonical pytest module that can be **parametrised** to the Regex pipeline
    package under ``nlp_pipeline.models.regex.pipeline``.
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

This version mirrors the SpaCy test, but for “regex”:
  1. Two modes: “raw” and “umls.”
  2. Evaluation files named eval_regex_raw.csv/.xlsx and eval_regex_umls.csv/.xlsx.
  3. Fairness CSVs under <mode>/fairness/{training_set, validation_set}/…
  4. A single top-level resources.csv, an emissions.csv, and pipeline_debug.log.
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

PIPELINE_NAME_MAP = {"regex": "regex"}

logger = logging.getLogger(__name__)

import sys

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
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
        default="regex",
        help="Pipeline name to test, e.g. bow, regex, sbert, etc.",
    )


@pytest.fixture(autouse=True)
def patch_logging(monkeypatch):
    pass  # Placeholder if you need to stub out configure_logging later


@pytest.fixture
def pipeline_name(request):
    return request.config.getoption("--pipeline_name")


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    """
    Synthetic dataset aligned with Regex pipeline expectations.
    Assume the regex pipeline requires exactly the columns below:
    - Two “report” text columns (for column-wise regex matching)
    - study_id (to collapse to patient level)
    - Patient_Has_IBD (gold label)
    - age_group, ethnicity, gender, imd_group (demographic keys for fairness)
    """
    return pd.DataFrame(
        {
            "report_col_1": [
                "Crohn's mention here",
                "Immunosuppressant mention",
                None,
                "No IBD",
            ],
            "report_col_2": [
                "No mention",
                "Ulcerative colitis suspected",
                "Irrelevant",
                "IBD exact",
            ],
            "study_id": [1, 2, 3, 4],
            "Patient_Has_IBD": [1, 1, 0, 0],
            "age_group": ["A", "B", "A", "B"],
            "ethnicity": ["X", "Y", "X", "Y"],
            "gender": ["M", "F", "M", "F"],
            "imd_group": ["low", "med", "low", "med"],
        }
    )


@pytest.fixture
def override_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Tuple[Path, Path]:
    """
    Redirect constants so that:
      - Input file paths point to dummy locations (not actually used, since we stub load)
      - ANALYSIS and RESULTS dirs live under tmp_path
      - Regex-specific mappings (IBD_COLUMNS, COLWISE_COLUMNS, DRUG_COLUMNS) match our dummy_df
    """
    from nlp_pipeline.config import constants

    # Create directories
    analysis_root = tmp_path / "analysis"
    results_root = tmp_path / "final"
    analysis_root.mkdir()
    results_root.mkdir()

    # Monkey‐patch the “regex”‐specific constants:
    #   - TRAIN/VAL file paths → dummy CSVs (never actually read, because load_and_preprocess is stubbed)
    #   - REGEX_ANALYSIS_DIR / FINAL_RESULTS_DIR → under tmp_path
    monkeypatch.setattr(constants, "TRAIN_FILE_PATH", str(tmp_path / "dummy_train.csv"))
    monkeypatch.setattr(constants, "VAL_FILE_PATH", str(tmp_path / "dummy_val.csv"))

    monkeypatch.setattr(constants, "REGEX_ANALYSIS_DIR", str(analysis_root))
    monkeypatch.setattr(constants, "FINAL_RESULTS_DIR", str(results_root))

    # For our dummy data, only “report_col_1” and “report_col_2” map to “Patient_Has_IBD”
    mapping = {"report_col_1": "Patient_Has_IBD", "report_col_2": "Patient_Has_IBD"}
    monkeypatch.setattr(constants, "IBD_COLUMNS", mapping)
    monkeypatch.setattr(constants, "COLWISE_COLUMNS", mapping)

    # ALSO override DRUG_COLUMNS so that the pipeline does not try to use any “real” histopathology_report, etc.
    monkeypatch.setattr(constants, "DRUG_COLUMNS", mapping)

    # Demographic keys for fairness
    monkeypatch.setattr(
        constants,
        "DEMOGRAPHICS_KEYS",
        ["age_group", "ethnicity", "gender", "imd_group"],
    )

    # Create dummy CSV files (headers only) so that if load_and_preprocess is ever called on them, no I/O error.
    (tmp_path / "dummy_train.csv").write_text(
        "report_col_1,report_col_2,study_id,Patient_Has_IBD,age_group,ethnicity,gender,imd_group\n"
    )
    (tmp_path / "dummy_val.csv").write_text(
        "report_col_1,report_col_2,study_id,Patient_Has_IBD,age_group,ethnicity,gender,imd_group\n"
    )

    return analysis_root, results_root


@pytest.fixture
def pipeline_module(
    request,
    dummy_df: pd.DataFrame,
    override_constants: Tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """
    Import the regex pipeline module and stub out any heavy/IO-bound functions:
      - load_and_preprocess → returns dummy_df
      - _assert_data_splits_ok → no-op
      - evaluate_fairness_dual → returns minimal aggregated/disaggregated dict
      - plot_fairness… and other plotting hooks → no-op
      - configure_logging → writes a “pipeline_debug.log” with “pipeline complete”
    """
    pipeline_name = request.param  # “regex”
    pkg = f"nlp_pipeline.models.{pipeline_name}.pipeline"
    pipeline = importlib.import_module(pkg)
    importlib.reload(pipeline)

    # 1. Stub load_and_preprocess to always return a fresh copy of dummy_df
    monkeypatch.setattr(
        pipeline,
        "load_and_preprocess",
        lambda path, exclude_combined_clinic=False: dummy_df.copy(),
        raising=True,
    )

    # 2. Stub _assert_data_splits_ok to a no-op
    if hasattr(pipeline, "_assert_data_splits_ok"):
        monkeypatch.setattr(
            pipeline, "_assert_data_splits_ok", lambda *a, **k: None, raising=True
        )

    # 3. Stub evaluate_fairness_dual to return a minimal structure
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

    # 4. Stub any plotting functions so they do nothing
    for fn in ["plot_fairness_metrics_aggregated", "plot_f1_scores_disaggregated"]:
        if hasattr(pipeline, fn):
            monkeypatch.setattr(pipeline, fn, lambda *a, **k: None, raising=True)

    # 5. Stub configure_logging so it creates pipeline_debug.log with “pipeline complete”
    if hasattr(pipeline, "configure_logging"):

        def fake_configure_logging(log_dir: str, custom_logger=None):
            log_path = Path(log_dir) / "pipeline_debug.log"
            log = logging.getLogger("pipeline_logger")
            log.setLevel(logging.INFO)
            log.propagate = False
            for h in list(log.handlers):
                log.removeHandler(h)
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            log.addHandler(handler)
            log.info("pipeline complete")

        monkeypatch.setattr(
            pipeline, "configure_logging", fake_configure_logging, raising=True
        )

    return pipeline


@pytest.fixture
def stub_pipeline_apis(monkeypatch: pytest.MonkeyPatch, pipeline_module: ModuleType):
    """
    Stub out all evaluation/resource/emissions hooks so that:
      - evaluate(...) returns a minimal one-row DataFrame each time
      - ResourceMonitor never errors and returns dummy metrics
      - EmissionsTracker writes a single emissions.csv
    """
    # 1. Stub evaluate(...) so every call yields a one‐row DataFrame
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

    # 2. Stub ResourceMonitor (no‐op start/stop, returns dummy metrics)
    class DummyMonitor:
        def __init__(self, interval):
            pass

        def start(self):
            pass

        def stop(self):
            self.elapsed = 1.23

        def get_metrics(self):
            return (0.11, 1.23)

    monkeypatch.setattr(pipeline_module, "ResourceMonitor", DummyMonitor)

    # 3. Stub EmissionsTracker so that stop() writes an emissions.csv
    class DummyTracker:
        def __init__(self, project_name, output_dir, allow_multiple_runs=False):
            self.out = Path(output_dir)

        def start(self):
            pass

        def stop(self):
            # Write a minimal emissions.csv under analysis_root
            (self.out / "emissions.csv").write_text("energy_kwh\n0.456\n")
            return 0.456

    monkeypatch.setattr(pipeline_module, "EmissionsTracker", DummyTracker)

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "pipeline_module", list(PIPELINE_NAME_MAP.values()), indirect=True
)
def test_regex_pipeline_creates_expected_artifacts(
    pipeline_module: ModuleType,
    override_constants: Tuple[Path, Path],
    stub_pipeline_apis,
):
    """
    Run `pipeline.main()` and assert that:
      1. eval_regex_raw.csv + .xlsx exist, and eval_regex_umls.csv + .xlsx exist.
      2. Each eval CSV has at least columns {"Dataset","Report_Column","Prediction_Type","Brier_Score"}.
      3. Fairness CSVs are written under:
           - raw/fairness/training_set/{aggregated.csv,disaggregated.csv}
           - raw/fairness/validation_set/{aggregated.csv,disaggregated.csv}
           - umls/fairness/training_set/{aggregated.csv,disaggregated.csv}
           - umls/fairness/validation_set/{aggregated.csv,disaggregated.csv}
      4. A single top-level resources.csv exists with columns
         {"cpu_pct","memory_mb","energy_kwh","time_sec"}.
      5. A single top-level emissions.csv exists with column {"energy_kwh"}.
      6. pipeline_debug.log exists and contains “pipeline complete.”
    """
    analysis_root, results_root = override_constants

    # Execute the pipeline (this now invokes our stubs)
    pipeline_module.main()

    # 1. Evaluation outputs

    for mode in ["raw", "umls"]:
        csv_path = analysis_root / f"eval_regex_{mode}.csv"
        xlsx_path = results_root / f"eval_regex_{mode}.xlsx"
        assert csv_path.exists(), f"Missing {csv_path}"
        assert xlsx_path.exists(), f"Missing {xlsx_path}"

        df = pd.read_csv(csv_path)
        assert not df.empty, f"Empty eval file: {csv_path}"
        assert {"Dataset", "Report_Column", "Prediction_Type", "Brier_Score"}.issubset(
            df.columns
        )

    # 2. Fairness CSVs

    # 2a. raw mode should write both training_set and validation_set fairness
    raw_train_agg = (
        analysis_root / "raw" / "fairness" / "training_set" / "aggregated.csv"
    )
    raw_train_dis = (
        analysis_root / "raw" / "fairness" / "training_set" / "disaggregated.csv"
    )
    raw_val_agg = (
        analysis_root / "raw" / "fairness" / "validation_set" / "aggregated.csv"
    )
    raw_val_dis = (
        analysis_root / "raw" / "fairness" / "validation_set" / "disaggregated.csv"
    )

    assert raw_train_agg.exists(), "Missing raw/fairness/training_set/aggregated.csv"
    assert raw_train_dis.exists(), "Missing raw/fairness/training_set/disaggregated.csv"
    assert raw_val_agg.exists(), "Missing raw/fairness/validation_set/aggregated.csv"
    assert raw_val_dis.exists(), "Missing raw/fairness/validation_set/disaggregated.csv"

    # 2b. umls mode should also write both training_set and validation_set
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
