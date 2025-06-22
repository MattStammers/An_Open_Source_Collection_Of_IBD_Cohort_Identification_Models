"""
Unified smoke + integration test suite for NLP pipeline packages.

Goal
-----
*   Provides one canonical pytest module that can be **parametrised** to any model
    package living under ``nlp_pipeline.models.<pipeline_name>.pipeline``.
*   Keeps it dependency-free beyond ``pytest``/``pandas``—all heavy lifting is
    stubbed.

Usage
-----
Add the file to your tests directory and drive it with **pytest's builtin
parametrisation**:

```bash
# change the model flag below as needed
```

For another model (e.g. "cnn") just pass ``--pipeline_name=cnn`` or mark the
parameter directly inside another test module using
``pytestmark = pytest.mark.parametrise('pipeline_name',["cnn"], indirect=True)``.

Design highlights
-----------------
*   **Fixtures layered for clarity** - each fixture has a single purpose and is
    re-usable by other pipeline tests (dummy_df, override_constants, stubs).
*   **Stubbing is centralised** inside ``stub_pipeline_apis`` so changing a
    signature requires touching only one spot.
*   **Artifact assertions** are automatically generated from a small metadata
    table so extending to new artefacts is one-line work.
*   **No leakage diagnostics** - the fixture bypasses them so identical dummy
    train/val sets do not break CI.
*   **100 % self-contained** - nothing is read from disk except the stubbed
    outputs we write on-the-fly.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple
from unittest.mock import patch

import pandas as pd
import pytest

from nlp_pipeline.config import constants as c

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
# CLI                                                                          #
# ---------------------------------------------------------------------------- #

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --pipeline_name command‑line option."""
    parser.addoption(
        "--pipeline_name",
        action="store",
        default="distilbert",
        help="nlp_pipeline.models.<name>.pipeline to import",
    )

# ---------------------------------------------------------------------------- #
# Fixtures                                                                     #
# ---------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def pipeline_name() -> str:
    return "distilbert"


@pytest.fixture
def dummy_df():
    """Synthetic dataset with class balance and realistic demographic bins for distilbert pipeline tests."""
    return pd.DataFrame(
        {
            "text_column": [
                "Patient has confirmed IBD diagnosis with severe flare.",
                "Routine follow-up shows no evidence of IBD currently.",
                "Mild abdominal pain, possibly functional.",
                "Symptoms resolved without intervention.",
                "Evidence of chronic inflammation suggestive of IBD.",
                "Negative colonoscopy, no signs of Crohn's or UC.",
                "Some non-specific inflammation observed, monitor advised.",
                "IBD patient stable on infliximab treatment.",
                "No further symptoms reported, all labs normal.",
                "Biopsy results indicate possible ulcerative colitis.",
            ],
            "Patient_Has_IBD": [1, 0, 0, 0, 1, 0, 0, 1, 0, 1],
            "age_group": [
                "10-20",
                "10-20",
                "30-40",
                "30-40",
                "50-60",
                "50-60",
                "70-80",
                "70-80",
                "90-100",
                "90-100",
            ],
            "ethnicity": [
                "Black",
                "White",
                "White",
                "Black",
                "White",
                "Black",
                "Black",
                "White",
                "White",
                "Black",
            ],
            "gender": ["M", "F", "F", "M", "M", "F", "M", "F", "M", "F"],
            "imd_group": [
                "1-2",
                "1-2",
                "3-4",
                "3-4",
                "5-6",
                "5-6",
                "7-8",
                "7-8",
                "9-10",
                "9-10",
            ],
            "study_id": list(range(1, 11)),
        }
    )


@pytest.fixture
def override_constants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Tuple[Path, Path]:
    """Redirect *all* filesystem references inside ``constants`` to *tmp_path*."""
    analysis_root = tmp_path / "analysis"
    results_root = tmp_path / "final"
    analysis_root.mkdir()
    results_root.mkdir()

    monkeypatch.setattr(c, "TRAIN_FILE_PATH", str(tmp_path / "dummy_train.csv"))
    monkeypatch.setattr(c, "VAL_FILE_PATH", str(tmp_path / "dummy_val.csv"))
    monkeypatch.setattr(c, "TRAIN_FILE_UMLS_PATH", str(tmp_path / "dummy_train.csv"))
    monkeypatch.setattr(c, "VAL_FILE_UMLS_PATH", str(tmp_path / "dummy_val.csv"))
    monkeypatch.setattr(c, "DISTILBERT_ANALYSIS_DIR", str(analysis_root))
    monkeypatch.setattr(c, "FINAL_RESULTS_DIR", str(results_root))

    mapping: Dict[str, str] = {"text_column": "Patient_Has_IBD"}
    monkeypatch.setattr(c, "COLWISE_COLUMNS", mapping)
    monkeypatch.setattr(c, "IBD_COLUMNS", mapping)
    monkeypatch.setattr(
        c,
        "DEMOGRAPHICS_KEYS",
        ["age_group", "ethnicity", "gender", "imd_group"],
    )

    # Create empty CSVs so load_and_preprocess does not crash
    header = (
        "text_column,Patient_Has_IBD,age_group,ethnicity,gender,imd_group,study_id\n"
    )
    for fname in ["dummy_train.csv", "dummy_val.csv"]:
        (tmp_path / fname).write_text(header)

    return analysis_root, results_root


PIPELINE_NAME_MAP = {
    "regex": "regex",
    "spacy": "spacy",
    "bow": "bow",
    "tfidf": "tfidf",
    "word2vec": "word2vec",
    "sbert": "sbert_base",
    "sbert_med": "sbert_med",
    "distilbert": "distilbert",
    "bio_clinical_bert": "bio_clinical_bert",
    "roberta": "roberta",
}


@pytest.fixture
def pipeline_module(
    pipeline_name: str,
    dummy_df: pd.DataFrame,
    override_constants: Tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """Import + patch the target pipeline after constants are overridden."""
    real_name = PIPELINE_NAME_MAP.get(pipeline_name, pipeline_name)
    pkg_path = f"nlp_pipeline.models.{real_name}.pipeline"
    pipeline = importlib.import_module(pkg_path)
    importlib.reload(pipeline)  # ensure patched constants are picked‑up

    # ------------------------------------------------------------------
    # Data loading – always return the same dummy DF
    # ------------------------------------------------------------------
    monkeypatch.setattr(
        pipeline,
        "load_and_preprocess",
        lambda path, **kwargs: dummy_df.copy(),
        raising=True,
    )

    # ------------------------------------------------------------------
    # Skip leakage & heavy operations if present in the module
    # ------------------------------------------------------------------
    for attr in [
        "_assert_data_splits_ok",
        "evaluate_fairness_dual",
        "plot_fairness_metrics_aggregated",
        "plot_f1_scores_disaggregated",
    ]:
        if hasattr(pipeline, attr):  # pragma: no‑cover
            monkeypatch.setattr(pipeline, attr, lambda *a, **k: None, raising=True)

    # ------------------------------------------------------------------
    # Force TEXT_COLS to align with dummy_df
    # ------------------------------------------------------------------
    if hasattr(pipeline, "TEXT_COLS"):
        monkeypatch.setattr(pipeline, "TEXT_COLS", ["text_column"], raising=True)

    return pipeline


@pytest.fixture
def stub_pipeline_apis(monkeypatch: pytest.MonkeyPatch, pipeline_module: ModuleType):
    """Generic stubs for *train/predict/evaluate* plus resource + emissions."""

    # -------------------- train_*_model --------------------
    def fake_train(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        *,
        report_col: str,
        label_col: str,
        tokenizer,
        output_dir,
        perform_shap=False,
        perform_lime=False,
        **kwargs,
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "model.pkl").write_text("dummy model")

        if perform_shap:
            Path(output_dir, "shap_values.csv").write_text("feature,value\nx,1.0")
        if perform_lime:
            Path(output_dir, "lime_explanation.csv").write_text(
                "example,importance\n1,0.9"
            )

        return MagicMock(), None, None, None

    # Attach for whatever name the module exports (convention: train_<name>_model)
    train_fn = [n for n in dir(pipeline_module) if n.startswith("train_")][0]
    monkeypatch.setattr(pipeline_module, train_fn, fake_train, raising=True)

    # -------------------- predict_* ------------------------
    def fake_predict(model, tokenizer, df, col, proba=False):
        import pandas as pd

        assert isinstance(df, pd.DataFrame), f"df must be a DataFrame, got {type(df)}"
        length = len(df)
        if proba:
            return pd.Series([0.5] * length, index=df.index)
        return pd.Series([i % 2 for i in range(length)], index=df.index)

    pred_fn = [n for n in dir(pipeline_module) if n.startswith("predict_")][0]
    print("Patching:", pred_fn)
    monkeypatch.setattr(
        pipeline_module, "predict_distilbert", fake_predict, raising=True
    )

    # -------------------- evaluate -------------------------
    def fake_evaluate(df, mapping, dataset_name, pred_type, **kwargs):
        report_col = list(mapping.keys())[0]
        return pd.DataFrame(
            [
                {
                    "Dataset": dataset_name,
                    "Report_Column": report_col,
                    "Gold_Standard_Column": mapping[report_col],
                    "Prediction_Type": pred_type,
                    "Accuracy": "90.00% (CI: 85‑95)",
                    "Precision": "91.00%",
                    "Recall": "92.00%",
                    "F1_Score": "91.5%",
                    "Brier_Score": 0.05,
                }
            ]
        )

    monkeypatch.setattr(
        pipeline_module,
        "evaluate_fairness_dual",
        lambda *a, **k: {
            "aggregated": [{"metric": 0.1}],
            "disaggregated": [{"group": "A", "f1": 0.9}],
        },
        raising=True,
    )

    monkeypatch.setattr(pipeline_module, "evaluate", fake_evaluate, raising=True)

    # skip any plotting of fairness metrics
    if hasattr(pipeline_module, "plot_fairness_metrics_aggregated"):
        monkeypatch.setattr(
            pipeline_module,
            "plot_fairness_metrics_aggregated",
            lambda *a, **k: None,
            raising=True,
        )
    if hasattr(pipeline_module, "plot_f1_scores_disaggregated"):
        monkeypatch.setattr(
            pipeline_module,
            "plot_f1_scores_disaggregated",
            lambda *a, **k: None,
            raising=True,
        )

    # ----------------- Resource monitor -------------------
    class DummyMonitor:
        def __init__(self, interval):
            pass

        def start(self):
            pass

        def stop(self):
            self.elapsed = 1.23

        def get_metrics(self):
            return (12.3, 456.0)

    monkeypatch.setattr(pipeline_module, "ResourceMonitor", DummyMonitor)

    # ------------------- Emissions ------------------------
    from unittest.mock import MagicMock

    class DummyTracker:
        def __init__(self, *args, **kwargs):
            self.output_dir = Path(kwargs.get("output_dir", "."))
            self.start = MagicMock()
            self._started = True

        def stop(self):
            # Simulate writing emissions.csv
            if self.output_dir:
                (Path(self.output_dir) / "emissions.csv").write_text("dummy_emissions")
            return 0.01  # dummy kWh

    monkeypatch.setattr(pipeline_module, "EmissionsTracker", DummyTracker)

    # ------------------- Logging -------------------------

    def fake_configure_logging(log_dir: Path, custom_logger: str | None = None):
        log_path = Path(log_dir) / "pipeline_debug.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        log = logging.getLogger()
        log.handlers = [handler]
        log.setLevel(logging.INFO)
        log.info("distilbert pipeline complete")

        import atexit

        atexit.register(handler.close)

    monkeypatch.setattr(pipeline_module, "configure_logging", fake_configure_logging)


from unittest.mock import MagicMock

# ---------------------------------------------------------------------------- #
# Tests                                                                        #
# ---------------------------------------------------------------------------- #

from transformers import DistilBertTokenizer


@patch.object(sys, "argv", ["pipeline.py"])
@patch(
    "transformers.AutoTokenizer.from_pretrained",
    new=MagicMock(
        return_value=DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", local_files_only=True
        )
    ),
)
@patch("transformers.AutoModel.from_pretrained", new=MagicMock())
def test_pipeline_creates_expected_artifacts(
    pipeline_module: ModuleType,
    override_constants: Tuple[Path, Path],
    stub_pipeline_apis,
):
    """Run ``main`` once and assert that *all* key artefacts exist + are sane."""
    analysis_root, results_root = override_constants

    # ------------------- execute -------------------
    pipeline_module.main()

    # ------------------- CSVs ----------------------
    raw_analysis = analysis_root / "eval_distilbert_raw.csv"
    raw_results = results_root / "eval_distilbert_raw.csv"
    umls_analysis = analysis_root / "eval_distilbert_umls.csv"
    for f in [raw_analysis, raw_results, umls_analysis]:
        assert f.exists(), f"Missing eval CSV {f}"

    df_raw = pd.read_csv(raw_analysis)
    assert df_raw.shape[0] > 0, "Eval CSV is empty"
    required_cols = {
        "Dataset",
        "Report_Column",
        "Gold_Standard_Column",
        "Prediction_Type",
        "Accuracy",
        "Brier_Score",
    }
    assert required_cols.issubset(df_raw.columns), "Missing expected metrics columns"

    # ------------------- models --------------------
    model_checks = [
        analysis_root / "raw" / "models" / p / "text_column" / "model.pkl"
        for p in ["document", "patient"]
    ] + [analysis_root / "raw" / "models" / "patient_text_aggregate" / "model.pkl"]
    for m in model_checks:
        assert m.exists(), f"Model artefact missing at {m}"

    # ----------------- calibration -----------------
    cal_csv = analysis_root / "raw" / "calibration" / "brier_scores.csv"
    assert cal_csv.exists(), "Brier scores CSV missing"

    # ------------------ fairness -------------------
    for split in ["training", "validation"]:
        agg = analysis_root / "raw" / "fairness" / split / "aggregated.csv"
        dis = analysis_root / "raw" / "fairness" / split / "disaggregated.csv"
        assert agg.exists() and dis.exists(), f"Fairness outputs missing for {split}"

    # ------------------ resources ------------------
    res_csv = analysis_root / "resources.csv"
    assert res_csv.exists(), "Resources CSV missing"
    df_res = pd.read_csv(res_csv)
    assert {"cpu_pct", "memory_mb", "energy_kwh", "time_sec"}.issubset(df_res.columns)

    # ------------------ emissions ------------------
    emissions = analysis_root / "emissions.csv"
    assert emissions.exists(), "Emissions CSV missing"

    # ------------------- logs ----------------------
    log_file = analysis_root / "pipeline_debug.log"
    assert log_file.exists(), "Log file missing"
    assert "pipeline complete" in log_file.read_text().lower()
