"""

- conftest.py

This script sets up the pytest test suite

"""

import importlib
import sys
from pathlib import Path
import pytest

# Add src to sys.path
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

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

def discover_pipeline_names():
    """Return list of logical pipeline names."""
    return list(PIPELINE_NAME_MAP.keys())

@pytest.fixture(scope="session", params=discover_pipeline_names())
def pipeline_module(request):
    """
    Parametrized fixture that loads each pipeline dynamically.
    """
    pipeline_name = request.param
    try:
        module_path = f"nlp_pipeline.models.{pipeline_name}.pipeline"
        module = importlib.import_module(module_path)
        return module
    except ModuleNotFoundError as e:
        raise ImportError(f"Could not load module: {module_path}. Error: {e}")

@pytest.fixture(autouse=True)
def disable_caplog(monkeypatch):
    monkeypatch.delattr("pytest._caplog", raising=False)
