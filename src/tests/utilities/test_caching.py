"""

test caching.py

Tests for the caching function

"""

import os
import shutil
import time

import pytest

from nlp_pipeline.common.caching import cached, memory


# --------------------------------------------------------------------------- #
# Fixture: Clear and isolate cache directory before each test run             #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clear_cache(tmp_path, monkeypatch):
    """
    Redirect the joblib.Memory cache to a temporary path to ensure test isolation.
    Clears any existing cache files before and after the test execution.
    """
    monkeypatch.setattr(memory, "location", str(tmp_path))
    memory.clear(warn=False)
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Cached test function                                                        #
# --------------------------------------------------------------------------- #
@cached
def add(a, b):
    """
    Test function decorated with caching. Introduces a delay to simulate
    an expensive computation for evaluating cache speedup.

    Args:
        a (int): First operand.
        b (int): Second operand.

    Returns:
        int: Sum of a and b.
    """
    time.sleep(0.1)
    return a + b


# --------------------------------------------------------------------------- #
# Test: Validate correctness of cached function result                        #
# --------------------------------------------------------------------------- #
def test_caching_returns_correct_result():
    """
    Ensures that the cached function returns the expected value.
    """
    assert add(2, 3) == 5


# --------------------------------------------------------------------------- #
# Test: Verify cache effectiveness through execution time reduction           #
# --------------------------------------------------------------------------- #
def test_caching_uses_cache_and_is_faster():
    """
    Measures execution time to confirm that subsequent calls are faster
    due to caching. Also verifies that cache files are created.
    """
    # First call - expected to delay due to sleep
    start = time.time()
    result1 = add(4, 5)
    duration1 = time.time() - start

    # Second call - should be instant due to cache
    start = time.time()
    result2 = add(4, 5)
    duration2 = time.time() - start

    assert result1 == result2 == 9
    assert duration2 < duration1

    # Check cache directory is populated
    cache_files = list(os.walk(memory.location))
    assert any(cache_files)
