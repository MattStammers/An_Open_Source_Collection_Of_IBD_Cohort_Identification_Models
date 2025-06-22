"""

test resource_monitor.py

Tests for the resource monitor

"""

import logging
import threading
import time

import numpy as np
import psutil
import pytest

from nlp_pipeline.common.resource_monitor import ResourceMonitor, timer

# --------------------------------------------------------------------------- #
# Dummy classes for psutil patching                                           #
# --------------------------------------------------------------------------- #
class DummyMemInfo:
    def __init__(self, rss):
        self.rss = rss


class DummyProcess:
    def __init__(self, mem_values):
        # mem_values is an iterator of rss values in bytes
        self._iter = iter(mem_values)

    def memory_info(self):
        # return a simple object with .rss
        try:
            return DummyMemInfo(next(self._iter))
        except StopIteration:
            # once exhausted, always return last value
            return DummyMemInfo(0)


# --------------------------------------------------------------------------- #
# Fixture: Patch psutil for consistent CPU and memory values                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_psutil(monkeypatch):
    """
    Monkeypatch psutil.cpu_percent to 42% and psutil.Process to yield fixed memory values.
    """
    # Always return 42% CPU
    monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: 42.0)
    # Memory in bytes: 100MB, 200MB, 300MB
    byte_vals = [100 * 1024**2, 200 * 1024**2, 300 * 1024**2]
    monkeypatch.setattr(psutil, "Process", lambda: DummyProcess(byte_vals))
    yield


# --------------------------------------------------------------------------- #
# Test: ResourceMonitor collects metrics and computes get_metrics properly    #
# --------------------------------------------------------------------------- #
def test_resource_monitor_collects_metrics_and_get_metrics():
    """
    Validates that ResourceMonitor collects CPU and memory samples at the given interval,
    and that get_metrics returns average CPU and peak memory in MB.
    """
    monitor = ResourceMonitor(interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()

    # Samples collected
    assert monitor.cpu_usage
    assert monitor.memory_usage

    # CPU always patched to 42.0
    assert all(cpu == 42.0 for cpu in monitor.cpu_usage)

    # Memory in MB matches first three patched values
    np.testing.assert_allclose(
        monitor.memory_usage[:3], [100.0, 200.0, 300.0], atol=1e-6
    )

    avg_cpu, peak_mem = monitor.get_metrics()
    assert avg_cpu == pytest.approx(42.0)
    assert peak_mem == pytest.approx(max(monitor.memory_usage))

# --------------------------------------------------------------------------- #
# Test: ResourceMonitor stops sampling after stop is called                   #
# --------------------------------------------------------------------------- #
def test_resource_monitor_stops_collecting_after_stop():
    """
    Ensures that no additional CPU or memory samples are recorded after stop().
    """
    monitor = ResourceMonitor(interval=0.01)
    monitor.start()
    time.sleep(0.03)
    monitor.stop()
    count_after_stop = len(monitor.cpu_usage)

    time.sleep(0.02)
    assert len(monitor.cpu_usage) == count_after_stop
    assert len(monitor.memory_usage) == count_after_stop


# --------------------------------------------------------------------------- #
# Test: timer decorator logs start and completion messages                    #
# --------------------------------------------------------------------------- #
def test_timer_decorator_logs_execution_time(caplog):
    """
    Validates that the timer decorator logs the start and duration of the function.
    """
    caplog.set_level(logging.INFO)

    @timer
    def fast_func(x, y):
        time.sleep(0.01)
        return x + y

    result = fast_func(3, 4)
    assert result == 7

    msgs = [r.message for r in caplog.records if "fast_func" in r.message]
    assert any("Starting 'fast_func'..." in m for m in msgs)
    assert any("Function 'fast_func' executed in" in m for m in msgs)