"""
Resource Monitoring Module

This module provides functionality for monitoring system resource usage (CPU and memory)
and timing function execution. It includes:
    - ResourceMonitor: A class that tracks CPU and memory usage at regular intervals using a background thread.
    - timer: A decorator that logs the start, end, and execution duration of any function it wraps.

Dependencies:
    - time: For time-based operations.
    - psutil: For retrieving system resource usage information.
    - threading: To run the monitor in a separate thread.
    - contextlib: To support context managers (if needed).
    - codecarbon.EmissionsTracker: Optional; for tracking carbon emissions (imported for extensibility).
    - numpy: To compute statistical metrics.
    - functools: For decorator utilities.
    - logging: To log messages.

Usage Example:

    # Monitoring resource usage:
    monitor = ResourceMonitor(interval=0.1)
    monitor.start()
    # ... run some resource-intensive operations ...
    monitor.stop()
    avg_cpu, peak_mem = monitor.get_metrics()
    print(f"Average CPU usage: {avg_cpu:.2f}%, Peak Memory Usage: {peak_mem:.2f} MB")

    # Timing function execution:
    @timer
    def sample_function():
        # Function body here
        time.sleep(1)

    sample_function()

Note:
    The EmissionsTracker from codecarbon is imported for potential extension into
    energy and carbon footprint tracking. It is not used in this snippet.
"""

import logging
import threading
import time
from contextlib import contextmanager
from functools import wraps

import numpy as np
import psutil
from codecarbon import EmissionsTracker

# Configure logging to output at the INFO level.
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Resource Monitor Class                                                      #
# --------------------------------------------------------------------------- #

class ResourceMonitor:
    """
    Monitors system resource usage (CPU and memory) over time.

    Attributes:
        interval (float): Time in seconds between metric collection.
        cpu_usage (list of float): Recorded CPU usage percentages.
        memory_usage (list of float): Recorded memory usage in megabytes.
    """

    def __init__(self, interval: float = 0.1) -> None:
        """
        Initialises the ResourceMonitor with a specified interval for data collection.

        Args:
            interval (float): Interval in seconds between each measurement. Defaults to 0.1 seconds.
        """
        self.interval = interval
        self.cpu_usage = []
        self.memory_usage = []
        self._monitoring = False
        self._thread = None

    def _monitor(self) -> None:
        """
        Private method that continuously collects CPU and memory usage data while monitoring is active.

        This method is intended to run in a separate thread.
        """
        process = psutil.Process()  # Get current process info for memory usage metrics.
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        self.memory_usage.append(process.memory_info().rss / (1024**2))
        while self._monitoring:
            # Append the current CPU usage percentage.
            self.cpu_usage.append(psutil.cpu_percent(interval=None))
            # Append the memory usage in MB.
            self.memory_usage.append(process.memory_info().rss / (1024**2))
            # Wait for the specified interval before the next measurement.
            time.sleep(self.interval)

    def start(self) -> None:
        """
        Starts the resource monitoring by initiating a separate thread to collect data.
        """
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor)
        self._thread.start()

    def stop(self) -> None:
        """
        Stops the resource monitoring and waits for the monitoring thread to complete.
        """
        self._monitoring = False
        if self._thread:
            self._thread.join()

    def get_metrics(self) -> tuple:
        """
        Computes and returns the average CPU usage and peak memory usage recorded.

        Returns:
            tuple: A tuple containing:
                - avg_cpu (float): The average CPU usage percentage.
                - peak_mem (float): The maximum memory usage in megabytes.
        """
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0.0
        peak_mem = np.max(self.memory_usage) if self.memory_usage else 0.0
        return avg_cpu, peak_mem

    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

# --------------------------------------------------------------------------- #
# Timer                                                                       #
# --------------------------------------------------------------------------- #

def timer(func):
    """
    Decorator that logs the execution time of the decorated function.

    The decorator logs:
      - A message indicating the function start.
      - A message with the total execution time upon function completion.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: The wrapped function that logs its execution time.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.info(f"Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logging.info(
            f"Function '{func.__name__}' executed in {elapsed_time:.2f} seconds."
        )
        return result

    return wrapper
