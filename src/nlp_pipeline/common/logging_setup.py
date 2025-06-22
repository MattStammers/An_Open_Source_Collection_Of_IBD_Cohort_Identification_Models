"""
Logging Configuration Module

This module provides a function to configure logging with a dual-handler setup:
    - A file handler that writes logs to a file with delayed opening to prevent lockups.
    - A console stream handler that outputs logs to standard error.

It is designed to:
    - Leave existing handlers (e.g., from pytest) unchanged.
    - Automatically determine the logging directory based on the main module or current working directory.
    - Log an initialisation banner at the INFO level.
"""

import logging
import os
import sys
from typing import Optional

# --------------------------------------------------------------------------- #
# Logging Configuration                                                       #
# --------------------------------------------------------------------------- #

def configure_logging(
    log_dir: Optional[str] = None,
    log_filename: str = "pipeline_debug.log",
    level: int = logging.DEBUG,
    console: bool = True,
    custom_logger: logging.Logger | None = None,
) -> None:
    """
    Configure logging with both a file handler and a console handler.

    This function:
      - Leaves any pre-existing logging handlers (e.g., from test runners) intact.
      - Adds a `FileHandler` with delayed file opening to avoid locking issues.
      - Adds a `StreamHandler` for console output at INFO level or higher.
      - Writes an initialisation message indicating where logs are stored.

    Args:
        log_dir (Optional[str], optional): The directory where the log file will be saved.
            If None, it is inferred from the `__main__` module or current working directory.
        log_filename (str): The name of the log file. Defaults to "pipeline_debug.log".
        level (int): Logging level. Defaults to `logging.DEBUG`.

    Returns:
        None
    """
    logger = custom_logger or logging.getLogger()
    logger.setLevel(level)
    # Determine directory for log file.
    if log_dir is None:
        main_mod = sys.modules.get("__main__")
        if main_mod and getattr(main_mod, "__file__", None):
            log_dir = os.path.dirname(os.path.abspath(main_mod.__file__))
        else:
            log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # Root logger configuration.
    logger = logging.getLogger()
    logger.setLevel(level)

    for lib in ("matplotlib", "urllib3", "font_manager"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # File handler with delayed write.
    fh = logging.FileHandler(log_path, mode="w", delay=True)
    fh.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Stream handler for console output
    if console:
        ch = logging.StreamHandler()  # defaults to stderr
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # Log initialisation banner.
    logger.info(f"Logging configured. Logs will be saved to: {log_path}")
