"""

test logging.py

Tests for logging

"""

import logging
import sys
from pathlib import Path
from typing import List

import pytest

from nlp_pipeline.common.logging_setup import configure_logging


# --------------------------------------------------------------------------- #
# Helper: Split file and stream handlers                                      #
# --------------------------------------------------------------------------- #
def split_handlers(handlers: List[logging.Handler]):
    """Return (file_handlers, stream_handlers)."""
    file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
    # a FileHandler is also a StreamHandler, so exclude those
    stream_handlers = [
        h
        for h in handlers
        if isinstance(h, logging.StreamHandler) and h not in file_handlers
    ]
    return file_handlers, stream_handlers


# --------------------------------------------------------------------------- #
# Helper: Buffer contents from logger                                         #
# --------------------------------------------------------------------------- #
def buffer_contents(root: logging.Logger) -> str | None:
    """Return whatever root.log_buffer contains, or None if unavailable."""
    buf = getattr(root, "log_buffer", None)
    if buf and callable(getattr(buf, "getvalue", None)):
        return buf.getvalue()
    return None


# --------------------------------------------------------------------------- #
# Test: configure_logging creates directory and file                          #
# --------------------------------------------------------------------------- #
def test_configure_logging_creates_directory_and_file(tmp_path, capsys):
    """
    Verifies that configure_logging creates the specified log directory and file,
    writes the configuration banner to the log file, and outputs the path to stderr.
    """
    log_dir = tmp_path / "logs"
    log_filename = "my_app.log"
    expected_path = log_dir / log_filename

    configure_logging(
        log_dir=str(log_dir), log_filename=log_filename, level=logging.DEBUG
    )

    captured = capsys.readouterr()

    # directory + file exist
    assert expected_path.is_file()

    text = expected_path.read_text()
    assert "Logging configured. Logs will be saved to:" in text
    assert str(expected_path) in text

    # console message went to stderr
    assert str(expected_path) in captured.err

    # in-memory buffer (if provided) contains same banner
    buf = buffer_contents(logging.getLogger())
    if buf is not None:
        assert str(expected_path) in buf


# --------------------------------------------------------------------------- #
# Test: Handlers attached to root logger                                      #
# --------------------------------------------------------------------------- #
def test_handlers_on_root_logger(tmp_path):
    """
    Ensures that configure_logging attaches at least one FileHandler and one
    StreamHandler to the root logger with appropriate levels.
    """
    log_dir = tmp_path / "logs2"
    log_filename = "another.log"

    root = logging.getLogger()
    root.handlers.clear()  # clean slate

    configure_logging(
        log_dir=str(log_dir), log_filename=log_filename, level=logging.WARNING
    )

    file_handlers, stream_handlers = split_handlers(root.handlers)

    # at least one of each
    assert file_handlers, "no FileHandler attached"
    assert stream_handlers, "no StreamHandler attached"

    # levels
    fh = file_handlers[0]
    sh = stream_handlers[0]

    assert fh.level in (0, logging.WARNING)  # 0 == NOTSET inherits root
    assert sh.level == logging.INFO
    assert root.level == logging.WARNING


# --------------------------------------------------------------------------- #
# Test: Default log directory falls back to CWD                               #
# --------------------------------------------------------------------------- #
def test_default_log_dir_falls_back_to_cwd(
    monkeypatch, capsys, tmp_path, tmp_path_factory
):
    """
    Validates that when log_dir is None, configure_logging falls back to the
    current working directory of the main module.
    """
    import sys as _sys

    dummy_main = type(_sys)("__main__")
    dummy_file = tmp_path / "script.py"
    dummy_file.write_text("# dummy")
    dummy_main.__file__ = str(dummy_file)
    monkeypatch.setitem(_sys.modules, "__main__", dummy_main)

    new_cwd = tmp_path_factory.mktemp("cwd")
    monkeypatch.chdir(new_cwd)

    configure_logging(log_dir=None, log_filename="fallback.log", level=logging.INFO)

    captured = capsys.readouterr()

    expected_path = Path(dummy_file).with_name("fallback.log")
    assert expected_path.is_file()
    assert str(expected_path) in captured.err
