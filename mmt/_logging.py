"""Logging configuration for the mmt package.

Errors (with full stack traces) are written to mmt_errors.log in the
current working directory.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a logger that appends WARNING+ entries (with tracebacks) to mmt_errors.log."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_path = Path(os.getcwd()) / "mmt_errors.log"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fmt = logging.Formatter(
            "%(asctime)s  %(name)s  %(levelname)s\n%(message)s\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.setLevel(logging.WARNING)
    return logger
