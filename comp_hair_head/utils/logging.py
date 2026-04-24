"""Logging utilities for CompHairHead."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOGGER_NAME = "comp_hair_head"
_configured = False


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Sub-logger name (e.g. "gaussian.renderer"). If None, returns root logger.

    Returns:
        Configured logger instance.
    """
    global _configured
    if not _configured:
        _setup_root_logger()
        _configured = True

    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def _setup_root_logger() -> None:
    """Configure root logger with console handler."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def setup_file_logging(log_dir: str | Path, level: int = logging.DEBUG) -> None:
    """Add file handler to root logger.

    Args:
        log_dir: Directory to write log files.
        level: Logging level for file handler.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    fh = logging.FileHandler(log_dir / "comp_hair_head.log")
    fh.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
