"""
Structured application logging to file and stderr.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.core.config import get_settings


def setup_logger(name: str = "risk_intel", log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure the root application logger with rotation and dual handlers.

    Args:
        name: Logger name.
        log_file: Path to log file; defaults to settings.log_file.

    Returns:
        Configured logger instance.
    """
    settings = get_settings()
    path = log_file or settings.log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


app_logger = setup_logger()
