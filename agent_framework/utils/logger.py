"""
logger.py

Logging utilities for PPO training runs.
"""

import logging

from datetime import datetime
from pathlib import Path

from config.config import config

LOG_FORMAT = "%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==========================================================
# Helpers
# ==========================================================


def safe_name(
    value: str,
) -> str:

    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def create_run_name(
    base_name: str,
) -> str:
    """
    Create a unique experiment/run identifier.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{safe_name(base_name)}_" f"{timestamp}"


# ==========================================================
# Run Directories
# ==========================================================


def prepare_run_directories(
    run_name: str,
):

    directories = {
        "logs": Path(config.logging.LOG_DIR) / run_name,
        "models": Path(config.logging.MODEL_DIR) / run_name,
        "metrics": Path(config.logging.METRICS_DIR) / run_name,
        "graphs": Path(config.logging.GRAPH_DIR) / run_name,
    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ==========================================================
# Run Logger
# ==========================================================


def get_run_logger(
    run_name: str,
    log_directory,
):

    logger_name = f"ppo.{safe_name(run_name)}"

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)

    logger.propagate = False

    # Remove previous handlers if reused.
    for handler in list(logger.handlers):

        handler.close()

        logger.removeHandler(handler)

    formatter = logging.Formatter(
        LOG_FORMAT,
        DATE_FORMAT,
    )

    # ------------------------------------------------------
    # Console
    # ------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # ------------------------------------------------------
    # File
    # ------------------------------------------------------

    log_file = Path(log_directory) / "training.log"

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def close_logger(
    logger,
):

    for handler in list(logger.handlers):

        handler.close()

        logger.removeHandler(handler)
