"""
logger.py

Centralized logging utilities for the finance RL agent project.

This module provides:

    - console logging
    - file logging
    - consistent log formatting
    - separate loggers for different project components

Metrics and experiment results should be handled by:
    - training.callbacks
    - utils.metrics
    - training.experiment

This module is only responsible for application/training logs.
"""

import logging
import os
from typing import Optional

# ==============================================================
# Constants
# ==============================================================

DEFAULT_LOG_LEVEL = logging.INFO

DEFAULT_LOG_DIRECTORY = "results/logs"

DEFAULT_LOG_FORMAT = "%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ==============================================================
# Logger Creation
# ==============================================================


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = DEFAULT_LOG_LEVEL,
    console: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured project logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_file : str, optional
        Path to the log file.

    level : int
        Logging level.

    console : bool
        Whether to log to the console.

    Returns
    -------
    logging.Logger
        Configured logger.
    """

    logger = logging.getLogger(name)

    logger.setLevel(level)

    logger.propagate = False

    # ----------------------------------------------------------
    # Avoid duplicate handlers
    # ----------------------------------------------------------

    if logger.handlers:

        return logger

    formatter = logging.Formatter(
        fmt=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )

    # ----------------------------------------------------------
    # Console Handler
    # ----------------------------------------------------------

    if console:

        console_handler = logging.StreamHandler()

        console_handler.setLevel(level)

        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    # ----------------------------------------------------------
    # File Handler
    # ----------------------------------------------------------

    if log_file:

        log_directory = os.path.dirname(log_file)

        if log_directory:

            os.makedirs(
                log_directory,
                exist_ok=True,
            )

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )

        file_handler.setLevel(level)

        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


# ==============================================================
# Experiment Logger
# ==============================================================


def get_experiment_logger(
    experiment_name: str,
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Create a logger for an experiment.

    Example
    -------
    logger = get_experiment_logger(
        "ppo_vs_llm_ppo"
    )
    """

    safe_name = experiment_name.lower().replace(" ", "_").replace("+", "_plus_")

    os.makedirs(
        log_directory,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_directory,
        f"{safe_name}.log",
    )

    return get_logger(
        name=f"experiment.{safe_name}",
        log_file=log_file,
        level=level,
        console=True,
    )


# ==============================================================
# Agent Logger
# ==============================================================


def get_agent_logger(
    agent_name: str,
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Create a logger for an individual agent.
    """

    safe_name = agent_name.lower().replace(" ", "_").replace("+", "_plus_")

    os.makedirs(
        log_directory,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_directory,
        f"{safe_name}.log",
    )

    return get_logger(
        name=f"agent.{safe_name}",
        log_file=log_file,
        level=level,
        console=True,
    )


# ==============================================================
# Environment Logger
# ==============================================================


def get_environment_logger(
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Create the logger used by the finance environment.
    """

    os.makedirs(
        log_directory,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_directory,
        "environment.log",
    )

    return get_logger(
        name="environment.finance",
        log_file=log_file,
        level=level,
        console=True,
    )


# ==============================================================
# LLM Logger
# ==============================================================


def get_llm_logger(
    log_directory: str = DEFAULT_LOG_DIRECTORY,
    level: int = DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Create the logger used by the LLM components.
    """

    os.makedirs(
        log_directory,
        exist_ok=True,
    )

    log_file = os.path.join(
        log_directory,
        "llm.log",
    )

    return get_logger(
        name="llm",
        log_file=log_file,
        level=level,
        console=True,
    )


# ==============================================================
# Utility Functions
# ==============================================================


def set_log_level(
    logger: logging.Logger,
    level: int,
):
    """
    Change the logging level for an existing logger
    and all of its handlers.
    """

    logger.setLevel(level)

    for handler in logger.handlers:

        handler.setLevel(level)


def close_logger(
    logger: logging.Logger,
):
    """
    Close and remove all handlers from a logger.

    Useful when an experiment finishes and a new experiment
    needs to create a logger using the same name.
    """

    for handler in logger.handlers:

        handler.close()

        logger.removeHandler(handler)
