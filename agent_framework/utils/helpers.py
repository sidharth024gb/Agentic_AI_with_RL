"""
helpers.py

General-purpose helper utilities for the finance RL agent project.

This module contains small reusable functions that do not belong
to a specific component such as the environment, agent, training
pipeline, or metrics system.
"""

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# ==============================================================
# Random Seed
# ==============================================================


def set_seed(
    seed: int,
) -> None:
    """
    Set random seeds for reproducibility.

    This affects:

        - Python random
        - NumPy
    """

    random.seed(seed)

    np.random.seed(seed)


# ==============================================================
# Directory Utilities
# ==============================================================


def ensure_directory(
    directory: str,
) -> str:
    """
    Create a directory if it does not already exist.

    Returns
    -------
    str
        The directory path.
    """

    os.makedirs(
        directory,
        exist_ok=True,
    )

    return directory


def ensure_parent_directory(
    file_path: str,
) -> str:
    """
    Create the parent directory of a file if necessary.

    Returns
    -------
    str
        The original file path.
    """

    parent = os.path.dirname(file_path)

    if parent:

        os.makedirs(
            parent,
            exist_ok=True,
        )

    return file_path


# ==============================================================
# JSON Utilities
# ==============================================================


def save_json(
    data: Dict[str, Any],
    file_path: str,
    indent: int = 4,
) -> str:
    """
    Save a dictionary as a JSON file.
    """

    ensure_parent_directory(file_path)

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=_json_serializer,
        )

    return file_path


def load_json(
    file_path: str,
) -> Dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==============================================================
# Timestamp Utilities
# ==============================================================


def get_timestamp(
    include_microseconds: bool = False,
) -> str:
    """
    Return the current local timestamp.

    Examples
    --------
    2026-08-09_10-45-30

    or:

    2026-08-09_10-45-30-123456
    """

    now = datetime.now()

    if include_microseconds:

        return now.strftime("%Y-%m-%d_%H-%M-%S-%f")

    return now.strftime("%Y-%m-%d_%H-%M-%S")


def get_timestamp_iso() -> str:
    """
    Return the current timestamp in ISO format.
    """

    return datetime.now().isoformat()


# ==============================================================
# Filename Utilities
# ==============================================================


def safe_filename(
    name: str,
) -> str:
    """
    Convert a string into a filesystem-safe filename.

    Example
    -------
    "PPO vs LLM + PPO"
        ->
    "ppo_vs_llm_plus_ppo"
    """

    invalid_characters = '\\/:*?"<>|'

    result = name.strip()

    for character in invalid_characters:

        result = result.replace(
            character,
            "_",
        )

    result = result.replace(
        "+",
        "_plus_",
    ).replace(
        " ",
        "_",
    )

    return result.lower()


# ==============================================================
# Nested Dictionary Utilities
# ==============================================================


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary.

    Example
    -------

    {
        "training": {
            "reward": 10
        }
    }

    becomes:

    {
        "training.reward": 10
    }
    """

    flattened = {}

    for key, value in data.items():

        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)

        if isinstance(
            value,
            dict,
        ):

            flattened.update(
                flatten_dict(
                    value,
                    new_key,
                    separator,
                )
            )

        else:

            flattened[new_key] = value

    return flattened


# ==============================================================
# Safe Value Conversion
# ==============================================================


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    if value is None:

        return default

    try:

        return float(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a value to int.
    """

    if value is None:

        return default

    try:

        return int(value)

    except (
        TypeError,
        ValueError,
    ):

        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    """
    Safely convert common values to bool.

    Handles strings such as:

        "true"
        "false"
        "yes"
        "no"
        "1"
        "0"
    """

    if value is None:

        return default

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        (int, float),
    ):

        return bool(value)

    if isinstance(
        value,
        str,
    ):

        normalized = value.strip().lower()

        if normalized in {
            "true",
            "yes",
            "y",
            "1",
        }:

            return True

        if normalized in {
            "false",
            "no",
            "n",
            "0",
        }:

            return False

    return default


# ==============================================================
# List Utilities
# ==============================================================


def chunk_list(
    values: list,
    chunk_size: int,
) -> list:
    """
    Split a list into smaller chunks.

    Example
    -------

    [1, 2, 3, 4, 5]

    with chunk_size=2 becomes:

    [
        [1, 2],
        [3, 4],
        [5]
    ]
    """

    if chunk_size <= 0:

        raise ValueError("chunk_size must be greater than zero.")

    return [
        values[index : index + chunk_size]
        for index in range(
            0,
            len(values),
            chunk_size,
        )
    ]


# ==============================================================
# Numeric Utilities
# ==============================================================


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Clamp a numeric value between minimum and maximum.
    """

    if minimum > maximum:

        raise ValueError("minimum cannot be greater than maximum.")

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ==============================================================
# Result Utilities
# ==============================================================


def create_run_id(
    prefix: Optional[str] = None,
) -> str:
    """
    Create a unique identifier for a training or experiment run.

    Example:

        ppo_2026-08-09_10-45-30
    """

    timestamp = get_timestamp()

    if prefix:

        return f"{safe_filename(prefix)}" f"_{timestamp}"

    return timestamp


# ==============================================================
# Environment Response Utilities
# ==============================================================


def extract_response_data(
    response: Any,
) -> Any:
    """
    Safely extract the 'data' field from a backend response.

    Expected backend response format:

        {
            "success": True,
            "data": {...}
        }

    If the expected structure is not present, the original
    response is returned.
    """

    if not isinstance(
        response,
        dict,
    ):

        return response

    if "data" in response:

        return response["data"]

    return response


def response_success(
    response: Any,
) -> bool:
    """
    Safely determine whether a backend response succeeded.
    """

    if not isinstance(
        response,
        dict,
    ):

        return False

    return bool(
        response.get(
            "success",
            False,
        )
    )


def response_message(
    response: Any,
    default: str = "",
) -> str:
    """
    Extract the message from a backend response.
    """

    if not isinstance(
        response,
        dict,
    ):

        return default

    message = response.get("message")

    if message is None:

        return default

    return str(message)


# ==============================================================
# Internal JSON Serializer
# ==============================================================


def _json_serializer(
    value: Any,
):
    """
    Convert common NumPy values into JSON-compatible values.
    """

    if isinstance(
        value,
        np.integer,
    ):

        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        datetime,
    ):

        return value.isoformat()

    raise TypeError(
        f"Object of type " f"{type(value).__name__} " f"is not JSON serializable."
    )
