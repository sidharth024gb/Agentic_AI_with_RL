"""
cache.py

Persistent cache for LLM-generated plans.

The cache reduces unnecessary Ollama calls when the same:
    - model
    - goal
    - state
    - prompt version

have already been planned previously.
"""

import hashlib
import json

from pathlib import Path
from typing import Any, Dict, Optional

from config.config import config


class LLMPlanCache:
    """
    Persistent JSON-file cache for LLM plans.

    Each cache entry is stored as a separate file using
    a SHA-256 key.
    """

    def __init__(
        self,
        cache_directory=None,
        enabled=None,
    ):

        if enabled is None:
            enabled = config.llm.USE_CACHE

        if cache_directory is None:
            cache_directory = config.llm.CACHE_DIR

        self.enabled = bool(
            enabled
        )

        self.cache_directory = Path(
            cache_directory
        )

        if self.enabled:

            self.cache_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.hits = 0

        self.misses = 0

        self.writes = 0
        
    # ==========================================================
    # Cache Key
    # ==========================================================

    @staticmethod
    def make_key(
        model,
        goal,
        state,
        prompt_version,
    ):
        """
        Build deterministic cache key.
        """

        payload = {
            "model": model,
            "goal": goal,
            "state": state,
            "prompt_version": prompt_version,
        }

        canonical = json.dumps(
            payload,
            sort_keys=True,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )

        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ==========================================================
    # Cache Path
    # ==========================================================

    def _path(
        self,
        key,
    ):

        return self.cache_directory / f"{key}.json"

    # ==========================================================
    # Read
    # ==========================================================

    def get(
        self,
        key,
    ) -> Optional[Dict[str, Any]]:

        if not self.enabled:

            return None

        path = self._path(key)

        if not path.exists():

            self.misses += 1

            return None

        try:

            with path.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            self.hits += 1

            return data

        except (
            OSError,
            json.JSONDecodeError,
        ):

            # Corrupt entries should not break training.
            self.misses += 1

            return None

    # ==========================================================
    # Write
    # ==========================================================

    def set(
        self,
        key,
        value,
    ):

        if not self.enabled:

            return

        path = self._path(key)

        temporary_path = path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                value,
                file,
                indent=2,
                default=str,
            )

        temporary_path.replace(path)

        self.writes += 1

    # ==========================================================
    # Delete Entry
    # ==========================================================

    def delete(
        self,
        key,
    ):

        if not self.enabled:

            return False

        path = self._path(key)

        if not path.exists():

            return False

        path.unlink()

        return True

    # ==========================================================
    # Clear Cache
    # ==========================================================

    def clear(
        self,
    ):

        if not self.cache_directory.exists():

            return

        for path in self.cache_directory.glob("*.json"):

            try:

                path.unlink()

            except OSError:

                pass

        self.hits = 0

        self.misses = 0

        self.writes = 0

    # ==========================================================
    # Statistics
    # ==========================================================

    def get_stats(
        self,
    ):

        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "cache_directory": str(self.cache_directory),
        }
