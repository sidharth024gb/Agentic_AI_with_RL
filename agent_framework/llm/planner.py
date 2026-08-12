"""
planner.py

Ollama-based high-level finance task planner.

Responsibilities:
    - build planner prompts
    - communicate with Ollama
    - parse and validate plans
    - use persistent plan cache
    - measure LLM latency

The planner does NOT execute environment actions.
"""

import os
import time

from typing import Optional

import requests

from config.config import config

from llm.cache import LLMPlanCache

from llm.parser import (
    PlanParseError,
    parse_plan_response,
)

from llm.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_planner_prompt,
)


class LLMPlanner:
    """
    High-level task planner using Ollama.
    """

    def __init__(
        self,
        model=None,
        base_url=None,
        timeout=None,
        temperature=None,
        use_cache=None,
        cache=None,
    ):

        # ==========================================================
        # LLM Configuration
        # ==========================================================

        self.model = (
            model
            if model is not None
            else config.llm.MODEL
        )

        self.base_url = (
            base_url
            if base_url is not None
            else config.llm.BASE_URL
        ).rstrip("/")

        self.timeout = int(
            timeout
            if timeout is not None
            else config.llm.TIMEOUT
        )

        self.temperature = float(
            temperature
            if temperature is not None
            else config.llm.TEMPERATURE
        )

        self.use_cache = bool(
            use_cache
            if use_cache is not None
            else config.llm.USE_CACHE
        )

        # ==========================================================
        # Ollama Endpoint
        # ==========================================================

        self.generate_endpoint = (
            f"{self.base_url}/api/generate"
        )

        # ==========================================================
        # HTTP Session
        # ==========================================================

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Content-Type":
                    "application/json",
            }
        )

        # ==========================================================
        # Cache
        # ==========================================================

        self.cache = (
            cache
            if cache is not None
            else LLMPlanCache(
                enabled=self.use_cache
            )
        )

        # ==========================================================
        # Planner Metrics
        # ==========================================================

        self.total_calls = 0

        self.failed_calls = 0

        self.total_latency_ms = 0.0

    # ==========================================================
    # Generate Cache Key
    # ==========================================================

    def _cache_key(
        self,
        goal,
        state,
    ):

        return self.cache.make_key(
            model=self.model,
            goal=goal,
            state=state,
            prompt_version=PROMPT_VERSION,
        )

    # ==========================================================
    # Ollama Request
    # ==========================================================

    def _call_ollama(
        self,
        prompt,
    ):
        """
        Send one request to Ollama.

        Returns
        -------
        tuple
            raw_text, latency_ms
        """

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            # Ask Ollama to produce JSON.
            "format": "json",
            "options": {
                "temperature": self.temperature,
            },
        }

        started = time.perf_counter()

        response = self.session.post(
            self.generate_endpoint,
            json=payload,
            timeout=self.timeout,
        )

        latency_ms = (time.perf_counter() - started) * 1000.0

        response.raise_for_status()

        data = response.json()

        raw_text = data.get("response")

        if not raw_text:

            raise RuntimeError("Ollama returned an empty response.")

        return (
            raw_text,
            latency_ms,
        )

    # ==========================================================
    # Plan
    # ==========================================================

    def plan(
        self,
        goal,
        state,
        force_refresh=False,
    ):
        """
        Generate a validated high-level plan.

        Parameters
        ----------
        goal : str

        state : dict
            Current binary environment state.

        force_refresh : bool
            Ignore any cached plan.

        Returns
        -------
        dict

        Success result:

        {
            "success": True,
            "cached": False,
            "model": "llama3",
            "action_names": [...],
            "action_ids": [...],
            "prerequisites": {...},
            "llm_plan": [...],
            "latency_ms": ...
        }

        Failure result:

        {
            "success": False,
            "error": "...",
            ...
        }
        """

        if (
            not isinstance(
                goal,
                str,
            )
            or not goal.strip()
        ):

            return {
                "success": False,
                "error": "Planner goal is empty.",
                "cached": False,
                "model": self.model,
            }

        if state is None:

            state = {}

        key = self._cache_key(
            goal,
            state,
        )

        # ======================================================
        # Cache Lookup
        # ======================================================

        if self.use_cache and not force_refresh:

            cached_result = self.cache.get(key)

            if cached_result is not None:

                cached_result["cached"] = True

                return cached_result

        # ======================================================
        # Build Prompt
        # ======================================================

        prompt = build_planner_prompt(
            goal=goal,
            state=state,
        )

        self.total_calls += 1

        # ======================================================
        # Ollama
        # ======================================================

        try:

            (
                raw_response,
                latency_ms,
            ) = self._call_ollama(prompt)

            self.total_latency_ms += latency_ms

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as exc:

            self.failed_calls += 1

            return {
                "success": False,
                "cached": False,
                "model": self.model,
                "error": f"Ollama request failed: {exc}",
            }

        # ======================================================
        # Parse Plan
        # ======================================================

        try:

            parsed_plan = parse_plan_response(raw_response)

        except PlanParseError as exc:

            self.failed_calls += 1

            return {
                "success": False,
                "cached": False,
                "model": self.model,
                "latency_ms": latency_ms,
                "raw_response": raw_response,
                "error": f"Plan parsing failed: {exc}",
            }

        # ======================================================
        # Result
        # ======================================================

        result = {
            "success": True,
            "cached": False,
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": float(latency_ms),
            "action_names": parsed_plan.action_names,
            "action_ids": parsed_plan.action_ids,
            "prerequisites": parsed_plan.prerequisites,
            # This can be stored directly in Episode.llmPlan
            "llm_plan": parsed_plan.action_names,
            "raw_response": raw_response,
        }

        # ======================================================
        # Cache Successful Plan
        # ======================================================

        if self.use_cache:

            self.cache.set(
                key,
                result,
            )

        return result

    # ==========================================================
    # Planner Metrics
    # ==========================================================

    def get_metrics(
        self,
    ):
        """
        Return LLM planner metrics.
        """

        successful_calls = self.total_calls - self.failed_calls

        average_latency_ms = 0.0

        if successful_calls > 0:

            average_latency_ms = self.total_latency_ms / successful_calls

        return {
            "model": self.model,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "successful_calls": successful_calls,
            "total_latency_ms": self.total_latency_ms,
            "average_latency_ms": average_latency_ms,
            "cache": self.cache.get_stats(),
        }

    # ==========================================================
    # Close
    # ==========================================================

    def close(
        self,
    ):

        self.session.close()
