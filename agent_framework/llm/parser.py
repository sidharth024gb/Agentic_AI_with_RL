"""
parser.py

Parser and validator for LLM-generated finance plans.

Responsibilities:
    - extract JSON from LLM responses
    - validate action names
    - convert action names to action IDs
    - remove accidental duplicates
    - generate prerequisite relationships

The parser DOES NOT correct or redesign the LLM plan.
"""

import json

from dataclasses import dataclass
from typing import Dict, List

from environment.action_space import FinanceAction

# ==========================================================
# Exceptions
# ==========================================================


class PlanParseError(ValueError):
    """Raised when an LLM plan cannot be parsed safely."""

    pass


# ==========================================================
# Parsed Plan
# ==========================================================


@dataclass
class ParsedPlan:
    """
    Validated representation of an LLM-generated plan.
    """

    action_names: List[str]

    action_ids: List[int]

    prerequisites: Dict[int, List[int]]

    raw_payload: dict

    # ======================================================
    # Helpers
    # ======================================================

    def to_dict(
        self,
    ):

        return {
            "action_names": self.action_names,
            "action_ids": self.action_ids,
            "prerequisites": self.prerequisites,
            "raw_payload": self.raw_payload,
        }

    def is_empty(
        self,
    ):

        return len(self.action_ids) == 0


# ==========================================================
# Action Mappings
# ==========================================================

ACTION_NAME_TO_ID = {action.name: int(action.value) for action in FinanceAction}


ACTION_ID_TO_NAME = {int(action.value): action.name for action in FinanceAction}


# ==========================================================
# Normalize Action
# ==========================================================


def normalize_action_name(
    action_name,
):
    """
    Normalize common formatting differences.

    Examples
    --------
    "get invoices"
        -> GET_INVOICES

    "check-budget"
        -> CHECK_BUDGET
    """

    if not isinstance(
        action_name,
        str,
    ):

        raise PlanParseError("Action name must be a string.")

    normalized = action_name.strip().upper().replace("-", "_").replace(" ", "_")

    return normalized


# ==========================================================
# JSON Extraction
# ==========================================================


def _extract_json(
    raw_text,
):
    """
    Extract JSON from an LLM response.

    Handles:
        pure JSON
        ```json ... ```
        extra text surrounding JSON
    """

    if not isinstance(
        raw_text,
        str,
    ):

        raise PlanParseError("LLM response must be text.")

    text = raw_text.strip()

    if not text:

        raise PlanParseError("LLM returned an empty response.")

    # ------------------------------------------------------
    # Remove Markdown fences if the model ignored the prompt
    # ------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:

            lines = lines[1:]

        if lines and lines[-1].strip() == "```":

            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # ------------------------------------------------------
    # First attempt: direct JSON
    # ------------------------------------------------------

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass

    # ------------------------------------------------------
    # Second attempt: locate first JSON object
    # ------------------------------------------------------

    start = text.find("{")

    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise PlanParseError("No JSON object found in LLM response.")

    candidate = text[start : end + 1]

    try:

        return json.loads(candidate)

    except json.JSONDecodeError as exc:

        raise PlanParseError("Unable to parse LLM plan as JSON.") from exc


# ==========================================================
# Extract Plan List
# ==========================================================


def _extract_plan_items(
    payload,
):
    """
    Extract the action sequence from supported JSON layouts.
    """

    if not isinstance(
        payload,
        dict,
    ):

        raise PlanParseError("Planner response must be a JSON object.")

    plan = payload.get("plan")

    # Some models may return these despite instructions.
    if plan is None:

        plan = payload.get("actions")

    if plan is None:

        plan = payload.get("steps")

    if not isinstance(
        plan,
        list,
    ):

        raise PlanParseError("Planner JSON does not contain a valid plan list.")

    return plan


# ==========================================================
# Parse Individual Item
# ==========================================================


def _parse_action_item(
    item,
):
    """
    Parse one item from the plan.

    Supports:

        "GET_INVOICES"

    and:

        {
            "action": "GET_INVOICES"
        }
    """

    if isinstance(
        item,
        str,
    ):

        return normalize_action_name(item)

    if isinstance(
        item,
        dict,
    ):

        action_name = item.get("action") or item.get("name")

        if action_name is None:

            raise PlanParseError("Plan step dictionary " "does not contain an action.")

        return normalize_action_name(action_name)

    raise PlanParseError(f"Unsupported plan item: {item}")


# ==========================================================
# Prerequisite Construction
# ==========================================================


def build_prerequisites(
    action_ids,
):
    """
    Convert an ordered sequence into prerequisite mappings.

    Example
    -------

    Sequence:

        [0, 2, 3, 5]

    becomes:

        {
            0: [],
            2: [0],
            3: [0, 2],
            5: [0, 2, 3]
        }

    This does not add domain knowledge.

    It simply represents the order supplied by the LLM.
    """

    prerequisites = {}

    completed_before = []

    for action_id in action_ids:

        prerequisites[action_id] = list(completed_before)

        completed_before.append(action_id)

    return prerequisites


# ==========================================================
# Main Parser
# ==========================================================


def parse_plan_response(
    raw_text,
):
    """
    Parse and validate an LLM planner response.

    Returns
    -------
    ParsedPlan
    """

    payload = _extract_json(raw_text)

    plan_items = _extract_plan_items(payload)

    action_names = []

    seen = set()

    for item in plan_items:

        action_name = _parse_action_item(item)

        # ------------------------------------------------------
        # Validate action
        # ------------------------------------------------------

        if action_name not in ACTION_NAME_TO_ID:

            raise PlanParseError("LLM generated unsupported action: " f"{action_name}")

        # ------------------------------------------------------
        # Remove accidental duplicates
        #
        # We do not reorder or replace actions.
        # ------------------------------------------------------

        if action_name in seen:

            continue

        seen.add(action_name)

        action_names.append(action_name)

    action_ids = [ACTION_NAME_TO_ID[action_name] for action_name in action_names]

    prerequisites = build_prerequisites(action_ids)

    return ParsedPlan(
        action_names=action_names,
        action_ids=action_ids,
        prerequisites=prerequisites,
        raw_payload=payload,
    )
