"""
prompts.py

Prompt definitions for the LLM finance planner.

The planner generates high-level procedural guidance for
the LLM + RL agent.

The LLM does NOT execute API calls.
It only proposes an ordered sequence of high-level actions.
"""

import json

from environment.action_space import FinanceAction

# ==========================================================
# Prompt Version
# ==========================================================

PROMPT_VERSION = "finance_planner_v2"


# ==========================================================
# Available Actions
# ==========================================================

ACTION_DESCRIPTIONS = {
    FinanceAction.GET_INVOICES.name: (
        "Load invoices from the finance environment and "
        "separate them into PAID, REJECTED, "
        "PENDING_APPROVAL and APPROVED groups."
    ),
    FinanceAction.CHECK_DUPLICATE.name: (
        "Check pending and approved invoices for duplicates. "
        "Duplicate invoices are removed from further processing."
    ),
    FinanceAction.CHECK_SUPPLIER.name: (
        "Validate suppliers associated with pending and "
        "approved invoices. Invoices with invalid suppliers "
        "are removed from further processing."
    ),
    FinanceAction.APPROVE_INVOICES.name: (
        "Approve valid invoices that currently have " "PENDING_APPROVAL status."
    ),
    FinanceAction.PAY_INVOICES.name: (
        "Pay invoices that are currently approved and " "eligible for payment."
    ),
    FinanceAction.CHECK_BUDGET.name: (
        "Check remaining department budgets and retain only "
        "approved invoices that can fit within the available "
        "budget."
    ),
    FinanceAction.GENERATE_REPORT.name: (
        "Generate a one-year transaction summary report."
    ),
    FinanceAction.CHECK_PAYMENT_COMPLETED.name: (
        "Verify whether all valid payable invoices have been "
        "processed and determine whether the task is complete."
    ),
}


# ==========================================================
# Business Context
# ==========================================================
BUSINESS_RULES = """
FINANCE ENVIRONMENT BUSINESS RULES

The following are HARD constraints. A valid plan MUST respect them.

1. GET_INVOICES must occur before any action that needs invoice data.

2. Duplicate checking must occur before an invoice is approved or paid.

3. Supplier validation must occur before an invoice is approved or paid.

4. A PENDING_APPROVAL invoice must be approved before it can be paid.

5. CHECK_BUDGET must occur BEFORE PAY_INVOICES.

6. PAY_INVOICES must only operate on invoices that:
   - are approved,
   - are not duplicates,
   - have valid suppliers,
   - fit within available department budget.

7. CHECK_PAYMENT_COMPLETED must occur after payment processing when
   determining whether the task has been completed.

The following are contextual rather than mandatory:

8. APPROVE_INVOICES is only useful when pending invoices exist.

9. GENERATE_REPORT is not required for the goal of paying valid invoices
   unless the task specifically asks for a report.

10. Actions that have already been completed according to the current
    environment state normally should not be repeated.

11. Do not include unnecessary actions.

12. The planner provides procedural guidance only. It does not execute
    actions and does not calculate RL rewards.
"""

# ==========================================================
# System Prompt
# ==========================================================

SYSTEM_PROMPT = f"""
You are a high-level task planner for a reinforcement learning
agent operating in a simulated finance environment.

Your responsibility is to generate an ordered procedure using
ONLY the provided high-level actions.

You are NOT the executor.

You must not:
- invent new actions
- call APIs
- change environment state
- calculate RL rewards
- return executable code

Your response MUST contain valid JSON only.

{BUSINESS_RULES}
""".strip()


# ==========================================================
# Build Action Description
# ==========================================================


def _format_actions():

    lines = []

    for action in FinanceAction:

        description = ACTION_DESCRIPTIONS.get(
            action.name,
            "",
        )

        lines.append(f"{int(action.value)}. " f"{action.name}: " f"{description}")

    return "\n".join(lines)


# ==========================================================
# Planner Prompt
# ==========================================================


def build_planner_prompt(
    goal,
    state,
):

    if state is None:
        state = {}

    state_json = json.dumps(
        state,
        indent=2,
        sort_keys=True,
    )

    available_actions = _format_actions()

    return f"""
TASK GOAL
---------
{goal}


CURRENT ENVIRONMENT STATE
-------------------------
{state_json}


AVAILABLE ACTIONS
-----------------
{available_actions}


PLANNING REQUIREMENTS
---------------------

Create the shortest valid high-level procedure for achieving the task.

IMPORTANT:

- HARD business constraints in the system prompt MUST be satisfied.
- Never place PAY_INVOICES before CHECK_BUDGET.
- Never approve or pay invoices before duplicate and supplier checks.
- Do not invent actions.
- Do not repeat actions unnecessarily.
- Do not include GENERATE_REPORT unless it is required by the task.
- Use the current environment state to omit actions that have already
  been completed and do not need repeating.

Before producing the JSON, internally verify:

1. Does every action use an allowed action name?
2. Are duplicate and supplier checks before approval/payment?
3. Is CHECK_BUDGET before PAY_INVOICES?
4. Is CHECK_PAYMENT_COMPLETED after payment?
5. Is every included action actually useful?


OUTPUT FORMAT
-------------

Return JSON only:

{{
    "plan": [
        "ACTION_NAME",
        "ACTION_NAME"
    ]
}}

Do not include explanations.
Do not include Markdown.
""".strip()
