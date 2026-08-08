"""
prompts.py

Prompt construction for the LLM planning component.

This module is responsible only for creating prompts.
It does not call the LLM and does not execute actions.
"""


class PromptBuilder:
    """
    Builds prompts used by the LLM planner.
    """

    # ==========================================================
    # System Prompt
    # ==========================================================

    SYSTEM_PROMPT = """
You are a planning assistant for a financial task execution
environment.

Your role is to analyse the current environment state and provide
a short, high-level planning recommendation to an RL agent.

You do NOT execute actions.

You do NOT call APIs.

You do NOT invent data that is not present in the state.

Your recommendation must be based only on the information
provided in the current state.

The RL agent remains responsible for selecting and executing
the actual action.

Focus on:

1. The current invoice workflow state.
2. Supplier validity and risk.
3. Budget availability.
4. Account balance.
5. The logical next step toward completing the task.

Return a concise recommendation that can be converted into
a planning hint for the RL agent.
"""

    # ==========================================================
    # Main Planning Prompt
    # ==========================================================

    @classmethod
    def build_planning_prompt(
        cls,
        state,
        goal,
    ):
        """
        Build a prompt for the LLM planner.

        Parameters
        ----------
        state : dict
            Current finance environment state.

        goal : str
            Current episode goal.

        Returns
        -------
        str
            Complete planning prompt.
        """

        return f"""
Current task goal:

{goal}


Current finance environment state:

{cls._format_state(state)}


Analyse the current state and determine the most useful
next step toward achieving the goal.

Consider:

- invoice status
- supplier status and risk
- budget availability
- account balance
- required workflow order
- whether an action would be premature or invalid

Provide one concise planning recommendation.

Do not execute the action.
Do not call an API.
Do not provide multiple alternative actions.

Recommendation:
"""

    # ==========================================================
    # State Formatting
    # ==========================================================

    @staticmethod
    def _format_state(
        state,
    ):
        """
        Convert the environment state into readable text.
        """

        if not state:

            return "No state information available."

        invoices = state.get("invoices", [])

        suppliers = state.get("suppliers", [])

        accounts = state.get("accounts", [])

        lines = []

        # ------------------------------------------------------
        # Invoices
        # ------------------------------------------------------

        lines.append("INVOICES:")

        if not invoices:

            lines.append("No invoices available.")

        else:

            for invoice in invoices:

                lines.append(
                    str(
                        {
                            "id": invoice.get("_id"),
                            "status": invoice.get("status"),
                            "amount": invoice.get("amount"),
                            "supplier": invoice.get("supplier"),
                        }
                    )
                )

        # ------------------------------------------------------
        # Suppliers
        # ------------------------------------------------------

        lines.append("\nSUPPLIERS:")

        if not suppliers:

            lines.append("No suppliers available.")

        else:

            for supplier in suppliers:

                lines.append(
                    str(
                        {
                            "id": supplier.get("_id"),
                            "active": supplier.get("isActive", supplier.get("active")),
                            "risk": supplier.get("riskScore"),
                        }
                    )
                )

        # ------------------------------------------------------
        # Accounts
        # ------------------------------------------------------

        lines.append("\nACCOUNTS:")

        if not accounts:

            lines.append("No accounts available.")

        else:

            for account in accounts:

                lines.append(
                    str(
                        {
                            "id": account.get("_id"),
                            "balance": account.get(
                                "currentBalance", account.get("balance")
                            ),
                        }
                    )
                )

        return "\n".join(lines)

    # ==========================================================
    # Compact Prompt
    # ==========================================================

    @classmethod
    def build_compact_prompt(
        cls,
        state,
        goal,
    ):
        """
        Build a shorter prompt for repeated LLM calls.

        Useful during large training experiments where prompt
        size and LLM cost need to be controlled.
        """

        return f"""
Goal:
{goal}

State:
{cls._format_state(state)}

Give one concise recommendation for the next logical step.
Do not execute any action or call any API.
"""
