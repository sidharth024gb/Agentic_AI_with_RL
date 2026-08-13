"""
state_encoder.py

Converts FinanceEnvironment state into a numerical PPO observation.

The observation intentionally remains 13-dimensional so the existing
LLMRLAgent input modes remain:

    base observation = 13
    guidance vector  = 8
    combined input   = 21

The semantics are changed to describe completed workflow progress
rather than merely attempted actions.
"""

import numpy as np


class StateEncoder:
    """Convert FinanceEnvironment state into a binary vector."""

    STATE_NAMES = [
        # Successfully reached workflow stages
        "get_invoices",
        "check_duplicate",
        "check_supplier",
        "approve_invoices",
        "check_budget",
        "pay_invoices",
        "generate_report",
        "check_payment_completed",
        # Current payable-work state
        "has_paid_invoices",
        "has_pending_approval_invoices",
        "has_approved_invoices",
        # Any invoice intentionally excluded because it is not valid/
        # payable in this episode (duplicate, supplier, or budget).
        "has_excluded_invoices",
        # Final task state
        "task_completed",
    ]

    STATE_SIZE = len(STATE_NAMES)

    def __init__(self):
        self.state_size = self.STATE_SIZE

    def encode(self, state):
        observation = [
            1.0 if bool(state.get(name, False)) else 0.0 for name in self.STATE_NAMES
        ]
        return np.asarray(observation, dtype=np.float32)

    def decode(self, observation):
        observation = np.asarray(observation).reshape(-1)

        if len(observation) != self.STATE_SIZE:
            raise ValueError(
                f"Expected {self.STATE_SIZE} states, " f"received {len(observation)}"
            )

        return {
            name: bool(value >= 0.5)
            for name, value in zip(
                self.STATE_NAMES,
                observation,
            )
        }

    def get_state_names(self):
        return list(self.STATE_NAMES)

    def get_state_size(self):
        return self.state_size

    def validate(self, observation):
        observation = np.asarray(observation).reshape(-1)

        if len(observation) != self.STATE_SIZE:
            return False

        if not np.all(np.isfinite(observation)):
            return False

        return bool(
            np.all(
                np.logical_or(
                    observation == 0.0,
                    observation == 1.0,
                )
            )
        )
