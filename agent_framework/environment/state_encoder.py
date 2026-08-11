"""
state_encoder.py

Converts backend environment state (JSON)
into numerical vectors for RL algorithms.
"""
import numpy as np


class StateEncoder:
    """
    Converts FinanceEnvironment state into a binary RL observation.

    Observation contains only 0/1 values.
    """

    STATE_NAMES = [
        # Action execution states
        "get_invoices",
        "check_duplicate",
        "check_supplier",
        "approve_invoices",
        "pay_invoices",
        "check_budget",
        "generate_report",
        "check_payment_completed",
        # Invoice condition states
        "has_paid_invoices",
        "has_rejected_invoices",
        "has_pending_approval_invoices",
        "has_approved_invoices",
        # Final task state
        "task_completed",
    ]

    STATE_SIZE = len(STATE_NAMES)

    def __init__(self):
        self.state_size = self.STATE_SIZE

    def encode(self, state):
        """
        Convert a state dictionary into a numpy binary vector.

        Returns:
            np.ndarray of shape (13,)
        """

        observation = []

        for state_name in self.STATE_NAMES:
            value = state.get(state_name, False)

            observation.append(1.0 if bool(value) else 0.0)

        return np.asarray(observation, dtype=np.float32)

    def decode(self, observation):
        """
        Convert an encoded observation back into a readable dictionary.

        Mainly useful for debugging and logging.
        """

        observation = np.asarray(observation).flatten()

        if len(observation) != self.STATE_SIZE:
            raise ValueError(
                f"Expected {self.STATE_SIZE} states, " f"received {len(observation)}"
            )

        return {
            state_name: bool(value >= 0.5)
            for state_name, value in zip(self.STATE_NAMES, observation)
        }

    def get_state_names(self):
        """
        Return ordered state names.
        """
        return list(self.STATE_NAMES)

    def get_state_size(self):
        """
        Return number of observation dimensions.
        """
        return self.state_size

    def validate(self, observation):
        """
        Validate that an observation contains only binary values.
        """

        observation = np.asarray(observation).flatten()

        if len(observation) != self.STATE_SIZE:
            return False

        return bool(np.all(np.logical_or(observation == 0, observation == 1)))
