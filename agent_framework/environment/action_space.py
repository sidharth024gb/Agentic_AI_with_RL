from enum import IntEnum


class FinanceAction(IntEnum):
    """
    Actions available to the RL agent.

    The integer values are used directly by PPO/DQN policies.
    """

    GET_INVOICES = 0
    CHECK_DUPLICATE = 1
    CHECK_SUPPLIER = 2
    APPROVE_INVOICES = 3
    PAY_INVOICES = 4
    CHECK_BUDGET = 5
    GENERATE_REPORT = 6
    CHECK_PAYMENT_COMPLETED = 7


class ActionSpace:
    """
    Maps RL action IDs to FinanceEnvironment operations.
    """

    ACTION_NAMES = {
        FinanceAction.GET_INVOICES: "GET_INVOICES",
        FinanceAction.CHECK_DUPLICATE: "CHECK_DUPLICATE",
        FinanceAction.CHECK_SUPPLIER: "CHECK_SUPPLIER",
        FinanceAction.APPROVE_INVOICES: "APPROVE_INVOICES",
        FinanceAction.PAY_INVOICES: "PAY_INVOICES",
        FinanceAction.CHECK_BUDGET: "CHECK_BUDGET",
        FinanceAction.GENERATE_REPORT: "GENERATE_REPORT",
        FinanceAction.CHECK_PAYMENT_COMPLETED: "CHECK_PAYMENT_COMPLETED",
    }

    def __init__(self):
        self.n = len(FinanceAction)

    def sample(self):
        """
        Return a random valid action.

        Useful for testing and random-policy baselines.
        """
        import random

        return random.randrange(self.n)

    def get_action_name(self, action):
        """
        Convert integer action ID to readable action name.
        """
        try:
            action = FinanceAction(int(action))
            return self.ACTION_NAMES[action]
        except (ValueError, TypeError):
            raise ValueError(f"Invalid action: {action}")

    def is_valid(self, action):
        """
        Check whether an action ID is valid.
        """
        try:
            FinanceAction(int(action))
            return True
        except (ValueError, TypeError):
            return False

    def execute(self, env, action):
        """
        Execute an action against FinanceEnvironment.

        The environment owns the workflow and DataFrames.
        ActionSpace only dispatches the action.
        """

        if not self.is_valid(action):
            raise ValueError(f"Invalid action: {action}")

        action = FinanceAction(int(action))

        if action == FinanceAction.GET_INVOICES:
            return env.get_invoices()

        if action == FinanceAction.CHECK_DUPLICATE:
            return env.check_duplicate()

        if action == FinanceAction.CHECK_SUPPLIER:
            return env.check_supplier()

        if action == FinanceAction.APPROVE_INVOICES:
            return env.approve_invoices()

        if action == FinanceAction.PAY_INVOICES:
            return env.pay_invoices()

        if action == FinanceAction.CHECK_BUDGET:
            return env.check_budget()

        if action == FinanceAction.GENERATE_REPORT:
            return env.generate_report()

        if action == FinanceAction.CHECK_PAYMENT_COMPLETED:
            return env.check_payment_completed()

        raise ValueError(f"Unhandled action: {action}")

    @property
    def action_count(self):
        return self.n

    @property
    def action_names(self):
        return list(self.ACTION_NAMES.values())
