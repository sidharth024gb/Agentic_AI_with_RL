"""
reward_processor.py

Centralized reward calculation for the Finance RL Environment.

The environment actions report WHAT happened.

RewardProcessor determines HOW that outcome should be rewarded.

This same base reward system is used for:

    - PPO
    - LLM + PPO

LLM-based experiments may additionally receive a positive
guidance bonus when the generated procedure is followed.

Important:
    Guidance can NEVER reduce or cancel a negative reward.
"""

from config.config import config


class RewardProcessor:
    """
    Central reward processor for the finance environment.
    """

    # ==========================================================
    # Positive Action Rewards
    #
    # These rewards are intentionally smaller than the
    # task-completion reward.
    # ==========================================================

    ACTION_REWARDS = {
        "GET_INVOICES": 2.0,
        "CHECK_DUPLICATE": 4.0,
        "CHECK_SUPPLIER": 4.0,
        "APPROVE_INVOICES": 5.0,
        "CHECK_BUDGET": 6.0,
        "PAY_INVOICES": 8.0,
        "GENERATE_REPORT": 1.0,
        "CHECK_PAYMENT_COMPLETED": 2.0,
    }

    # ==========================================================
    # Goal Reward
    #
    # Completing the task is the most important objective.
    # ==========================================================

    TASK_COMPLETION_BONUS = 50.0

    # ==========================================================
    # General Penalties
    # ==========================================================

    NO_OP_PENALTY = -1.0

    REPEATED_ACTION_PENALTY = -2.0

    DEFAULT_FAILURE_PENALTY = -5.0

    # ==========================================================
    # Business Error Penalties
    #
    # These use backend errorType values.
    #
    # We deliberately do NOT use backend reward values.
    # ==========================================================

    ERROR_PENALTIES = {
        # ------------------------------------------------------
        # General invalid action / workflow
        # ------------------------------------------------------
        "INVALID_ACTION": -5.0,
        "INVALID_WORKFLOW": -8.0,
        "MISSING_REQUIRED_FIELD": -5.0,
        "INVALID_STATE_DATA": -5.0,
        "INVALID_RESPONSE": -5.0,
        # ------------------------------------------------------
        # Invoice
        # ------------------------------------------------------
        "INVOICE_NOT_FOUND": -8.0,
        "INVOICE_NOT_APPROVED": -10.0,
        "DUPLICATE_INVOICE": -12.0,
        # ------------------------------------------------------
        # Supplier
        # ------------------------------------------------------
        "SUPPLIER_NOT_FOUND": -8.0,
        "SUPPLIER_INACTIVE": -12.0,
        "SUPPLIER_HIGH_RISK": -10.0,
        # ------------------------------------------------------
        # Budget
        # ------------------------------------------------------
        "BUDGET_NOT_FOUND": -8.0,
        "BUDGET_EXCEEDED": -15.0,
        # ------------------------------------------------------
        # Account
        # ------------------------------------------------------
        "ACCOUNT_NOT_FOUND": -8.0,
        "ACCOUNT_FROZEN": -12.0,
        "INSUFFICIENT_BALANCE": -15.0,
        # ------------------------------------------------------
        # Payment
        # ------------------------------------------------------
        "PAYMENT_FAILED": -12.0,
        # ------------------------------------------------------
        # Other
        # ------------------------------------------------------
        "REPORT_FAILED": -5.0,
        "UNKNOWN_BUSINESS_ERROR": -5.0,
    }

    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(
        self,
        use_guidance=None,
        guidance_bonus=None,
    ):

        if use_guidance is None:

            use_guidance = (
                config.agent.AGENT_TYPE == "LLM_RL"
                and config.experiment.GUIDANCE_MODE
                in {
                    "REWARD_SHAPING",
                    "INPUT_AND_REWARD",
                }
            )

        if guidance_bonus is None:

            guidance_bonus = config.experiment.GUIDANCE_BONUS

        self.use_guidance = bool(use_guidance)

        self.guidance_bonus = float(guidance_bonus)

    # ==========================================================
    # Configure Guidance
    # ==========================================================

    def configure_guidance(
        self,
        use_guidance,
        guidance_bonus=None,
    ):

        self.use_guidance = bool(use_guidance)

        if guidance_bonus is not None:

            self.guidance_bonus = float(guidance_bonus)

    # ==========================================================
    # Error Penalty
    # ==========================================================

    def get_error_penalty(
        self,
        error_type,
    ):

        if not error_type:

            return self.DEFAULT_FAILURE_PENALTY

        return float(
            self.ERROR_PENALTIES.get(
                str(error_type).upper(),
                self.DEFAULT_FAILURE_PENALTY,
            )
        )

    # ==========================================================
    # Reward Processing
    # ==========================================================

    def process(
        self,
        action_name,
        action_success=True,
        useful_action=True,
        environment_error=False,
        error_type=None,
        task_completed=False,
        procedure_followed=None,
        repeated_action=False,
    ):
        """
        Calculate the final RL reward.

        Returns a complete reward breakdown.

        Rules
        -----

        Environment error:
            reward = 0
            PPO transition should be ignored.

        Failed business action:
            negative reward based on errorType.

        Successful but useless action:
            small negative reward.

        Repeated rewarded action:
            negative reward.

        Successful useful action:
            positive base reward.

        Task completion:
            large additional completion reward.

        LLM guidance:
            added ONLY to positive successful useful actions.

            It can NEVER modify a negative reward.
        """

        # ======================================================
        # Infrastructure / Environment Failure
        # ======================================================

        if environment_error:

            return {
                "base_reward": 0.0,
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": 0.0,
                "penalty_type": None,
                "trainable": False,
            }

        # ======================================================
        # Failed Business Action
        # ======================================================

        if not action_success:

            penalty = self.get_error_penalty(error_type)

            return {
                "base_reward": penalty,
                # IMPORTANT:
                # Never add guidance to a negative reward.
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": penalty,
                "penalty_type": error_type or "ACTION_FAILED",
                "trainable": True,
            }

        # ======================================================
        # Repeated Action
        # ======================================================

        if repeated_action:

            return {
                "base_reward": self.REPEATED_ACTION_PENALTY,
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": self.REPEATED_ACTION_PENALTY,
                "penalty_type": "REPEATED_ACTION",
                "trainable": True,
            }

        # ======================================================
        # Successful But Useless Action
        # ======================================================

        if not useful_action:

            return {
                "base_reward": self.NO_OP_PENALTY,
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": self.NO_OP_PENALTY,
                "penalty_type": "NO_OP",
                "trainable": True,
            }

        # ======================================================
        # Successful Useful Action
        # ======================================================

        base_reward = float(
            self.ACTION_REWARDS.get(
                action_name,
                0.0,
            )
        )

        # ======================================================
        # Goal Completion
        # ======================================================

        completion_bonus = self.TASK_COMPLETION_BONUS if task_completed else 0.0

        # ======================================================
        # LLM Guidance Bonus
        #
        # Guidance is positive-only.
        #
        # It is deliberately impossible for this bonus to:
        #
        #     - reduce a penalty
        #     - cancel a penalty
        #     - turn a failed action positive
        # ======================================================

        guidance_bonus = 0.0

        if (
            self.use_guidance
            and procedure_followed is True
            and action_success
            and useful_action
            and base_reward > 0
        ):

            guidance_bonus = self.guidance_bonus

        final_reward = base_reward + completion_bonus + guidance_bonus

        return {
            "base_reward": float(base_reward),
            "guidance_bonus": float(guidance_bonus),
            "completion_bonus": float(completion_bonus),
            "reward": float(final_reward),
            "penalty_type": None,
            "trainable": True,
        }
