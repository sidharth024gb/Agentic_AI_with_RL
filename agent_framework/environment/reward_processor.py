"""
reward_processor.py

Progress-based reward calculation for the Finance RL Environment.

Design principles
-----------------
1. Successful useful progress receives a small positive reward.
2. A valid action with nothing to process receives zero.
3. Repeating a valid action without progress receives a very small penalty.\n   Exception: when an LLM-guided agent successfully follows the current\n   LLM recommendation, that artificial repeat penalty is suppressed.
4. Genuine workflow/action mistakes receive only a small penalty.
5. Business validation findings (duplicate, high-risk supplier,
   over-budget invoice) should normally be represented by the
   environment as successful validation + filtering, not failure.
6. Infrastructure/backend failures are never learned from.
7. LLM guidance is positive-only.
"""

from config.config import config


class RewardProcessor:
    """Central reward processor shared by PPO and LLM + PPO."""

    ACTION_REWARDS = {
        "GET_INVOICES": 1.0,
        "CHECK_DUPLICATE": 2.0,
        "CHECK_SUPPLIER": 2.0,
        "APPROVE_INVOICES": 3.0,
        "CHECK_BUDGET": 2.0,
        "PAY_INVOICES": 5.0,
        "GENERATE_REPORT": 1.0,
        # Completion itself is rewarded by TASK_COMPLETION_BONUS.
        "CHECK_PAYMENT_COMPLETED": 0.0,
    }

    TASK_COMPLETION_BONUS = 25.0

    # A valid first no-op remains neutral.
    NO_OP_REWARD = 0.0

    # Repeating an already-completed/no-progress action gets
    # a very small penalty to prevent zero-reward loops.
    REPEATED_ACTION_REWARD = -0.1

    # Genuine action/workflow mistakes are intentionally mild.
    DEFAULT_FAILURE_PENALTY = -0.5

    ERROR_PENALTIES = {
        "INVALID_ACTION": -1.0,
        "INVALID_WORKFLOW": -0.5,
        "INVALID_REQUEST": -1.0,
        "MISSING_REQUIRED_FIELD": -1.0,
        "INVALID_STATE_DATA": -1.0,
        "INVALID_RESPONSE": -1.0,
        "INVALID_STATUS": -1.0,
        "INVOICE_NOT_FOUND": -0.5,
        "INVOICE_NOT_APPROVED": -0.5,
        "INVOICE_ALREADY_PAID": -0.5,
        # These should normally be filtered by validation actions.
        # They remain here as protection if the agent jumps directly
        # to PAY_INVOICES before validating.
        "DUPLICATE_INVOICE": -0.5,
        "SUPPLIER_NOT_FOUND": -0.5,
        "SUPPLIER_INACTIVE": -0.5,
        "SUPPLIER_HIGH_RISK": -0.5,
        "BUDGET_NOT_FOUND": -0.5,
        "BUDGET_EXCEEDED": -0.5,
        "ACCOUNT_NOT_FOUND": -0.5,
        "ACCOUNT_FROZEN": -0.5,
        "INSUFFICIENT_BALANCE": -0.5,
        "PAYMENT_FAILED": -0.5,
        "REPORT_FAILED": -0.5,
        "TRANSACTION_NOT_FOUND": -0.5,
        "TRANSFER_LIMIT_EXCEEDED": -0.5,
        "UNKNOWN_BUSINESS_ERROR": -0.5,
    }

    def __init__(self, use_guidance=None, guidance_bonus=None):
        if use_guidance is None:
            use_guidance = (
                config.agent.AGENT_TYPE == "LLM_RL"
                and config.experiment.GUIDANCE_MODE
                in {"REWARD_SHAPING", "INPUT_AND_REWARD"}
            )

        if guidance_bonus is None:
            guidance_bonus = config.experiment.GUIDANCE_BONUS

        self.use_guidance = bool(use_guidance)
        self.guidance_bonus = float(guidance_bonus)

    def configure_guidance(self, use_guidance, guidance_bonus=None):
        self.use_guidance = bool(use_guidance)

        if guidance_bonus is not None:
            self.guidance_bonus = float(guidance_bonus)

    def get_error_penalty(self, error_type):
        if not error_type:
            return float(self.DEFAULT_FAILURE_PENALTY)

        return float(
            self.ERROR_PENALTIES.get(
                str(error_type).upper(),
                self.DEFAULT_FAILURE_PENALTY,
            )
        )

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
        guided_repeated_action=False,
    ):
        """
        Return a complete reward breakdown.

        ``trainable`` is False only for environment/infrastructure
        failures. All valid environment outcomes remain trainable.

        ``guided_repeated_action`` means the action is both:

            - a consecutive no-progress repeat, and
            - the current LLM recommendation that PPO successfully followed.

        Such an action is exempt from REPEATED_ACTION_REWARD and falls
        back to the normal no-op reward (0.0). It does not automatically
        receive a guidance bonus; guidance remains positive-only for
        useful progress or task completion.
        """

        if environment_error:
            return {
                "base_reward": 0.0,
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": 0.0,
                "penalty_type": None,
                "trainable": False,
            }

        if not action_success:
            penalty = self.get_error_penalty(error_type)

            return {
                "base_reward": penalty,
                "guidance_bonus": 0.0,
                "completion_bonus": 0.0,
                "reward": penalty,
                "penalty_type": error_type or "ACTION_FAILED",
                "trainable": True,
            }

        completion_bonus = self.TASK_COMPLETION_BONUS if task_completed else 0.0

        # ------------------------------------------------------
        # Repeated no-progress action
        #
        # Normally this receives the tiny efficiency penalty.
        #
        # For an actively LLM-guided agent, however, if the repeated
        # action is the current recommendation and the recommendation
        # was successfully followed, do not punish the agent for
        # following that guidance. The no-progress action remains
        # neutral rather than receiving a positive reward.
        # ------------------------------------------------------

        if repeated_action and not guided_repeated_action:
            base_reward = self.REPEATED_ACTION_REWARD
        elif not useful_action:
            base_reward = self.NO_OP_REWARD
        else:
            base_reward = float(
                self.ACTION_REWARDS.get(
                    action_name,
                    0.0,
                )
            )

        guidance_bonus = 0.0

        # Positive-only guidance. It can reinforce a useful action or
        # the final successful completion check, but never rescue a
        # failed action.
        if (
            self.use_guidance
            and procedure_followed is True
            and action_success
            and (useful_action or task_completed)
        ):
            guidance_bonus = self.guidance_bonus

        final_reward = (
            float(base_reward) + float(completion_bonus) + float(guidance_bonus)
        )

        return {
            "base_reward": float(base_reward),
            "guidance_bonus": float(guidance_bonus),
            "completion_bonus": float(completion_bonus),
            "reward": float(final_reward),
            "penalty_type": None,
            "trainable": True,
        }
