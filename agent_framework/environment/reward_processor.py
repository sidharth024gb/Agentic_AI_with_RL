"""
reward_processor.py

Processes rewards returned by the Finance RL Backend.

The backend is the source of truth for reward calculation.
This module only interprets and validates the response.
"""

from typing import Any, Dict, Tuple


class RewardProcessor:
    """
    Handles reward extraction and processing.
    """

    def __init__(self):

        self.total_reward = 0

        self.step_rewards = []

    # ==========================================================
    # Main Processing
    # ==========================================================

    def process(self, response: Dict[str, Any]) -> Tuple[float, Dict]:
        """
        Process backend response.

        Returns:

            reward,
            info dictionary

        """

        # --------------------------------------
        # Environment failure
        # --------------------------------------

        if response.get("environment_error", False):

            return (
                0,
                {
                    "reward_error": True,
                    "reason": "Environment failure",
                },
            )

        data = response.get("data", {})

        # --------------------------------------
        # Extract reward
        # --------------------------------------

        reward = data.get("reward", 0)

        # Backend may return null for system errors

        if reward is None:

            return (
                0,
                {
                    "reward_error": True,
                    "reason": "System error reward",
                },
            )

        reward = float(reward)

        self.total_reward += reward

        self.step_rewards.append(reward)

        info = {
            "reward_source": "backend",
            "step_reward": reward,
            "total_reward": self.total_reward,
            "reward_type": data.get("rewardType"),
            "message": data.get("message"),
        }

        return reward, info

    # ==========================================================
    # Episode Reset
    # ==========================================================

    def reset(self):
        """
        Reset reward tracking
        at the beginning of an episode.
        """

        self.total_reward = 0

        self.step_rewards = []

    # ==========================================================
    # Statistics
    # ==========================================================

    def get_total_reward(self):

        return self.total_reward

    def get_step_rewards(self):

        return self.step_rewards

    def get_average_reward(self):

        if not self.step_rewards:

            return 0

        return sum(self.step_rewards) / len(self.step_rewards)
