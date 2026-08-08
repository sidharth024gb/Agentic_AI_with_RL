"""
policy_network.py

Actor (Policy) Network for PPO.

Maps an environment observation to a probability
distribution over the available actions.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Actor network used by PPO.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                action_size,
            ),
        )

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def forward(self, state):
        """
        Returns raw action logits.

        Parameters
        ----------
        state : torch.Tensor
            Encoded environment state.

        Returns
        -------
        torch.Tensor
            Action logits.
        """

        return self.network(state)

    # ==========================================================
    # Action Distribution
    # ==========================================================

    def get_distribution(self, state):
        """
        Create a categorical distribution over actions.

        Parameters
        ----------
        state : torch.Tensor

        Returns
        -------
        torch.distributions.Categorical
        """

        logits = self.forward(state)

        return torch.distributions.Categorical(logits=logits)

    # ==========================================================
    # Sample Action
    # ==========================================================

    def sample_action(self, state):
        """
        Sample an action from the current policy.

        Returns
        -------
        action
        log_probability
        entropy
        """

        distribution = self.get_distribution(state)

        action = distribution.sample()

        log_probability = distribution.log_prob(action)

        entropy = distribution.entropy()

        return (
            action,
            log_probability,
            entropy,
        )

    # ==========================================================
    # Greedy Action
    # ==========================================================

    def predict(self, state):
        """
        Deterministic action selection.

        Used during evaluation.
        """

        logits = self.forward(state)

        return torch.argmax(
            logits,
            dim=-1,
        )
