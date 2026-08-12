"""
policy_network.py

Actor (Policy) Network for PPO.

Maps the encoded environment observation to a categorical
probability distribution over the available discrete actions.

Current environment:
    Observation: binary vector
    Actions: discrete high-level finance actions
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Actor network used by PPO.

    Input
    -----
    Environment observation vector.

    Output
    ------
    Raw logits for each discrete action.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        if observation_size <= 0:
            raise ValueError("observation_size must be greater than 0.")

        if action_size <= 0:
            raise ValueError("action_size must be greater than 0.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")

        self.observation_size = int(observation_size)

        self.action_size = int(action_size)

        self.hidden_size = int(hidden_size)

        # ==========================================================
        # Network
        # ==========================================================

        self.network = nn.Sequential(
            nn.Linear(
                self.observation_size,
                self.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size,
                self.hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                self.hidden_size,
                self.action_size,
            ),
        )

        # ==========================================================
        # PPO-Friendly Initialization
        # ==========================================================

        self._initialize_weights()

    # ==========================================================
    # Weight Initialization
    # ==========================================================

    def _initialize_weights(self):
        """
        Orthogonal initialization is commonly used with PPO.

        Hidden layers use sqrt(2) gain.

        The policy output layer uses a small gain so the
        initial action distribution begins close to uniform
        rather than strongly preferring arbitrary actions.
        """

        # First hidden layer
        nn.init.orthogonal_(
            self.network[0].weight,
            gain=math.sqrt(2),
        )

        nn.init.constant_(
            self.network[0].bias,
            0.0,
        )

        # Second hidden layer
        nn.init.orthogonal_(
            self.network[2].weight,
            gain=math.sqrt(2),
        )

        nn.init.constant_(
            self.network[2].bias,
            0.0,
        )

        # Policy output layer
        nn.init.orthogonal_(
            self.network[4].weight,
            gain=0.01,
        )

        nn.init.constant_(
            self.network[4].bias,
            0.0,
        )

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def forward(
        self,
        state,
    ):
        """
        Return raw action logits.

        Parameters
        ----------
        state : torch.Tensor

            Expected shape:

                (batch_size, observation_size)

            or:

                (observation_size,)

        Returns
        -------
        torch.Tensor

            Action logits:

                (batch_size, action_size)
        """

        if state.shape[-1] != self.observation_size:

            raise ValueError(
                "PolicyNetwork expected observation size "
                f"{self.observation_size}, "
                f"received {state.shape[-1]}."
            )

        return self.network(state)

    # ==========================================================
    # Action Distribution
    # ==========================================================

    def get_distribution(
        self,
        state,
    ):
        """
        Create categorical probability distribution over
        the discrete finance actions.
        """

        logits = self.forward(state)

        return Categorical(logits=logits)

    # ==========================================================
    # Sample Action
    # ==========================================================

    def sample_action(
        self,
        state,
    ):
        """
        Sample an action from the policy.

        Mainly useful during PPO training.

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
    # Deterministic Action
    # ==========================================================

    def predict(
        self,
        state,
    ):
        """
        Return the highest-probability action.

        Used during evaluation.
        """

        distribution = self.get_distribution(state)

        return torch.argmax(
            distribution.probs,
            dim=-1,
        )
