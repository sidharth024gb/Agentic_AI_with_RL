"""
value_network.py

Critic (Value) Network for PPO.

Estimates the expected discounted future return V(s)
from the current environment observation.
"""

import math

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Critic network used by PPO.

    Input
    -----
    Environment observation.

    Output
    ------
    Scalar value estimate V(s).
    """

    def __init__(
        self,
        observation_size: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        if observation_size <= 0:
            raise ValueError("observation_size must be greater than 0.")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")

        self.observation_size = int(observation_size)

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
                1,
            ),
        )

        # ==========================================================
        # Initialization
        # ==========================================================

        self._initialize_weights()

    # ==========================================================
    # Weight Initialization
    # ==========================================================

    def _initialize_weights(self):
        """
        PPO-friendly orthogonal initialization.
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

        # Value output layer
        nn.init.orthogonal_(
            self.network[4].weight,
            gain=1.0,
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
        Estimate V(state).

        Parameters
        ----------
        state : torch.Tensor

            Shape:

                (batch_size, observation_size)

        Returns
        -------
        torch.Tensor

            Shape:

                (batch_size, 1)

        PPOAgent performs squeeze(-1) where needed.
        """

        if state.shape[-1] != self.observation_size:

            raise ValueError(
                "ValueNetwork expected observation size "
                f"{self.observation_size}, "
                f"received {state.shape[-1]}."
            )

        return self.network(state)
