"""
value_network.py

Critic (Value) Network for PPO.

Estimates the state value V(s), which represents the expected
future return from the current environment state.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    Critic network used by PPO.
    """

    def __init__(
        self,
        observation_size: int,
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
                1,
            ),
        )

    # ==========================================================
    # Forward Pass
    # ==========================================================

    def forward(self, state):
        """
        Estimate the value of a state.

        Parameters
        ----------
        state : torch.Tensor
            Encoded environment state.

        Returns
        -------
        torch.Tensor
            Estimated state value.
        """

        value = self.network(state)

        return value.squeeze(-1)
