"""
rollout_buffer.py

Trajectory storage for PPO.

Stores one rollout collected from the environment.
The buffer is cleared after every PPO update.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """
    Single PPO transition.
    """

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool

    log_prob: float

    value: float

    # Added after GAE calculation
    advantage: float = 0.0

    return_value: float = 0.0


class RolloutBuffer:
    """
    Stores PPO trajectories.

    PPO is an on-policy algorithm,
    therefore data is discarded after update.
    """

    def __init__(self):

        self.clear()

    # ==========================================================
    # Add Experience
    # ==========================================================

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_prob,
        value,
    ):

        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        )

        self.buffer.append(transition)

    # ==========================================================
    # Add PPO Calculations
    # ==========================================================

    def add_advantages(
        self,
        advantages,
        returns,
    ):
        """
        Store GAE advantages and return targets.

        Called after rollout collection.
        """

        for transition, advantage, return_value in zip(
            self.buffer,
            advantages,
            returns,
        ):

            transition.advantage = advantage

            transition.return_value = return_value

    # ==========================================================
    # Tensor Conversion
    # ==========================================================

    def get_tensors(
        self,
        device,
    ):

        states = torch.tensor(
            np.array([t.state for t in self.buffer]),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.tensor(
            [t.action for t in self.buffer],
            dtype=torch.long,
            device=device,
        )

        rewards = torch.tensor(
            [t.reward for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.tensor(
            np.array([t.next_state for t in self.buffer]),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.tensor(
            [t.done for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        old_log_probs = torch.tensor(
            [t.log_prob for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        values = torch.tensor(
            [t.value for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        advantages = torch.tensor(
            [t.advantage for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        returns = torch.tensor(
            [t.return_value for t in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "old_log_probs": old_log_probs,
            "values": values,
            "advantages": advantages,
            "returns": returns,
        }

    # ==========================================================
    # Buffer Information
    # ==========================================================

    def __len__(self):

        return len(self.buffer)

    def is_empty(self):

        return len(self.buffer) == 0

    # ==========================================================
    # Clear Rollout
    # ==========================================================

    def clear(self):

        self.buffer = []
