"""
rollout_buffer.py

On-policy rollout storage for PPO.

Stores valid transitions collected from FinanceEnvironment.

The buffer is cleared after every PPO update because PPO is
an on-policy reinforcement learning algorithm.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# ==========================================================
# Transition
# ==========================================================


@dataclass
class Transition:
    """
    One valid PPO transition.
    """

    state: np.ndarray

    action: int

    reward: float

    next_state: np.ndarray

    done: bool

    log_prob: float

    value: float

    # ------------------------------------------------------
    # Added later by GAE calculation
    # ------------------------------------------------------

    advantage: float = 0.0

    return_value: float = 0.0


# ==========================================================
# Rollout Buffer
# ==========================================================


class RolloutBuffer:
    """
    Stores PPO on-policy transitions.

    Data flow:

        environment step
              ↓
        RolloutBuffer.add()
              ↓
        PPO.compute_advantages()
              ↓
        add_advantages()
              ↓
        get_tensors()
              ↓
        PPO update
              ↓
        clear()
    """

    def __init__(
        self,
        observation_size: Optional[int] = None,
    ):

        self.observation_size = (
            int(observation_size) if observation_size is not None else None
        )

        self.buffer = []

        self.advantages_ready = False

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
        """
        Add one transition to the rollout.

        Observations are copied so subsequent environment
        changes cannot accidentally modify stored PPO data.
        """

        # ------------------------------------------------------
        # Convert observations
        # ------------------------------------------------------

        state = (
            np.asarray(
                state,
                dtype=np.float32,
            )
            .reshape(-1)
            .copy()
        )

        next_state = (
            np.asarray(
                next_state,
                dtype=np.float32,
            )
            .reshape(-1)
            .copy()
        )

        # ------------------------------------------------------
        # Infer observation size if necessary
        # ------------------------------------------------------

        if self.observation_size is None:

            self.observation_size = state.shape[0]

        # ------------------------------------------------------
        # Validate observation dimensions
        # ------------------------------------------------------

        if state.shape[0] != self.observation_size:

            raise ValueError(
                "Invalid state size. "
                f"Expected {self.observation_size}, "
                f"received {state.shape[0]}."
            )

        if next_state.shape[0] != self.observation_size:

            raise ValueError(
                "Invalid next_state size. "
                f"Expected {self.observation_size}, "
                f"received {next_state.shape[0]}."
            )

        # ------------------------------------------------------
        # Validate numerical data
        # ------------------------------------------------------

        if not np.all(np.isfinite(state)):

            raise ValueError("State contains NaN or infinite values.")

        if not np.all(np.isfinite(next_state)):

            raise ValueError("next_state contains NaN or infinite values.")

        reward = float(reward)

        log_prob = float(log_prob)

        value = float(value)

        if not np.isfinite(reward):

            raise ValueError("Reward must be finite.")

        if not np.isfinite(log_prob):

            raise ValueError("log_prob must be finite.")

        if not np.isfinite(value):

            raise ValueError("Value estimate must be finite.")

        # ------------------------------------------------------
        # Create transition
        # ------------------------------------------------------

        transition = Transition(
            state=state,
            action=int(action),
            reward=reward,
            next_state=next_state,
            done=bool(done),
            log_prob=log_prob,
            value=value,
        )

        self.buffer.append(transition)

        # New transitions invalidate previously calculated GAE.
        self.advantages_ready = False

    # ==========================================================
    # Mark Episode Boundary
    # ==========================================================

    def mark_last_done(
        self,
    ):
        """
        Mark the most recently stored transition as terminal.

        This is useful when the NEXT environment action fails
        because of an infrastructure/environment error.

        The failed transition itself should not enter PPO
        training, but GAE must still know that the previous
        rollout segment ended before the next episode starts.

        Returns
        -------
        bool
            True if a transition was marked.
            False if the buffer was empty.
        """

        if not self.buffer:

            return False

        self.buffer[-1].done = True

        self.advantages_ready = False

        return True

    # ==========================================================
    # Add PPO Calculations
    # ==========================================================

    def add_advantages(
        self,
        advantages,
        returns,
    ):
        """
        Store Generalized Advantage Estimates and value targets.
        """

        if len(advantages) != len(self.buffer):

            raise ValueError(
                "Number of advantages does not match " "the rollout buffer length."
            )

        if len(returns) != len(self.buffer):

            raise ValueError(
                "Number of returns does not match " "the rollout buffer length."
            )

        for (
            transition,
            advantage,
            return_value,
        ) in zip(
            self.buffer,
            advantages,
            returns,
        ):

            advantage = float(advantage)

            return_value = float(return_value)

            if not np.isfinite(advantage):

                raise ValueError("Advantage contains NaN " "or infinite value.")

            if not np.isfinite(return_value):

                raise ValueError("Return contains NaN " "or infinite value.")

            transition.advantage = advantage

            transition.return_value = return_value

        self.advantages_ready = True

    # ==========================================================
    # Tensor Conversion
    # ==========================================================

    def get_tensors(
        self,
        device,
    ):
        """
        Convert the current rollout into PyTorch tensors.

        Returns
        -------
        dict
            Batched tensors used by PPOAgent.learn().
        """

        if self.is_empty():

            raise ValueError("Cannot create tensors from " "an empty rollout buffer.")

        if not self.advantages_ready:

            raise RuntimeError(
                "Advantages have not been calculated. "
                "Call compute_advantages() first."
            )

        # ------------------------------------------------------
        # Observations
        # ------------------------------------------------------

        states_np = np.stack(
            [transition.state for transition in self.buffer],
            axis=0,
        )

        next_states_np = np.stack(
            [transition.next_state for transition in self.buffer],
            axis=0,
        )

        states = torch.as_tensor(
            states_np,
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.as_tensor(
            next_states_np,
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # Actions
        # ------------------------------------------------------

        actions = torch.as_tensor(
            [transition.action for transition in self.buffer],
            dtype=torch.long,
            device=device,
        )

        # ------------------------------------------------------
        # Rewards
        # ------------------------------------------------------

        rewards = torch.as_tensor(
            [transition.reward for transition in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # Done Flags
        # ------------------------------------------------------

        dones = torch.as_tensor(
            [transition.done for transition in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # Old Policy Log Probabilities
        # ------------------------------------------------------

        old_log_probs = torch.as_tensor(
            [transition.log_prob for transition in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # Old Value Estimates
        # ------------------------------------------------------

        values = torch.as_tensor(
            [transition.value for transition in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # GAE Advantages
        # ------------------------------------------------------

        advantages = torch.as_tensor(
            [transition.advantage for transition in self.buffer],
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------
        # Return Targets
        # ------------------------------------------------------

        returns = torch.as_tensor(
            [transition.return_value for transition in self.buffer],
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

    def __len__(
        self,
    ):

        return len(self.buffer)

    def is_empty(
        self,
    ):

        return len(self.buffer) == 0

    @property
    def size(
        self,
    ):

        return len(self.buffer)

    # ==========================================================
    # Last Transition
    # ==========================================================

    def last(
        self,
    ):
        """
        Return the most recently stored transition.
        """

        if not self.buffer:

            return None

        return self.buffer[-1]

    # ==========================================================
    # Clear Rollout
    # ==========================================================

    def clear(
        self,
    ):
        """
        Remove transitions after a PPO update.

        observation_size is intentionally preserved because the
        agent architecture does not change between rollouts.
        """

        self.buffer.clear()

        self.advantages_ready = False
