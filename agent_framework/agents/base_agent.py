"""
base_agent.py

Abstract base class for all RL agents.

Every agent implementation (Q-Learning, DQN, PPO, LLM+RL)
must inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch

from config.config import config


class BaseAgent(ABC):
    """
    Base class for all reinforcement learning agents.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
    ):
        self.observation_size = observation_size
        self.action_size = action_size

        self.device = torch.device(config.agent.DEVICE)

        self.training = True

        self.episode = 0

        self.total_steps = 0

    # ==========================================================
    # Core RL Methods
    # ==========================================================

    @abstractmethod
    def select_action(self, state):
        """
        Select an action from the current state.

        Parameters
        ----------
        state : np.ndarray

        Returns
        -------
        int
            Selected action index.
        """
        pass

    @abstractmethod
    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Update the agent after one transition.

        Parameters
        ----------
        state
        action
        reward
        next_state
        done
        """
        pass

    # ==========================================================
    # Model Persistence
    # ==========================================================

    @abstractmethod
    def save(self, path: str):
        """
        Save model.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model.
        """
        pass

    # ==========================================================
    # Training / Evaluation
    # ==========================================================

    def train(self):
        """
        Enable training mode.
        """

        self.training = True

    def eval(self):
        """
        Enable evaluation mode.
        """

        self.training = False

    # ==========================================================
    # Episode Tracking
    # ==========================================================

    def start_episode(self):

        self.episode += 1

    def increment_step(self):

        self.total_steps += 1

    # ==========================================================
    # Information
    # ==========================================================

    def get_info(self) -> Dict[str, Any]:

        return {
            "algorithm": self.__class__.__name__,
            "device": str(self.device),
            "episode": self.episode,
            "total_steps": self.total_steps,
            "training": self.training,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
        }

    # ==========================================================
    # Utilities
    # ==========================================================

    def create_checkpoint_directory(self):

        Path(config.logging.MODEL_DIR).mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Task Planning
    # ==========================================================

    def plan(self, state):
        """
        planning step.

        Classical RL agents return None.
        LLM+RL agents can override this method
        to generate a high-level plan before
        selecting an action.
        """
        return None