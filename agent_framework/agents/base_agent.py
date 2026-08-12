"""
base_agent.py

Abstract base class for all reinforcement learning agents.

Every agent implementation such as PPO, DQN, Q-Learning,
and LLM-enhanced RL agents inherits from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch

from config.config import config


class BaseAgent(ABC):
    """
    Base class for reinforcement learning agents.

    The environment is responsible for:
        - executing actions
        - calculating rewards
        - detecting episode completion
        - detecting environment/backend errors
        - recording backend episode history

    The agent is responsible for:
        - selecting actions
        - storing valid learning transitions
        - updating its model
        - saving/loading learned parameters
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
    ):

        if observation_size <= 0:
            raise ValueError("observation_size must be greater than 0.")

        if action_size <= 0:
            raise ValueError("action_size must be greater than 0.")

        self.observation_size = int(observation_size)

        self.action_size = int(action_size)

        # ==========================================================
        # Device
        # ==========================================================

        self.device = torch.device(config.agent.DEVICE)

        # ==========================================================
        # Reproducibility
        # ==========================================================

        self.seed = config.environment.RANDOM_SEED

        torch.manual_seed(self.seed)

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(self.seed)

        # ==========================================================
        # Agent State
        # ==========================================================

        self.training = True

        self.episode = 0

        self.total_steps = 0

    # ==========================================================
    # Core RL Methods
    # ==========================================================

    @abstractmethod
    def select_action(
        self,
        state,
    ):
        """
        Select an action from the current observation.

        Parameters
        ----------
        state
            Encoded environment observation.

        Returns
        -------
        Implementation dependent.

        PPO normally returns:
            action, log_probability, value

        Other agents may return only:
            action
        """

        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        *args,
        **kwargs,
    ):
        """
        Perform a learning update.

        Different RL algorithms require different
        learning interfaces.

        PPO learns from its rollout buffer.

        DQN/Q-Learning may learn directly from
        transitions or replay memory.
        """

        raise NotImplementedError

    # ==========================================================
    # Model Persistence
    # ==========================================================

    @abstractmethod
    def save(
        self,
        path: str,
    ):
        """
        Save model/checkpoint.
        """

        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        path: str,
    ):
        """
        Load model/checkpoint.
        """

        raise NotImplementedError

    # ==========================================================
    # Training / Evaluation
    # ==========================================================

    def train(self):
        """
        Enable training behaviour.
        """

        self.training = True

        return self

    def eval(self):
        """
        Enable evaluation behaviour.
        """

        self.training = False

        return self

    # ==========================================================
    # Episode Tracking
    # ==========================================================

    def start_episode(self):
        """
        Record the beginning of an agent episode.

        Backend episode creation is handled by
        FinanceEnvironment.reset().
        """

        self.episode += 1

    def increment_step(
        self,
        count: int = 1,
    ):
        """
        Increment total agent interaction steps.
        """

        self.total_steps += int(count)

    # ==========================================================
    # Information
    # ==========================================================

    def get_info(
        self,
    ) -> Dict[str, Any]:
        """
        Return general agent information.
        """

        return {
            "algorithm": self.__class__.__name__,
            "configured_algorithm": config.agent.ALGORITHM,
            "agent_type": config.agent.AGENT_TYPE,
            "device": str(self.device),
            "episode": self.episode,
            "total_steps": self.total_steps,
            "training": self.training,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "seed": self.seed,
        }

    # ==========================================================
    # Checkpoint Utilities
    # ==========================================================

    def create_checkpoint_directory(
        self,
    ):
        """
        Ensure model output directory exists.
        """

        Path(config.logging.MODEL_DIR).mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def ensure_parent_directory(
        path,
    ):
        """
        Ensure the parent directory of a checkpoint exists.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    # ==========================================================
    # Task Planning
    # ==========================================================

    def plan(
        self,
        state,
    ):
        """
        Optional planning step.

        Classical RL agents return None.

        LLM-enhanced agents can override this method
        to generate high-level procedural guidance.
        """

        return None
