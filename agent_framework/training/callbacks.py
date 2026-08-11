"""
callbacks.py

Training callbacks for the finance RL agents.

The callbacks are intentionally independent of PPO and LLM logic.
They receive training events and record useful experiment metrics.

Tracked information includes:
    - episode reward
    - episode length
    - episode completion
    - episode termination reason
    - running success rate
    - best reward
    - convergence information
    - optional model checkpoints
"""

import json
import os
import time
from typing import Any, Dict, Optional


class TrainingCallback:
    """
    Base callback interface.

    The training loop can call these methods without knowing
    whether the agent is PPO or LLM + PPO.
    """

    def on_training_start(
        self,
        agent=None,
        environment=None,
    ):
        """
        Called once before training starts.
        """
        pass

    def on_episode_start(
        self,
        episode_number: int,
        state=None,
    ):
        """
        Called at the beginning of an episode.
        """
        pass

    def on_step(
        self,
        step: int,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Called after every environment step.
        """
        pass

    def on_episode_end(
        self,
        episode_number: int,
        reward: float,
        steps: int,
        completed: bool,
        terminated_reason: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ):
        """
        Called when an episode finishes.
        """
        pass

    def on_training_end(self):
        """
        Called after training finishes.
        """
        pass


class TrainingMetricsCallback(TrainingCallback):
    """
    Collects metrics during training.

    These metrics can later be consumed by metrics.py,
    visualization.py, or experiment.py.
    """

    def __init__(
        self,
        convergence_window: int = 20,
        convergence_threshold: float = 0.8,
    ):
        """
        Parameters
        ----------
        convergence_window : int
            Number of recent episodes used to determine convergence.

        convergence_threshold : float
            Required success rate within the window.
        """

        self.convergence_window = convergence_window

        self.convergence_threshold = convergence_threshold

        self.reset()

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):
        """
        Reset all collected metrics.
        """

        self.start_time = None

        self.end_time = None

        self.episode_rewards = []

        self.episode_lengths = []

        self.episode_completed = []

        self.termination_reasons = []

        self.step_rewards = []

        self.total_steps = 0

        self.total_episodes = 0

        self.success_count = 0

        self.best_reward = None

        self.best_episode = None

        self.convergence_episode = None

    # ==========================================================
    # Training Start
    # ==========================================================

    def on_training_start(
        self,
        agent=None,
        environment=None,
    ):
        self.reset()

        self.start_time = time.perf_counter()

    # ==========================================================
    # Episode Start
    # ==========================================================

    def on_episode_start(
        self,
        episode_number: int,
        state=None,
    ):
        pass

    # ==========================================================
    # Step
    # ==========================================================

    def on_step(
        self,
        step: int,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        self.total_steps += 1

        self.step_rewards.append(reward)

    # ==========================================================
    # Episode End
    # ==========================================================

    def on_episode_end(
        self,
        episode_number: int,
        reward: float,
        steps: int,
        completed: bool,
        terminated_reason: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ):

        self.total_episodes += 1

        self.episode_rewards.append(reward)

        self.episode_lengths.append(steps)

        self.episode_completed.append(completed)

        self.termination_reasons.append(terminated_reason)

        if completed:

            self.success_count += 1

        # ------------------------------------------------------
        # Best Reward
        # ------------------------------------------------------

        if self.best_reward is None or reward > self.best_reward:

            self.best_reward = reward

            self.best_episode = episode_number

        # ------------------------------------------------------
        # Convergence
        # ------------------------------------------------------

        self._check_convergence(episode_number)

    # ==========================================================
    # Convergence
    # ==========================================================

    def _check_convergence(
        self,
        episode_number: int,
    ):
        """
        Check whether the agent has reached the configured
        success-rate threshold over the recent episode window.

        Once convergence is detected, the first episode at which
        it occurred is retained.
        """

        if self.convergence_episode is not None:
            return

        if len(self.episode_completed) < self.convergence_window:
            return

        recent_results = self.episode_completed[-self.convergence_window :]

        recent_success_rate = sum(recent_results) / len(recent_results)

        if recent_success_rate >= self.convergence_threshold:

            self.convergence_episode = episode_number

    # ==========================================================
    # Training End
    # ==========================================================

    def on_training_end(self):

        self.end_time = time.perf_counter()

    # ==========================================================
    # Statistics
    # ==========================================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the training run.
        """

        training_time = 0.0

        if self.start_time is not None and self.end_time is not None:

            training_time = self.end_time - self.start_time

        average_reward = 0.0

        if self.episode_rewards:

            average_reward = sum(self.episode_rewards) / len(self.episode_rewards)

        average_steps = 0.0

        if self.episode_lengths:

            average_steps = sum(self.episode_lengths) / len(self.episode_lengths)

        success_rate = 0.0

        if self.total_episodes > 0:

            success_rate = self.success_count / self.total_episodes

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "success_rate": success_rate,
            "success_count": self.success_count,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "convergence_episode": self.convergence_episode,
            "training_time": training_time,
        }

    # ==========================================================
    # Save Metrics
    # ==========================================================

    def save(
        self,
        path: str,
    ):
        """
        Save training metrics as JSON.
        """

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        data = {
            "summary": self.get_summary(),
            "episode_rewards": (self.episode_rewards),
            "episode_lengths": (self.episode_lengths),
            "episode_completed": (self.episode_completed),
            "termination_reasons": (self.termination_reasons),
            "step_rewards": (self.step_rewards),
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
            )


class CheckpointCallback(TrainingCallback):
    """
    Periodically saves the agent.

    The agent must implement a save(path) method.
    """

    def __init__(
        self,
        save_directory: str,
        save_frequency: int = 100,
    ):
        """
        Parameters
        ----------
        save_directory : str
            Directory where checkpoints are stored.

        save_frequency : int
            Save every N episodes.
        """

        self.save_directory = save_directory

        self.save_frequency = max(
            1,
            save_frequency,
        )

        self.agent = None

        os.makedirs(
            self.save_directory,
            exist_ok=True,
        )

    # ==========================================================
    # Training Start
    # ==========================================================

    def on_training_start(
        self,
        agent=None,
        environment=None,
    ):

        self.agent = agent

    # ==========================================================
    # Episode End
    # ==========================================================

    def on_episode_end(
        self,
        episode_number: int,
        reward: float,
        steps: int,
        completed: bool,
        terminated_reason: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ):

        if self.agent is None:
            return

        if episode_number % self.save_frequency != 0:
            return

        checkpoint_path = os.path.join(
            self.save_directory,
            f"episode_{episode_number}.pt",
        )

        self.agent.save(checkpoint_path)


class CallbackList(TrainingCallback):
    """
    Combines multiple callbacks.

    This allows train.py to use:

        callbacks = CallbackList([
            metrics_callback,
            checkpoint_callback,
        ])

    instead of handling each callback separately.
    """

    def __init__(
        self,
        callbacks=None,
    ):

        self.callbacks = callbacks if callbacks is not None else []

    # ==========================================================
    # Training Start
    # ==========================================================

    def on_training_start(
        self,
        agent=None,
        environment=None,
    ):

        for callback in self.callbacks:

            callback.on_training_start(
                agent=agent,
                environment=environment,
            )

    # ==========================================================
    # Episode Start
    # ==========================================================

    def on_episode_start(
        self,
        episode_number: int,
        state=None,
    ):

        for callback in self.callbacks:

            callback.on_episode_start(
                episode_number=episode_number,
                state=state,
            )

    # ==========================================================
    # Step
    # ==========================================================

    def on_step(
        self,
        step: int,
        reward: float,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):

        for callback in self.callbacks:

            callback.on_step(
                step=step,
                reward=reward,
                done=done,
                info=info,
            )

    # ==========================================================
    # Episode End
    # ==========================================================

    def on_episode_end(
        self,
        episode_number: int,
        reward: float,
        steps: int,
        completed: bool,
        terminated_reason: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ):

        for callback in self.callbacks:

            callback.on_episode_end(
                episode_number=episode_number,
                reward=reward,
                steps=steps,
                completed=completed,
                terminated_reason=terminated_reason,
                info=info,
            )

    # ==========================================================
    # Training End
    # ==========================================================

    def on_training_end(self):

        for callback in self.callbacks:

            callback.on_training_end()
