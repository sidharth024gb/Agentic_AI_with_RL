"""
train.py

Training loop for the finance RL agents.

The trainer is intentionally agent-agnostic.

It can train:
    - PPOAgent
    - LLMRLAgent

The environment is responsible for:
    - state transitions
    - rewards
    - termination

The agent is responsible for:
    - action selection
    - storing transitions
    - learning

The callbacks are responsible for:
    - metrics
    - checkpoints
    - logging
"""

import os
import time
from typing import Any, Dict, Optional


from environment.finance_env import FinanceEnv

from agents.base_agent import BaseAgent

from training.callbacks import (
    TrainingCallback,
    CallbackList,
)


class Trainer:
    """
    Generic trainer for PPO-based finance agents.
    """

    def __init__(
        self,
        env: FinanceEnv,
        agent: BaseAgent,
        episodes: int = 1000,
        max_steps_per_episode: Optional[int] = None,
        callbacks: Optional[TrainingCallback] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        env : FinanceEnv
            Finance RL environment.

        agent : BaseAgent
            Agent being trained.

        episodes : int
            Number of training episodes.

        max_steps_per_episode : int, optional
            Optional additional safety limit.

        callbacks : TrainingCallback, optional
            Training callbacks.

        verbose : bool
            Whether to print training progress.
        """

        self.env = env

        self.agent = agent

        self.episodes = episodes

        self.max_steps_per_episode = max_steps_per_episode

        self.callbacks = callbacks if callbacks is not None else CallbackList([])

        self.verbose = verbose

        self.training_start_time = None

        self.training_end_time = None

    # ==========================================================
    # Training
    # ==========================================================

    def train(self):
        """
        Run the complete training process.

        Returns
        -------
        TrainingCallback
            Callback object containing training metrics.
        """

        self.training_start_time = time.perf_counter()

        self.callbacks.on_training_start(
            agent=self.agent,
            environment=self.env,
        )

        try:

            for episode in range(
                1,
                self.episodes + 1,
            ):

                self._run_episode(episode)

        finally:

            self.training_end_time = time.perf_counter()

            self.callbacks.on_training_end()

            self._close_environment()

        return self.callbacks

    # ==========================================================
    # Episode
    # ==========================================================

    def _run_episode(
        self,
        episode_number: int,
    ):
        """
        Run one complete training episode.
        """

        state = self.env.reset()

        # ------------------------------------------------------
        # Reset LLM planning state if available
        # ------------------------------------------------------

        self._reset_agent_episode()

        self.callbacks.on_episode_start(
            episode_number=episode_number,
            state=state,
        )

        total_reward = 0.0

        step_count = 0

        done = False

        completed = False

        terminated_reason = None

        episode_info = {}

        # ------------------------------------------------------
        # Environment interaction
        # ------------------------------------------------------

        while not done:

            step_count += 1

            # --------------------------------------------------
            # Safety step limit
            # --------------------------------------------------

            if (
                self.max_steps_per_episode is not None
                and step_count > self.max_steps_per_episode
            ):

                done = True

                terminated_reason = "MAX_STEPS"

                break

            # --------------------------------------------------
            # Select Action
            # --------------------------------------------------

            action_data = self._select_action(state)

            action = action_data["action"]

            log_prob = action_data["log_prob"]

            value = action_data["value"]

            # --------------------------------------------------
            # Environment Step
            # --------------------------------------------------

            step_result = self.env.step(action)

            (
                next_state,
                reward,
                done,
                info,
            ) = self._parse_step_result(step_result)

            episode_info = info

            # --------------------------------------------------
            # Environment Error
            # --------------------------------------------------

            if self._is_environment_error(info):

                # Do NOT store this transition.
                #
                # The backend design separates infrastructure
                # failures from agent mistakes. A server/database
                # failure should therefore not teach PPO that the
                # chosen action was bad.

                if self.verbose:

                    print(
                        f"[Episode {episode_number}] "
                        f"Environment error: "
                        f"{info.get('message', 'Unknown error')}"
                    )

                # Keep the current state and retry the step.

                step_count -= 1

                continue

            # --------------------------------------------------
            # Store PPO Transition
            # --------------------------------------------------

            self._store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            # --------------------------------------------------
            # Callback
            # --------------------------------------------------

            self.callbacks.on_step(
                step=step_count,
                reward=reward,
                done=done,
                info=info,
            )

            # --------------------------------------------------
            # Update episode statistics
            # --------------------------------------------------

            total_reward += reward

            state = next_state

            # --------------------------------------------------
            # Completion
            # --------------------------------------------------

            completed = self._get_completed(info)

            terminated_reason = self._get_termination_reason(
                info,
                terminated_reason,
            )

        # ======================================================
        # Learn
        # ======================================================

        self.agent.learn()

        # ======================================================
        # Episode Callback
        # ======================================================

        self.callbacks.on_episode_end(
            episode_number=episode_number,
            reward=total_reward,
            steps=step_count,
            completed=completed,
            terminated_reason=terminated_reason,
            info=episode_info,
        )

        # ======================================================
        # Logging
        # ======================================================

        if self.verbose:

            self._print_episode_summary(
                episode_number=episode_number,
                reward=total_reward,
                steps=step_count,
                completed=completed,
                terminated_reason=terminated_reason,
            )

    # ==========================================================
    # Action Selection
    # ==========================================================

    def _select_action(
        self,
        state,
    ) -> Dict[str, Any]:
        """
        Select an action from the agent.

        Standard PPOAgent returns:

            action, log_prob, value

        LLMRLAgent may use additional information internally.
        """

        result = self.agent.select_action(state)

        # ------------------------------------------------------
        # Tuple interface
        # ------------------------------------------------------

        if isinstance(
            result,
            tuple,
        ):

            if len(result) == 3:

                action, log_prob, value = result

                return {
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                }

            if len(result) == 2:

                action, log_prob = result

                return {
                    "action": action,
                    "log_prob": log_prob,
                    "value": 0.0,
                }

        # ------------------------------------------------------
        # Dictionary interface
        # ------------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            return {
                "action": result["action"],
                "log_prob": result.get(
                    "log_prob",
                    0.0,
                ),
                "value": result.get(
                    "value",
                    0.0,
                ),
            }

        raise TypeError(
            "Agent select_action() must return "
            "(action, log_prob, value), "
            "(action, log_prob), or a dictionary."
        )

    # ==========================================================
    # Store Transition
    # ==========================================================

    def _store_transition(
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
        Store a transition using the agent interface.
        """

        if hasattr(
            self.agent,
            "store_transition",
        ):

            self.agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
            )

            return

        raise AttributeError("Agent does not implement " "store_transition().")

    # ==========================================================
    # Step Result Parsing
    # ==========================================================

    @staticmethod
    def _parse_step_result(
        step_result,
    ):
        """
        Handle the environment's step return.

        Expected current interface:

            next_state,
            reward,
            done,
            info
        """

        if not isinstance(
            step_result,
            tuple,
        ):

            raise TypeError(
                "Environment step() must return " "(next_state, reward, done, info)."
            )

        if len(step_result) != 4:

            raise ValueError("Environment step() must return " "exactly four values.")

        next_state, reward, done, info = step_result

        if info is None:

            info = {}

        return (
            next_state,
            reward,
            done,
            info,
        )

    # ==========================================================
    # Environment Errors
    # ==========================================================

    @staticmethod
    def _is_environment_error(
        info: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a step failed because of the
        environment rather than because of an agent decision.
        """

        return bool(
            info.get(
                "environmentError",
                False,
            )
        )

    # ==========================================================
    # Completion
    # ==========================================================

    @staticmethod
    def _get_completed(
        info: Dict[str, Any],
    ) -> bool:
        """
        Determine whether the episode goal was completed.
        """

        return bool(
            info.get(
                "completed",
                False,
            )
        )

    # ==========================================================
    # Termination Reason
    # ==========================================================

    @staticmethod
    def _get_termination_reason(
        info: Dict[str, Any],
        current_reason: Optional[str],
    ) -> Optional[str]:
        """
        Get the termination reason returned by the environment.
        """

        reason = info.get("terminatedReason")

        if reason is not None:

            return reason

        return current_reason

    # ==========================================================
    # Reset Agent Episode State
    # ==========================================================

    def _reset_agent_episode(self):
        """
        Reset episode-specific state for agents that need it.

        This is particularly useful for LLMRLAgent, which keeps
        a current planning hint.
        """

        if hasattr(
            self.agent,
            "reset_planning",
        ):

            self.agent.reset_planning()

    # ==========================================================
    # Episode Logging
    # ==========================================================

    def _print_episode_summary(
        self,
        episode_number: int,
        reward: float,
        steps: int,
        completed: bool,
        terminated_reason: Optional[str],
    ):
        """
        Print a compact training summary.
        """

        status = "COMPLETED" if completed else "TERMINATED"

        reason = terminated_reason if terminated_reason is not None else "-"

        print(
            f"Episode {episode_number:04d} | "
            f"Reward: {reward:8.2f} | "
            f"Steps: {steps:3d} | "
            f"{status:10s} | "
            f"Reason: {reason}"
        )

    # ==========================================================
    # Close Environment
    # ==========================================================

    def _close_environment(self):
        """
        Close the environment safely.
        """

        try:

            self.env.close("TRAINING_COMPLETE")

        except TypeError:

            self.env.close()

        except Exception as exc:

            if self.verbose:

                print(f"Environment close error: {exc}")

    # ==========================================================
    # Training Statistics
    # ==========================================================

    def get_training_time(self) -> float:
        """
        Return total training time in seconds.
        """

        if self.training_start_time is None or self.training_end_time is None:

            return 0.0

        return self.training_end_time - self.training_start_time


# ==============================================================
# Convenience Function
# ==============================================================


def train_agent(
    env: FinanceEnv,
    agent: BaseAgent,
    episodes: int = 1000,
    max_steps_per_episode: Optional[int] = None,
    callbacks: Optional[TrainingCallback] = None,
    verbose: bool = True,
):
    """
    Convenience wrapper around Trainer.

    Example
    -------
    trainer = train_agent(
        env=env,
        agent=agent,
        episodes=1000,
    )
    """

    trainer = Trainer(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps_per_episode=(max_steps_per_episode),
        callbacks=callbacks,
        verbose=verbose,
    )

    trainer.train()

    return trainer
