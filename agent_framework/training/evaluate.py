"""
evaluate.py

Evaluation utilities for trained finance RL agents.

Evaluation is performed without learning.

The evaluator measures:
    - episode reward
    - episode steps
    - goal completion
    - success rate
    - termination reason
    - total evaluation time

The same evaluator can be used for:
    - PPOAgent
    - LLMRLAgent
"""

import time
from typing import Any, Dict, List, Optional


class Evaluator:
    """
    Evaluates a trained RL agent without updating its parameters.
    """

    def __init__(
        self,
        env,
        agent,
        episodes: int = 100,
        max_steps_per_episode: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        env :
            Finance environment.

        agent :
            Trained PPO or LLM+PPO agent.

        episodes : int
            Number of evaluation episodes.

        max_steps_per_episode : int, optional
            Additional safety limit for evaluation.

        verbose : bool
            Print episode results.
        """

        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.verbose = verbose

        self.results: List[Dict[str, Any]] = []

    # ==========================================================
    # Evaluate
    # ==========================================================

    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation episodes.

        Returns
        -------
        dict
            Aggregate evaluation metrics.
        """

        self.results = []

        start_time = time.perf_counter()

        for episode in range(
            1,
            self.episodes + 1,
        ):

            result = self._run_episode(episode)

            self.results.append(result)

        end_time = time.perf_counter()

        summary = self._calculate_summary(end_time - start_time)

        self._close_environment()

        return summary

    # ==========================================================
    # Episode
    # ==========================================================

    def _run_episode(
        self,
        episode_number: int,
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        """

        state = self.env.reset()

        self._reset_agent_episode()

        total_reward = 0.0

        steps = 0

        done = False

        completed = False

        terminated_reason = None

        environment_errors = 0

        info = {}

        start_time = time.perf_counter()

        while not done:

            steps += 1

            # --------------------------------------------------
            # Safety limit
            # --------------------------------------------------

            if (
                self.max_steps_per_episode is not None
                and steps > self.max_steps_per_episode
            ):

                done = True

                terminated_reason = "MAX_STEPS"

                break

            # --------------------------------------------------
            # Select action
            # --------------------------------------------------

            action = self._select_action(state)

            # --------------------------------------------------
            # Environment step
            # --------------------------------------------------

            step_result = self.env.step(action)

            (
                next_state,
                reward,
                done,
                info,
            ) = self._parse_step_result(step_result)

            # --------------------------------------------------
            # Environment error
            # --------------------------------------------------

            if self._is_environment_error(info):

                environment_errors += 1

                # Environment failures are not counted as
                # successful or failed agent decisions.

                steps -= 1

                continue

            # --------------------------------------------------
            # Update episode state
            # --------------------------------------------------

            total_reward += reward

            completed = self._get_completed(info)

            reason = info.get("terminatedReason")

            if reason is not None:

                terminated_reason = reason

            state = next_state

        episode_time = time.perf_counter() - start_time

        result = {
            "episode": episode_number,
            "reward": total_reward,
            "steps": steps,
            "completed": completed,
            "terminated_reason": terminated_reason,
            "environment_errors": (environment_errors),
            "evaluation_time": episode_time,
        }

        if self.verbose:

            self._print_episode_result(result)

        return result

    # ==========================================================
    # Action Selection
    # ==========================================================

    def _select_action(
        self,
        state,
    ):
        """
        Select an action for evaluation.

        PPOAgent.select_action() returns:

            action, log_prob, value

        We only need the action during evaluation.
        """

        result = self.agent.select_action(state)

        if isinstance(
            result,
            tuple,
        ):

            return result[0]

        if isinstance(
            result,
            dict,
        ):

            return result["action"]

        return result

    # ==========================================================
    # Agent Episode Reset
    # ==========================================================

    def _reset_agent_episode(self):
        """
        Reset episode-specific agent state.

        Useful for LLMRLAgent because it may maintain
        a planning hint between steps.
        """

        if hasattr(
            self.agent,
            "reset_planning",
        ):

            self.agent.reset_planning()

    # ==========================================================
    # Step Result Parsing
    # ==========================================================

    @staticmethod
    def _parse_step_result(
        step_result,
    ):
        """
        Parse the FinanceEnv step result.

        Expected interface:

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
    # Environment Error
    # ==========================================================

    @staticmethod
    def _is_environment_error(
        info: Dict[str, Any],
    ) -> bool:
        """
        Check whether the backend/environment reported
        an infrastructure error.
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
        Determine whether the goal was completed.
        """

        return bool(
            info.get(
                "completed",
                False,
            )
        )

    # ==========================================================
    # Summary
    # ==========================================================

    def _calculate_summary(
        self,
        evaluation_time: float,
    ) -> Dict[str, Any]:
        """
        Calculate aggregate evaluation metrics.
        """

        if not self.results:

            return {
                "episodes": 0,
                "average_reward": 0.0,
                "average_steps": 0.0,
                "success_rate": 0.0,
                "completed_episodes": 0,
                "total_environment_errors": 0,
                "evaluation_time": evaluation_time,
            }

        rewards = [result["reward"] for result in self.results]

        steps = [result["steps"] for result in self.results]

        completed = [result["completed"] for result in self.results]

        environment_errors = [result["environment_errors"] for result in self.results]

        completed_count = sum(completed)

        episode_count = len(self.results)

        return {
            "episodes": episode_count,
            "average_reward": (sum(rewards) / episode_count),
            "average_steps": (sum(steps) / episode_count),
            "success_rate": (completed_count / episode_count),
            "completed_episodes": (completed_count),
            "total_environment_errors": (sum(environment_errors)),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "evaluation_time": (evaluation_time),
        }

    # ==========================================================
    # Episode Output
    # ==========================================================

    def _print_episode_result(
        self,
        result: Dict[str, Any],
    ):
        """
        Print one evaluation episode.
        """

        status = "COMPLETED" if result["completed"] else "NOT COMPLETED"

        reason = result["terminated_reason"] if result["terminated_reason"] else "-"

        print(
            f"Evaluation Episode "
            f"{result['episode']:04d} | "
            f"Reward: "
            f"{result['reward']:8.2f} | "
            f"Steps: "
            f"{result['steps']:3d} | "
            f"{status:13s} | "
            f"Reason: {reason}"
        )

    # ==========================================================
    # Environment Close
    # ==========================================================

    def _close_environment(self):
        """
        Close the evaluation environment.
        """

        try:

            self.env.close("EVALUATION_COMPLETE")

        except TypeError:

            self.env.close()

        except Exception as exc:

            if self.verbose:

                print(f"Environment close error: {exc}")


# ==============================================================
# Convenience Function
# ==============================================================


def evaluate_agent(
    env,
    agent,
    episodes: int = 100,
    max_steps_per_episode: Optional[int] = None,
    verbose: bool = True,
):
    """
    Convenience function for evaluating an agent.

    Example
    -------

    results = evaluate_agent(
        env,
        agent,
        episodes=100,
    )
    """

    evaluator = Evaluator(
        env=env,
        agent=agent,
        episodes=episodes,
        max_steps_per_episode=(max_steps_per_episode),
        verbose=verbose,
    )

    return evaluator.evaluate()
 