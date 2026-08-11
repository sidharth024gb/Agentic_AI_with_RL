"""
experiment.py

Experiment runner for comparing RL agents.

Primary comparison:

    PPO
    vs
    LLM + PPO

The experiment runner is responsible for:

    - creating independent training runs
    - training each agent
    - evaluating each trained agent
    - collecting training metrics
    - collecting evaluation metrics
    - collecting LLM-specific metrics when available
    - saving experiment results

The actual agent logic remains inside the agent classes.
"""

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from training.train import Trainer
from training.evaluate import Evaluator
from training.callbacks import (
    TrainingMetricsCallback,
    CheckpointCallback,
    CallbackList,
)


class ExperimentRunner:
    """
    Runs repeated experiments for one or more agent types.

    The runner uses factory functions so that each run receives
    a fresh environment and a fresh agent.
    """

    def __init__(
        self,
        experiment_name: str,
        agent_factories: Dict[str, Callable],
        environment_factory: Callable,
        training_episodes: int = 1000,
        evaluation_episodes: int = 100,
        runs: int = 3,
        max_steps_per_episode: Optional[int] = None,
        results_directory: str = "results/metrics",
        model_directory: str = "results/models",
        checkpoint_frequency: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        experiment_name : str
            Name of the experiment.

        agent_factories : dict
            Mapping of agent names to factory functions.

            Example:

                {
                    "PPO": create_ppo_agent,
                    "LLM_PPO": create_llm_ppo_agent
                }

        environment_factory : callable
            Function that creates a fresh FinanceEnv.

        training_episodes : int
            Number of training episodes per run.

        evaluation_episodes : int
            Number of evaluation episodes per run.

        runs : int
            Number of independent runs per agent.

        max_steps_per_episode : int, optional
            Additional safety limit.

        results_directory : str
            Directory for experiment metrics.

        model_directory : str
            Directory for trained models.

        checkpoint_frequency : int, optional
            Save checkpoints every N episodes.

        verbose : bool
            Print experiment progress.
        """

        self.experiment_name = experiment_name

        self.agent_factories = agent_factories

        self.environment_factory = environment_factory

        self.training_episodes = training_episodes

        self.evaluation_episodes = evaluation_episodes

        self.runs = max(
            1,
            runs,
        )

        self.max_steps_per_episode = max_steps_per_episode

        self.results_directory = results_directory

        self.model_directory = model_directory

        self.checkpoint_frequency = checkpoint_frequency

        self.verbose = verbose

        self.results: List[Dict[str, Any]] = []

        os.makedirs(
            self.results_directory,
            exist_ok=True,
        )

        os.makedirs(
            self.model_directory,
            exist_ok=True,
        )

    # ==========================================================
    # Run Experiment
    # ==========================================================

    def run(self) -> Dict[str, Any]:
        """
        Run the complete experiment.

        Returns
        -------
        dict
            Complete experiment results.
        """

        experiment_start = time.perf_counter()

        self.results = []

        # ------------------------------------------------------
        # Run each agent
        # ------------------------------------------------------

        for agent_name, agent_factory in self.agent_factories.items():

            if self.verbose:

                print()
                print("=" * 70)
                print(f"Agent: {agent_name}")
                print("=" * 70)

            for run_number in range(
                1,
                self.runs + 1,
            ):

                result = self._run_single_experiment(
                    agent_name=agent_name,
                    agent_factory=agent_factory,
                    run_number=run_number,
                )

                self.results.append(result)

        experiment_time = time.perf_counter() - experiment_start

        summary = self._build_experiment_summary(experiment_time)

        self._save_results(summary)

        return summary

    # ==========================================================
    # Single Run
    # ==========================================================

    def _run_single_experiment(
        self,
        agent_name: str,
        agent_factory: Callable,
        run_number: int,
    ) -> Dict[str, Any]:
        """
        Run one independent training + evaluation cycle.
        """

        if self.verbose:

            print()
            print(f"Run {run_number}/{self.runs}")

        # ------------------------------------------------------
        # Create fresh environment
        # ------------------------------------------------------

        train_env = self.environment_factory()

        # ------------------------------------------------------
        # Create fresh agent
        # ------------------------------------------------------

        agent = agent_factory()

        # ------------------------------------------------------
        # Metrics callback
        # ------------------------------------------------------

        metrics_callback = TrainingMetricsCallback()

        # ------------------------------------------------------
        # Checkpoint callback
        # ------------------------------------------------------

        callbacks = [metrics_callback]

        if self.checkpoint_frequency is not None:

            checkpoint_directory = os.path.join(
                self.model_directory,
                self._safe_name(agent_name),
                f"run_{run_number}",
            )

            checkpoint_callback = CheckpointCallback(
                save_directory=(checkpoint_directory),
                save_frequency=(self.checkpoint_frequency),
            )

            callbacks.append(checkpoint_callback)

        callback_list = CallbackList(callbacks)

        # ------------------------------------------------------
        # Train
        # ------------------------------------------------------

        trainer = Trainer(
            env=train_env,
            agent=agent,
            episodes=self.training_episodes,
            max_steps_per_episode=(self.max_steps_per_episode),
            callbacks=callback_list,
            verbose=self.verbose,
        )

        trainer.train()

        training_summary = metrics_callback.get_summary()

        # ------------------------------------------------------
        # Save training metrics
        # ------------------------------------------------------

        training_metrics_path = os.path.join(
            self.results_directory,
            self._safe_name(agent_name),
            f"run_{run_number}_training.json",
        )

        metrics_callback.save(training_metrics_path)

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------

        evaluation_summary = self._evaluate_agent(
            agent_name=agent_name,
            agent=agent,
            run_number=run_number,
        )

        # ------------------------------------------------------
        # LLM statistics
        # ------------------------------------------------------

        llm_statistics = self._get_llm_statistics(agent)

        # ------------------------------------------------------
        # Build run result
        # ------------------------------------------------------

        result = {
            "agent": agent_name,
            "run": run_number,
            "training": training_summary,
            "evaluation": evaluation_summary,
            "llm": llm_statistics,
        }

        # ------------------------------------------------------
        # Print summary
        # ------------------------------------------------------

        if self.verbose:

            self._print_run_summary(result)

        return result

    # ==========================================================
    # Evaluation
    # ==========================================================

    def _evaluate_agent(
        self,
        agent_name: str,
        agent,
        run_number: int,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained agent using a fresh environment.
        """

        evaluation_env = self.environment_factory()

        evaluator = Evaluator(
            env=evaluation_env,
            agent=agent,
            episodes=self.evaluation_episodes,
            max_steps_per_episode=(self.max_steps_per_episode),
            verbose=self.verbose,
        )

        evaluation_summary = evaluator.evaluate()

        evaluation_path = os.path.join(
            self.results_directory,
            self._safe_name(agent_name),
            f"run_{run_number}_evaluation.json",
        )

        self._ensure_parent_directory(evaluation_path)

        self._save_json(
            evaluation_path,
            evaluation_summary,
        )

        return evaluation_summary

    # ==========================================================
    # LLM Statistics
    # ==========================================================

    @staticmethod
    def _get_llm_statistics(
        agent,
    ) -> Optional[Dict[str, Any]]:
        """
        Collect LLM-specific statistics when the agent exposes
        get_llm_statistics().

        Normal PPO agents simply return None.
        """

        if not hasattr(
            agent,
            "get_llm_statistics",
        ):

            return None

        return agent.get_llm_statistics()

    # ==========================================================
    # Build Summary
    # ==========================================================

    def _build_experiment_summary(
        self,
        experiment_time: float,
    ) -> Dict[str, Any]:
        """
        Build the final experiment result.
        """

        summary = {
            "experiment": (self.experiment_name),
            "configuration": {
                "training_episodes": (self.training_episodes),
                "evaluation_episodes": (self.evaluation_episodes),
                "runs": self.runs,
                "max_steps_per_episode": (self.max_steps_per_episode),
            },
            "experiment_time": (experiment_time),
            "runs": self.results,
            "agent_summary": (self._aggregate_agent_results()),
        }

        return summary

    # ==========================================================
    # Aggregate Results
    # ==========================================================

    def _aggregate_agent_results(
        self,
    ) -> Dict[str, Any]:
        """
        Aggregate results across independent runs for each agent.
        """

        grouped = {}

        for result in self.results:

            agent_name = result["agent"]

            if agent_name not in grouped:

                grouped[agent_name] = []

            grouped[agent_name].append(result)

        summary = {}

        for agent_name, runs in grouped.items():

            training_rewards = [run["training"]["average_reward"] for run in runs]

            training_success = [run["training"]["success_rate"] for run in runs]

            training_steps = [run["training"]["average_steps"] for run in runs]

            convergence = [
                run["training"]["convergence_episode"]
                for run in runs
                if run["training"]["convergence_episode"] is not None
            ]

            evaluation_rewards = [run["evaluation"]["average_reward"] for run in runs]

            evaluation_success = [run["evaluation"]["success_rate"] for run in runs]

            evaluation_steps = [run["evaluation"]["average_steps"] for run in runs]

            training_times = [run["training"]["training_time"] for run in runs]

            evaluation_times = [run["evaluation"]["evaluation_time"] for run in runs]

            summary[agent_name] = {
                "runs": len(runs),
                "training": {
                    "mean_average_reward": (self._mean(training_rewards)),
                    "mean_success_rate": (self._mean(training_success)),
                    "mean_average_steps": (self._mean(training_steps)),
                    "mean_convergence_episode": (self._mean(convergence)),
                    "mean_training_time": (self._mean(training_times)),
                },
                "evaluation": {
                    "mean_average_reward": (self._mean(evaluation_rewards)),
                    "mean_success_rate": (self._mean(evaluation_success)),
                    "mean_average_steps": (self._mean(evaluation_steps)),
                    "mean_evaluation_time": (self._mean(evaluation_times)),
                },
                "llm": (self._aggregate_llm_results(runs)),
            }

        return summary

    # ==========================================================
    # Aggregate LLM Results
    # ==========================================================

    @staticmethod
    def _aggregate_llm_results(
        runs: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Aggregate LLM metrics when available.
        """

        llm_results = [run["llm"] for run in runs if run["llm"] is not None]

        if not llm_results:

            return None

        return {
            "mean_llm_calls": (
                ExperimentRunner._mean([result["llm_calls"] for result in llm_results])
            ),
            "mean_llm_failures": (
                ExperimentRunner._mean(
                    [result["llm_failures"] for result in llm_results]
                )
            ),
            "mean_llm_success_rate": (
                ExperimentRunner._mean(
                    [result["llm_success_rate"] for result in llm_results]
                )
            ),
            "mean_llm_latency": (
                ExperimentRunner._mean(
                    [result["average_llm_latency"] for result in llm_results]
                )
            ),
            "mean_total_llm_latency": (
                ExperimentRunner._mean(
                    [result["total_llm_latency"] for result in llm_results]
                )
            ),
        }

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _mean(
        values: List[float],
    ) -> float:
        """
        Calculate the mean of a list.

        None values are ignored.
        """

        values = [value for value in values if value is not None]

        if not values:

            return 0.0

        return sum(values) / len(values)

    @staticmethod
    def _safe_name(
        name: str,
    ) -> str:
        """
        Convert an agent name into a filesystem-safe name.
        """

        return name.lower().replace(" ", "_").replace("+", "_plus_")

    @staticmethod
    def _ensure_parent_directory(
        path: str,
    ):
        """
        Create the parent directory for a file.
        """

        directory = os.path.dirname(path)

        if directory:

            os.makedirs(
                directory,
                exist_ok=True,
            )

    @staticmethod
    def _save_json(
        path: str,
        data: Dict[str, Any],
    ):
        """
        Save dictionary data as JSON.
        """

        ExperimentRunner._ensure_parent_directory(path)

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

    # ==========================================================
    # Print Summary
    # ==========================================================

    def _print_run_summary(
        self,
        result: Dict[str, Any],
    ):
        """
        Print a concise summary for one run.
        """

        training = result["training"]

        evaluation = result["evaluation"]

        print()
        print(f"--- {result['agent']} " f"Run {result['run']} ---")

        print(f"Training reward: " f"{training['average_reward']:.3f}")

        print(f"Training success: " f"{training['success_rate']:.3f}")

        print(f"Training steps: " f"{training['average_steps']:.3f}")

        print(f"Evaluation reward: " f"{evaluation['average_reward']:.3f}")

        print(f"Evaluation success: " f"{evaluation['success_rate']:.3f}")

        print(f"Evaluation steps: " f"{evaluation['average_steps']:.3f}")

        if result["llm"] is not None:

            llm = result["llm"]

            print(f"LLM calls: " f"{llm['llm_calls']}")

            print(f"Average LLM latency: " f"{llm['average_llm_latency']:.3f}s")

    # ==========================================================
    # Save Complete Results
    # ==========================================================

    def _save_results(
        self,
        summary: Dict[str, Any],
    ):
        """
        Save the complete experiment result.
        """

        filename = f"{self._safe_name(self.experiment_name)}" "_results.json"

        path = os.path.join(
            self.results_directory,
            filename,
        )

        self._save_json(
            path,
            summary,
        )
