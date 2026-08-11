"""
metrics.py

Metrics utilities for the finance RL agent project.

This module calculates and aggregates metrics used to evaluate
and compare:

    - PPO
    - LLM + PPO

Main research metrics include:

    - episode reward
    - success rate
    - average steps
    - convergence episode
    - training time
    - evaluation time
    - LLM calls
    - LLM latency
    - LLM overhead
"""

from typing import Any, Dict, List, Optional

# ==============================================================
# Basic Statistical Functions
# ==============================================================


def mean(
    values: List[float],
) -> float:
    """
    Calculate the arithmetic mean.

    None values are ignored.
    """

    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return sum(valid_values) / len(valid_values)


def median(
    values: List[float],
) -> float:
    """
    Calculate the median.

    None values are ignored.
    """

    valid_values = sorted(value for value in values if value is not None)

    if not valid_values:
        return 0.0

    count = len(valid_values)

    middle = count // 2

    if count % 2 == 0:

        return (valid_values[middle - 1] + valid_values[middle]) / 2

    return valid_values[middle]


def minimum(
    values: List[float],
) -> float:
    """
    Return the minimum value.
    """

    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return min(valid_values)


def maximum(
    values: List[float],
) -> float:
    """
    Return the maximum value.
    """

    valid_values = [value for value in values if value is not None]

    if not valid_values:
        return 0.0

    return max(valid_values)


# ==============================================================
# Episode Metrics
# ==============================================================


def calculate_episode_metrics(
    episode_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate aggregate metrics from episode results.

    Expected episode result format:

        {
            "reward": 10,
            "steps": 5,
            "completed": True,
            "terminated_reason": "GOAL_ACHIEVED"
        }
    """

    if not episode_results:

        return {
            "episodes": 0,
            "average_reward": 0.0,
            "median_reward": 0.0,
            "best_reward": 0.0,
            "worst_reward": 0.0,
            "average_steps": 0.0,
            "median_steps": 0.0,
            "success_rate": 0.0,
            "completed_episodes": 0,
        }

    rewards = [
        episode.get(
            "reward",
            0.0,
        )
        for episode in episode_results
    ]

    steps = [
        episode.get(
            "steps",
            0,
        )
        for episode in episode_results
    ]

    completed = [
        bool(
            episode.get(
                "completed",
                False,
            )
        )
        for episode in episode_results
    ]

    completed_count = sum(completed)

    episode_count = len(episode_results)

    return {
        "episodes": episode_count,
        "average_reward": mean(rewards),
        "median_reward": median(rewards),
        "best_reward": maximum(rewards),
        "worst_reward": minimum(rewards),
        "average_steps": mean(steps),
        "median_steps": median(steps),
        "success_rate": (completed_count / episode_count),
        "completed_episodes": (completed_count),
    }


# ==============================================================
# Success Rate
# ==============================================================


def calculate_success_rate(
    completed_episodes: int,
    total_episodes: int,
) -> float:
    """
    Calculate episode success rate.
    """

    if total_episodes <= 0:

        return 0.0

    return completed_episodes / total_episodes


# ==============================================================
# Convergence
# ==============================================================


def calculate_convergence_episode(
    rewards: List[float],
    success_flags: Optional[List[bool]] = None,
    window_size: int = 50,
    reward_threshold: Optional[float] = None,
    success_threshold: float = 1.0,
) -> Optional[int]:
    """
    Estimate the episode at which the agent converges.

    Convergence is detected when a moving window satisfies:

        - reward threshold, if supplied
        - success-rate threshold, if supplied

    If success_flags are not supplied, convergence is based
    only on reward.

    Parameters
    ----------
    rewards :
        Reward obtained at each episode.

    success_flags :
        Whether each episode completed successfully.

    window_size :
        Number of episodes in the moving window.

    reward_threshold :
        Required average reward.

    success_threshold :
        Required moving-window success rate.

    Returns
    -------
    int or None
        First episode satisfying the convergence criteria.
    """

    if not rewards:

        return None

    if window_size <= 0:

        raise ValueError("window_size must be greater than zero.")

    if success_flags is not None and len(success_flags) != len(rewards):

        raise ValueError("rewards and success_flags " "must have the same length.")

    if len(rewards) < window_size:

        return None

    for index in range(
        window_size,
        len(rewards) + 1,
    ):

        reward_window = rewards[index - window_size : index]

        average_reward = mean(reward_window)

        # ------------------------------------------------------
        # Reward criterion
        # ------------------------------------------------------

        if reward_threshold is not None and average_reward < reward_threshold:

            continue

        # ------------------------------------------------------
        # Success criterion
        # ------------------------------------------------------

        if success_flags is not None:

            success_window = success_flags[index - window_size : index]

            success_rate = sum(success_window) / window_size

            if success_rate < success_threshold:

                continue

        return index

    return None


# ==============================================================
# Training Metrics
# ==============================================================


def calculate_training_metrics(
    episode_results: List[Dict[str, Any]],
    training_time: float,
    convergence_window: int = 50,
    reward_threshold: Optional[float] = None,
    success_threshold: float = 1.0,
) -> Dict[str, Any]:
    """
    Calculate complete training metrics.

    This is intended to be used by the training pipeline
    after a training run has completed.
    """

    base_metrics = calculate_episode_metrics(episode_results)

    rewards = [
        episode.get(
            "reward",
            0.0,
        )
        for episode in episode_results
    ]

    success_flags = [
        bool(
            episode.get(
                "completed",
                False,
            )
        )
        for episode in episode_results
    ]

    convergence_episode = calculate_convergence_episode(
        rewards=rewards,
        success_flags=success_flags,
        window_size=convergence_window,
        reward_threshold=reward_threshold,
        success_threshold=success_threshold,
    )

    base_metrics["convergence_episode"] = convergence_episode

    base_metrics["training_time"] = training_time

    return base_metrics


# ==============================================================
# Evaluation Metrics
# ==============================================================


def calculate_evaluation_metrics(
    episode_results: List[Dict[str, Any]],
    evaluation_time: float,
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics.

    Evaluation does not modify the agent.
    """

    metrics = calculate_episode_metrics(episode_results)

    metrics["evaluation_time"] = evaluation_time

    return metrics


# ==============================================================
# LLM Metrics
# ==============================================================


def calculate_llm_metrics(
    llm_calls: int,
    llm_failures: int,
    total_llm_latency: float,
) -> Dict[str, Any]:
    """
    Calculate LLM-related metrics.

    These metrics are particularly important for comparing
    PPO against LLM + PPO.
    """

    if llm_calls > 0:

        average_latency = total_llm_latency / llm_calls

        llm_success_rate = (llm_calls - llm_failures) / llm_calls

    else:

        average_latency = 0.0

        llm_success_rate = 0.0

    return {
        "llm_calls": llm_calls,
        "llm_failures": (llm_failures),
        "llm_success_rate": (llm_success_rate),
        "average_llm_latency": (average_latency),
        "total_llm_latency": (total_llm_latency),
    }


# ==============================================================
# Training Efficiency
# ==============================================================


def calculate_training_efficiency(
    convergence_episode: Optional[int],
    training_time: float,
) -> Dict[str, Any]:
    """
    Calculate training efficiency.

    Lower convergence episode generally indicates faster
    learning.

    Lower training time indicates lower computational cost.
    """

    if convergence_episode is not None and convergence_episode > 0:

        episodes_per_second = (
            convergence_episode / training_time if training_time > 0 else 0.0
        )

    else:

        episodes_per_second = 0.0

    return {
        "convergence_episode": (convergence_episode),
        "training_time": (training_time),
        "episodes_to_convergence": (convergence_episode),
        "episodes_per_second": (episodes_per_second),
    }


# ==============================================================
# LLM Overhead
# ==============================================================


def calculate_llm_overhead(
    ppo_training_time: float,
    llm_ppo_training_time: float,
    llm_latency: float,
) -> Dict[str, Any]:
    """
    Calculate the additional cost introduced by LLM guidance.

    This allows the experiment to distinguish:

        learning improvement

    from:

        additional computational/latency cost.
    """

    training_time_difference = llm_ppo_training_time - ppo_training_time

    if ppo_training_time > 0:

        percentage_overhead = (training_time_difference / ppo_training_time) * 100.0

    else:

        percentage_overhead = 0.0

    return {
        "ppo_training_time": (ppo_training_time),
        "llm_ppo_training_time": (llm_ppo_training_time),
        "llm_latency": (llm_latency),
        "additional_training_time": (training_time_difference),
        "training_time_overhead_percent": (percentage_overhead),
    }


# ==============================================================
# Agent Comparison
# ==============================================================


def compare_agents(
    ppo_metrics: Dict[str, Any],
    llm_ppo_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare PPO against LLM + PPO.

    Positive improvement means the LLM + PPO result is better
    for that metric.

    For average steps and convergence episode, lower is better.

    For reward and success rate, higher is better.
    """

    ppo_training = ppo_metrics.get("training", {})

    llm_training = llm_ppo_metrics.get("training", {})

    ppo_evaluation = ppo_metrics.get("evaluation", {})

    llm_evaluation = llm_ppo_metrics.get("evaluation", {})

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    training_reward_improvement = _percentage_change(
        ppo_training.get(
            "average_reward",
            0.0,
        ),
        llm_training.get(
            "average_reward",
            0.0,
        ),
    )

    training_success_improvement = _percentage_change(
        ppo_training.get(
            "success_rate",
            0.0,
        ),
        llm_training.get(
            "success_rate",
            0.0,
        ),
    )

    training_steps_improvement = _percentage_reduction(
        ppo_training.get(
            "average_steps",
            0.0,
        ),
        llm_training.get(
            "average_steps",
            0.0,
        ),
    )

    # ----------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------

    evaluation_reward_improvement = _percentage_change(
        ppo_evaluation.get(
            "average_reward",
            0.0,
        ),
        llm_evaluation.get(
            "average_reward",
            0.0,
        ),
    )

    evaluation_success_improvement = _percentage_change(
        ppo_evaluation.get(
            "success_rate",
            0.0,
        ),
        llm_evaluation.get(
            "success_rate",
            0.0,
        ),
    )

    evaluation_steps_improvement = _percentage_reduction(
        ppo_evaluation.get(
            "average_steps",
            0.0,
        ),
        llm_evaluation.get(
            "average_steps",
            0.0,
        ),
    )

    # ----------------------------------------------------------
    # Convergence
    # ----------------------------------------------------------

    ppo_convergence = ppo_training.get("convergence_episode")

    llm_convergence = llm_training.get("convergence_episode")

    convergence_improvement = _percentage_reduction(
        ppo_convergence,
        llm_convergence,
    )

    # ----------------------------------------------------------
    # Training time
    # ----------------------------------------------------------

    training_time_difference = llm_training.get(
        "training_time",
        0.0,
    ) - ppo_training.get(
        "training_time",
        0.0,
    )

    training_time_overhead = _percentage_change(
        ppo_training.get(
            "training_time",
            0.0,
        ),
        llm_training.get(
            "training_time",
            0.0,
        ),
    )

    return {
        "training": {
            "reward_improvement_percent": (training_reward_improvement),
            "success_rate_improvement_percent": (training_success_improvement),
            "steps_reduction_percent": (training_steps_improvement),
            "convergence_improvement_percent": (convergence_improvement),
        },
        "evaluation": {
            "reward_improvement_percent": (evaluation_reward_improvement),
            "success_rate_improvement_percent": (evaluation_success_improvement),
            "steps_reduction_percent": (evaluation_steps_improvement),
        },
        "cost": {
            "additional_training_time": (training_time_difference),
            "training_time_overhead_percent": (training_time_overhead),
        },
    }


# ==============================================================
# Percentage Helpers
# ==============================================================


def _percentage_change(
    baseline: Optional[float],
    new_value: Optional[float],
) -> float:
    """
    Calculate percentage change:

        ((new - baseline) / baseline) * 100

    Used when higher values are generally better.
    """

    if baseline is None:
        return 0.0

    if baseline == 0:

        return 0.0

    if new_value is None:

        return 0.0

    return ((new_value - baseline) / abs(baseline)) * 100.0


def _percentage_reduction(
    baseline: Optional[float],
    new_value: Optional[float],
) -> float:
    """
    Calculate percentage reduction:

        ((baseline - new) / baseline) * 100

    Used when lower values are better.
    """

    if baseline is None:
        return 0.0

    if baseline == 0:

        return 0.0

    if new_value is None:

        return 0.0

    return ((baseline - new_value) / abs(baseline)) * 100.0


# ==============================================================
# Run-Level Aggregation
# ==============================================================


def aggregate_runs(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple independent runs.

    Each run is expected to contain:

        {
            "training": {...},
            "evaluation": {...}
        }
    """

    if not runs:

        return {}

    training = [run.get("training", {}) for run in runs]

    evaluation = [run.get("evaluation", {}) for run in runs]

    return {
        "runs": len(runs),
        "training": {
            "average_reward": mean([item.get("average_reward") for item in training]),
            "success_rate": mean([item.get("success_rate") for item in training]),
            "average_steps": mean([item.get("average_steps") for item in training]),
            "convergence_episode": mean(
                [item.get("convergence_episode") for item in training]
            ),
            "training_time": mean([item.get("training_time") for item in training]),
        },
        "evaluation": {
            "average_reward": mean([item.get("average_reward") for item in evaluation]),
            "success_rate": mean([item.get("success_rate") for item in evaluation]),
            "average_steps": mean([item.get("average_steps") for item in evaluation]),
            "evaluation_time": mean(
                [item.get("evaluation_time") for item in evaluation]
            ),
        },
    }


"""
| Metric                            | Interpretation                                   |
| --------------------------------- | ------------------------------------------------ |
| `success_rate`                    | Does the agent achieve the goal?                 |
| `average_reward`                  | How good is the behaviour?                       |
| `average_steps`                   | How efficiently is the goal reached?             |
| `convergence_episode`             | How quickly does learning stabilize?             |
| `training_time`                   | How expensive is training?                       |
| `llm_latency`                     | What cost does LLM guidance introduce?           |
| `steps_reduction_percent`         | Does LLM guidance make execution more efficient? |
| `convergence_improvement_percent` | Does LLM guidance speed up learning?             |
| `training_time_overhead_percent`  | How much extra cost does the LLM introduce?      |

"""
