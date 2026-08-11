"""
visualization.py

Visualization utilities for the finance RL agent project.

This module generates plots for comparing:

    - PPO
    - LLM + PPO

Main visualizations:

    - reward curves
    - success-rate curves
    - episode-step curves
    - convergence comparison
    - training-time comparison
    - LLM latency
    - agent performance comparison

Plots are saved under:

    results/graphs/
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

# ==============================================================
# Constants
# ==============================================================

DEFAULT_GRAPH_DIRECTORY = "results/graphs"


# ==============================================================
# Directory Helper
# ==============================================================


def _ensure_directory(
    directory: str,
) -> None:
    """
    Create the output directory if it does not exist.
    """

    os.makedirs(
        directory,
        exist_ok=True,
    )


def _save_figure(
    figure,
    filename: str,
    output_directory: str,
) -> str:
    """
    Save a matplotlib figure and close it.
    """

    _ensure_directory(output_directory)

    path = os.path.join(
        output_directory,
        filename,
    )

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return path


# ==============================================================
# Reward Curve
# ==============================================================


def plot_reward_curve(
    rewards: List[float],
    agent_name: str,
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: Optional[str] = None,
    moving_average_window: Optional[int] = None,
) -> str:
    """
    Plot reward against training episode.

    Parameters
    ----------
    rewards:
        Reward obtained for every episode.

    agent_name:
        Name of the agent.

    moving_average_window:
        Optional moving-average window.

    Returns
    -------
    str
        Path of the saved graph.
    """

    if filename is None:

        filename = f"{_safe_filename(agent_name)}" "_reward_curve.png"

    episodes = list(
        range(
            1,
            len(rewards) + 1,
        )
    )

    figure = plt.figure(figsize=(10, 6))

    plt.plot(
        episodes,
        rewards,
        label="Episode Reward",
    )

    if (
        moving_average_window is not None
        and moving_average_window > 1
        and len(rewards) >= moving_average_window
    ):

        moving_average = []

        for index in range(
            moving_average_window,
            len(rewards) + 1,
        ):

            window = rewards[index - moving_average_window : index]

            moving_average.append(sum(window) / len(window))

        moving_episodes = list(
            range(
                moving_average_window,
                len(rewards) + 1,
            )
        )

        plt.plot(
            moving_episodes,
            moving_average,
            label=(f"Moving Average " f"({moving_average_window})"),
        )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title(f"{agent_name} - Training Reward")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Success Rate Curve
# ==============================================================


def plot_success_rate_curve(
    success_flags: List[bool],
    agent_name: str,
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: Optional[str] = None,
    moving_average_window: int = 50,
) -> str:
    """
    Plot rolling success rate against episode.
    """

    if filename is None:

        filename = f"{_safe_filename(agent_name)}" "_success_rate.png"

    if not success_flags:

        raise ValueError("success_flags cannot be empty.")

    success_values = [1 if value else 0 for value in success_flags]

    episodes = list(
        range(
            1,
            len(success_values) + 1,
        )
    )

    figure = plt.figure(figsize=(10, 6))

    if moving_average_window > 1 and len(success_values) >= moving_average_window:

        rolling_rates = []

        for index in range(
            moving_average_window,
            len(success_values) + 1,
        ):

            window = success_values[index - moving_average_window : index]

            rolling_rates.append(sum(window) / len(window))

        rolling_episodes = list(
            range(
                moving_average_window,
                len(success_values) + 1,
            )
        )

        plt.plot(
            rolling_episodes,
            rolling_rates,
            label=(f"Success Rate " f"({moving_average_window}-episode)"),
        )

    else:

        plt.plot(
            episodes,
            success_values,
            label="Success",
        )

    plt.xlabel("Episode")

    plt.ylabel("Success Rate")

    plt.title(f"{agent_name} - Success Rate")

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Steps Curve
# ==============================================================


def plot_steps_curve(
    steps: List[int],
    agent_name: str,
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: Optional[str] = None,
    moving_average_window: Optional[int] = None,
) -> str:
    """
    Plot episode steps against training episode.
    """

    if filename is None:

        filename = f"{_safe_filename(agent_name)}" "_steps_curve.png"

    episodes = list(
        range(
            1,
            len(steps) + 1,
        )
    )

    figure = plt.figure(figsize=(10, 6))

    plt.plot(
        episodes,
        steps,
        label="Episode Steps",
    )

    if (
        moving_average_window is not None
        and moving_average_window > 1
        and len(steps) >= moving_average_window
    ):

        moving_average = []

        for index in range(
            moving_average_window,
            len(steps) + 1,
        ):

            window = steps[index - moving_average_window : index]

            moving_average.append(sum(window) / len(window))

        moving_episodes = list(
            range(
                moving_average_window,
                len(steps) + 1,
            )
        )

        plt.plot(
            moving_episodes,
            moving_average,
            label=(f"Moving Average " f"({moving_average_window})"),
        )

    plt.xlabel("Episode")

    plt.ylabel("Steps")

    plt.title(f"{agent_name} - Steps per Episode")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Reward Comparison
# ==============================================================


def plot_reward_comparison(
    ppo_rewards: List[float],
    llm_ppo_rewards: List[float],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_reward.png",
) -> str:
    """
    Compare PPO and LLM + PPO reward curves.
    """

    figure = plt.figure(figsize=(10, 6))

    ppo_episodes = list(
        range(
            1,
            len(ppo_rewards) + 1,
        )
    )

    llm_episodes = list(
        range(
            1,
            len(llm_ppo_rewards) + 1,
        )
    )

    plt.plot(
        ppo_episodes,
        ppo_rewards,
        label="PPO",
    )

    plt.plot(
        llm_episodes,
        llm_ppo_rewards,
        label="LLM + PPO",
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("PPO vs LLM + PPO - Reward")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Success Rate Comparison
# ==============================================================


def plot_success_rate_comparison(
    ppo_success: List[bool],
    llm_ppo_success: List[bool],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_success_rate.png",
    moving_average_window: int = 50,
) -> str:
    """
    Compare success-rate curves for PPO and LLM + PPO.
    """

    figure = plt.figure(figsize=(10, 6))

    ppo_values = [1 if value else 0 for value in ppo_success]

    llm_values = [1 if value else 0 for value in llm_ppo_success]

    if moving_average_window > 1 and len(ppo_values) >= moving_average_window:

        ppo_rates = _moving_average(
            ppo_values,
            moving_average_window,
        )

        ppo_episodes = list(
            range(
                moving_average_window,
                len(ppo_values) + 1,
            )
        )

        plt.plot(
            ppo_episodes,
            ppo_rates,
            label="PPO",
        )

    else:

        plt.plot(
            range(
                1,
                len(ppo_values) + 1,
            ),
            ppo_values,
            label="PPO",
        )

    if moving_average_window > 1 and len(llm_values) >= moving_average_window:

        llm_rates = _moving_average(
            llm_values,
            moving_average_window,
        )

        llm_episodes = list(
            range(
                moving_average_window,
                len(llm_values) + 1,
            )
        )

        plt.plot(
            llm_episodes,
            llm_rates,
            label="LLM + PPO",
        )

    else:

        plt.plot(
            range(
                1,
                len(llm_values) + 1,
            ),
            llm_values,
            label="LLM + PPO",
        )

    plt.xlabel("Episode")

    plt.ylabel("Success Rate")

    plt.title("PPO vs LLM + PPO - Success Rate")

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Steps Comparison
# ==============================================================


def plot_steps_comparison(
    ppo_steps: List[int],
    llm_ppo_steps: List[int],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_steps.png",
) -> str:
    """
    Compare episode steps between PPO and LLM + PPO.
    """

    figure = plt.figure(figsize=(10, 6))

    plt.plot(
        range(
            1,
            len(ppo_steps) + 1,
        ),
        ppo_steps,
        label="PPO",
    )

    plt.plot(
        range(
            1,
            len(llm_ppo_steps) + 1,
        ),
        llm_ppo_steps,
        label="LLM + PPO",
    )

    plt.xlabel("Episode")

    plt.ylabel("Steps")

    plt.title("PPO vs LLM + PPO - Steps per Episode")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Convergence Comparison
# ==============================================================


def plot_convergence_comparison(
    ppo_convergence: Optional[int],
    llm_ppo_convergence: Optional[int],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_convergence.png",
) -> str:
    """
    Compare the episode at which PPO and LLM + PPO converge.
    """

    figure = plt.figure(figsize=(8, 6))

    labels = [
        "PPO",
        "LLM + PPO",
    ]

    values = [
        (ppo_convergence if ppo_convergence is not None else 0),
        (llm_ppo_convergence if llm_ppo_convergence is not None else 0),
    ]

    plt.bar(
        labels,
        values,
    )

    plt.ylabel("Episode")

    plt.title("Convergence Episode Comparison")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Training Time Comparison
# ==============================================================


def plot_training_time_comparison(
    ppo_training_time: float,
    llm_ppo_training_time: float,
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_training_time.png",
) -> str:
    """
    Compare wall-clock training time.
    """

    figure = plt.figure(figsize=(8, 6))

    labels = [
        "PPO",
        "LLM + PPO",
    ]

    values = [
        ppo_training_time,
        llm_ppo_training_time,
    ]

    plt.bar(
        labels,
        values,
    )

    plt.ylabel("Training Time (seconds)")

    plt.title("Training Time Comparison")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# LLM Latency
# ==============================================================


def plot_llm_latency(
    latencies: List[float],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "llm_latency.png",
) -> str:
    """
    Plot LLM response latency for each LLM call.
    """

    figure = plt.figure(figsize=(10, 6))

    calls = list(
        range(
            1,
            len(latencies) + 1,
        )
    )

    plt.plot(
        calls,
        latencies,
        label="LLM Latency",
    )

    plt.xlabel("LLM Call")

    plt.ylabel("Latency (seconds)")

    plt.title("LLM Response Latency")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Performance Summary
# ==============================================================


def plot_performance_summary(
    ppo_metrics: Dict[str, float],
    llm_ppo_metrics: Dict[str, float],
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
    filename: str = "ppo_vs_llm_ppo_summary.png",
) -> str:
    """
    Create a summary comparison of key evaluation metrics.

    Expected metrics:

        {
            "average_reward": ...,
            "success_rate": ...,
            "average_steps": ...
        }

    Note:
        Steps are inverted for display so that larger values
        represent better performance in the comparison.
    """

    metric_names = [
        "Average Reward",
        "Success Rate",
        "Average Steps",
    ]

    ppo_reward = ppo_metrics.get(
        "average_reward",
        0.0,
    )

    llm_reward = llm_ppo_metrics.get(
        "average_reward",
        0.0,
    )

    ppo_success = ppo_metrics.get(
        "success_rate",
        0.0,
    )

    llm_success = llm_ppo_metrics.get(
        "success_rate",
        0.0,
    )

    ppo_steps = ppo_metrics.get(
        "average_steps",
        0.0,
    )

    llm_steps = llm_ppo_metrics.get(
        "average_steps",
        0.0,
    )

    # ----------------------------------------------------------
    # Normalize values
    # ----------------------------------------------------------

    ppo_values = [
        ppo_reward,
        ppo_success,
        _normalize_lower_is_better(
            ppo_steps,
            ppo_steps,
            llm_steps,
        ),
    ]

    llm_values = [
        llm_reward,
        llm_success,
        _normalize_lower_is_better(
            llm_steps,
            ppo_steps,
            llm_steps,
        ),
    ]

    figure = plt.figure(figsize=(10, 6))

    positions = list(range(len(metric_names)))

    width = 0.35

    ppo_positions = [position - width / 2 for position in positions]

    llm_positions = [position + width / 2 for position in positions]

    plt.bar(
        ppo_positions,
        ppo_values,
        width,
        label="PPO",
    )

    plt.bar(
        llm_positions,
        llm_values,
        width,
        label="LLM + PPO",
    )

    plt.xticks(
        positions,
        metric_names,
    )

    plt.ylabel("Metric Value")

    plt.title("PPO vs LLM + PPO - Performance Summary")

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    return _save_figure(
        figure,
        filename,
        output_directory,
    )


# ==============================================================
# Complete Experiment Visualization
# ==============================================================


def generate_experiment_plots(
    ppo_results: Dict,
    llm_ppo_results: Dict,
    output_directory: str = DEFAULT_GRAPH_DIRECTORY,
) -> Dict[str, str]:
    """
    Generate all major plots for a PPO vs LLM + PPO experiment.

    Expected structure:

        ppo_results = {
            "rewards": [...],
            "success": [...],
            "steps": [...],
            "training_time": ...,
            "convergence_episode": ...
        }

        llm_ppo_results = {
            "rewards": [...],
            "success": [...],
            "steps": [...],
            "training_time": ...,
            "convergence_episode": ...
        }

    Returns
    -------
    dict
        Mapping of graph name to saved file path.
    """

    paths = {}

    paths["reward"] = plot_reward_comparison(
        ppo_results.get(
            "rewards",
            [],
        ),
        llm_ppo_results.get(
            "rewards",
            [],
        ),
        output_directory,
    )

    paths["success_rate"] = plot_success_rate_comparison(
        ppo_results.get(
            "success",
            [],
        ),
        llm_ppo_results.get(
            "success",
            [],
        ),
        output_directory,
    )

    paths["steps"] = plot_steps_comparison(
        ppo_results.get(
            "steps",
            [],
        ),
        llm_ppo_results.get(
            "steps",
            [],
        ),
        output_directory,
    )

    paths["convergence"] = plot_convergence_comparison(
        ppo_results.get("convergence_episode"),
        llm_ppo_results.get("convergence_episode"),
        output_directory,
    )

    paths["training_time"] = plot_training_time_comparison(
        ppo_results.get(
            "training_time",
            0.0,
        ),
        llm_ppo_results.get(
            "training_time",
            0.0,
        ),
        output_directory,
    )

    return paths


# ==============================================================
# Internal Helpers
# ==============================================================


def _moving_average(
    values: List[float],
    window_size: int,
) -> List[float]:
    """
    Calculate a simple moving average.
    """

    if window_size <= 0:

        raise ValueError("window_size must be greater than zero.")

    if len(values) < window_size:

        return []

    averages = []

    for index in range(
        window_size,
        len(values) + 1,
    ):

        window = values[index - window_size : index]

        averages.append(sum(window) / len(window))

    return averages


def _normalize_lower_is_better(
    value: float,
    ppo_value: float,
    llm_value: float,
) -> float:
    """
    Normalize a lower-is-better metric.

    Used only for visualization.

    The best value becomes 1 and the other value is
    scaled relative to it.
    """

    minimum_value = min(
        ppo_value,
        llm_value,
    )

    if value <= 0:

        return 0.0

    if minimum_value <= 0:

        return 0.0

    return minimum_value / value


def _safe_filename(
    name: str,
) -> str:
    """
    Convert an agent name into a safe filename.
    """

    return (
        name.lower()
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("/", "_")
        .replace("\\", "_")
    )
