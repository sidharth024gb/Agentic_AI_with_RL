"""
visualization.py

Visualizations for PPO training/evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.config import config

# ==========================================================
# Save Helper
# ==========================================================


def _save(
    figure,
    output_directory,
    filename,
):

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_directory / filename

    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return str(path)


# ==========================================================
# Reward Curve
# ==========================================================


def plot_reward_curve(
    training_df,
    output_directory,
    window,
):

    dataframe = training_df.copy()

    dataframe["movingReward"] = (
        dataframe["totalReward"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe["episodeNumber"],
        dataframe["totalReward"],
        label="Episode reward",
        alpha=0.45,
    )

    plt.plot(
        dataframe["episodeNumber"],
        dataframe["movingReward"],
        label=f"{window}-episode moving average",
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title("PPO Training Reward")

    plt.legend()

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_reward_curve.png",
    )


# ==========================================================
# Success Curve
# ==========================================================


def plot_success_curve(
    training_df,
    output_directory,
    window,
):

    values = (
        training_df["completed"]
        .astype(float)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        training_df["episodeNumber"],
        values,
    )

    plt.xlabel("Episode")

    plt.ylabel("Rolling success rate")

    plt.ylim(
        0,
        1.05,
    )

    plt.title("PPO Training Success Rate")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_success_rate.png",
    )


# ==========================================================
# Steps Curve
# ==========================================================


def plot_steps_curve(
    training_df,
    output_directory,
    window,
):

    dataframe = training_df.copy()

    dataframe["movingSteps"] = (
        dataframe["totalSteps"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe["episodeNumber"],
        dataframe["totalSteps"],
        label="Episode steps",
        alpha=0.45,
    )

    plt.plot(
        dataframe["episodeNumber"],
        dataframe["movingSteps"],
        label=f"{window}-episode moving average",
    )

    plt.xlabel("Episode")

    plt.ylabel("Steps")

    plt.title("PPO Steps per Episode")

    plt.legend()

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_steps_curve.png",
    )


# ==========================================================
# PPO Loss
# ==========================================================


def plot_policy_loss(
    updates_df,
    output_directory,
):

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        updates_df["update"],
        updates_df["policy_loss"],
    )

    plt.xlabel("PPO Update")

    plt.ylabel("Policy loss")

    plt.title("PPO Policy Loss")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_policy_loss.png",
    )


def plot_value_loss(
    updates_df,
    output_directory,
):

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        updates_df["update"],
        updates_df["value_loss"],
    )

    plt.xlabel("PPO Update")

    plt.ylabel("Value loss")

    plt.title("PPO Value Loss")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_value_loss.png",
    )


def plot_entropy(
    updates_df,
    output_directory,
):

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        updates_df["update"],
        updates_df["entropy"],
    )

    plt.xlabel("PPO Update")

    plt.ylabel("Policy entropy")

    plt.title("PPO Policy Entropy")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_entropy.png",
    )


# ==========================================================
# Action Frequency
# ==========================================================


def plot_action_frequency(
    steps_df,
    output_directory,
):

    training_steps = steps_df[steps_df["phase"] == "TRAIN"]

    counts = training_steps["action"].value_counts().sort_values()

    figure = plt.figure(figsize=(11, 7))

    plt.barh(
        counts.index,
        counts.values,
    )

    plt.xlabel("Action count")

    plt.ylabel("Action")

    plt.title("PPO Training Action Frequency")

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    return _save(
        figure,
        output_directory,
        "ppo_action_frequency.png",
    )


# ==========================================================
# Termination Reasons
# ==========================================================


def plot_termination_reasons(
    training_df,
    output_directory,
):

    counts = training_df["terminatedReason"].fillna("UNKNOWN").value_counts()

    figure = plt.figure(figsize=(9, 6))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.xlabel("Termination reason")

    plt.ylabel("Episodes")

    plt.title("PPO Training Episode Termination")

    plt.xticks(
        rotation=30,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    return _save(
        figure,
        output_directory,
        "ppo_termination_reasons.png",
    )


# ==========================================================
# Evaluation Reward
# ==========================================================


def plot_evaluation_rewards(
    evaluation_df,
    output_directory,
):

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        range(
            1,
            len(evaluation_df) + 1,
        ),
        evaluation_df["totalReward"],
    )

    plt.xlabel("Evaluation episode")

    plt.ylabel("Reward")

    plt.title("PPO Deterministic Evaluation Reward")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        "ppo_evaluation_reward.png",
    )


# ==========================================================
# Generate Everything
# ==========================================================


def generate_ppo_visualizations(
    tables,
    output_directory,
):

    paths = {}

    training_df = tables["training_episodes"]

    evaluation_df = tables["evaluation_episodes"]

    updates_df = tables["ppo_updates"]

    steps_df = tables["steps"]

    window = config.training.MOVING_AVERAGE_WINDOW

    if not training_df.empty:

        paths["reward"] = plot_reward_curve(
            training_df,
            output_directory,
            window,
        )

        paths["success"] = plot_success_curve(
            training_df,
            output_directory,
            window,
        )

        paths["steps"] = plot_steps_curve(
            training_df,
            output_directory,
            window,
        )

        paths["termination"] = plot_termination_reasons(
            training_df,
            output_directory,
        )

    if not updates_df.empty:

        paths["policy_loss"] = plot_policy_loss(
            updates_df,
            output_directory,
        )

        paths["value_loss"] = plot_value_loss(
            updates_df,
            output_directory,
        )

        paths["entropy"] = plot_entropy(
            updates_df,
            output_directory,
        )

    if not steps_df.empty:

        paths["actions"] = plot_action_frequency(
            steps_df,
            output_directory,
        )

    if not evaluation_df.empty:

        paths["evaluation_reward"] = plot_evaluation_rewards(
            evaluation_df,
            output_directory,
        )

    return paths
