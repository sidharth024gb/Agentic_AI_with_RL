"""
visualization.py

Visualizations shared by:

    - PPO
    - LLM + PPO
"""

from pathlib import Path

import matplotlib.pyplot as plt

from config.config import config

# ==========================================================
# Helpers
# ==========================================================


def _safe_name(
    value,
):

    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace("+", "_plus_")
        .replace("/", "_")
        .replace("\\", "_")
    )


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
# Reward
# ==========================================================


def plot_reward_curve(
    training_df,
    output_directory,
    window,
    agent_name,
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
        label=(f"{window}-episode " "moving average"),
    )

    plt.xlabel("Episode")

    plt.ylabel("Reward")

    plt.title(f"{agent_name} Training Reward")

    plt.legend()

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_reward_curve.png"),
    )


# ==========================================================
# Success
# ==========================================================


def plot_success_curve(
    training_df,
    output_directory,
    window,
    agent_name,
):

    rolling = (
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
        rolling,
    )

    plt.xlabel("Episode")

    plt.ylabel("Rolling success rate")

    plt.ylim(
        0,
        1.05,
    )

    plt.title(f"{agent_name} Training Success Rate")

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_success_rate.png"),
    )


# ==========================================================
# Steps
# ==========================================================


def plot_steps_curve(
    training_df,
    output_directory,
    window,
    agent_name,
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
        label=(f"{window}-episode " "moving average"),
    )

    plt.xlabel("Episode")

    plt.ylabel("Steps")

    plt.title(f"{agent_name} Steps per Episode")

    plt.legend()

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_steps_curve.png"),
    )


# ==========================================================
# PPO Updates
# ==========================================================


def _plot_update_metric(
    updates_df,
    column,
    ylabel,
    title,
    filename,
    output_directory,
):

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        updates_df["update"],
        updates_df[column],
    )

    plt.xlabel("PPO Update")

    plt.ylabel(ylabel)

    plt.title(title)

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        filename,
    )


# ==========================================================
# Action Frequency
# ==========================================================


def plot_action_frequency(
    steps_df,
    output_directory,
    agent_name,
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

    plt.title(f"{agent_name} Training Action Frequency")

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_action_frequency.png"),
    )


# ==========================================================
# Termination
# ==========================================================


def plot_termination_reasons(
    training_df,
    output_directory,
    agent_name,
):

    counts = training_df["terminatedReason"].fillna("UNKNOWN").value_counts()

    figure = plt.figure(figsize=(9, 6))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.xlabel("Termination reason")

    plt.ylabel("Episodes")

    plt.title(f"{agent_name} Episode Termination")

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
        (f"{_safe_name(agent_name)}" "_termination.png"),
    )


# ==========================================================
# Evaluation
# ==========================================================


def plot_evaluation_rewards(
    evaluation_df,
    output_directory,
    agent_name,
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

    plt.title((f"{agent_name} " "Deterministic Evaluation"))

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_evaluation_reward.png"),
    )


# ==========================================================
# LLM Procedure Adherence
# ==========================================================


def plot_procedure_adherence(
    steps_df,
    output_directory,
    agent_name,
):

    guided = steps_df[
        (steps_df["phase"] == "TRAIN") & (steps_df["procedureFollowed"].notna())
    ].copy()

    if guided.empty:

        return None

    guided["procedureFollowed"] = guided["procedureFollowed"].astype(float)

    episode_adherence = guided.groupby("episodeNumber")["procedureFollowed"].mean()

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        episode_adherence.index,
        episode_adherence.values,
    )

    plt.xlabel("Episode")

    plt.ylabel("Procedure adherence")

    plt.ylim(
        0,
        1.05,
    )

    plt.title((f"{agent_name} " "LLM Procedure Adherence"))

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_procedure_adherence.png"),
    )


# ==========================================================
# Guidance Bonus
# ==========================================================


def plot_guidance_bonus(
    steps_df,
    output_directory,
    agent_name,
):

    guided = steps_df[steps_df["phase"] == "TRAIN"]

    if guided.empty or guided["guidanceBonus"].sum() == 0:

        return None

    bonuses = guided.groupby("episodeNumber")["guidanceBonus"].sum()

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        bonuses.index,
        bonuses.values,
    )

    plt.xlabel("Episode")

    plt.ylabel("Guidance bonus")

    plt.title((f"{agent_name} " "Guidance Reward per Episode"))

    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        (f"{_safe_name(agent_name)}" "_guidance_bonus.png"),
    )


# ==========================================================
# Generate
# ==========================================================


def generate_agent_visualizations(
    tables,
    output_directory,
    agent_name,
):

    paths = {}

    training_df = tables["training_episodes"]

    evaluation_df = tables["evaluation_episodes"]

    updates_df = tables["ppo_updates"]

    steps_df = tables["steps"]

    window = config.training.MOVING_AVERAGE_WINDOW

    # ======================================================
    # Episode Graphs
    # ======================================================

    if not training_df.empty:

        paths["reward"] = plot_reward_curve(
            training_df,
            output_directory,
            window,
            agent_name,
        )

        paths["success"] = plot_success_curve(
            training_df,
            output_directory,
            window,
            agent_name,
        )

        paths["steps"] = plot_steps_curve(
            training_df,
            output_directory,
            window,
            agent_name,
        )

        paths["termination"] = plot_termination_reasons(
            training_df,
            output_directory,
            agent_name,
        )

    # ======================================================
    # PPO Graphs
    # ======================================================

    if not updates_df.empty:

        safe = _safe_name(agent_name)

        paths["policy_loss"] = _plot_update_metric(
            updates_df,
            "policy_loss",
            "Policy loss",
            (f"{agent_name} " "PPO Policy Loss"),
            f"{safe}_policy_loss.png",
            output_directory,
        )

        paths["value_loss"] = _plot_update_metric(
            updates_df,
            "value_loss",
            "Value loss",
            (f"{agent_name} " "PPO Value Loss"),
            f"{safe}_value_loss.png",
            output_directory,
        )

        paths["entropy"] = _plot_update_metric(
            updates_df,
            "entropy",
            "Policy entropy",
            (f"{agent_name} " "PPO Policy Entropy"),
            f"{safe}_entropy.png",
            output_directory,
        )

    # ======================================================
    # Actions
    # ======================================================

    if not steps_df.empty:

        paths["actions"] = plot_action_frequency(
            steps_df,
            output_directory,
            agent_name,
        )

        adherence_path = plot_procedure_adherence(
            steps_df,
            output_directory,
            agent_name,
        )

        if adherence_path:

            paths["procedure_adherence"] = adherence_path

        bonus_path = plot_guidance_bonus(
            steps_df,
            output_directory,
            agent_name,
        )

        if bonus_path:

            paths["guidance_bonus"] = bonus_path

    # ======================================================
    # Evaluation
    # ======================================================

    if not evaluation_df.empty:

        paths["evaluation_reward"] = plot_evaluation_rewards(
            evaluation_df,
            output_directory,
            agent_name,
        )

    return paths


# ==========================================================
# Backwards Compatibility
# ==========================================================


def generate_ppo_visualizations(
    tables,
    output_directory,
):

    return generate_agent_visualizations(
        tables=tables,
        output_directory=output_directory,
        agent_name="PPO",
    )
