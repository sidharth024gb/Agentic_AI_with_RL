"""
visualization.py

Visualizations shared by:

    - PPO
    - LLM + PPO

Agent-performance plots exclude episodes terminated by
backend/environment infrastructure errors. Diagnostic
termination plots intentionally retain those episodes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.config import config

# ==========================================================
# Helpers
# ==========================================================


def _safe_name(value):
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


def _valid_episode_rows(dataframe):
    """Exclude infrastructure-error episodes from performance plots."""

    if dataframe.empty:
        return dataframe.copy()

    if "validAgentEpisode" in dataframe.columns:
        mask = dataframe["validAgentEpisode"].astype(bool)
    else:
        environment_errors = dataframe.get(
            "environmentErrors",
            pd.Series(0, index=dataframe.index),
        ).fillna(0)

        terminated_reason = (
            dataframe.get(
                "terminatedReason",
                pd.Series("", index=dataframe.index),
            )
            .fillna("")
            .astype(str)
            .str.upper()
        )

        mask = (environment_errors.astype(float) <= 0) & (
            terminated_reason != "ENVIRONMENT_ERROR"
        )

    return dataframe.loc[mask].copy()


def _valid_step_rows(dataframe):
    """Exclude infrastructure-error episodes/steps from behaviour plots."""

    if dataframe.empty:
        return dataframe.copy()

    mask = pd.Series(True, index=dataframe.index)

    if "episodeEnvironmentError" in dataframe.columns:
        mask &= ~dataframe["episodeEnvironmentError"].astype(bool)

    if "environmentError" in dataframe.columns:
        mask &= ~dataframe["environmentError"].astype(bool)

    return dataframe.loc[mask].copy()


def _episode_axis_column(dataframe):
    """
    Prefer the run-local episode index for experiment plots.

    ``episodeNumber`` is backend-global and is retained only as a
    backwards-compatible fallback for older metric workbooks.
    """

    if "runEpisode" in dataframe.columns:
        return "runEpisode"

    return "episodeNumber"


# ==========================================================
# Reward
# ==========================================================


def plot_reward_curve(
    training_df,
    output_directory,
    window,
    agent_name,
):
    dataframe = _valid_episode_rows(training_df)

    if dataframe.empty:
        return None

    dataframe["movingReward"] = (
        dataframe["totalReward"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    episode_column = _episode_axis_column(dataframe)

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe[episode_column],
        dataframe["totalReward"],
        label="Episode reward",
        alpha=0.45,
    )

    plt.plot(
        dataframe[episode_column],
        dataframe["movingReward"],
        label=f"{window}-valid-episode moving average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{agent_name} Training Reward")
    plt.legend()
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_reward_curve.png",
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
    dataframe = _valid_episode_rows(training_df)

    if dataframe.empty:
        return None

    rolling = (
        dataframe["completed"]
        .astype(float)
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    episode_column = _episode_axis_column(dataframe)

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe[episode_column],
        rolling,
    )

    plt.xlabel("Episode")
    plt.ylabel("Rolling success rate")
    plt.ylim(0, 1.05)
    plt.title(f"{agent_name} Training Success Rate")
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_success_rate.png",
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
    dataframe = _valid_episode_rows(training_df)

    if dataframe.empty:
        return None

    dataframe["movingSteps"] = (
        dataframe["totalSteps"]
        .rolling(
            window=window,
            min_periods=1,
        )
        .mean()
    )

    episode_column = _episode_axis_column(dataframe)

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe[episode_column],
        dataframe["totalSteps"],
        label="Episode steps",
        alpha=0.45,
    )

    plt.plot(
        dataframe[episode_column],
        dataframe["movingSteps"],
        label=f"{window}-valid-episode moving average",
    )

    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(f"{agent_name} Steps per Episode")
    plt.legend()
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_steps_curve.png",
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
    if updates_df.empty or column not in updates_df.columns:
        return None

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
    valid_steps = _valid_step_rows(steps_df)

    training_steps = valid_steps[valid_steps["phase"] == "TRAIN"]

    if training_steps.empty:
        return None

    counts = training_steps["action"].value_counts().sort_values()

    if counts.empty:
        return None

    figure = plt.figure(figsize=(11, 7))

    plt.barh(
        counts.index,
        counts.values,
    )

    plt.xlabel("Action count")
    plt.ylabel("Action")
    plt.title(f"{agent_name} Training Action Frequency")
    plt.grid(axis="x", alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_action_frequency.png",
    )


# ==========================================================
# Termination
# ==========================================================


def plot_termination_reasons(
    training_df,
    output_directory,
    agent_name,
):
    """
    Diagnostic plot: intentionally includes environment errors.
    """

    if training_df.empty:
        return None

    counts = training_df["terminatedReason"].fillna("UNKNOWN").value_counts()

    if counts.empty:
        return None

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

    plt.grid(axis="y", alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_termination.png",
    )


# ==========================================================
# Evaluation
# ==========================================================


def plot_evaluation_rewards(
    evaluation_df,
    output_directory,
    agent_name,
):
    dataframe = _valid_episode_rows(evaluation_df)

    if dataframe.empty:
        return None

    episode_column = _episode_axis_column(dataframe)

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        dataframe[episode_column],
        dataframe["totalReward"],
    )

    plt.xlabel("Evaluation episode")
    plt.ylabel("Reward")
    plt.title(f"{agent_name} Deterministic Evaluation")
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_evaluation_reward.png",
    )


# ==========================================================
# LLM Procedure Adherence
# ==========================================================


def plot_procedure_adherence(
    steps_df,
    output_directory,
    agent_name,
):
    valid_steps = _valid_step_rows(steps_df)

    guided = valid_steps[
        (valid_steps["phase"] == "TRAIN") & (valid_steps["procedureFollowed"].notna())
    ].copy()

    if guided.empty:
        return None

    guided["procedureFollowed"] = guided["procedureFollowed"].astype(float)

    episode_column = _episode_axis_column(guided)

    episode_adherence = (
        guided.groupby(episode_column)["procedureFollowed"].mean().sort_index()
    )

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        episode_adherence.index,
        episode_adherence.values,
    )

    plt.xlabel("Episode")
    plt.ylabel("Procedure adherence")
    plt.ylim(0, 1.05)
    plt.title(f"{agent_name} LLM Procedure Adherence")
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_procedure_adherence.png",
    )


# ==========================================================
# Guidance Bonus
# ==========================================================


def plot_guidance_bonus(
    steps_df,
    output_directory,
    agent_name,
):
    valid_steps = _valid_step_rows(steps_df)

    guided = valid_steps[valid_steps["phase"] == "TRAIN"]

    if guided.empty or guided["guidanceBonus"].sum() == 0:
        return None

    episode_column = _episode_axis_column(guided)

    bonuses = guided.groupby(episode_column)["guidanceBonus"].sum().sort_index()

    figure = plt.figure(figsize=(11, 6))

    plt.plot(
        bonuses.index,
        bonuses.values,
    )

    plt.xlabel("Episode")
    plt.ylabel("Guidance bonus")
    plt.title(f"{agent_name} Guidance Reward per Episode")
    plt.grid(alpha=0.25)

    return _save(
        figure,
        output_directory,
        f"{_safe_name(agent_name)}_guidance_bonus.png",
    )


# ==========================================================
# Generate
# ==========================================================


def _add_path(paths, key, path):
    if path:
        paths[key] = path


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
        _add_path(
            paths,
            "reward",
            plot_reward_curve(
                training_df,
                output_directory,
                window,
                agent_name,
            ),
        )

        _add_path(
            paths,
            "success",
            plot_success_curve(
                training_df,
                output_directory,
                window,
                agent_name,
            ),
        )

        _add_path(
            paths,
            "steps",
            plot_steps_curve(
                training_df,
                output_directory,
                window,
                agent_name,
            ),
        )

        _add_path(
            paths,
            "termination",
            plot_termination_reasons(
                training_df,
                output_directory,
                agent_name,
            ),
        )

    # ======================================================
    # PPO Graphs
    # ======================================================

    if not updates_df.empty:
        safe = _safe_name(agent_name)

        _add_path(
            paths,
            "policy_loss",
            _plot_update_metric(
                updates_df,
                "policy_loss",
                "Policy loss",
                f"{agent_name} PPO Policy Loss",
                f"{safe}_policy_loss.png",
                output_directory,
            ),
        )

        _add_path(
            paths,
            "value_loss",
            _plot_update_metric(
                updates_df,
                "value_loss",
                "Value loss",
                f"{agent_name} PPO Value Loss",
                f"{safe}_value_loss.png",
                output_directory,
            ),
        )

        _add_path(
            paths,
            "entropy",
            _plot_update_metric(
                updates_df,
                "entropy",
                "Policy entropy",
                f"{agent_name} PPO Policy Entropy",
                f"{safe}_entropy.png",
                output_directory,
            ),
        )

    # ======================================================
    # Actions
    # ======================================================

    if not steps_df.empty:
        _add_path(
            paths,
            "actions",
            plot_action_frequency(
                steps_df,
                output_directory,
                agent_name,
            ),
        )

        _add_path(
            paths,
            "procedure_adherence",
            plot_procedure_adherence(
                steps_df,
                output_directory,
                agent_name,
            ),
        )

        _add_path(
            paths,
            "guidance_bonus",
            plot_guidance_bonus(
                steps_df,
                output_directory,
                agent_name,
            ),
        )

    # ======================================================
    # Evaluation
    # ======================================================

    if not evaluation_df.empty:
        _add_path(
            paths,
            "evaluation_reward",
            plot_evaluation_rewards(
                evaluation_df,
                output_directory,
                agent_name,
            ),
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
