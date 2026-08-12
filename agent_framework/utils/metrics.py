"""
metrics.py

PPO metrics and Excel export.

Backend Episode records are treated as the authoritative
source for episode-level performance.
"""

import json

from pathlib import Path

import numpy as np
import pandas as pd

from openpyxl.styles import (
    Alignment,
    Font,
    PatternFill,
)

from config.config import config

# ==========================================================
# Helpers
# ==========================================================


def _json_string(
    value,
):

    if value is None:
        return ""

    return json.dumps(
        value,
        default=str,
        sort_keys=True,
    )


# ==========================================================
# Episode DataFrame
# ==========================================================


def episodes_to_dataframe(
    episodes,
):

    rows = []

    for episode in episodes:

        rows.append(
            {
                "episodeId": str(
                    episode.get(
                        "_id",
                        "",
                    )
                ),
                "episodeNumber": episode.get("episodeNumber"),
                "experimentName": episode.get("experimentName"),
                "phase": episode.get("phase"),
                "agentType": episode.get("agentType"),
                "algorithm": episode.get("algorithm"),
                "seed": episode.get("seed"),
                "goal": episode.get("goal"),
                "totalReward": float(
                    episode.get(
                        "totalReward",
                        0,
                    )
                    or 0
                ),
                "totalBaseReward": float(
                    episode.get(
                        "totalBaseReward",
                        0,
                    )
                    or 0
                ),
                "totalGuidanceBonus": float(
                    episode.get(
                        "totalGuidanceBonus",
                        0,
                    )
                    or 0
                ),
                "totalSteps": int(
                    episode.get(
                        "totalSteps",
                        0,
                    )
                    or 0
                ),
                "successfulActions": int(
                    episode.get(
                        "successfulActions",
                        0,
                    )
                    or 0
                ),
                "failedActions": int(
                    episode.get(
                        "failedActions",
                        0,
                    )
                    or 0
                ),
                "noOpActions": int(
                    episode.get(
                        "noOpActions",
                        0,
                    )
                    or 0
                ),
                "environmentErrors": int(
                    episode.get(
                        "environmentErrors",
                        0,
                    )
                    or 0
                ),
                "completed": bool(
                    episode.get(
                        "completed",
                        False,
                    )
                ),
                "terminatedReason": episode.get("terminatedReason"),
                "executionTimeMs": float(
                    episode.get(
                        "executionTimeMs",
                        0,
                    )
                    or 0
                ),
                "initialState": _json_string(episode.get("initialState")),
                "finalState": _json_string(episode.get("finalState")),
                "createdAt": episode.get("createdAt"),
            }
        )

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty and "episodeNumber" in dataframe.columns:

        dataframe = dataframe.sort_values("episodeNumber").reset_index(drop=True)

    return dataframe


# ==========================================================
# Steps DataFrame
# ==========================================================


def steps_to_dataframe(
    episodes,
):

    rows = []

    for episode in episodes:

        episode_number = episode.get("episodeNumber")

        phase = episode.get("phase")

        for step in episode.get(
            "actionSequence",
            [],
        ):

            rows.append(
                {
                    "episodeNumber": episode_number,
                    "phase": phase,
                    "stepNumber": step.get("stepNumber"),
                    "action": step.get("action"),
                    "endpoint": step.get("endpoint"),
                    "baseReward": float(
                        step.get(
                            "baseReward",
                            0,
                        )
                        or 0
                    ),
                    "guidanceBonus": float(
                        step.get(
                            "guidanceBonus",
                            0,
                        )
                        or 0
                    ),
                    "reward": float(
                        step.get(
                            "reward",
                            0,
                        )
                        or 0
                    ),
                    "success": bool(
                        step.get(
                            "success",
                            False,
                        )
                    ),
                    "usefulAction": bool(
                        step.get(
                            "usefulAction",
                            True,
                        )
                    ),
                    "environmentError": bool(
                        step.get(
                            "environmentError",
                            False,
                        )
                    ),
                    "procedureFollowed": step.get("procedureFollowed"),
                    "durationMs": float(
                        step.get(
                            "durationMs",
                            0,
                        )
                        or 0
                    ),
                    "message": step.get("message"),
                    "stateBefore": _json_string(step.get("stateBefore")),
                    "stateAfter": _json_string(step.get("stateAfter")),
                }
            )

    return pd.DataFrame(rows)


# ==========================================================
# Convergence
# ==========================================================


def calculate_convergence_episode(
    training_df,
    window_size=None,
    success_threshold=None,
):

    if training_df.empty:
        return None

    window_size = window_size or config.training.CONVERGENCE_WINDOW

    success_threshold = (
        success_threshold
        if success_threshold is not None
        else config.training.CONVERGENCE_SUCCESS_THRESHOLD
    )

    if len(training_df) < window_size:

        return None

    rolling_success = (
        training_df["completed"]
        .astype(float)
        .rolling(
            window=window_size,
        )
        .mean()
    )

    matches = rolling_success[rolling_success >= success_threshold]

    if matches.empty:

        return None

    index = matches.index[0]

    return int(
        training_df.loc[
            index,
            "episodeNumber",
        ]
    )


# ==========================================================
# Summary
# ==========================================================


def calculate_episode_summary(
    dataframe,
):

    if dataframe.empty:

        return {
            "episodes": 0,
            "completed": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "median_reward": 0.0,
            "reward_std": 0.0,
            "best_reward": 0.0,
            "worst_reward": 0.0,
            "average_steps": 0.0,
            "median_steps": 0.0,
            "average_execution_ms": 0.0,
            "environment_errors": 0,
            "no_op_actions": 0,
            "failed_actions": 0,
        }

    completed = int(dataframe["completed"].sum())

    return {
        "episodes": len(dataframe),
        "completed": completed,
        "success_rate": completed / len(dataframe),
        "average_reward": float(dataframe["totalReward"].mean()),
        "median_reward": float(dataframe["totalReward"].median()),
        "reward_std": float(dataframe["totalReward"].std(ddof=0)),
        "best_reward": float(dataframe["totalReward"].max()),
        "worst_reward": float(dataframe["totalReward"].min()),
        "average_steps": float(dataframe["totalSteps"].mean()),
        "median_steps": float(dataframe["totalSteps"].median()),
        "average_execution_ms": float(dataframe["executionTimeMs"].mean()),
        "environment_errors": int(dataframe["environmentErrors"].sum()),
        "no_op_actions": int(dataframe["noOpActions"].sum()),
        "failed_actions": int(dataframe["failedActions"].sum()),
    }


# ==========================================================
# Full Metrics Tables
# ==========================================================


def build_metric_tables(
    training_episodes,
    evaluation_episodes,
    ppo_updates,
    training_time,
    evaluation_time,
):

    training_df = episodes_to_dataframe(training_episodes)

    evaluation_df = episodes_to_dataframe(evaluation_episodes)

    all_episodes = list(training_episodes) + list(evaluation_episodes)

    steps_df = steps_to_dataframe(all_episodes)

    updates_df = pd.DataFrame(ppo_updates)

    training_summary = calculate_episode_summary(training_df)

    evaluation_summary = calculate_episode_summary(evaluation_df)

    convergence_episode = calculate_convergence_episode(training_df)

    training_summary["convergence_episode"] = convergence_episode

    training_summary["wall_clock_seconds"] = training_time

    evaluation_summary["wall_clock_seconds"] = evaluation_time

    # ------------------------------------------------------
    # Summary table
    # ------------------------------------------------------

    metrics = sorted(set(training_summary.keys()) | set(evaluation_summary.keys()))

    summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "training": training_summary.get(metric),
                "evaluation": evaluation_summary.get(metric),
            }
            for metric in metrics
        ]
    )

    # ------------------------------------------------------
    # Action summary
    # ------------------------------------------------------

    if not steps_df.empty:

        action_summary = (
            steps_df.groupby(
                [
                    "phase",
                    "action",
                ],
                dropna=False,
            )
            .agg(
                count=(
                    "action",
                    "size",
                ),
                successful=(
                    "success",
                    "sum",
                ),
                useful=(
                    "usefulAction",
                    "sum",
                ),
                environment_errors=(
                    "environmentError",
                    "sum",
                ),
                total_reward=(
                    "reward",
                    "sum",
                ),
                average_reward=(
                    "reward",
                    "mean",
                ),
                average_duration_ms=(
                    "durationMs",
                    "mean",
                ),
            )
            .reset_index()
        )

    else:

        action_summary = pd.DataFrame()

    # ------------------------------------------------------
    # Termination summary
    # ------------------------------------------------------

    combined_df = pd.concat(
        [
            training_df,
            evaluation_df,
        ],
        ignore_index=True,
    )

    if not combined_df.empty:

        termination_summary = (
            combined_df.groupby(
                [
                    "phase",
                    "terminatedReason",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

    else:

        termination_summary = pd.DataFrame()

    return {
        "summary": summary_df,
        "training_episodes": training_df,
        "evaluation_episodes": evaluation_df,
        "steps": steps_df,
        "ppo_updates": updates_df,
        "action_summary": action_summary,
        "termination_summary": termination_summary,
    }


# ==========================================================
# Config Snapshot
# ==========================================================


def config_dataframe():

    values = {
        "TOTAL_EPISODES": config.training.TOTAL_EPISODES,
        "EVALUATION_EPISODES": config.training.EVALUATION_EPISODES,
        "GAMMA": config.training.GAMMA,
        "GAE_LAMBDA": config.training.GAE_LAMBDA,
        "CLIP_EPSILON": config.training.CLIP_EPSILON,
        "LEARNING_RATE": config.training.LEARNING_RATE,
        "BATCH_SIZE": config.training.BATCH_SIZE,
        "UPDATE_INTERVAL": config.training.UPDATE_INTERVAL,
        "EPOCHS": config.training.EPOCHS,
        "HIDDEN_NEURON_SIZE": config.training.HIDDEN_NEURON_SIZE,
        "MAX_STEPS_PER_EPISODE": config.environment.MAX_STEPS_PER_EPISODE,
        "RANDOM_SEED": config.environment.RANDOM_SEED,
        "ACTION_SPACE_SIZE": config.environment.ACTION_SPACE_SIZE,
    }

    return pd.DataFrame(
        [
            {
                "parameter": key,
                "value": value,
            }
            for key, value in values.items()
        ]
    )


# ==========================================================
# Excel Export
# ==========================================================


def export_metrics_excel(
    tables,
    output_path,
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        tables["summary"].to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        tables["training_episodes"].to_excel(
            writer,
            sheet_name="Training Episodes",
            index=False,
        )

        tables["evaluation_episodes"].to_excel(
            writer,
            sheet_name="Evaluation Episodes",
            index=False,
        )

        tables["ppo_updates"].to_excel(
            writer,
            sheet_name="PPO Updates",
            index=False,
        )

        tables["action_summary"].to_excel(
            writer,
            sheet_name="Action Summary",
            index=False,
        )

        tables["termination_summary"].to_excel(
            writer,
            sheet_name="Termination Summary",
            index=False,
        )

        tables["steps"].to_excel(
            writer,
            sheet_name="Steps",
            index=False,
        )

        config_dataframe().to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )

        workbook = writer.book

        # ======================================================
        # Workbook Styling
        # ======================================================

        header_fill = PatternFill(
            "solid",
            fgColor="D9EAF7",
        )

        header_font = Font(bold=True)

        for worksheet in workbook.worksheets:

            worksheet.freeze_panes = "A2"

            worksheet.auto_filter.ref = worksheet.dimensions

            for cell in worksheet[1]:

                cell.font = header_font

                cell.fill = header_fill

                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )

            for column_cells in worksheet.columns:

                max_length = 0

                column_letter = column_cells[0].column_letter

                for cell in column_cells:

                    try:

                        value = "" if cell.value is None else str(cell.value)

                        max_length = max(
                            max_length,
                            len(value),
                        )

                    except Exception:

                        pass

                worksheet.column_dimensions[column_letter].width = min(
                    max(
                        max_length + 2,
                        12,
                    ),
                    50,
                )

    return str(output_path)
