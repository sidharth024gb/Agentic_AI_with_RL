"""
metrics.py

Metrics and Excel reporting for:

    - PPO
    - LLM + PPO

Backend Episode records are treated as the authoritative
record of what happened in the finance environment.

Important metric rule
---------------------
Episodes terminated by infrastructure/backend/environment
failures remain visible for diagnostics, but they are excluded
from agent-performance calculations such as:

    - success rate
    - reward statistics
    - convergence
    - average steps
    - guidance adherence

This matches the PPO training rule that environment-error
transitions are not trainable.
"""

import json

from pathlib import Path

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


def _json_string(value):
    if value is None:
        return ""

    return json.dumps(
        value,
        default=str,
        sort_keys=True,
    )


def _episode_has_environment_error(episode):
    """Return True when an episode ended because of infrastructure failure."""

    if not isinstance(episode, dict):
        return False

    environment_errors = int(
        episode.get(
            "environmentErrors",
            0,
        )
        or 0
    )

    terminated_reason = str(
        episode.get(
            "terminatedReason",
            "",
        )
        or ""
    ).upper()

    return environment_errors > 0 or terminated_reason == "ENVIRONMENT_ERROR"


def _valid_episode_dataframe(dataframe):
    """Return only episodes valid for agent-performance metrics."""

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


def _valid_step_dataframe(dataframe):
    """Return steps that belong to valid episodes and are not env-error steps."""

    if dataframe.empty:
        return dataframe.copy()

    mask = pd.Series(True, index=dataframe.index)

    if "episodeEnvironmentError" in dataframe.columns:
        mask &= ~dataframe["episodeEnvironmentError"].astype(bool)

    if "environmentError" in dataframe.columns:
        mask &= ~dataframe["environmentError"].astype(bool)

    return dataframe.loc[mask].copy()


# ==========================================================
# Episodes
# ==========================================================


def episodes_to_dataframe(episodes):
    rows = []

    for episode in episodes:
        episode_environment_error = _episode_has_environment_error(episode)

        rows.append(
            {
                "episodeId": str(episode.get("_id", "")),
                "episodeNumber": episode.get("episodeNumber"),
                "experimentName": episode.get("experimentName"),
                "phase": episode.get("phase"),
                "agentType": episode.get("agentType"),
                "algorithm": episode.get("algorithm"),
                "seed": episode.get("seed"),
                "goal": episode.get("goal"),
                # ----------------------------------------------
                # LLM metadata
                # ----------------------------------------------
                "llmModel": episode.get("llmModel"),
                "guidanceMode": episode.get("guidanceMode"),
                "promptVersion": episode.get("promptVersion"),
                "llmPlanCached": episode.get("llmPlanCached"),
                "llmPlanningTimeMs": episode.get("llmPlanningTimeMs"),
                "llmPlan": _json_string(episode.get("llmPlan")),
                # ----------------------------------------------
                # Rewards
                # ----------------------------------------------
                "totalReward": float(episode.get("totalReward", 0) or 0),
                "totalBaseReward": float(episode.get("totalBaseReward", 0) or 0),
                "totalGuidanceBonus": float(episode.get("totalGuidanceBonus", 0) or 0),
                # ----------------------------------------------
                # Actions
                # ----------------------------------------------
                "totalSteps": int(episode.get("totalSteps", 0) or 0),
                "successfulActions": int(episode.get("successfulActions", 0) or 0),
                "failedActions": int(episode.get("failedActions", 0) or 0),
                "noOpActions": int(episode.get("noOpActions", 0) or 0),
                "environmentErrors": int(episode.get("environmentErrors", 0) or 0),
                # ----------------------------------------------
                # Completion
                # ----------------------------------------------
                "completed": bool(episode.get("completed", False)),
                "terminatedReason": episode.get("terminatedReason"),
                "validAgentEpisode": not episode_environment_error,
                "executionTimeMs": float(episode.get("executionTimeMs", 0) or 0),
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
# Steps
# ==========================================================


def steps_to_dataframe(episodes):
    rows = []

    for episode in episodes:
        episode_number = episode.get("episodeNumber")
        phase = episode.get("phase")
        agent_type = episode.get("agentType")
        guidance_mode = episode.get("guidanceMode")
        episode_environment_error = _episode_has_environment_error(episode)

        for step in episode.get("actionSequence", []):
            step_environment_error = bool(
                step.get(
                    "environmentError",
                    False,
                )
            )

            rows.append(
                {
                    "episodeNumber": episode_number,
                    "phase": phase,
                    "agentType": agent_type,
                    "guidanceMode": guidance_mode,
                    "stepNumber": step.get("stepNumber"),
                    "action": step.get("action"),
                    "endpoint": step.get("endpoint"),
                    "baseReward": float(step.get("baseReward", 0) or 0),
                    "guidanceBonus": float(step.get("guidanceBonus", 0) or 0),
                    "reward": float(step.get("reward", 0) or 0),
                    "success": bool(step.get("success", False)),
                    "usefulAction": bool(step.get("usefulAction", True)),
                    "environmentError": step_environment_error,
                    "episodeEnvironmentError": episode_environment_error,
                    "validAgentStep": bool(
                        not episode_environment_error and not step_environment_error
                    ),
                    "procedureFollowed": step.get("procedureFollowed"),
                    "durationMs": float(step.get("durationMs", 0) or 0),
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
    """
    Return the original episode number at which a rolling window
    of VALID episodes first reaches the target completion rate.
    """

    if training_df.empty:
        return None

    valid_df = _valid_episode_dataframe(training_df)

    if valid_df.empty:
        return None

    window_size = window_size or config.training.CONVERGENCE_WINDOW

    success_threshold = (
        success_threshold
        if success_threshold is not None
        else config.training.CONVERGENCE_SUCCESS_THRESHOLD
    )

    if len(valid_df) < window_size:
        return None

    valid_df = valid_df.reset_index(drop=True)

    rolling_success = (
        valid_df["completed"].astype(float).rolling(window=window_size).mean()
    )

    matches = rolling_success[rolling_success >= success_threshold]

    if matches.empty:
        return None

    index = matches.index[0]

    return int(valid_df.loc[index, "episodeNumber"])


# ==========================================================
# Episode Summary
# ==========================================================


def _empty_episode_summary(total_episodes=0, environment_error_episodes=0):
    return {
        "episodes": int(total_episodes),
        "valid_episodes": 0,
        "environment_error_episodes": int(environment_error_episodes),
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
        "total_guidance_bonus": 0.0,
    }


def calculate_episode_summary(dataframe):
    if dataframe.empty:
        return _empty_episode_summary()

    total_episodes = len(dataframe)

    valid_df = _valid_episode_dataframe(dataframe)

    environment_error_episodes = total_episodes - len(valid_df)

    total_environment_errors = int(
        dataframe["environmentErrors"].sum()
        if "environmentErrors" in dataframe.columns
        else 0
    )

    if valid_df.empty:
        summary = _empty_episode_summary(
            total_episodes=total_episodes,
            environment_error_episodes=environment_error_episodes,
        )

        summary["environment_errors"] = total_environment_errors

        return summary

    completed = int(valid_df["completed"].sum())

    return {
        # Total rows remain visible for diagnostics.
        "episodes": total_episodes,
        # Performance denominator.
        "valid_episodes": len(valid_df),
        "environment_error_episodes": environment_error_episodes,
        "completed": completed,
        "success_rate": completed / len(valid_df),
        "average_reward": float(valid_df["totalReward"].mean()),
        "median_reward": float(valid_df["totalReward"].median()),
        "reward_std": float(valid_df["totalReward"].std(ddof=0)),
        "best_reward": float(valid_df["totalReward"].max()),
        "worst_reward": float(valid_df["totalReward"].min()),
        "average_steps": float(valid_df["totalSteps"].mean()),
        "median_steps": float(valid_df["totalSteps"].median()),
        "average_execution_ms": float(valid_df["executionTimeMs"].mean()),
        # Environment failures are still reported diagnostically.
        "environment_errors": total_environment_errors,
        # Agent behaviour statistics use valid episodes only.
        "no_op_actions": int(valid_df["noOpActions"].sum()),
        "failed_actions": int(valid_df["failedActions"].sum()),
        "total_guidance_bonus": float(valid_df["totalGuidanceBonus"].sum()),
    }


# ==========================================================
# Guidance Summary
# ==========================================================


def build_guidance_summary(steps_df):
    columns = [
        "phase",
        "procedure_attempts",
        "procedure_followed",
        "procedure_adherence_rate",
        "total_guidance_bonus",
        "average_guidance_bonus",
    ]

    if steps_df.empty:
        return pd.DataFrame(columns=columns)

    valid_steps = _valid_step_dataframe(steps_df)

    if valid_steps.empty:
        return pd.DataFrame(columns=columns)

    guided = valid_steps[valid_steps["procedureFollowed"].notna()].copy()

    if guided.empty:
        return pd.DataFrame(columns=columns)

    guided["procedureFollowed"] = guided["procedureFollowed"].astype(bool)

    rows = []

    for phase, frame in guided.groupby("phase"):
        attempts = len(frame)
        followed = int(frame["procedureFollowed"].sum())

        rows.append(
            {
                "phase": phase,
                "procedure_attempts": attempts,
                "procedure_followed": followed,
                "procedure_adherence_rate": (followed / attempts if attempts else 0.0),
                "total_guidance_bonus": float(frame["guidanceBonus"].sum()),
                "average_guidance_bonus": float(frame["guidanceBonus"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


# ==========================================================
# LLM Metrics
# ==========================================================


def build_llm_metrics_dataframe(
    training_llm_metrics=None,
    evaluation_llm_metrics=None,
):
    training_llm_metrics = training_llm_metrics or {}
    evaluation_llm_metrics = evaluation_llm_metrics or {}

    keys = sorted(set(training_llm_metrics.keys()) | set(evaluation_llm_metrics.keys()))

    return pd.DataFrame(
        [
            {
                "metric": key,
                "training": training_llm_metrics.get(key),
                "evaluation": evaluation_llm_metrics.get(key),
            }
            for key in keys
        ],
        columns=[
            "metric",
            "training",
            "evaluation",
        ],
    )


# ==========================================================
# Metric Tables
# ==========================================================


def build_metric_tables(
    training_episodes,
    evaluation_episodes,
    ppo_updates,
    training_time,
    evaluation_time,
    training_llm_metrics=None,
    evaluation_llm_metrics=None,
):
    training_df = episodes_to_dataframe(training_episodes)
    evaluation_df = episodes_to_dataframe(evaluation_episodes)

    all_episodes = list(training_episodes) + list(evaluation_episodes)

    steps_df = steps_to_dataframe(all_episodes)

    updates_df = pd.DataFrame(ppo_updates)

    # ======================================================
    # Episode Metrics
    # ======================================================

    training_summary = calculate_episode_summary(training_df)
    evaluation_summary = calculate_episode_summary(evaluation_df)

    training_summary["convergence_episode"] = calculate_convergence_episode(training_df)

    training_summary["wall_clock_seconds"] = training_time
    evaluation_summary["wall_clock_seconds"] = evaluation_time

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

    # ======================================================
    # Action Summary
    #
    # Reward/success statistics use only valid agent steps.
    # Infrastructure-error step counts remain visible through
    # the environment_errors column.
    # ======================================================

    if not steps_df.empty:
        valid_steps = _valid_step_dataframe(steps_df)

        if not valid_steps.empty:
            action_summary = (
                valid_steps.groupby(
                    ["phase", "action"],
                    dropna=False,
                )
                .agg(
                    count=("action", "size"),
                    successful=("success", "sum"),
                    useful=("usefulAction", "sum"),
                    total_reward=("reward", "sum"),
                    average_reward=("reward", "mean"),
                    total_guidance_bonus=("guidanceBonus", "sum"),
                    average_duration_ms=("durationMs", "mean"),
                )
                .reset_index()
            )
        else:
            action_summary = pd.DataFrame(
                columns=[
                    "phase",
                    "action",
                    "count",
                    "successful",
                    "useful",
                    "total_reward",
                    "average_reward",
                    "total_guidance_bonus",
                    "average_duration_ms",
                ]
            )

        environment_error_steps = (
            steps_df.groupby(
                ["phase", "action"],
                dropna=False,
            )["environmentError"]
            .sum()
            .reset_index(name="environment_errors")
        )

        action_summary = action_summary.merge(
            environment_error_steps,
            on=["phase", "action"],
            how="outer",
        )

        numeric_columns = [
            "count",
            "successful",
            "useful",
            "total_reward",
            "average_reward",
            "total_guidance_bonus",
            "average_duration_ms",
            "environment_errors",
        ]

        for column in numeric_columns:
            if column in action_summary.columns:
                action_summary[column] = action_summary[column].fillna(0)

    else:
        action_summary = pd.DataFrame()

    # ======================================================
    # Terminations
    #
    # Keep ALL episodes here, including environment errors.
    # This is intentionally a diagnostic table.
    # ======================================================

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
                ["phase", "terminatedReason"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
    else:
        termination_summary = pd.DataFrame()

    # ======================================================
    # Guidance
    # ======================================================

    guidance_summary = build_guidance_summary(steps_df)

    # ======================================================
    # LLM Planner
    # ======================================================

    llm_metrics_df = build_llm_metrics_dataframe(
        training_llm_metrics,
        evaluation_llm_metrics,
    )

    return {
        "summary": summary_df,
        "training_episodes": training_df,
        "evaluation_episodes": evaluation_df,
        "steps": steps_df,
        "ppo_updates": updates_df,
        "action_summary": action_summary,
        "termination_summary": termination_summary,
        "guidance_summary": guidance_summary,
        "llm_metrics": llm_metrics_df,
    }


# ==========================================================
# Configuration Snapshot
# ==========================================================


def config_dataframe(agent_label=None):
    values = {
        # ------------------------------------------------------
        # Agent
        # ------------------------------------------------------
        "RUN_AGENT": agent_label,
        "CONFIG_AGENT_TYPE": config.agent.AGENT_TYPE,
        "ALGORITHM": config.agent.ALGORITHM,
        "TASK": config.agent.TASK,
        # ------------------------------------------------------
        # Experiment
        # ------------------------------------------------------
        "EXPERIMENT_NAME": config.experiment.EXPERIMENT_NAME,
        "GUIDANCE_MODE": config.experiment.GUIDANCE_MODE,
        "GUIDANCE_BONUS": config.experiment.GUIDANCE_BONUS,
        "ENVIRONMENT_VERSION": getattr(
            config.experiment,
            "ENVIRONMENT_VERSION",
            None,
        ),
        # ------------------------------------------------------
        # LLM
        # ------------------------------------------------------
        "LLM_MODEL": config.llm.MODEL,
        "LLM_BASE_URL": config.llm.BASE_URL,
        "LLM_TEMPERATURE": config.llm.TEMPERATURE,
        "LLM_USE_CACHE": config.llm.USE_CACHE,
        # ------------------------------------------------------
        # PPO
        # ------------------------------------------------------
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
        "ENTROPY_COEF": getattr(
            config.training,
            "ENTROPY_COEF",
            None,
        ),
        "MAX_GRAD_NORM": getattr(
            config.training,
            "MAX_GRAD_NORM",
            None,
        ),
        # ------------------------------------------------------
        # Environment
        # ------------------------------------------------------
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
# Excel
# ==========================================================


def export_metrics_excel(
    tables,
    output_path,
    agent_label=None,
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
        sheet_map = {
            "summary": "Summary",
            "training_episodes": "Training Episodes",
            "evaluation_episodes": "Evaluation Episodes",
            "ppo_updates": "PPO Updates",
            "action_summary": "Action Summary",
            "termination_summary": "Termination Summary",
            "guidance_summary": "Guidance Summary",
            "llm_metrics": "LLM Metrics",
            "steps": "Steps",
        }

        for key, sheet_name in sheet_map.items():
            tables.get(
                key,
                pd.DataFrame(),
            ).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        config_dataframe(agent_label).to_excel(
            writer,
            sheet_name="Configuration",
            index=False,
        )

        # ======================================================
        # Styling
        # ======================================================

        workbook = writer.book

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
                    value = "" if cell.value is None else str(cell.value)

                    max_length = max(
                        max_length,
                        len(value),
                    )

                worksheet.column_dimensions[column_letter].width = min(
                    max(
                        max_length + 2,
                        12,
                    ),
                    50,
                )

    return str(output_path)
