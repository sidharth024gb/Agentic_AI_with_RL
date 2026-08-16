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


def _excel_safe_value(value):
    """Preserve scalar values and serialize nested structures for Excel."""

    if isinstance(value, (dict, list, tuple, set)):
        return _json_string(value)

    return value


def records_to_dataframe(records):
    """Convert arbitrary runtime records without silently dropping fields.

    This is intentionally schema-flexible. If later training/evaluation code adds
    a diagnostic field, it automatically appears in the individual workbook and,
    through the comparison layer, in the combined workbook as well.
    """

    rows = []

    for record in records or []:
        if not isinstance(record, dict):
            continue

        rows.append({key: _excel_safe_value(value) for key, value in record.items()})

    return pd.DataFrame(rows)


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
    """Flatten backend episodes while retaining every top-level field.

    Known fields are normalized to stable names/types, but unknown/new backend
    fields are retained automatically. ``actionSequence`` is omitted here only
    because it has its own full ``Steps`` table.
    """

    rows = []

    for episode in episodes or []:
        if not isinstance(episode, dict):
            continue

        episode_environment_error = _episode_has_environment_error(episode)

        row = {
            key: _excel_safe_value(value)
            for key, value in episode.items()
            if key != "actionSequence"
        }

        row.update(
            {
                "episodeId": str(episode.get("_id", "")),
                "episodeNumber": episode.get("episodeNumber"),
                "experimentName": episode.get("experimentName"),
                "phase": episode.get("phase"),
                "agentType": episode.get("agentType"),
                "algorithm": episode.get("algorithm"),
                "seed": episode.get("seed"),
                "goal": episode.get("goal"),
                "llmModel": episode.get("llmModel"),
                "guidanceMode": episode.get("guidanceMode"),
                "promptVersion": episode.get("promptVersion"),
                "llmPlanCached": episode.get("llmPlanCached"),
                "llmPlanningTimeMs": episode.get("llmPlanningTimeMs"),
                "llmPlan": _json_string(episode.get("llmPlan")),
                "llmPrerequisites": _json_string(episode.get("llmPrerequisites")),
                "totalReward": float(episode.get("totalReward", 0) or 0),
                "totalBaseReward": float(episode.get("totalBaseReward", 0) or 0),
                "totalGuidanceBonus": float(episode.get("totalGuidanceBonus", 0) or 0),
                "totalSteps": int(episode.get("totalSteps", 0) or 0),
                "successfulActions": int(episode.get("successfulActions", 0) or 0),
                "failedActions": int(episode.get("failedActions", 0) or 0),
                "noOpActions": int(episode.get("noOpActions", 0) or 0),
                "environmentErrors": int(episode.get("environmentErrors", 0) or 0),
                "completed": bool(episode.get("completed", False)),
                "terminatedReason": episode.get("terminatedReason"),
                "validAgentEpisode": not episode_environment_error,
                "executionTimeMs": float(episode.get("executionTimeMs", 0) or 0),
                "initialState": _json_string(episode.get("initialState")),
                "finalState": _json_string(episode.get("finalState")),
                "createdAt": episode.get("createdAt"),
            }
        )

        rows.append(row)

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        if "episodeNumber" in dataframe.columns:
            dataframe = dataframe.sort_values(
                "episodeNumber",
                na_position="last",
            ).reset_index(drop=True)
        else:
            dataframe = dataframe.reset_index(drop=True)

        dataframe.insert(
            1,
            "runEpisode",
            range(1, len(dataframe) + 1),
        )

    return dataframe


# ==========================================================
# Steps
# ==========================================================


def steps_to_dataframe(episodes):
    """Flatten every backend step while retaining all step diagnostics.

    This deliberately starts from the complete step dictionary. Fields such as
    repeated-action diagnostics, procedure information, error types, or future
    backend additions are therefore not discarded by the reporting layer.
    """

    rows = []
    phase_episode_numbers = {}

    for episode in episodes or []:
        phase = episode.get("phase")
        episode_number = episode.get("episodeNumber")
        phase_episode_numbers.setdefault(phase, []).append(episode_number)

    run_episode_lookup = {}

    for phase, episode_numbers in phase_episode_numbers.items():
        ordered_numbers = sorted(
            episode_numbers,
            key=lambda value: (
                value is None,
                value if value is not None else 0,
            ),
        )

        for run_episode, episode_number in enumerate(
            ordered_numbers,
            start=1,
        ):
            run_episode_lookup[(phase, episode_number)] = run_episode

    for episode in episodes or []:
        episode_number = episode.get("episodeNumber")
        phase = episode.get("phase")
        run_episode = run_episode_lookup.get((phase, episode_number))
        agent_type = episode.get("agentType")
        guidance_mode = episode.get("guidanceMode")
        episode_environment_error = _episode_has_environment_error(episode)

        for step in episode.get("actionSequence", []) or []:
            if not isinstance(step, dict):
                continue

            step_environment_error = bool(step.get("environmentError", False))

            row = {key: _excel_safe_value(value) for key, value in step.items()}

            row.update(
                {
                    "episodeNumber": episode_number,
                    "runEpisode": run_episode,
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

            rows.append(row)

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        sort_columns = [
            column
            for column in ["phase", "runEpisode", "stepNumber"]
            if column in dataframe.columns
        ]

        if sort_columns:
            dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)

    return dataframe


# ==========================================================
# Convergence
# ==========================================================


def calculate_convergence_episode(
    training_df,
    window_size=None,
    success_threshold=None,
):
    """
    Return the RUN-LOCAL episode at which a rolling window of
    VALID episodes first reaches the target completion rate.

    ``episodeNumber`` is backend-global and may continue increasing
    across experiments. ``runEpisode`` always starts at 1 for the
    current experiment and is therefore the correct convergence index.

    Environment-error episodes remain excluded from the rolling
    success calculation, but the returned ``runEpisode`` still
    represents the episode's true position within the full run.
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

    if "runEpisode" in valid_df.columns:
        return int(valid_df.loc[index, "runEpisode"])

    # Compatibility fallback for older dataframes that were created
    # before runEpisode was introduced.
    return int(index + 1)


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
        "total_reward": 0.0,
        "average_reward": 0.0,
        "median_reward": 0.0,
        "reward_std": 0.0,
        "best_reward": 0.0,
        "worst_reward": 0.0,
        "total_base_reward": 0.0,
        "average_base_reward": 0.0,
        "total_guidance_bonus": 0.0,
        "average_guidance_bonus": 0.0,
        "guidance_bonus_reward_fraction": 0.0,
        "total_steps": 0,
        "average_steps": 0.0,
        "median_steps": 0.0,
        "steps_std": 0.0,
        "min_steps": 0.0,
        "max_steps": 0.0,
        "reward_per_step": 0.0,
        "successful_actions": 0,
        "failed_actions": 0,
        "no_op_actions": 0,
        "action_success_rate": 0.0,
        "average_failed_actions": 0.0,
        "average_no_op_actions": 0.0,
        "environment_errors": 0,
        "average_execution_ms": 0.0,
        "median_execution_ms": 0.0,
        "execution_ms_std": 0.0,
        "total_execution_seconds": 0.0,
        "successful_episode_average_reward": 0.0,
        "failed_episode_average_reward": 0.0,
        "successful_episode_average_steps": 0.0,
        "failed_episode_average_steps": 0.0,
        "first_100_success_rate": 0.0,
        "last_100_success_rate": 0.0,
        "first_100_average_reward": 0.0,
        "last_100_average_reward": 0.0,
        "first_100_average_steps": 0.0,
        "last_100_average_steps": 0.0,
    }


def calculate_episode_summary(dataframe):
    if dataframe.empty:
        return _empty_episode_summary()

    total_episodes = len(dataframe)
    valid_df = _valid_episode_dataframe(dataframe)
    environment_error_episodes = total_episodes - len(valid_df)

    total_environment_errors = int(
        pd.to_numeric(
            dataframe.get(
                "environmentErrors",
                pd.Series(0, index=dataframe.index),
            ),
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    if valid_df.empty:
        summary = _empty_episode_summary(
            total_episodes=total_episodes,
            environment_error_episodes=environment_error_episodes,
        )
        summary["environment_errors"] = total_environment_errors
        return summary

    def numeric(column, default=0.0):
        if column not in valid_df.columns:
            return pd.Series(default, index=valid_df.index, dtype=float)
        return pd.to_numeric(valid_df[column], errors="coerce").fillna(default)

    rewards = numeric("totalReward")
    base_rewards = numeric("totalBaseReward")
    guidance = numeric("totalGuidanceBonus")
    steps = numeric("totalSteps")
    successful_actions = numeric("successfulActions")
    failed_actions = numeric("failedActions")
    no_ops = numeric("noOpActions")
    execution = numeric("executionTimeMs")
    completed_series = valid_df["completed"].astype(bool)

    completed = int(completed_series.sum())
    action_total = float(successful_actions.sum() + failed_actions.sum())
    reward_total = float(rewards.sum())
    step_total = float(steps.sum())
    guidance_total = float(guidance.sum())

    successful_frame = valid_df.loc[completed_series]
    failed_frame = valid_df.loc[~completed_series]

    def frame_mean(frame, column):
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())

    first_100 = valid_df.head(min(100, len(valid_df)))
    last_100 = valid_df.tail(min(100, len(valid_df)))

    def success_rate(frame):
        if frame.empty:
            return 0.0
        return float(frame["completed"].astype(bool).mean())

    def average(frame, column):
        if frame.empty or column not in frame.columns:
            return 0.0
        return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())

    return {
        "episodes": total_episodes,
        "valid_episodes": len(valid_df),
        "environment_error_episodes": environment_error_episodes,
        "completed": completed,
        "success_rate": completed / len(valid_df),
        "total_reward": reward_total,
        "average_reward": float(rewards.mean()),
        "median_reward": float(rewards.median()),
        "reward_std": float(rewards.std(ddof=0)),
        "best_reward": float(rewards.max()),
        "worst_reward": float(rewards.min()),
        "total_base_reward": float(base_rewards.sum()),
        "average_base_reward": float(base_rewards.mean()),
        "total_guidance_bonus": guidance_total,
        "average_guidance_bonus": float(guidance.mean()),
        "guidance_bonus_reward_fraction": (
            guidance_total / reward_total if abs(reward_total) > 1e-12 else 0.0
        ),
        "total_steps": int(step_total),
        "average_steps": float(steps.mean()),
        "median_steps": float(steps.median()),
        "steps_std": float(steps.std(ddof=0)),
        "min_steps": float(steps.min()),
        "max_steps": float(steps.max()),
        "reward_per_step": reward_total / step_total if step_total > 0 else 0.0,
        "successful_actions": int(successful_actions.sum()),
        "failed_actions": int(failed_actions.sum()),
        "no_op_actions": int(no_ops.sum()),
        "action_success_rate": (
            float(successful_actions.sum()) / action_total if action_total > 0 else 0.0
        ),
        "average_failed_actions": float(failed_actions.mean()),
        "average_no_op_actions": float(no_ops.mean()),
        "environment_errors": total_environment_errors,
        "average_execution_ms": float(execution.mean()),
        "median_execution_ms": float(execution.median()),
        "execution_ms_std": float(execution.std(ddof=0)),
        "total_execution_seconds": float(execution.sum() / 1000.0),
        "successful_episode_average_reward": frame_mean(
            successful_frame, "totalReward"
        ),
        "failed_episode_average_reward": frame_mean(failed_frame, "totalReward"),
        "successful_episode_average_steps": frame_mean(successful_frame, "totalSteps"),
        "failed_episode_average_steps": frame_mean(failed_frame, "totalSteps"),
        "first_100_success_rate": success_rate(first_100),
        "last_100_success_rate": success_rate(last_100),
        "first_100_average_reward": average(first_100, "totalReward"),
        "last_100_average_reward": average(last_100, "totalReward"),
        "first_100_average_steps": average(first_100, "totalSteps"),
        "last_100_average_steps": average(last_100, "totalSteps"),
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
# Procedure Action Summary
# ==========================================================


def build_procedure_action_summary(steps_df):
    """Detailed procedure-adherence metrics by phase and action."""

    columns = [
        "phase",
        "action",
        "procedure_attempts",
        "procedure_followed",
        "procedure_not_followed",
        "procedure_adherence_rate",
        "successful",
        "useful",
        "success_rate",
        "useful_rate",
        "total_reward",
        "average_reward",
        "total_guidance_bonus",
        "average_guidance_bonus",
        "average_duration_ms",
    ]

    if steps_df.empty:
        return pd.DataFrame(columns=columns)

    valid_steps = _valid_step_dataframe(steps_df)
    guided = valid_steps[valid_steps["procedureFollowed"].notna()].copy()

    if guided.empty:
        return pd.DataFrame(columns=columns)

    guided["procedureFollowed"] = guided["procedureFollowed"].astype(bool)

    rows = []

    for (phase, action), frame in guided.groupby(
        ["phase", "action"],
        dropna=False,
    ):
        attempts = len(frame)
        followed = int(frame["procedureFollowed"].sum())
        successful = int(frame["success"].astype(bool).sum())
        useful = int(frame["usefulAction"].astype(bool).sum())

        rows.append(
            {
                "phase": phase,
                "action": action,
                "procedure_attempts": attempts,
                "procedure_followed": followed,
                "procedure_not_followed": attempts - followed,
                "procedure_adherence_rate": followed / attempts if attempts else 0.0,
                "successful": successful,
                "useful": useful,
                "success_rate": successful / attempts if attempts else 0.0,
                "useful_rate": useful / attempts if attempts else 0.0,
                "total_reward": float(frame["reward"].sum()),
                "average_reward": float(frame["reward"].mean()),
                "total_guidance_bonus": float(frame["guidanceBonus"].sum()),
                "average_guidance_bonus": float(frame["guidanceBonus"].mean()),
                "average_duration_ms": float(frame["durationMs"].mean()),
            }
        )

    return pd.DataFrame(rows, columns=columns)


# ==========================================================
# PPO Diagnostic Summary
# ==========================================================


def build_ppo_diagnostics(updates_df):
    """Summarize every numeric PPO update field, not just known losses.

    If PPOAgent later adds another numeric diagnostic to its update dictionary,
    it automatically appears here and in combined reports.
    """

    columns = [
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "first",
        "last",
        "change",
    ]

    if updates_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []

    for column in updates_df.columns:
        series = pd.to_numeric(updates_df[column], errors="coerce").dropna()

        if series.empty:
            continue

        rows.append(
            {
                "metric": column,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "first": float(series.iloc[0]),
                "last": float(series.iloc[-1]),
                "change": float(series.iloc[-1] - series.iloc[0]),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_metric_inventory(tables):
    rows = []

    for table_name, dataframe in tables.items():
        if not isinstance(dataframe, pd.DataFrame):
            continue

        if dataframe.empty and len(dataframe.columns) == 0:
            rows.append(
                {
                    "table": table_name,
                    "rows": 0,
                    "columns": 0,
                    "column": None,
                    "dtype": None,
                }
            )
            continue

        for column in dataframe.columns:
            rows.append(
                {
                    "table": table_name,
                    "rows": len(dataframe),
                    "columns": len(dataframe.columns),
                    "column": column,
                    "dtype": str(dataframe[column].dtype),
                }
            )

    return pd.DataFrame(rows)


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
    training_local_episodes=None,
    evaluation_local_episodes=None,
    agent_label=None,
    runtime_values=None,
):
    """Build the complete individual-run metric bundle.

    The bundle intentionally contains both authoritative backend records and
    local runtime records. This preserves procedure, planner, reward-component,
    PPO, entropy and timing diagnostics for later re-analysis.
    """

    training_df = episodes_to_dataframe(training_episodes)
    evaluation_df = episodes_to_dataframe(evaluation_episodes)
    all_episodes = list(training_episodes or []) + list(evaluation_episodes or [])
    steps_df = steps_to_dataframe(all_episodes)
    updates_df = records_to_dataframe(ppo_updates)
    local_training_df = records_to_dataframe(training_local_episodes)
    local_evaluation_df = records_to_dataframe(evaluation_local_episodes)

    training_summary = calculate_episode_summary(training_df)
    evaluation_summary = calculate_episode_summary(evaluation_df)

    training_summary["convergence_episode"] = calculate_convergence_episode(training_df)
    training_summary["wall_clock_seconds"] = float(training_time or 0.0)
    evaluation_summary["wall_clock_seconds"] = float(evaluation_time or 0.0)

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
                    total_base_reward=("baseReward", "sum"),
                    average_base_reward=("baseReward", "mean"),
                    total_guidance_bonus=("guidanceBonus", "sum"),
                    average_guidance_bonus=("guidanceBonus", "mean"),
                    average_duration_ms=("durationMs", "mean"),
                )
                .reset_index()
            )
        else:
            action_summary = pd.DataFrame()

        if "environmentError" in steps_df.columns:
            environment_error_steps = (
                steps_df.groupby(
                    ["phase", "action"],
                    dropna=False,
                )["environmentError"]
                .sum()
                .reset_index(name="environment_errors")
            )

            if action_summary.empty:
                action_summary = environment_error_steps
            else:
                action_summary = action_summary.merge(
                    environment_error_steps,
                    on=["phase", "action"],
                    how="outer",
                )
    else:
        action_summary = pd.DataFrame()

    combined_df = pd.concat(
        [training_df, evaluation_df],
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

    guidance_summary = build_guidance_summary(steps_df)
    procedure_action_summary = build_procedure_action_summary(steps_df)
    llm_metrics_df = build_llm_metrics_dataframe(
        training_llm_metrics,
        evaluation_llm_metrics,
    )
    ppo_diagnostics = build_ppo_diagnostics(updates_df)

    tables = {
        "summary": summary_df,
        "training_episodes": training_df,
        "evaluation_episodes": evaluation_df,
        "local_training_episodes": local_training_df,
        "local_evaluation_episodes": local_evaluation_df,
        "steps": steps_df,
        "ppo_updates": updates_df,
        "ppo_diagnostics": ppo_diagnostics,
        "action_summary": action_summary,
        "termination_summary": termination_summary,
        "guidance_summary": guidance_summary,
        "procedure_action_summary": procedure_action_summary,
        "llm_metrics": llm_metrics_df,
        "configuration": config_dataframe(
            agent_label=agent_label,
            runtime_values=runtime_values,
        ),
    }

    tables["metric_inventory"] = build_metric_inventory(tables)

    return tables


# ==========================================================
# Configuration Snapshot
# ==========================================================


def config_dataframe(agent_label=None, runtime_values=None):
    runtime_values = runtime_values or {}

    values = {
        "RUN_AGENT": agent_label,
        "CONFIG_AGENT_TYPE": config.agent.AGENT_TYPE,
        "ALGORITHM": config.agent.ALGORITHM,
        "TASK": config.agent.TASK,
        "DEVICE": config.agent.DEVICE,
        "EXPERIMENT_NAME": config.experiment.EXPERIMENT_NAME,
        "EXPERIMENT_DESCRIPTION": config.experiment.DESCRIPTION,
        "GUIDANCE_MODE": config.experiment.GUIDANCE_MODE,
        "GUIDANCE_BONUS": config.experiment.GUIDANCE_BONUS,
        "ENVIRONMENT_VERSION": getattr(
            config.experiment,
            "ENVIRONMENT_VERSION",
            None,
        ),
        "EXPERIMENT_SUITE_NAME": getattr(
            config.experiment,
            "SUITE_NAME",
            None,
        ),
        "EXPERIMENT_SEEDS": ",".join(
            str(seed) for seed in getattr(config.experiment, "SEEDS", [])
        ),
        "LLM_MODEL": config.llm.MODEL,
        "LLM_BASE_URL": config.llm.BASE_URL,
        "LLM_TIMEOUT": config.llm.TIMEOUT,
        "LLM_TEMPERATURE": config.llm.TEMPERATURE,
        "LLM_USE_CACHE": config.llm.USE_CACHE,
        "LLM_CACHE_DIR": str(config.llm.CACHE_DIR),
        "TOTAL_EPISODES": runtime_values.get(
            "TOTAL_EPISODES",
            config.training.TOTAL_EPISODES,
        ),
        "EVALUATION_EPISODES": runtime_values.get(
            "EVALUATION_EPISODES",
            config.training.EVALUATION_EPISODES,
        ),
        "GAMMA": config.training.GAMMA,
        "GAE_LAMBDA": config.training.GAE_LAMBDA,
        "CLIP_EPSILON": config.training.CLIP_EPSILON,
        "LEARNING_RATE": config.training.LEARNING_RATE,
        "BATCH_SIZE": config.training.BATCH_SIZE,
        "UPDATE_INTERVAL": config.training.UPDATE_INTERVAL,
        "EPOCHS": config.training.EPOCHS,
        "HIDDEN_NEURON_SIZE": config.training.HIDDEN_NEURON_SIZE,
        "ENTROPY_COEF": getattr(config.training, "ENTROPY_COEF", None),
        "MAX_GRAD_NORM": getattr(config.training, "MAX_GRAD_NORM", None),
        "SAVE_EVERY": config.training.SAVE_EVERY,
        "LOG_EVERY": config.training.LOG_EVERY,
        "MOVING_AVERAGE_WINDOW": config.training.MOVING_AVERAGE_WINDOW,
        "CONVERGENCE_WINDOW": config.training.CONVERGENCE_WINDOW,
        "CONVERGENCE_SUCCESS_THRESHOLD": (
            config.training.CONVERGENCE_SUCCESS_THRESHOLD
        ),
        "ENV_NAME": config.environment.ENV_NAME,
        "OBSERVATION_TYPE": config.environment.OBSERVATION_TYPE,
        "MAX_STEPS_PER_EPISODE": config.environment.MAX_STEPS_PER_EPISODE,
        "RANDOM_SEED": runtime_values.get(
            "RANDOM_SEED",
            config.environment.RANDOM_SEED,
        ),
        "ACTION_SPACE_SIZE": config.environment.ACTION_SPACE_SIZE,
        "BACKEND_BASE_URL": config.backend.BASE_URL,
        "BACKEND_TIMEOUT": config.backend.TIMEOUT,
    }

    for key, value in runtime_values.items():
        values.setdefault(key, value)

    return pd.DataFrame(
        [
            {"parameter": key, "value": _excel_safe_value(value)}
            for key, value in values.items()
        ]
    )


# ==========================================================
# Excel
# ==========================================================


def _safe_sheet_name(name, used_names):
    base = str(name).replace("/", "_").replace("\\", "_")
    base = base[:31] or "Sheet"
    candidate = base
    counter = 2

    while candidate in used_names:
        suffix = f"_{counter}"
        candidate = f"{base[:31-len(suffix)]}{suffix}"
        counter += 1

    used_names.add(candidate)
    return candidate


def export_raw_run_data(
    output_directory,
    training_episodes,
    evaluation_episodes,
    training_result,
    evaluation_result,
    runtime_config=None,
):
    """Save lossless JSON inputs so later metrics can be recomputed.

    This is the main safeguard against having to rerun expensive experiments
    simply because a new derived metric is wanted later.
    """

    output_directory = Path(output_directory) / "raw"
    output_directory.mkdir(parents=True, exist_ok=True)

    payloads = {
        "backend_training_episodes.json": training_episodes or [],
        "backend_evaluation_episodes.json": evaluation_episodes or [],
        "training_runtime.json": training_result or {},
        "evaluation_runtime.json": evaluation_result or {},
        "runtime_config.json": runtime_config or {},
    }

    paths = {}

    for filename, payload in payloads.items():
        path = output_directory / filename
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, default=str)
        paths[filename] = str(path)

    return paths


def export_metrics_excel(
    tables,
    output_path,
    agent_label=None,
):
    """Export every DataFrame in ``tables``.

    The exporter is intentionally dynamic. New tables added to the metric bundle
    are automatically written rather than being silently omitted.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_names = {
        "summary": "Summary",
        "training_episodes": "Training Episodes",
        "evaluation_episodes": "Evaluation Episodes",
        "local_training_episodes": "Local Training",
        "local_evaluation_episodes": "Local Evaluation",
        "ppo_updates": "PPO Updates",
        "ppo_diagnostics": "PPO Diagnostics",
        "action_summary": "Action Summary",
        "termination_summary": "Termination Summary",
        "guidance_summary": "Guidance Summary",
        "procedure_action_summary": "Procedure Actions",
        "llm_metrics": "LLM Metrics",
        "steps": "Steps",
        "configuration": "Configuration",
        "metric_inventory": "Metric Inventory",
    }

    ordered_keys = list(preferred_names)
    ordered_keys.extend(key for key in tables.keys() if key not in preferred_names)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        used_names = set()

        for key in ordered_keys:
            dataframe = tables.get(key)
            if not isinstance(dataframe, pd.DataFrame):
                continue

            requested = preferred_names.get(
                key,
                str(key).replace("_", " ").title(),
            )
            sheet_name = _safe_sheet_name(requested, used_names)
            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        workbook = writer.book
        header_fill = PatternFill("solid", fgColor="D9EAF7")
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
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[column_letter].width = min(
                    max(max_length + 2, 12),
                    50,
                )

    return str(output_path)
