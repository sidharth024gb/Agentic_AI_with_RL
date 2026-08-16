"""
comparison_visualization.py

Comparison figures for per-seed and cross-seed experiment suites.

The individual visualization module remains responsible for one run. This module
focuses on PPO vs the three LLM ablations and, when multiple seeds are present,
plots mean trajectories with standard-deviation bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.config import config
from utils.comparison_metrics import build_run_metric_matrix

# ==========================================================
# Helpers
# ==========================================================


def _save(figure, output_directory, filename):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return str(path)


def _label(run_result):
    return run_result.get("condition_label") or run_result.get("condition")


def _valid_episodes(dataframe):
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    frame = dataframe.copy()

    if "validAgentEpisode" in frame.columns:
        frame = frame[frame["validAgentEpisode"].astype(bool)]

    return frame


def _run_episode_column(dataframe):
    if "runEpisode" in dataframe.columns:
        return "runEpisode"
    if "episode" in dataframe.columns:
        return "episode"
    return "episodeNumber"


def _condition_curves(
    run_results,
    table_name,
    value_column,
    rolling_window=None,
):
    rows = []

    for run in run_results:
        tables = run.get("tables", {}) or {}
        dataframe = tables.get(table_name)

        if dataframe is None or dataframe.empty:
            continue

        if table_name in {"training_episodes", "evaluation_episodes"}:
            dataframe = _valid_episodes(dataframe)

        if dataframe.empty or value_column not in dataframe.columns:
            continue

        episode_column = _run_episode_column(dataframe)
        frame = dataframe[[episode_column, value_column]].copy()
        frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
        frame = frame.dropna(subset=[value_column])

        if rolling_window:
            frame[value_column] = (
                frame[value_column].rolling(rolling_window, min_periods=1).mean()
            )

        frame["seed"] = run.get("seed")
        frame["condition"] = run.get("condition")
        frame["conditionLabel"] = _label(run)
        frame = frame.rename(columns={episode_column: "x", value_column: "y"})
        rows.append(frame)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def _plot_condition_curves(
    curve_df,
    output_directory,
    filename,
    title,
    ylabel,
):
    if curve_df.empty:
        return None

    figure = plt.figure(figsize=(11, 6))

    for condition, frame in curve_df.groupby("condition", dropna=False):
        label = frame["conditionLabel"].dropna().iloc[0]
        grouped = frame.groupby("x")["y"]
        mean = grouped.mean()
        std = grouped.std(ddof=0).fillna(0.0)

        plt.plot(mean.index, mean.values, label=label)

        # A one-seed comparison naturally has a zero-width band.
        if frame["seed"].nunique() > 1:
            plt.fill_between(
                mean.index,
                mean - std,
                mean + std,
                alpha=0.15,
            )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.25)
    return _save(figure, output_directory, filename)


def _metric_from_summary(run, metric, phase):
    summary = (run.get("tables") or {}).get("summary")

    if summary is None or summary.empty:
        return None

    matched = summary[summary["metric"] == metric]
    if matched.empty or phase not in matched.columns:
        return None

    return matched.iloc[0][phase]


def _bar_metric(
    run_results,
    metric,
    phase,
    output_directory,
    filename,
    title,
    ylabel,
):
    rows = []

    for run in run_results:
        value = _metric_from_summary(run, metric, phase)
        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

        if pd.isna(value):
            continue

        rows.append(
            {
                "condition": run.get("condition"),
                "conditionLabel": _label(run),
                "seed": run.get("seed"),
                "value": float(value),
            }
        )

    if not rows:
        return None

    dataframe = pd.DataFrame(rows)
    aggregate = (
        dataframe.groupby(["condition", "conditionLabel"], dropna=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate["std"] = aggregate["std"].fillna(0.0)

    figure = plt.figure(figsize=(10, 6))
    plt.bar(
        aggregate["conditionLabel"],
        aggregate["mean"],
        yerr=aggregate["std"],
        capsize=4,
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return _save(figure, output_directory, filename)


def _ppo_metric_curves(run_results, metric, output_directory, filename, title, ylabel):
    rows = []

    for run in run_results:
        updates = (run.get("tables") or {}).get("ppo_updates")
        if updates is None or updates.empty or metric not in updates.columns:
            continue

        x_column = "episode" if "episode" in updates.columns else "update"
        frame = updates[[x_column, metric]].copy()
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        frame = frame.dropna(subset=[metric])
        frame["seed"] = run.get("seed")
        frame["condition"] = run.get("condition")
        frame["conditionLabel"] = _label(run)
        frame = frame.rename(columns={x_column: "x", metric: "y"})
        rows.append(frame)

    if not rows:
        return None

    curve_df = pd.concat(rows, ignore_index=True)
    return _plot_condition_curves(
        curve_df,
        output_directory,
        filename,
        title,
        ylabel,
    )


def _guidance_adherence_bar(run_results, output_directory):
    rows = []

    for run in run_results:
        guidance = (run.get("tables") or {}).get("guidance_summary")
        if guidance is None or guidance.empty:
            continue

        for _, row in guidance.iterrows():
            rows.append(
                {
                    "condition": run.get("condition"),
                    "conditionLabel": _label(run),
                    "seed": run.get("seed"),
                    "phase": str(row.get("phase")),
                    "value": pd.to_numeric(
                        pd.Series([row.get("procedure_adherence_rate")]),
                        errors="coerce",
                    ).iloc[0],
                }
            )

    dataframe = pd.DataFrame(rows).dropna(subset=["value"]) if rows else pd.DataFrame()
    if dataframe.empty:
        return None

    aggregate = (
        dataframe.groupby(
            ["condition", "conditionLabel", "phase"],
            dropna=False,
        )["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate["std"] = aggregate["std"].fillna(0.0)

    pivot = aggregate.pivot(
        index="conditionLabel",
        columns="phase",
        values="mean",
    ).fillna(0.0)

    figure = plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", ax=plt.gca())
    plt.title("LLM Procedure Adherence Comparison")
    plt.ylabel("Procedure adherence rate")
    plt.xlabel("Experiment")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return _save(
        figure,
        output_directory,
        "procedure_adherence_comparison.png",
    )


def _llm_metric_bar(run_results, metric, output_directory):
    rows = []

    for run in run_results:
        table = (run.get("tables") or {}).get("llm_metrics")
        if table is None or table.empty:
            continue

        matched = table[table["metric"] == metric]
        if matched.empty:
            continue

        for phase in ("training", "evaluation"):
            value = matched.iloc[0].get(phase)
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric):
                continue
            rows.append(
                {
                    "condition": run.get("condition"),
                    "conditionLabel": _label(run),
                    "seed": run.get("seed"),
                    "phase": phase,
                    "value": float(numeric),
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return None

    aggregate = (
        dataframe.groupby(["conditionLabel", "phase"], dropna=False)["value"]
        .mean()
        .unstack(fill_value=0.0)
    )

    figure = plt.figure(figsize=(10, 6))
    aggregate.plot(kind="bar", ax=plt.gca())
    plt.title(f"LLM Metric: {metric}")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xlabel("Experiment")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return _save(
        figure,
        output_directory,
        f"llm_{metric}.png",
    )


def _action_frequency(run_results, output_directory, phase="TRAIN"):
    rows = []

    for run in run_results:
        table = (run.get("tables") or {}).get("action_summary")
        if table is None or table.empty:
            continue

        frame = table.copy()
        if "phase" in frame.columns:
            frame = frame[frame["phase"].astype(str).str.upper() == phase]

        for _, row in frame.iterrows():
            rows.append(
                {
                    "conditionLabel": _label(run),
                    "seed": run.get("seed"),
                    "action": row.get("action"),
                    "count": pd.to_numeric(
                        pd.Series([row.get("count")]), errors="coerce"
                    )
                    .fillna(0.0)
                    .iloc[0],
                }
            )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return None

    per_run_total = dataframe.groupby(["conditionLabel", "seed"], dropna=False)[
        "count"
    ].transform("sum")
    dataframe["frequency"] = dataframe["count"] / per_run_total.replace(0, 1)

    aggregate = (
        dataframe.groupby(["action", "conditionLabel"], dropna=False)["frequency"]
        .mean()
        .unstack(fill_value=0.0)
    )

    figure = plt.figure(figsize=(12, 7))
    aggregate.plot(kind="bar", ax=plt.gca())
    plt.title(f"{phase.title()} Action Frequency Comparison")
    plt.ylabel("Mean fraction of actions")
    plt.xlabel("Action")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    return _save(
        figure,
        output_directory,
        f"{phase.lower()}_action_frequency_comparison.png",
    )


# ==========================================================
# Public API
# ==========================================================


def generate_combined_visualizations(
    run_results,
    output_directory,
    scope_name="combined",
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {}

    window = config.training.MOVING_AVERAGE_WINDOW

    curve_specs = [
        (
            "training_episodes",
            "totalReward",
            window,
            "training_reward_comparison.png",
            f"{scope_name}: Training Reward",
            "Rolling mean episode reward",
        ),
        (
            "training_episodes",
            "completed",
            window,
            "training_success_comparison.png",
            f"{scope_name}: Training Success",
            "Rolling success rate",
        ),
        (
            "training_episodes",
            "totalSteps",
            window,
            "training_steps_comparison.png",
            f"{scope_name}: Training Episode Length",
            "Rolling mean steps",
        ),
        (
            "evaluation_episodes",
            "totalReward",
            None,
            "evaluation_reward_comparison.png",
            f"{scope_name}: Deterministic Evaluation Reward",
            "Episode reward",
        ),
    ]

    for table_name, value, rolling, filename, title, ylabel in curve_specs:
        curve = _condition_curves(
            run_results,
            table_name,
            value,
            rolling_window=rolling,
        )
        path = _plot_condition_curves(
            curve,
            output_directory,
            filename,
            title,
            ylabel,
        )
        if path:
            paths[filename] = path

    bar_specs = [
        (
            "success_rate",
            "training",
            "training_success_bar.png",
            "Training Success Rate",
            "Success rate",
        ),
        (
            "success_rate",
            "evaluation",
            "evaluation_success_bar.png",
            "Deterministic Evaluation Success Rate",
            "Success rate",
        ),
        (
            "last_100_success_rate",
            "training",
            "training_last_100_success_bar.png",
            "Final-100 Training Success Rate",
            "Success rate",
        ),
        (
            "average_reward",
            "training",
            "training_average_reward_bar.png",
            "Average Training Reward",
            "Average reward",
        ),
        (
            "average_reward",
            "evaluation",
            "evaluation_average_reward_bar.png",
            "Average Evaluation Reward",
            "Average reward",
        ),
        (
            "average_steps",
            "evaluation",
            "evaluation_steps_bar.png",
            "Average Evaluation Episode Length",
            "Average steps",
        ),
        (
            "convergence_episode",
            "training",
            "convergence_episode_bar.png",
            "First Attainment of Convergence Criterion",
            "Run episode",
        ),
        (
            "wall_clock_seconds",
            "training",
            "training_wall_clock_bar.png",
            "Training Wall-Clock Time",
            "Seconds",
        ),
    ]

    for metric, phase, filename, title, ylabel in bar_specs:
        path = _bar_metric(
            run_results,
            metric,
            phase,
            output_directory,
            filename,
            title,
            ylabel,
        )
        if path:
            paths[filename] = path

    ppo_specs = [
        ("policy_loss", "ppo_policy_loss.png", "PPO Policy Loss", "Policy loss"),
        ("value_loss", "ppo_value_loss.png", "PPO Value Loss", "Value loss"),
        ("entropy", "ppo_entropy.png", "PPO Policy Entropy", "Entropy"),
        (
            "normalized_entropy",
            "ppo_normalized_entropy.png",
            "PPO Normalized Policy Entropy",
            "Normalized entropy",
        ),
        ("approx_kl", "ppo_approx_kl.png", "PPO Approximate KL", "Approx. KL"),
        (
            "clip_fraction",
            "ppo_clip_fraction.png",
            "PPO Clip Fraction",
            "Clip fraction",
        ),
        (
            "explained_variance",
            "ppo_explained_variance.png",
            "Critic Explained Variance",
            "Explained variance",
        ),
    ]

    for metric, filename, title, ylabel in ppo_specs:
        path = _ppo_metric_curves(
            run_results,
            metric,
            output_directory,
            filename,
            title,
            ylabel,
        )
        if path:
            paths[filename] = path

    for key, path in {
        "procedure_adherence": _guidance_adherence_bar(run_results, output_directory),
        "train_action_frequency": _action_frequency(
            run_results, output_directory, phase="TRAIN"
        ),
        "eval_action_frequency": _action_frequency(
            run_results, output_directory, phase="EVALUATION"
        ),
        "llm_average_latency": _llm_metric_bar(
            run_results,
            "average_llm_latency_ms",
            output_directory,
        ),
        "llm_total_latency": _llm_metric_bar(
            run_results,
            "total_llm_latency_ms",
            output_directory,
        ),
        "llm_cache_hits": _llm_metric_bar(
            run_results,
            "cache_hits",
            output_directory,
        ),
    }.items():
        if path:
            paths[key] = path

    # Saving the wide matrix beside the figures is handy for plotting elsewhere.
    matrix = build_run_metric_matrix(run_results)
    if not matrix.empty:
        csv_path = output_directory / "combined_run_matrix.csv"
        matrix.to_csv(csv_path, index=False)
        paths["run_matrix_csv"] = str(csv_path)

    return paths
