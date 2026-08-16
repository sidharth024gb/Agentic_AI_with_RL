"""
comparison_metrics.py

Multi-run / multi-seed reporting for the Finance PPO experiment suite.

Design goals
------------
1. Preserve every DataFrame produced by an individual run.
2. Add run/seed/condition metadata to every combined row.
3. Build a one-row-per-run metric matrix for easy dissertation comparison.
4. Aggregate every numeric run metric across seeds (mean/std/min/max/median).
5. Keep the exporter schema-flexible so later individual metric tables are not
   silently omitted from combined workbooks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

# ==========================================================
# Helpers
# ==========================================================


def _safe_value(value):
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, sort_keys=True)
    return value


def _run_metadata(run_result):
    runtime = run_result.get("runtime_config", {}) or {}

    return {
        "suiteName": run_result.get("suite_name"),
        "runSeed": run_result.get(
            "seed",
            runtime.get("seed"),
        ),
        "condition": run_result.get("condition"),
        "conditionLabel": run_result.get("condition_label"),
        "runName": run_result.get("run_name"),
        "agent": run_result.get("agent"),
        "agentType": runtime.get("agent_type"),
        "algorithm": runtime.get("algorithm"),
        "guidanceMode": runtime.get("guidance_mode"),
        "guidanceBonus": runtime.get("guidance_bonus"),
        "experimentName": runtime.get("experiment_name"),
    }


def _prepend_metadata(dataframe, metadata):
    if dataframe is None:
        dataframe = pd.DataFrame()

    dataframe = dataframe.copy()

    for key in reversed(list(metadata.keys())):
        if key in dataframe.columns:
            continue
        dataframe.insert(0, key, metadata.get(key))

    return dataframe


def _safe_sheet_name(name, used_names):
    invalid = set("[]:*?/\\")
    cleaned = "".join(
        "_" if character in invalid else character for character in str(name)
    )
    cleaned = cleaned[:31] or "Sheet"

    candidate = cleaned
    index = 2

    while candidate in used_names:
        suffix = f"_{index}"
        candidate = f"{cleaned[:31-len(suffix)]}{suffix}"
        index += 1

    used_names.add(candidate)
    return candidate


def _style_workbook(workbook):
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
                55,
            )


# ==========================================================
# Preserve Every Individual Table
# ==========================================================


def combine_all_run_tables(run_results):
    """Concatenate every individual run table by table name.

    This is intentionally dynamic. If ``metrics.py`` later adds another table,
    the combined report receives it automatically as long as the suite was run
    with that version of the code.
    """

    buckets: Dict[str, List[pd.DataFrame]] = {}

    for run_result in run_results:
        metadata = _run_metadata(run_result)
        tables = run_result.get("tables", {}) or {}

        for table_name, dataframe in tables.items():
            if not isinstance(dataframe, pd.DataFrame):
                continue

            buckets.setdefault(table_name, []).append(
                _prepend_metadata(dataframe, metadata)
            )

    combined = {}

    for table_name, frames in buckets.items():
        if not frames:
            combined[table_name] = pd.DataFrame()
            continue

        combined[table_name] = pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

    return combined


# ==========================================================
# One Row Per Run
# ==========================================================


def _add_long_metric_table(row, dataframe, prefix, phase_columns=True):
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        return

    if "metric" not in dataframe.columns:
        return

    for _, metric_row in dataframe.iterrows():
        metric = str(metric_row.get("metric"))

        if phase_columns:
            for phase in ("training", "evaluation"):
                if phase in dataframe.columns:
                    row[f"{prefix}_{phase}_{metric}"] = metric_row.get(phase)
        else:
            for column in dataframe.columns:
                if column == "metric":
                    continue
                row[f"{prefix}_{metric}_{column}"] = metric_row.get(column)


def build_run_metric_matrix(run_results):
    """Create one wide row per experiment run with every summary metric."""

    rows = []

    for run_result in run_results:
        row = dict(_run_metadata(run_result))
        row["seed"] = row.get("runSeed")
        tables = run_result.get("tables", {}) or {}

        row["metricsExcel"] = run_result.get("metrics_excel")
        row["finalCheckpoint"] = (run_result.get("training") or {}).get(
            "final_checkpoint"
        )

        _add_long_metric_table(
            row,
            tables.get("summary"),
            prefix="summary",
            phase_columns=True,
        )

        # Guidance summary is phase-row based rather than metric-row based.
        guidance = tables.get("guidance_summary")
        if isinstance(guidance, pd.DataFrame) and not guidance.empty:
            for _, guidance_row in guidance.iterrows():
                phase = str(guidance_row.get("phase", "unknown")).lower()
                for column in guidance.columns:
                    if column == "phase":
                        continue
                    row[f"guidance_{phase}_{column}"] = guidance_row.get(column)

        # LLM Metrics uses the same long metric format as Summary.
        _add_long_metric_table(
            row,
            tables.get("llm_metrics"),
            prefix="llm",
            phase_columns=True,
        )

        # PPO diagnostics are metric rows with statistics columns.
        _add_long_metric_table(
            row,
            tables.get("ppo_diagnostics"),
            prefix="ppo",
            phase_columns=False,
        )

        # Configuration is parameter/value. Store every parameter.
        configuration = tables.get("configuration")
        if isinstance(configuration, pd.DataFrame) and not configuration.empty:
            if {"parameter", "value"}.issubset(configuration.columns):
                for _, config_row in configuration.iterrows():
                    parameter = str(config_row.get("parameter"))
                    row[f"config_{parameter}"] = config_row.get("value")

        rows.append({key: _safe_value(value) for key, value in row.items()})

    dataframe = pd.DataFrame(rows)

    if not dataframe.empty:
        preferred = [
            "suiteName",
            "seed",
            "runSeed",
            "condition",
            "conditionLabel",
            "runName",
            "agent",
            "agentType",
            "algorithm",
            "guidanceMode",
            "guidanceBonus",
            "experimentName",
        ]
        columns = [c for c in preferred if c in dataframe.columns]
        columns += [c for c in dataframe.columns if c not in columns]
        dataframe = dataframe[columns]

    return dataframe


# ==========================================================
# Across-Seed Numeric Aggregation
# ==========================================================


def build_numeric_aggregate(run_matrix):
    """Aggregate every numeric run-level metric for each condition."""

    columns = [
        "condition",
        "conditionLabel",
        "metric",
        "n",
        "mean",
        "std",
        "min",
        "max",
        "median",
    ]

    if run_matrix.empty:
        return pd.DataFrame(columns=columns)

    metadata_columns = {
        "suiteName",
        "seed",
        "runSeed",
        "condition",
        "conditionLabel",
        "runName",
        "agent",
        "agentType",
        "algorithm",
        "guidanceMode",
        "experimentName",
        "metricsExcel",
        "finalCheckpoint",
    }

    rows = []

    for condition, frame in run_matrix.groupby("condition", dropna=False):
        label = (
            frame["conditionLabel"].dropna().iloc[0]
            if "conditionLabel" in frame.columns
            and not frame["conditionLabel"].dropna().empty
            else condition
        )

        for column in frame.columns:
            if column in metadata_columns:
                continue

            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).dropna()

            if numeric.empty:
                continue

            rows.append(
                {
                    "condition": condition,
                    "conditionLabel": label,
                    "metric": column,
                    "n": int(numeric.count()),
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std(ddof=0)),
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "median": float(numeric.median()),
                }
            )

    return pd.DataFrame(rows, columns=columns)


# ==========================================================
# Paper-Friendly Core Summary
# ==========================================================


def build_publication_summary(run_matrix):
    """Compact mean ± SD table for the metrics most likely used in the paper."""

    metric_map = {
        "train_success_rate": "summary_training_success_rate",
        "eval_success_rate": "summary_evaluation_success_rate",
        "train_average_reward": "summary_training_average_reward",
        "eval_average_reward": "summary_evaluation_average_reward",
        "train_average_steps": "summary_training_average_steps",
        "eval_average_steps": "summary_evaluation_average_steps",
        "convergence_episode": "summary_training_convergence_episode",
        "train_last_100_success_rate": "summary_training_last_100_success_rate",
        "train_last_100_average_reward": "summary_training_last_100_average_reward",
        "train_last_100_average_steps": "summary_training_last_100_average_steps",
        "train_failed_actions": "summary_training_failed_actions",
        "train_no_op_actions": "summary_training_no_op_actions",
        "eval_failed_actions": "summary_evaluation_failed_actions",
        "eval_no_op_actions": "summary_evaluation_no_op_actions",
        "train_wall_clock_seconds": "summary_training_wall_clock_seconds",
        "eval_wall_clock_seconds": "summary_evaluation_wall_clock_seconds",
        "train_procedure_adherence": ("guidance_train_procedure_adherence_rate"),
        "eval_procedure_adherence": ("guidance_evaluation_procedure_adherence_rate"),
        "ppo_entropy_mean": "ppo_entropy_mean",
        "ppo_entropy_last": "ppo_entropy_last",
        "ppo_normalized_entropy_mean": "ppo_normalized_entropy_mean",
        "ppo_policy_loss_mean": "ppo_policy_loss_mean",
        "ppo_value_loss_mean": "ppo_value_loss_mean",
        "ppo_approx_kl_mean": "ppo_approx_kl_mean",
        "ppo_clip_fraction_mean": "ppo_clip_fraction_mean",
        "ppo_explained_variance_last": "ppo_explained_variance_last",
    }

    if run_matrix.empty:
        return pd.DataFrame()

    rows = []

    for condition, frame in run_matrix.groupby("condition", dropna=False):
        row = {
            "condition": condition,
            "conditionLabel": (
                frame["conditionLabel"].dropna().iloc[0]
                if "conditionLabel" in frame.columns
                and not frame["conditionLabel"].dropna().empty
                else condition
            ),
            "seeds": ",".join(
                str(int(seed))
                for seed in sorted(
                    pd.to_numeric(frame["seed"], errors="coerce").dropna().unique()
                )
            ),
            "n_runs": len(frame),
        }

        for output_name, column in metric_map.items():
            if column not in frame.columns:
                continue

            numeric = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).dropna()

            if numeric.empty:
                continue

            row[f"{output_name}_mean"] = float(numeric.mean())
            row[f"{output_name}_std"] = float(numeric.std(ddof=0))
            row[f"{output_name}_min"] = float(numeric.min())
            row[f"{output_name}_max"] = float(numeric.max())

        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================================
# Inventory
# ==========================================================


def build_combined_inventory(combined_tables):
    rows = []

    for name, dataframe in combined_tables.items():
        if not isinstance(dataframe, pd.DataFrame):
            continue

        for column in dataframe.columns:
            rows.append(
                {
                    "table": name,
                    "rows": len(dataframe),
                    "columns": len(dataframe.columns),
                    "column": column,
                    "dtype": str(dataframe[column].dtype),
                }
            )

    return pd.DataFrame(rows)


# ==========================================================
# Build / Export
# ==========================================================


def build_combined_metric_tables(run_results):
    combined = combine_all_run_tables(run_results)
    run_matrix = build_run_metric_matrix(run_results)
    numeric_aggregate = build_numeric_aggregate(run_matrix)
    publication_summary = build_publication_summary(run_matrix)

    # Put comparison sheets first, then every individual table family.
    result = {
        "run_matrix": run_matrix,
        "publication_summary": publication_summary,
        "numeric_aggregate": numeric_aggregate,
    }
    result.update(combined)
    result["combined_inventory"] = build_combined_inventory(result)
    return result


def export_combined_metrics(
    run_results,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tables = build_combined_metric_tables(run_results)

    preferred_names = {
        "run_matrix": "Run Matrix",
        "publication_summary": "Paper Summary",
        "numeric_aggregate": "All Numeric Aggregate",
        "summary": "All Summaries",
        "training_episodes": "Training Episodes",
        "evaluation_episodes": "Evaluation Episodes",
        "local_training_episodes": "Local Training",
        "local_evaluation_episodes": "Local Evaluation",
        "steps": "All Steps",
        "ppo_updates": "PPO Updates",
        "ppo_diagnostics": "PPO Diagnostics",
        "action_summary": "Action Summary",
        "termination_summary": "Termination Summary",
        "guidance_summary": "Guidance Summary",
        "procedure_action_summary": "Procedure Actions",
        "llm_metrics": "LLM Metrics",
        "configuration": "Configurations",
        "metric_inventory": "Run Metric Inventory",
        "combined_inventory": "Combined Inventory",
    }

    ordered = list(preferred_names)
    ordered.extend(key for key in tables.keys() if key not in preferred_names)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        used_names = set()

        for table_name in ordered:
            dataframe = tables.get(table_name)
            if not isinstance(dataframe, pd.DataFrame):
                continue

            requested = preferred_names.get(
                table_name,
                str(table_name).replace("_", " ").title(),
            )
            sheet_name = _safe_sheet_name(requested, used_names)
            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

        _style_workbook(writer.book)

    return str(output_path), tables
