"""
experiment_suite.py

Runs the complete four-condition PPO/LLM ablation across multiple seeds and
creates both per-seed and all-seed comparison outputs.

The single-run pipeline remains owned by main.py. To avoid circular imports,
``run_experiment_suite`` receives that function as ``run_training_fn``.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.config import config
from utils.logger import (
    close_logger,
    create_run_name,
    get_run_logger,
    safe_name,
)
from utils.comparison_metrics import export_combined_metrics
from utils.comparison_visualization import generate_combined_visualizations

EXPERIMENT_CONDITIONS = [
    {
        "key": "ppo_baseline",
        "label": "PPO Baseline",
        "agent": "ppo",
        "guidance_mode": "none",
        "uses_guidance_bonus": False,
    },
    {
        "key": "llm_ppo_input",
        "label": "LLM Input",
        "agent": "llm_rl",
        "guidance_mode": "input",
        "uses_guidance_bonus": False,
    },
    {
        "key": "llm_ppo_reward",
        "label": "LLM Reward Shaping",
        "agent": "llm_rl",
        "guidance_mode": "reward_shaping",
        "uses_guidance_bonus": True,
    },
    {
        "key": "llm_ppo_input_reward",
        "label": "LLM Input + Reward",
        "agent": "llm_rl",
        "guidance_mode": "input_and_reward",
        "uses_guidance_bonus": True,
    },
]


# ==========================================================
# Directories / Manifest
# ==========================================================


def _prepare_suite_directories(suite_run_name):
    roots = {
        "logs": Path(config.logging.LOG_DIR),
        "metrics": Path(config.logging.METRICS_DIR),
        "graphs": Path(config.logging.GRAPH_DIR),
    }

    directories = {}

    for key, root in roots.items():
        path = root / "combined" / safe_name(suite_run_name)
        path.mkdir(parents=True, exist_ok=True)
        directories[key] = path

    return directories


def _scope_directories(suite_directories, scope):
    result = {}

    for key, root in suite_directories.items():
        path = Path(root) / safe_name(scope)
        path.mkdir(parents=True, exist_ok=True)
        result[key] = path

    return result


def _json_safe_run(run):
    return {
        "suite_name": run.get("suite_name"),
        "seed": run.get("seed"),
        "condition": run.get("condition"),
        "condition_label": run.get("condition_label"),
        "run_name": run.get("run_name"),
        "agent": run.get("agent"),
        "runtime_config": run.get("runtime_config"),
        "metrics_excel": run.get("metrics_excel"),
        "graphs": run.get("graphs"),
        "directories": run.get("directories"),
        "raw_data": run.get("raw_data"),
        "final_checkpoint": ((run.get("training") or {}).get("final_checkpoint")),
    }


def _write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)

    return str(path)


# ==========================================================
# Comparison Outputs
# ==========================================================


def _generate_scope_outputs(
    run_results,
    directories,
    scope_name,
):
    excel_path = (
        Path(directories["metrics"]) / f"{safe_name(scope_name)}_comparison.xlsx"
    )

    excel_path, combined_tables = export_combined_metrics(
        run_results=run_results,
        output_path=excel_path,
    )

    graph_paths = generate_combined_visualizations(
        run_results=run_results,
        output_directory=directories["graphs"],
        scope_name=scope_name,
    )

    manifest_path = _write_manifest(
        Path(directories["logs"]) / "manifest.json",
        {
            "scope": scope_name,
            "runs": [_json_safe_run(run) for run in run_results],
            "combined_metrics": excel_path,
            "combined_graphs": graph_paths,
            "combined_table_names": list(combined_tables.keys()),
        },
    )

    return {
        "metrics_excel": excel_path,
        "graphs": graph_paths,
        "manifest": manifest_path,
    }


# ==========================================================
# Suite Runner
# ==========================================================


def run_experiment_suite(
    run_training_fn,
    seeds=None,
    total_episodes=None,
    evaluation_episodes=None,
    guidance_bonus=None,
    suite_name=None,
    continue_on_error=False,
):
    """Run PPO + three LLM ablations for every supplied seed."""

    seeds = list(seeds or config.experiment.SEEDS)

    if not seeds:
        raise ValueError("At least one experiment seed is required.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment seeds must be unique.")

    resolved_bonus = (
        float(guidance_bonus)
        if guidance_bonus is not None
        else float(config.experiment.GUIDANCE_BONUS)
    )

    suite_base_name = suite_name or config.experiment.SUITE_NAME
    suite_run_name = create_run_name(suite_base_name)
    suite_directories = _prepare_suite_directories(suite_run_name)
    all_scope = _scope_directories(suite_directories, "all_seeds")

    suite_logger = get_run_logger(
        run_name=f"suite_{suite_run_name}",
        log_directory=all_scope["logs"],
    )

    previous_seed = config.environment.RANDOM_SEED
    previous_suite_name = config.experiment.SUITE_NAME
    config.experiment.SUITE_NAME = suite_run_name

    all_runs = []
    seed_outputs = {}
    failures = []

    try:
        suite_logger.info("========================================")
        suite_logger.info("STARTING EXPERIMENT SUITE: %s", suite_run_name)
        suite_logger.info("Seeds: %s", seeds)
        suite_logger.info("Conditions: %s", [c["key"] for c in EXPERIMENT_CONDITIONS])
        suite_logger.info(
            "Training episodes: %s", total_episodes or config.training.TOTAL_EPISODES
        )
        suite_logger.info(
            "Evaluation episodes: %s",
            evaluation_episodes or config.training.EVALUATION_EPISODES,
        )
        suite_logger.info("Guidance bonus: %s", resolved_bonus)
        suite_logger.info("========================================")

        for seed in seeds:
            seed = int(seed)
            seed_runs = []
            seed_scope = _scope_directories(
                suite_directories,
                f"seed_{seed}",
            )

            suite_logger.info("Starting seed %s", seed)

            for condition in EXPERIMENT_CONDITIONS:
                experiment_name = f"{condition['key']}_seed_{seed}"
                bonus = resolved_bonus if condition["uses_guidance_bonus"] else 0.0

                suite_logger.info(
                    "Running seed=%s condition=%s guidance=%s bonus=%s",
                    seed,
                    condition["key"],
                    condition["guidance_mode"],
                    bonus,
                )

                try:
                    result = run_training_fn(
                        agent_name=condition["agent"],
                        total_episodes=total_episodes,
                        evaluation_episodes=evaluation_episodes,
                        experiment_name=experiment_name,
                        guidance_mode=condition["guidance_mode"],
                        guidance_bonus=bonus,
                        random_seed=seed,
                        suite_name=suite_run_name,
                    )

                    result["suite_name"] = suite_run_name
                    result["seed"] = seed
                    result["condition"] = condition["key"]
                    result["condition_label"] = condition["label"]

                    seed_runs.append(result)
                    all_runs.append(result)

                except Exception as exc:
                    failure = {
                        "seed": seed,
                        "condition": condition["key"],
                        "error": str(exc),
                    }
                    failures.append(failure)
                    suite_logger.exception(
                        "Run failed | seed=%s condition=%s",
                        seed,
                        condition["key"],
                    )

                    if not continue_on_error:
                        _write_manifest(
                            Path(all_scope["logs"]) / "failures.json",
                            failures,
                        )
                        raise

            # Persist a comparison as soon as this seed is finished. If a later
            # seed fails, earlier expensive experiments still have combined data.
            if seed_runs:
                seed_outputs[str(seed)] = _generate_scope_outputs(
                    run_results=seed_runs,
                    directories=seed_scope,
                    scope_name=f"seed_{seed}",
                )

            suite_logger.info(
                "Seed %s finished | successful runs=%s",
                seed,
                len(seed_runs),
            )

        if all_runs:
            all_outputs = _generate_scope_outputs(
                run_results=all_runs,
                directories=all_scope,
                scope_name="all_seeds",
            )
        else:
            all_outputs = {}

        suite_manifest = {
            "suite_name": suite_run_name,
            "seeds": seeds,
            "conditions": EXPERIMENT_CONDITIONS,
            "run_count": len(all_runs),
            "expected_run_count": len(seeds) * len(EXPERIMENT_CONDITIONS),
            "runs": [_json_safe_run(run) for run in all_runs],
            "seed_outputs": seed_outputs,
            "all_seed_outputs": all_outputs,
            "failures": failures,
        }

        suite_manifest_path = _write_manifest(
            Path(all_scope["logs"]) / "suite_manifest.json",
            suite_manifest,
        )

        suite_logger.info("========================================")
        suite_logger.info("EXPERIMENT SUITE COMPLETE")
        suite_logger.info("Runs completed: %s", len(all_runs))
        suite_logger.info("Failures: %s", len(failures))
        suite_logger.info("Manifest: %s", suite_manifest_path)
        suite_logger.info("Combined metrics: %s", all_outputs.get("metrics_excel"))
        suite_logger.info("Combined graphs: %s", all_scope["graphs"])
        suite_logger.info("========================================")

        return suite_manifest

    finally:
        config.environment.RANDOM_SEED = previous_seed
        config.experiment.SUITE_NAME = previous_suite_name
        close_logger(suite_logger)
