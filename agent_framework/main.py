"""
main.py

Finance RL experiment runner.

Supported agents:

    PPO
    LLM_RL

Examples
--------

Baseline PPO:

    python main.py train --agent ppo

LLM + PPO input guidance:

    python main.py train --agent llm_rl --guidance-mode input

LLM + PPO reward shaping:

    python main.py train --agent llm_rl --guidance-mode reward_shaping

LLM + PPO input + reward:

    python main.py train --agent llm_rl --guidance-mode input_and_reward

Short smoke test:

    python main.py train --agent ppo --episodes 50 --eval-episodes 10
"""

import argparse
import sys
from pathlib import Path

from agents.ppo_agent import PPOAgent
from agents.llm_rl_agent import LLMRLAgent

from config.config import config

from environment.api_client import APIClient
from environment.finance_env import FinanceEnvironment

from training.train import train_agent
from training.evaluate import evaluate_agent

from utils.logger import (
    close_logger,
    create_run_name,
    get_run_logger,
    prepare_run_directories,
)

from utils.metrics import (
    build_metric_tables,
    export_metrics_excel,
)

from utils.visualization import (
    generate_agent_visualizations,
)

# ==========================================================
# Guidance Modes
# ==========================================================


GUIDANCE_MODES = {
    "none": "NONE",
    "input": "INPUT",
    "reward_shaping": "REWARD_SHAPING",
    "input_and_reward": "INPUT_AND_REWARD",
}


# ==========================================================
# Response Helper
# ==========================================================


def _response_payload(response):
    """Extract the useful backend JSON payload."""

    if not isinstance(response, dict):
        return {}

    data = response.get("data", {})

    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]

    return data if isinstance(data, dict) else {}


# ==========================================================
# API
# ==========================================================


def create_api_client(logger):
    """Create and authenticate the AGENT_BOT API client."""

    client = APIClient()

    logger.info("Authenticating AGENT_BOT.")

    response = client.login()

    if response.get("environment_error", False):
        raise RuntimeError("Backend environment error during login.")

    if not response.get("success", False):
        raise RuntimeError("Agent authentication failed.")

    payload = _response_payload(response)

    if isinstance(payload, dict):
        user = payload.get("user", {})

        if isinstance(user, dict):
            logger.info(
                "Backend authentication successful | role=%s",
                user.get("role", "unknown"),
            )
        else:
            logger.info("Backend authentication successful.")
    else:
        logger.info("Backend authentication successful.")

    return client


# ==========================================================
# Agent Settings
# ==========================================================


def normalize_agent_name(agent_name):
    value = str(agent_name).strip().lower()

    if value in {"ppo", "rl"}:
        return "ppo"

    if value in {
        "llm_rl",
        "llm-rl",
        "llm_ppo",
        "llm+ppo",
    }:
        return "llm_rl"

    raise ValueError(f"Unknown agent: {agent_name}")


def normalize_guidance_mode(mode):
    if mode is None:
        return None

    value = str(mode).strip().lower().replace("-", "_")

    if value not in GUIDANCE_MODES:
        raise ValueError(
            "Unknown guidance mode: "
            f"{mode}. Expected one of: "
            f"{sorted(GUIDANCE_MODES)}"
        )

    return GUIDANCE_MODES[value]


def backend_agent_type(agent_name):
    return "LLM_RL" if agent_name == "llm_rl" else "RL"


def display_agent_name(agent_name):
    return "LLM + PPO" if agent_name == "llm_rl" else "PPO"


# ==========================================================
# Runtime Experiment Configuration
# ==========================================================


def default_experiment_name(
    agent_name,
    guidance_mode,
):
    """Return a stable experiment prefix for each ablation."""

    if agent_name == "ppo":
        return "ppo_baseline"

    names = {
        "NONE": "llm_ppo_none",
        "INPUT": "llm_ppo_input",
        "REWARD_SHAPING": "llm_ppo_reward",
        "INPUT_AND_REWARD": "llm_ppo_input_reward",
    }

    return names[guidance_mode]


def configure_runtime_experiment(
    agent_name,
    guidance_mode=None,
    guidance_bonus=None,
    experiment_name=None,
):
    """Synchronize the global config with the CLI run.

    This prevents the old metadata mismatch where a PPO baseline
    could still have CONFIG_AGENT_TYPE=LLM_RL merely because that
    was the static value in config.py.
    """

    agent_name = normalize_agent_name(agent_name)

    if agent_name == "ppo":
        resolved_guidance = "NONE"
        resolved_bonus = 0.0
    else:
        resolved_guidance = (
            normalize_guidance_mode(guidance_mode)
            if guidance_mode is not None
            else str(config.experiment.GUIDANCE_MODE).upper()
        )

        if resolved_guidance not in set(GUIDANCE_MODES.values()):
            raise ValueError(
                "Invalid configured guidance mode: " f"{resolved_guidance}"
            )

        resolved_bonus = (
            float(guidance_bonus)
            if guidance_bonus is not None
            else float(config.experiment.GUIDANCE_BONUS)
        )

        # No reward shaping means bonus is irrelevant.
        if resolved_guidance in {"NONE", "INPUT"}:
            resolved_bonus = 0.0

    resolved_experiment_name = (
        experiment_name
        if experiment_name
        else default_experiment_name(
            agent_name,
            resolved_guidance,
        )
    )

    # Config dataclasses are deliberately mutable at runtime.
    config.agent.AGENT_TYPE = backend_agent_type(agent_name)
    config.agent.ALGORITHM = "PPO"

    config.experiment.EXPERIMENT_NAME = resolved_experiment_name
    config.experiment.GUIDANCE_MODE = resolved_guidance
    config.experiment.GUIDANCE_BONUS = resolved_bonus

    if agent_name == "ppo":
        config.experiment.DESCRIPTION = (
            "Baseline PPO agent without LLM planning " "or guidance."
        )
    elif resolved_guidance == "INPUT":
        config.experiment.DESCRIPTION = (
            "PPO agent with LLM-generated procedural "
            "guidance appended to the policy input."
        )
    elif resolved_guidance == "REWARD_SHAPING":
        config.experiment.DESCRIPTION = (
            "PPO agent with LLM-generated procedural "
            "guidance used for positive reward shaping."
        )
    elif resolved_guidance == "INPUT_AND_REWARD":
        config.experiment.DESCRIPTION = (
            "PPO agent with LLM-generated procedural "
            "guidance used as policy input and positive "
            "reward shaping."
        )
    else:
        config.experiment.DESCRIPTION = (
            "LLM planner enabled without policy-input or " "reward guidance."
        )

    return {
        "agent_name": agent_name,
        "agent_type": config.agent.AGENT_TYPE,
        "algorithm": config.agent.ALGORITHM,
        "guidance_mode": resolved_guidance,
        "guidance_bonus": resolved_bonus,
        "experiment_name": resolved_experiment_name,
    }


# ==========================================================
# Create Agent
# ==========================================================


def create_agent(
    agent_name,
    observation_size,
    action_size,
):
    if agent_name == "ppo":
        return PPOAgent(
            observation_size=observation_size,
            action_size=action_size,
        )

    if agent_name == "llm_rl":
        return LLMRLAgent(
            observation_size=observation_size,
            action_size=action_size,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


# ==========================================================
# Fetch Backend Episodes
# ==========================================================


def fetch_backend_episodes(
    client,
    run_name,
    phase,
    agent_name,
):
    response = client.get_episodes(
        experiment_name=run_name,
        phase=phase,
        agent_type=backend_agent_type(agent_name),
        algorithm="PPO",
    )

    if response.get("environment_error", False):
        raise RuntimeError("Environment error while fetching " f"{phase} episodes.")

    if not response.get("success", False):
        raise RuntimeError(f"Failed to fetch {phase} episodes.")

    payload = _response_payload(response)

    return payload.get("episodes", [])


# ==========================================================
# Experiment Pipeline
# ==========================================================


def run_training(
    agent_name,
    total_episodes=None,
    evaluation_episodes=None,
    experiment_name=None,
    guidance_mode=None,
    guidance_bonus=None,
):
    # Resolve CLI -> config BEFORE constructing env/agent.
    runtime = configure_runtime_experiment(
        agent_name=agent_name,
        guidance_mode=guidance_mode,
        guidance_bonus=guidance_bonus,
        experiment_name=experiment_name,
    )

    agent_name = runtime["agent_name"]

    run_name = create_run_name(runtime["experiment_name"])

    directories = prepare_run_directories(run_name)

    logger = get_run_logger(
        run_name=run_name,
        log_directory=directories["logs"],
    )

    api_client = None
    env = None
    agent = None

    display_name = display_agent_name(agent_name)

    try:
        logger.info("========================================")
        logger.info(
            "STARTING %s EXPERIMENT",
            display_name,
        )
        logger.info("Run name: %s", run_name)
        logger.info(
            "Runtime config | agentType=%s | algorithm=%s | "
            "guidance=%s | guidanceBonus=%.2f",
            runtime["agent_type"],
            runtime["algorithm"],
            runtime["guidance_mode"],
            runtime["guidance_bonus"],
        )
        logger.info(
            "PPO config | gamma=%s | lr=%s | rollout=%s | "
            "batch=%s | epochs=%s | entropy=%s | gradClip=%s",
            config.training.GAMMA,
            config.training.LEARNING_RATE,
            config.training.UPDATE_INTERVAL,
            config.training.BATCH_SIZE,
            config.training.EPOCHS,
            config.training.ENTROPY_COEF,
            config.training.MAX_GRAD_NORM,
        )
        logger.info("========================================")

        # ==================================================
        # API
        # ==================================================

        api_client = create_api_client(logger)

        # ==================================================
        # Environment
        # ==================================================

        env = FinanceEnvironment(api_client=api_client)

        observation_size = env.state_encoder.get_state_size()

        action_size = env.action_space_handler.action_count

        if action_size != config.environment.ACTION_SPACE_SIZE:
            raise RuntimeError(
                "Action space mismatch: "
                f"environment={action_size}, "
                "config="
                f"{config.environment.ACTION_SPACE_SIZE}"
            )

        # ==================================================
        # Agent
        # ==================================================

        agent = create_agent(
            agent_name=agent_name,
            observation_size=observation_size,
            action_size=action_size,
        )

        logger.info(
            "Agent ready | type=%s | base_observation=%s | "
            "policy_observation=%s | actions=%s | device=%s",
            display_name,
            observation_size,
            agent.observation_size,
            action_size,
            agent.device,
        )

        if agent_name == "llm_rl":
            logger.info(
                "LLM configuration | model=%s | guidance=%s | "
                "temperature=%s | cache=%s",
                config.llm.MODEL,
                config.experiment.GUIDANCE_MODE,
                config.llm.TEMPERATURE,
                config.llm.USE_CACHE,
            )

        # ==================================================
        # Train
        # ==================================================

        training_result = train_agent(
            agent=agent,
            env=env,
            run_name=run_name,
            model_directory=directories["models"],
            logger=logger,
            total_episodes=total_episodes,
            agent_label=agent_name,
        )

        # ==================================================
        # Evaluate
        # ==================================================

        evaluation_result = evaluate_agent(
            agent=agent,
            env=env,
            run_name=run_name,
            logger=logger,
            evaluation_episodes=evaluation_episodes,
            agent_label=agent_name,
        )

        logger.info(
            "Evaluation accounting | valid=%s | envErrors=%s | "
            "success=%s | successRate=%.2f%%",
            evaluation_result.get(
                "valid_episode_count",
                len(
                    evaluation_result.get(
                        "local_episodes",
                        [],
                    )
                ),
            ),
            evaluation_result.get(
                "environment_error_episodes",
                0,
            ),
            evaluation_result.get(
                "success_count",
                0,
            ),
            float(
                evaluation_result.get(
                    "success_rate",
                    0.0,
                )
            )
            * 100.0,
        )

        # ==================================================
        # Backend Episodes
        # ==================================================

        logger.info("Fetching authoritative TRAIN episodes.")

        training_episodes = fetch_backend_episodes(
            client=api_client,
            run_name=run_name,
            phase="TRAIN",
            agent_name=agent_name,
        )

        logger.info(
            "Training episodes retrieved: %s",
            len(training_episodes),
        )

        logger.info("Fetching authoritative EVALUATION episodes.")

        evaluation_backend = fetch_backend_episodes(
            client=api_client,
            run_name=run_name,
            phase="EVALUATION",
            agent_name=agent_name,
        )

        logger.info(
            "Evaluation episodes retrieved: %s",
            len(evaluation_backend),
        )

        # ==================================================
        # Metrics
        # ==================================================

        tables = build_metric_tables(
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_backend,
            ppo_updates=training_result["ppo_updates"],
            training_time=training_result["training_time"],
            evaluation_time=evaluation_result["evaluation_time"],
            training_llm_metrics=training_result.get("llm_metrics"),
            evaluation_llm_metrics=evaluation_result.get("llm_metrics"),
        )

        # ==================================================
        # Excel
        # ==================================================

        excel_path = Path(directories["metrics"]) / f"{agent_name}_results.xlsx"

        excel_path = export_metrics_excel(
            tables=tables,
            output_path=excel_path,
            agent_label=agent_name,
        )

        # ==================================================
        # Visualizations
        # ==================================================

        graph_paths = generate_agent_visualizations(
            tables=tables,
            output_directory=directories["graphs"],
            agent_name=display_name,
        )

        # ==================================================
        # Summary
        # ==================================================

        logger.info(
            "\n%s",
            tables["summary"].to_string(index=False),
        )

        if not tables["guidance_summary"].empty:
            logger.info(
                "\nLLM GUIDANCE SUMMARY\n%s",
                tables["guidance_summary"].to_string(index=False),
            )

        logger.info("========================================")
        logger.info("%s RUN COMPLETE", display_name)
        logger.info(
            "Model   : %s",
            training_result["final_checkpoint"],
        )
        logger.info("Metrics : %s", excel_path)
        logger.info(
            "Graphs  : %s",
            directories["graphs"],
        )
        logger.info("========================================")

        return {
            "run_name": run_name,
            "agent": agent_name,
            "runtime_config": runtime,
            "training": training_result,
            "evaluation": evaluation_result,
            "metrics_excel": excel_path,
            "graphs": graph_paths,
        }

    finally:
        # LLM planner owns a separate HTTP session.
        if agent is not None and agent_name == "llm_rl":
            try:
                agent.close()
            except Exception:
                pass

        if env is not None:
            env.close(terminated_reason="FAILED")

        elif api_client is not None and hasattr(api_client, "session"):
            api_client.session.close()

        close_logger(logger)


# ==========================================================
# CLI
# ==========================================================


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Train and analyse PPO or LLM-enhanced PPO.")
    )

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--agent",
        choices=["ppo", "llm_rl"],
        required=True,
        help=("ppo = baseline RL; llm_rl = PPO with LLM plan."),
    )

    train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=("Training episodes. Defaults to config.training." "TOTAL_EPISODES."),
    )

    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=(
            "Evaluation episodes. Defaults to config.training." "EVALUATION_EPISODES."
        ),
    )

    train_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Optional experiment prefix. A timestamp/run suffix "
            "is still created by create_run_name()."
        ),
    )

    train_parser.add_argument(
        "--guidance-mode",
        choices=[
            "none",
            "input",
            "reward_shaping",
            "input_and_reward",
        ],
        default=None,
        help=(
            "LLM guidance ablation. Ignored for baseline PPO, "
            "which always uses NONE."
        ),
    )

    train_parser.add_argument(
        "--guidance-bonus",
        type=float,
        default=None,
        help=(
            "Positive bonus for followed LLM procedure steps. "
            "Only applies to reward_shaping/input_and_reward."
        ),
    )

    return parser


# ==========================================================
# Main
# ==========================================================


def main(argv=None):
    parser = build_parser()

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "train":
            if args.episodes is not None and args.episodes <= 0:
                raise ValueError("--episodes must be greater than 0.")

            if args.eval_episodes is not None and args.eval_episodes <= 0:
                raise ValueError("--eval-episodes must be greater than 0.")

            if args.guidance_bonus is not None and args.guidance_bonus < 0:
                raise ValueError("--guidance-bonus must be >= 0.")

            run_training(
                agent_name=args.agent,
                total_episodes=args.episodes,
                evaluation_episodes=args.eval_episodes,
                experiment_name=args.experiment_name,
                guidance_mode=args.guidance_mode,
                guidance_bonus=args.guidance_bonus,
            )

            return 0

        return 1

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        return 130

    except Exception as exc:
        print(f"\nExperiment failed: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
