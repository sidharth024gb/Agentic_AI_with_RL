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

LLM + PPO:

    python main.py train --agent llm_rl

Short test:

    python main.py train --agent llm_rl --episodes 50 --eval-episodes 10
"""

import argparse
import sys

from pathlib import Path


from agents.ppo_agent import (
    PPOAgent,
)

from agents.llm_rl_agent import (
    LLMRLAgent,
)

from config.config import config

from environment.api_client import (
    APIClient,
)

from environment.finance_env import (
    FinanceEnvironment,
)

from training.train import (
    train_agent,
)

from training.evaluate import (
    evaluate_agent,
)

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
# Response Helper
# ==========================================================


def _response_payload(
    response,
):

    if not isinstance(
        response,
        dict,
    ):

        return {}

    data = response.get(
        "data",
        {},
    )

    if isinstance(
        data,
        dict,
    ) and isinstance(
        data.get("data"),
        dict,
    ):

        return data["data"]

    return (
        data
        if isinstance(
            data,
            dict,
        )
        else {}
    )


# ==========================================================
# API
# ==========================================================


def create_api_client(
    logger,
):

    client = APIClient()

    logger.info("Authenticating AGENT_BOT.")

    response = client.login()

    if response.get(
        "environment_error",
        False,
    ):

        raise RuntimeError(("Backend environment " "error during login."))

    if not response.get(
        "success",
        False,
    ):

        raise RuntimeError("Agent authentication failed.")

    logger.info("Backend authentication successful.")

    return client


# ==========================================================
# Agent Settings
# ==========================================================


def normalize_agent_name(
    agent_name,
):

    value = str(agent_name).strip().lower()

    if value in {
        "ppo",
        "rl",
    }:

        return "ppo"

    if value in {
        "llm_rl",
        "llm-rl",
        "llm_ppo",
        "llm+ppo",
    }:

        return "llm_rl"

    raise ValueError(f"Unknown agent: {agent_name}")


def backend_agent_type(
    agent_name,
):

    return "LLM_RL" if agent_name == "llm_rl" else "RL"


def display_agent_name(
    agent_name,
):

    return "LLM + PPO" if agent_name == "llm_rl" else "PPO"


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
# Experiment Name
# ==========================================================


def default_experiment_name(
    agent_name,
):

    if agent_name == "ppo":

        return "ppo_baseline"

    return config.experiment.EXPERIMENT_NAME


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

    if response.get(
        "environment_error",
        False,
    ):

        raise RuntimeError(("Environment error while " f"fetching {phase} episodes."))

    if not response.get(
        "success",
        False,
    ):

        raise RuntimeError(("Failed to fetch " f"{phase} episodes."))

    payload = _response_payload(response)

    return payload.get(
        "episodes",
        [],
    )


# ==========================================================
# Experiment Pipeline
# ==========================================================


def run_training(
    agent_name,
    total_episodes=None,
    evaluation_episodes=None,
    experiment_name=None,
):

    agent_name = normalize_agent_name(agent_name)

    if experiment_name is None:

        experiment_name = default_experiment_name(agent_name)

    run_name = create_run_name(experiment_name)

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

        logger.info(
            "Run name: %s",
            run_name,
        )

        logger.info("========================================")

        # ======================================================
        # API
        # ======================================================

        api_client = create_api_client(logger)

        # ======================================================
        # Environment
        # ======================================================

        env = FinanceEnvironment(api_client=api_client)

        observation_size = env.state_encoder.get_state_size()

        action_size = env.action_space_handler.action_count

        if action_size != config.environment.ACTION_SPACE_SIZE:

            raise RuntimeError(
                (
                    "Action space mismatch: "
                    f"environment={action_size}, "
                    "config="
                    f"{config.environment.ACTION_SPACE_SIZE}"
                )
            )

        # ======================================================
        # Agent
        # ======================================================

        agent = create_agent(
            agent_name=agent_name,
            observation_size=observation_size,
            action_size=action_size,
        )

        logger.info(
            (
                "Agent ready | "
                "type=%s | "
                "base_observation=%s | "
                "policy_observation=%s | "
                "actions=%s | "
                "device=%s"
            ),
            display_name,
            observation_size,
            agent.observation_size,
            action_size,
            agent.device,
        )

        if agent_name == "llm_rl":

            logger.info(
                (
                    "LLM configuration | "
                    "model=%s | "
                    "guidance=%s | "
                    "temperature=%s | "
                    "cache=%s"
                ),
                config.llm.MODEL,
                config.experiment.GUIDANCE_MODE,
                config.llm.TEMPERATURE,
                config.llm.USE_CACHE,
            )

        # ======================================================
        # Train
        # ======================================================

        training_result = train_agent(
            agent=agent,
            env=env,
            run_name=run_name,
            model_directory=directories["models"],
            logger=logger,
            total_episodes=total_episodes,
            agent_label=agent_name,
        )

        # ======================================================
        # Evaluate
        # ======================================================

        evaluation_result = evaluate_agent(
            agent=agent,
            env=env,
            run_name=run_name,
            logger=logger,
            evaluation_episodes=evaluation_episodes,
            agent_label=agent_name,
        )

        # ======================================================
        # Backend Episodes
        # ======================================================

        logger.info(("Fetching authoritative " "TRAIN episodes."))

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

        logger.info(("Fetching authoritative " "EVALUATION episodes."))

        evaluation_backend = fetch_backend_episodes(
            client=api_client,
            run_name=run_name,
            phase="EVALUATION",
            agent_name=agent_name,
        )

        logger.info(
            ("Evaluation episodes " "retrieved: %s"),
            len(evaluation_backend),
        )

        # ======================================================
        # Metrics
        # ======================================================

        tables = build_metric_tables(
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_backend,
            ppo_updates=training_result["ppo_updates"],
            training_time=training_result["training_time"],
            evaluation_time=evaluation_result["evaluation_time"],
            training_llm_metrics=training_result.get("llm_metrics"),
            evaluation_llm_metrics=evaluation_result.get("llm_metrics"),
        )

        # ======================================================
        # Excel
        # ======================================================

        excel_path = Path(directories["metrics"]) / (f"{agent_name}" "_results.xlsx")

        excel_path = export_metrics_excel(
            tables=tables,
            output_path=excel_path,
            agent_label=agent_name,
        )

        # ======================================================
        # Visualizations
        # ======================================================

        graph_paths = generate_agent_visualizations(
            tables=tables,
            output_directory=directories["graphs"],
            agent_name=display_name,
        )

        # ======================================================
        # Summary
        # ======================================================

        logger.info(
            "\n%s",
            tables["summary"].to_string(index=False),
        )

        if not tables["guidance_summary"].empty:

            logger.info(
                ("\nLLM GUIDANCE SUMMARY\n%s"),
                tables["guidance_summary"].to_string(index=False),
            )

        logger.info("========================================")

        logger.info(
            "%s RUN COMPLETE",
            display_name,
        )

        logger.info(
            "Model   : %s",
            training_result["final_checkpoint"],
        )

        logger.info(
            "Metrics : %s",
            excel_path,
        )

        logger.info(
            "Graphs  : %s",
            directories["graphs"],
        )

        logger.info("========================================")

        return {
            "run_name": run_name,
            "agent": agent_name,
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

        elif api_client is not None and hasattr(
            api_client,
            "session",
        ):

            api_client.session.close()

        close_logger(logger)


# ==========================================================
# CLI
# ==========================================================


def build_parser():

    parser = argparse.ArgumentParser(
        description=("Train and analyse PPO " "or LLM-enhanced PPO.")
    )

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train")

    train_parser.add_argument(
        "--agent",
        choices=[
            "ppo",
            "llm_rl",
        ],
        required=True,
    )

    train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
    )

    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
    )

    train_parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
    )

    return parser


# ==========================================================
# Main
# ==========================================================


def main(
    argv=None,
):

    parser = build_parser()

    args = parser.parse_args(argv)

    if args.command is None:

        parser.print_help()

        return 0

    try:

        if args.command == "train":

            run_training(
                agent_name=args.agent,
                total_episodes=args.episodes,
                evaluation_episodes=args.eval_episodes,
                experiment_name=args.experiment_name,
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
