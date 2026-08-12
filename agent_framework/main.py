"""
main.py

Complete PPO training pipeline.

Usage:

    python main.py train

Optional:

    python main.py train --episodes 100
    python main.py train --episodes 1000 --eval-episodes 100
"""

import argparse
import sys

from pathlib import Path


from agents.ppo_agent import PPOAgent

from config.config import config

from environment.api_client import APIClient
from environment.finance_env import FinanceEnvironment

from training.train import train_ppo
from training.evaluate import evaluate_ppo

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
    generate_ppo_visualizations,
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
# Authentication
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

        raise RuntimeError("Backend environment error during login.")

    if not response.get(
        "success",
        False,
    ):

        raise RuntimeError("Agent authentication failed.")

    logger.info("Backend authentication successful.")

    return client


# ==========================================================
# Fetch Backend Episodes
# ==========================================================


def fetch_backend_episodes(
    client,
    run_name,
    phase,
):

    response = client.get_episodes(
        experiment_name=run_name,
        phase=phase,
        agent_type="RL",
        algorithm="PPO",
    )

    if response.get(
        "environment_error",
        False,
    ):

        raise RuntimeError(f"Environment error while fetching {phase} episodes.")

    if not response.get(
        "success",
        False,
    ):

        raise RuntimeError(f"Failed to fetch {phase} episodes.")

    payload = _response_payload(response)

    return payload.get(
        "episodes",
        [],
    )


# ==========================================================
# PPO Pipeline
# ==========================================================


def run_ppo_training(
    total_episodes=None,
    evaluation_episodes=None,
):

    run_name = create_run_name(config.experiment.EXPERIMENT_NAME)

    directories = prepare_run_directories(run_name)

    logger = get_run_logger(
        run_name=run_name,
        log_directory=directories["logs"],
    )

    api_client = None

    env = None

    try:

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
                    f"config="
                    f"{config.environment.ACTION_SPACE_SIZE}"
                )
            )

        logger.info(
            ("Environment ready | " "observation_size=%s | " "action_size=%s"),
            observation_size,
            action_size,
        )

        # ======================================================
        # PPO Agent
        # ======================================================

        agent = PPOAgent(
            observation_size=observation_size,
            action_size=action_size,
        )

        logger.info(
            "PPO agent created | device=%s",
            agent.device,
        )

        # ======================================================
        # Train
        # ======================================================

        training_result = train_ppo(
            agent=agent,
            env=env,
            run_name=run_name,
            model_directory=directories["models"],
            logger=logger,
            total_episodes=total_episodes,
        )

        # ======================================================
        # Deterministic Evaluation
        # ======================================================

        evaluation_result = evaluate_ppo(
            agent=agent,
            env=env,
            run_name=run_name,
            logger=logger,
            evaluation_episodes=evaluation_episodes,
        )

        # ======================================================
        # Fetch authoritative backend episodes
        # ======================================================

        logger.info("Fetching training episodes from backend.")

        training_episodes = fetch_backend_episodes(
            api_client,
            run_name,
            "TRAIN",
        )

        logger.info(
            "Training episodes retrieved: %s",
            len(training_episodes),
        )

        logger.info("Fetching evaluation episodes from backend.")

        evaluation_episodes_backend = fetch_backend_episodes(
            api_client,
            run_name,
            "EVALUATION",
        )

        logger.info(
            "Evaluation episodes retrieved: %s",
            len(evaluation_episodes_backend),
        )

        # ======================================================
        # Metrics
        # ======================================================

        tables = build_metric_tables(
            training_episodes=training_episodes,
            evaluation_episodes=evaluation_episodes_backend,
            ppo_updates=training_result["ppo_updates"],
            training_time=training_result["training_time"],
            evaluation_time=evaluation_result["evaluation_time"],
        )

        # ======================================================
        # Excel
        # ======================================================

        excel_path = Path(directories["metrics"]) / "ppo_results.xlsx"

        excel_path = export_metrics_excel(
            tables,
            excel_path,
        )

        logger.info(
            "Excel metrics exported: %s",
            excel_path,
        )

        # ======================================================
        # Visualizations
        # ======================================================

        graph_paths = generate_ppo_visualizations(
            tables=tables,
            output_directory=directories["graphs"],
        )

        for (
            graph_name,
            graph_path,
        ) in graph_paths.items():

            logger.info(
                "Graph %-20s %s",
                graph_name,
                graph_path,
            )

        # ======================================================
        # Console Summary
        # ======================================================

        summary = tables["summary"]

        logger.info(
            "\n%s",
            summary.to_string(index=False),
        )

        logger.info("========================================")

        logger.info("PPO RUN COMPLETE")

        logger.info(
            "Run name: %s",
            run_name,
        )

        logger.info(
            "Model: %s",
            training_result["final_checkpoint"],
        )

        logger.info(
            "Metrics: %s",
            excel_path,
        )

        logger.info(
            "Graphs: %s",
            directories["graphs"],
        )

        logger.info("========================================")

        return {
            "run_name": run_name,
            "training": training_result,
            "evaluation": evaluation_result,
            "metrics_excel": excel_path,
            "graphs": graph_paths,
        }

    finally:

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
        description=("Train and analyse the " "Finance PPO agent.")
    )

    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help=("Train PPO, evaluate it, " "and generate results."),
    )

    train_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=("Override TOTAL_EPISODES."),
    )

    train_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help=("Override EVALUATION_EPISODES."),
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

            run_ppo_training(
                total_episodes=args.episodes,
                evaluation_episodes=args.eval_episodes,
            )

            return 0

        return 1

    except KeyboardInterrupt:

        print("\nTraining interrupted.")

        return 130

    except Exception as exc:

        print(f"\nPPO run failed: {exc}")

        raise


if __name__ == "__main__":

    sys.exit(main())
