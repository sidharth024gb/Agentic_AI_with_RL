"""
main.py

Main entry point for the Finance RL Agent project.

The project focuses on comparing:

    1. PPO
    2. LLM + PPO

The actual training, evaluation, and experiment logic is
implemented in the training package.

This file only orchestrates those components.
"""

import argparse
import sys
from typing import Optional

# ==============================================================
# Project Imports
# ==============================================================

from environment.finance_env import FinanceEnv

from agents.ppo_agent import PPOAgent
from agents.llm_rl_agent import LLMRLAgent

from training.train import train_agent
from training.evaluate import evaluate_agent
from training.experiment import run_experiment

from utils.logger import get_logger

# ==============================================================
# Logger
# ==============================================================

logger = get_logger(name="main")


# ==============================================================
# Agent Creation
# ==============================================================


def create_agent(
    agent_type: str,
    env: FinanceEnv,
):
    """
    Create the requested agent.

    Parameters
    ----------
    agent_type:
        "ppo" or "llm_ppo"

    env:
        Finance RL environment.

    Returns
    -------
    Agent instance
    """

    agent_type = agent_type.lower()

    # ----------------------------------------------------------
    # PPO
    # ----------------------------------------------------------

    if agent_type == "ppo":

        logger.info("Creating PPO agent.")

        return PPOAgent(env=env)

    # ----------------------------------------------------------
    # LLM + PPO
    # ----------------------------------------------------------

    if agent_type in {
        "llm_ppo",
        "llm+ppo",
        "llm",
    }:

        logger.info("Creating LLM + PPO agent.")

        return LLMRLAgent(env=env)

    raise ValueError(f"Unknown agent type: {agent_type}")


# ==============================================================
# Environment Creation
# ==============================================================


def create_environment():
    """
    Create and return the Finance RL environment.
    """

    logger.info("Creating finance environment.")

    return FinanceEnv()


# ==============================================================
# Training
# ==============================================================


def run_training(
    agent_type: str,
):
    """
    Train a single agent.
    """

    env = create_environment()

    try:

        agent = create_agent(
            agent_type,
            env,
        )

        logger.info(
            "Starting training for %s.",
            agent_type,
        )

        results = train_agent(
            agent=agent,
            env=env,
        )

        logger.info(
            "Training completed for %s.",
            agent_type,
        )

        return results

    finally:

        env.close("TRAINING_COMPLETE")


# ==============================================================
# Evaluation
# ==============================================================


def run_evaluation(
    agent_type: str,
):
    """
    Evaluate a trained agent.
    """

    env = create_environment()

    try:

        agent = create_agent(
            agent_type,
            env,
        )

        logger.info(
            "Starting evaluation for %s.",
            agent_type,
        )

        results = evaluate_agent(
            agent=agent,
            env=env,
        )

        logger.info(
            "Evaluation completed for %s.",
            agent_type,
        )

        return results

    finally:

        env.close("EVALUATION_COMPLETE")


# ==============================================================
# Experiment
# ==============================================================


def run_comparison():
    """
    Run the main PPO vs LLM + PPO experiment.

    The experiment module is responsible for:

        - running both agents
        - collecting metrics
        - comparing results
        - generating experiment outputs
    """

    logger.info("Starting PPO vs LLM + PPO experiment.")

    results = run_experiment()

    logger.info("PPO vs LLM + PPO experiment completed.")

    return results


# ==============================================================
# Argument Parser
# ==============================================================


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=("Finance RL Agent - " "PPO and LLM + PPO experiments")
    )

    subparsers = parser.add_subparsers(dest="command")

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------

    train_parser = subparsers.add_parser(
        "train",
        help="Train an agent.",
    )

    train_parser.add_argument(
        "--agent",
        choices=[
            "ppo",
            "llm_ppo",
        ],
        required=True,
        help=("Agent to train."),
    )

    # ----------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained agent.",
    )

    evaluate_parser.add_argument(
        "--agent",
        choices=[
            "ppo",
            "llm_ppo",
        ],
        required=True,
        help=("Agent to evaluate."),
    )

    # ----------------------------------------------------------
    # Experiment
    # ----------------------------------------------------------

    subparsers.add_parser(
        "experiment",
        help=("Run PPO vs LLM + PPO comparison."),
    )

    return parser


# ==============================================================
# Main
# ==============================================================


def main(
    argv: Optional[list] = None,
) -> int:
    """
    Main application entry point.

    Returns
    -------
    int
        Process exit code.
    """

    parser = build_parser()

    args = parser.parse_args(argv)

    # ----------------------------------------------------------
    # No command
    # ----------------------------------------------------------

    if args.command is None:

        parser.print_help()

        return 0

    try:

        # ------------------------------------------------------
        # Training
        # ------------------------------------------------------

        if args.command == "train":

            run_training(args.agent)

            return 0

        # ------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------

        if args.command == "evaluate":

            run_evaluation(args.agent)

            return 0

        # ------------------------------------------------------
        # Experiment
        # ------------------------------------------------------

        if args.command == "experiment":

            run_comparison()

            return 0

        logger.error(
            "Unknown command: %s",
            args.command,
        )

        return 1

    except KeyboardInterrupt:

        logger.warning("Execution interrupted by user.")

        return 130

    except Exception as exc:

        logger.exception(
            "Execution failed: %s",
            exc,
        )

        return 1


# ==============================================================
# Script Entry Point
# ==============================================================

if __name__ == "__main__":

    sys.exit(main())

"""
python main.py train --agent ppo
python main.py train --agent llm_ppo
python main.py evaluate --agent ppo
python main.py evaluate --agent llm_ppo
python main.py experiment
"""
