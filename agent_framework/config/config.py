"""
config.py

Central configuration for the Finance PPO / LLM+PPO project.

The command-line runner may override agent/experiment fields at runtime
so that PPO and LLM+PPO runs are logged with the correct metadata.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

# Keep the user's existing .env location and variable names.
load_dotenv(dotenv_path="./config/.env")


def _parse_int_list(value, default=(42, 43, 44)):
    """Parse a comma/space separated integer list from configuration."""

    if value is None:
        return list(default)

    if isinstance(value, (list, tuple, set)):
        raw_values = value
    else:
        raw_values = str(value).replace(";", ",").replace(" ", ",").split(",")

    result = []

    for item in raw_values:
        text = str(item).strip()
        if not text:
            continue
        number = int(text)
        if number not in result:
            result.append(number)

    return result or list(default)


# ==========================================================
# Backend Configuration
# ==========================================================


@dataclass
class BackendConfig:
    """Backend API configuration."""

    BASE_URL: str = os.getenv(
        "BASE_URL",
        "http://localhost:5000/api",
    )

    TIMEOUT: int = int(os.getenv("BACKEND_TIMEOUT", "30"))

    # ------------------------------------------------------
    # Authentication
    # ------------------------------------------------------

    EMAIL: str = os.getenv(
        "EMAIL",
        "",
    )

    PASSWORD: str = os.getenv(
        "PASSWORD",
        "",
    )

    # ------------------------------------------------------
    # Authentication Endpoints
    # ------------------------------------------------------

    LOGIN_ENDPOINT: str = "/auth/login"
    PROFILE_ENDPOINT: str = "/auth/me"

    AUTH_REGISTER: str = "/auth/register"
    AUTH_LOGIN: str = "/auth/login"
    AUTH_ME: str = "/auth/me"

    # ------------------------------------------------------
    # Episode Endpoints
    # ------------------------------------------------------

    EPISODE: str = "/episode"
    EPISODE_START: str = "/episode/start"
    EPISODE_STEP: str = "/step"
    EPISODE_END: str = "/end"

    # ------------------------------------------------------
    # Sandbox Endpoints
    # ------------------------------------------------------

    SANDBOX_RESET: str = "/sandbox/reset"
    SANDBOX_STATE: str = "/sandbox/state"
    SANDBOX_REWARD: str = "/sandbox/reward"

    # ------------------------------------------------------
    # Supplier Endpoints
    # ------------------------------------------------------

    SUPPLIER: str = "/supplier"
    SUPPLIER_VALIDATE: str = "/supplier/validate"

    # ------------------------------------------------------
    # Invoice Endpoints
    # ------------------------------------------------------

    INVOICE: str = "/invoice"
    INVOICE_DUPLICATE_CHECK: str = "/invoice/duplicate-check"

    # ------------------------------------------------------
    # Account Endpoints
    # ------------------------------------------------------

    ACCOUNT: str = "/account"
    ACCOUNT_BUDGET_CHECK: str = "/account/budget/check"
    ACCOUNT_CASH_POSITION: str = "/account/cash-position"

    # ------------------------------------------------------
    # Approval Endpoints
    # ------------------------------------------------------

    APPROVAL_APPROVE: str = "/approval/approve"

    # ------------------------------------------------------
    # Report Endpoints
    # ------------------------------------------------------

    REPORT_TRANSACTIONS: str = "/report/transactions"
    REPORT_GENERATE_REPORT: str = "/report/generate-report"

    # ------------------------------------------------------
    # Payment Endpoints
    # ------------------------------------------------------

    PAYMENT_PAY: str = "/payment/pay"
    PAYMENT_CANCEL_PAYMENT: str = "/payment/cancel-payment"
    PAYMENT_RETRY_PAYMENT: str = "/payment/retry-payment"


# ==========================================================
# Environment Configuration
# ==========================================================


@dataclass
class EnvironmentConfig:
    """RL environment configuration."""

    ENV_NAME: str = "FinanceSandbox-v1"

    MAX_STEPS_PER_EPISODE: int = int(os.getenv("MAX_STEPS_PER_EPISODE", "20"))

    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

    OBSERVATION_TYPE: str = "vector"

    # ------------------------------------------------------
    # Current Action Space
    #
    # 0 GET_INVOICES
    # 1 CHECK_DUPLICATE
    # 2 CHECK_SUPPLIER
    # 3 APPROVE_INVOICES
    # 4 PAY_INVOICES
    # 5 CHECK_BUDGET
    # 6 GENERATE_REPORT
    # 7 CHECK_PAYMENT_COMPLETED
    # ------------------------------------------------------

    ACTION_SPACE_SIZE: int = 8


# ==========================================================
# Training Configuration
# ==========================================================


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""

    TOTAL_EPISODES: int = int(os.getenv("TOTAL_EPISODES", "1000"))

    EVALUATION_EPISODES: int = int(os.getenv("EVALUATION_EPISODES", "100"))

    GAMMA: float = float(os.getenv("GAMMA", "0.99"))

    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.0003"))

    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "64"))

    # Important for this short-horizon environment.
    # 2048 meant roughly 100 twenty-step episodes could pass
    # before the first PPO update. 256 gives much more frequent
    # on-policy updates while still collecting a useful rollout.
    UPDATE_INTERVAL: int = int(os.getenv("UPDATE_INTERVAL", "256"))

    SAVE_EVERY: int = int(os.getenv("SAVE_EVERY", "100"))

    LOG_EVERY: int = int(os.getenv("LOG_EVERY", "10"))

    GAE_LAMBDA: float = float(os.getenv("GAE_LAMBDA", "0.95"))

    CLIP_EPSILON: float = float(os.getenv("CLIP_EPSILON", "0.2"))

    EPOCHS: int = int(os.getenv("PPO_EPOCHS", "10"))

    HIDDEN_NEURON_SIZE: int = int(os.getenv("HIDDEN_NEURON_SIZE", "256"))

    # ------------------------------------------------------
    # PPO exploration / stability
    # ------------------------------------------------------

    # This is now actually used in PPOAgent policy loss.
    ENTROPY_COEF: float = float(os.getenv("ENTROPY_COEF", "0.01"))

    MAX_GRAD_NORM: float = float(os.getenv("MAX_GRAD_NORM", "0.5"))

    # ------------------------------------------------------
    # Metrics / Visualization
    # ------------------------------------------------------

    MOVING_AVERAGE_WINDOW: int = 50
    CONVERGENCE_WINDOW: int = 50
    CONVERGENCE_SUCCESS_THRESHOLD: float = 0.90


# ==========================================================
# Agent Configuration
# ==========================================================


@dataclass
class AgentConfig:
    """Agent settings.

    main.py updates AGENT_TYPE at runtime from --agent, ensuring
    PPO runs are recorded as RL and LLM runs as LLM_RL.
    """

    AGENT_TYPE: str = "LLM_RL"
    ALGORITHM: str = "PPO"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    TASK: str = "Pay all valid invoices"


# ==========================================================
# LLM Configuration
# ==========================================================


@dataclass
class LLMConfig:
    """Ollama planner configuration."""

    MODEL: str = os.getenv(
        "LLM_MODEL",
        "llama3",
    )

    BASE_URL: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    CACHE_DIR: Path = Path.cwd() / "results" / "llm_cache"


# ==========================================================
# Experiment Configuration
# ==========================================================


@dataclass
class ExperimentConfig:
    """Experiment metadata.

    These values are mutable because main.py deliberately applies
    runtime CLI overrides before the environment and agent are made.
    """

    # Default LLM experiment if --guidance-mode is omitted.
    EXPERIMENT_NAME: str = "llm_ppo_input_reward"

    DESCRIPTION: str = (
        "PPO agent with LLM-generated procedural guidance "
        "provided as policy input and positive reward shaping."
    )

    # NONE / INPUT / REWARD_SHAPING / INPUT_AND_REWARD
    GUIDANCE_MODE: str = "INPUT_AND_REWARD"

    PHASE: str = "TRAIN"

    # Smaller than the earlier +5 because the corrected base reward
    # scale is also smaller. Guidance should help, not dominate PPO.
    GUIDANCE_BONUS: float = float(os.getenv("GUIDANCE_BONUS", "1.0"))

    ENVIRONMENT_VERSION: str = "v2-reward-fix"

    # ------------------------------------------------------
    # Multi-seed experiment suite
    # ------------------------------------------------------

    # .env example:
    #     EXPERIMENT_SEEDS=42,43,44
    #
    # ``RANDOM_SEED`` remains the base seed for a normal single run.
    # ``EXPERIMENT_SEEDS`` is used by ``python main.py run-all``.
    SEEDS: list[int] = field(
        default_factory=lambda: _parse_int_list(
            os.getenv("EXPERIMENT_SEEDS", "42,43,44")
        )
    )

    SUITE_NAME: str = os.getenv(
        "EXPERIMENT_SUITE_NAME",
        "final_comparison",
    )


# ==========================================================
# Logging Configuration
# ==========================================================


@dataclass
class LoggingConfig:
    """Logging and output directories."""

    ROOT_DIR: Path = Path.cwd()

    RESULTS_DIR: Path = ROOT_DIR / "results"
    LOG_DIR: Path = RESULTS_DIR / "logs"
    MODEL_DIR: Path = RESULTS_DIR / "models"
    METRICS_DIR: Path = RESULTS_DIR / "metrics"
    GRAPH_DIR: Path = RESULTS_DIR / "graphs"


# ==========================================================
# Combined Configuration
# ==========================================================


@dataclass
class Config:
    """Main project configuration."""

    backend: BackendConfig = field(default_factory=BackendConfig)

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    agent: AgentConfig = field(default_factory=AgentConfig)

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    llm: LLMConfig = field(default_factory=LLMConfig)

    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ==========================================================
# Global Configuration Instance
# ==========================================================


config = Config()
