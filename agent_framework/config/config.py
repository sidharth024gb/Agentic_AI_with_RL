import os

import torch

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path="./config/.env")


# ==========================================================
# Backend Configuration
# ==========================================================


@dataclass(frozen=True)
class BackendConfig:
    """Backend API configuration."""

    BASE_URL: str = os.getenv(
        "BASE_URL",
        "http://localhost:5000/api",
    )

    TIMEOUT: int = 30

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


@dataclass(frozen=True)
class EnvironmentConfig:
    """RL environment configuration."""

    ENV_NAME: str = "FinanceSandbox-v1"

    MAX_STEPS_PER_EPISODE: int = 20

    RANDOM_SEED: int = 42

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


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    TOTAL_EPISODES: int = 1000

    EVALUATION_EPISODES: int = 100

    GAMMA: float = 0.99

    LEARNING_RATE: float = 3e-4

    BATCH_SIZE: int = 64

    UPDATE_INTERVAL: int = 2048

    SAVE_EVERY: int = 100

    LOG_EVERY: int = 10

    GAE_LAMBDA: float = 0.95

    CLIP_EPSILON: float = 0.2

    EPOCHS: int = 10

    HIDDEN_NEURON_SIZE: int = 256

    # Metrics / visualization
    MOVING_AVERAGE_WINDOW: int = 50

    CONVERGENCE_WINDOW: int = 50

    CONVERGENCE_SUCCESS_THRESHOLD: float = 0.90


# ==========================================================
# Agent Configuration
# ==========================================================


@dataclass(frozen=True)
class AgentConfig:
    """Agent settings."""

    # RL / LLM_RL
    AGENT_TYPE: str = "LLM_RL"

    # PPO / DQN / Q_LEARNING / SAC
    ALGORITHM: str = "PPO"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Task/goal stored in Episode.goal
    TASK: str = "Pay all valid invoices"


# ==========================================================
# LLM Configuration
# ==========================================================


@dataclass(frozen=True)
class LLMConfig:
    """LLM planner configuration."""

    # ------------------------------------------------------
    # Ollama
    # ------------------------------------------------------

    MODEL: str = "llama3"

    BASE_URL: str = "http://localhost:11434"

    TIMEOUT: int = 120

    TEMPERATURE: float = 0.1

    # ------------------------------------------------------
    # Cache
    # ------------------------------------------------------

    USE_CACHE: bool = True

    CACHE_DIR: Path = Path.cwd() / "results" / "llm_cache"


# ==========================================================
# Experiment Configuration
# ==========================================================


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata."""

    EXPERIMENT_NAME: str = "llm_ppo" # llm_ppo / ppo_baseline

    DESCRIPTION: str = "Baseline PPO agent with LLM planning."

    # ------------------------------------------------------
    # TRAIN / EVALUATION / TEST
    # ------------------------------------------------------

    PHASE: str = "TRAIN"

    # ------------------------------------------------------
    # Guidance Mode
    #
    # NONE
    # INPUT
    # REWARD_SHAPING
    # INPUT_AND_REWARD
    # ------------------------------------------------------

    GUIDANCE_MODE: str = "INPUT_AND_REWARD"

    # ------------------------------------------------------
    # Extra reward given when the LLM procedure is followed.
    #
    # Only used for:
    # REWARD_SHAPING
    # INPUT_AND_REWARD
    # ------------------------------------------------------

    GUIDANCE_BONUS: float = 5.0

    ENVIRONMENT_VERSION: str = "v1"


# ==========================================================
# Logging Configuration
# ==========================================================


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
