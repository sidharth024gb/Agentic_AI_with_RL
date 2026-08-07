import torch
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================
# Backend Configuration
# ============================


@dataclass(frozen=True)
class BackendConfig:
    """Backend API configuration."""

    BASE_URL: str = os.getenv("BASE_URL")
    TIMEOUT: int = 30

    # Authentication
    USERNAME: str = os.getenv("USERNAME")
    PASSWORD: str = os.getenv("PASSWORD")

    # Endpoints
    LOGIN_ENDPOINT: str = "/auth/login"
    PROFILE_ENDPOINT: str = "/auth/me"

    AUTH_REGISTER = "/auth/register"
    AUTH_LOGIN = "/auth/login"
    AUTH_ME = "/auth/me"

    EPISODE = "/episode"
    EPISODE_START = "/episode/start"
    EPISODE_STEP = "/step"
    EPISODE_END = "/end"

    SANDBOX_RESET = "/sandbox/reset"
    SANDBOX_STATE = "/sandbox/state"
    SANDBOX_REWARD = "/sandbox/reward"

    SUPPLIER = "/supplier"
    SUPPLIER_VALIDATE = "/supplier/validate"

    INVOICE = "/invoice" # get invoices and invoice by id [/invoice/:id]
    INVOICE_DUPLICATE_CHECK = "/invoice/duplicate-check"

    ACCOUNT = "/account"
    ACCOUNT_BUDGET_CHECK = "/account/budget/check"
    ACCOUNT_CASH_POSITION = "/account/cash-position"

    APPROVAL_APPROVE = "/approval/approve"

    REPORT_TRANSACTIONS = "/report/transactions"
    REPORT_GENERATE_REPORT = "/report/generate-report"

    PAYMENT_PAY = "/payment/pay"
    PAYMENT_CANCEL_PAYMENT = "/payment/cancel-payment"
    PAYMENT_RETRY_PAYMENT = "/payment/retry-payment"


# ============================
# Environment Configuration
# ============================


@dataclass(frozen=True)
class EnvironmentConfig:
    """RL environment configuration."""

    ENV_NAME: str = "FinanceSandbox-v1"

    MAX_STEPS_PER_EPISODE: int = 50

    RANDOM_SEED: int = 42

    OBSERVATION_TYPE: str = "vector"

    ACTION_SPACE_SIZE: int = 6


# ============================
# Training Configuration
# ============================


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters."""

    TOTAL_EPISODES: int = 1000

    GAMMA: float = 0.99

    LEARNING_RATE: float = 3e-4

    BATCH_SIZE: int = 64

    UPDATE_INTERVAL: int = 2048

    SAVE_EVERY: int = 100

    EVALUATE_EVERY: int = 50


# ============================
# Agent Configuration
# ============================


@dataclass(frozen=True)
class AgentConfig:
    """Agent settings."""

    AGENT_TYPE: str = "RL"  # RL / LLM_RL

    ALGORITHM: str = "PPO"  # PPO / DQN / Q_LEARNING

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    TASK: str = ""


# ============================
# Experiment Configuration
# ============================


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata."""

    EXPERIMENT_NAME: str = "ppo_baseline"

    DESCRIPTION: str = "Baseline PPO agent without LLM planning."

    LLM_MODEL: str | None = None

    ENVIRONMENT_VERSION: str = "v1"


# ============================
# Logging Configuration
# ============================


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and output directories."""

    ROOT_DIR: Path = Path.cwd()

    RESULTS_DIR: Path = ROOT_DIR / "results"

    LOG_DIR: Path = RESULTS_DIR / "logs"

    MODEL_DIR: Path = RESULTS_DIR / "models"

    METRICS_DIR: Path = RESULTS_DIR / "metrics"

    GRAPH_DIR: Path = RESULTS_DIR / "graphs"


# ============================
# Combined Configuration
# ============================


@dataclass(frozen=True)
class Config:
    """Main project configuration."""

    backend: BackendConfig = field(default_factory=BackendConfig)

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)

    agent: AgentConfig = field(default_factory=AgentConfig)

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    logging: LoggingConfig = field(default_factory=LoggingConfig)


# Global configuration instance
config = Config()
