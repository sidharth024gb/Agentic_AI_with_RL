"""
constants.py

Shared constants used across the finance RL agent project.

Project-specific configuration values should remain in:
    config/config.py
    config/api_config.py
    config/experiment_config.py

This file contains stable constants used by multiple modules.
"""

# ==============================================================
# Agent Types
# ==============================================================

PPO_AGENT = "PPO"

LLM_PPO_AGENT = "LLM_PPO"


# ==============================================================
# Algorithms
# ==============================================================

PPO = "PPO"


# ==============================================================
# Environment
# ==============================================================

DEFAULT_MAX_EPISODE_STEPS = 50


# ==============================================================
# Episode Status
# ==============================================================

EPISODE_RUNNING = "RUNNING"

EPISODE_COMPLETED = "COMPLETED"

EPISODE_TERMINATED = "TERMINATED"

EPISODE_MAX_STEPS = "MAX_STEPS"

EPISODE_ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"


# ==============================================================
# Termination Reasons
# ==============================================================

TERMINATED_GOAL_ACHIEVED = "GOAL_ACHIEVED"

TERMINATED_MAX_STEPS = "MAX_STEPS"

TERMINATED_ENVIRONMENT_ERROR = "ENVIRONMENT_ERROR"

TERMINATED_TRAINING_COMPLETE = "TRAINING_COMPLETE"

TERMINATED_EVALUATION_COMPLETE = "EVALUATION_COMPLETE"


# ==============================================================
# Reward Constants
# ==============================================================

# Positive rewards

REWARD_SUCCESS = 10

REWARD_INVOICE_FOUND = 5

REWARD_SUPPLIER_VALIDATED = 10

REWARD_BUDGET_CHECK_SUCCESS = 10

REWARD_PAYMENT_SUCCESS = 30

REWARD_REPORT_GENERATED = 5


# Negative rewards

REWARD_INVALID_ACTION = -5

REWARD_INVOICE_NOT_FOUND = -10

REWARD_DUPLICATE_INVOICE = -15

REWARD_SUPPLIER_INACTIVE = -20

REWARD_SUPPLIER_HIGH_RISK = -15

REWARD_BUDGET_EXCEEDED = -25

REWARD_INSUFFICIENT_BALANCE = -30

REWARD_PAYMENT_FAILED = -20

REWARD_UNAUTHORIZED_ACTION = -50


# Neutral / system rewards

REWARD_NONE = 0

REWARD_SYSTEM_ERROR = None


# ==============================================================
# Reward Names
# ==============================================================

REWARD_NAME_SUCCESS = "SUCCESS"

REWARD_NAME_INVOICE_FOUND = "INVOICE_FOUND"

REWARD_NAME_SUPPLIER_VALIDATED = "SUPPLIER_VALIDATED"

REWARD_NAME_BUDGET_CHECK_SUCCESS = "BUDGET_CHECK_SUCCESS"

REWARD_NAME_PAYMENT_SUCCESS = "PAYMENT_SUCCESS"

REWARD_NAME_REPORT_GENERATED = "REPORT_GENERATED"

REWARD_NAME_INVALID_ACTION = "INVALID_ACTION"

REWARD_NAME_INVOICE_NOT_FOUND = "INVOICE_NOT_FOUND"

REWARD_NAME_DUPLICATE_INVOICE = "DUPLICATE_INVOICE"

REWARD_NAME_SUPPLIER_INACTIVE = "SUPPLIER_INACTIVE"

REWARD_NAME_SUPPLIER_HIGH_RISK = "SUPPLIER_HIGH_RISK"

REWARD_NAME_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

REWARD_NAME_INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"

REWARD_NAME_PAYMENT_FAILED = "PAYMENT_FAILED"

REWARD_NAME_UNAUTHORIZED_ACTION = "UNAUTHORIZED_ACTION"

REWARD_NAME_SYSTEM_ERROR = "SYSTEM_ERROR"

REWARD_NAME_NONE = "NONE"


# ==============================================================
# Reward Mapping
# ==============================================================

REWARDS = {
    REWARD_NAME_SUCCESS: REWARD_SUCCESS,
    REWARD_NAME_INVOICE_FOUND: (REWARD_INVOICE_FOUND),
    REWARD_NAME_SUPPLIER_VALIDATED: (REWARD_SUPPLIER_VALIDATED),
    REWARD_NAME_BUDGET_CHECK_SUCCESS: (REWARD_BUDGET_CHECK_SUCCESS),
    REWARD_NAME_PAYMENT_SUCCESS: (REWARD_PAYMENT_SUCCESS),
    REWARD_NAME_REPORT_GENERATED: (REWARD_REPORT_GENERATED),
    REWARD_NAME_INVALID_ACTION: (REWARD_INVALID_ACTION),
    REWARD_NAME_INVOICE_NOT_FOUND: (REWARD_INVOICE_NOT_FOUND),
    REWARD_NAME_DUPLICATE_INVOICE: (REWARD_DUPLICATE_INVOICE),
    REWARD_NAME_SUPPLIER_INACTIVE: (REWARD_SUPPLIER_INACTIVE),
    REWARD_NAME_SUPPLIER_HIGH_RISK: (REWARD_SUPPLIER_HIGH_RISK),
    REWARD_NAME_BUDGET_EXCEEDED: (REWARD_BUDGET_EXCEEDED),
    REWARD_NAME_INSUFFICIENT_BALANCE: (REWARD_INSUFFICIENT_BALANCE),
    REWARD_NAME_PAYMENT_FAILED: (REWARD_PAYMENT_FAILED),
    REWARD_NAME_UNAUTHORIZED_ACTION: (REWARD_UNAUTHORIZED_ACTION),
    REWARD_NAME_SYSTEM_ERROR: (REWARD_SYSTEM_ERROR),
    REWARD_NAME_NONE: (REWARD_NONE),
}


# ==============================================================
# Action Categories
# ==============================================================

ACTION_INVOICE = "INVOICE"

ACTION_APPROVAL = "APPROVAL"

ACTION_PAYMENT = "PAYMENT"

ACTION_SUPPLIER = "SUPPLIER"

ACTION_ACCOUNT = "ACCOUNT"

ACTION_REPORT = "REPORT"


# ==============================================================
# Experiment Result Keys
# ==============================================================

METRIC_REWARD = "reward"

METRIC_SUCCESS_RATE = "success_rate"

METRIC_AVERAGE_STEPS = "average_steps"

METRIC_CONVERGENCE_EPISODE = "convergence_episode"

METRIC_TRAINING_TIME = "training_time"

METRIC_EVALUATION_TIME = "evaluation_time"

METRIC_LLM_CALLS = "llm_calls"

METRIC_LLM_LATENCY = "average_llm_latency"

METRIC_LLM_TOTAL_LATENCY = "total_llm_latency"


# ==============================================================
# File / Directory Names
# ==============================================================

RESULTS_DIRECTORY = "results"

LOGS_DIRECTORY = "results/logs"

MODELS_DIRECTORY = "results/models"

METRICS_DIRECTORY = "results/metrics"

GRAPHS_DIRECTORY = "results/graphs"

CHECKPOINT_DIRECTORY = "models/checkpoints"


# ==============================================================
# Experiment Names
# ==============================================================

EXPERIMENT_PPO = "ppo"

EXPERIMENT_LLM_PPO = "llm_ppo"

EXPERIMENT_PPO_COMPARISON = "ppo_vs_llm_ppo"
