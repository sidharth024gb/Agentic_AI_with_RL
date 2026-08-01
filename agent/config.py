import os

class Config:
    # Backend Server Configuration
    BASE_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
    AUTH_LOGIN_URL = f"{BASE_URL}/api/auth/login"
    SANDBOX_RESET_URL = f"{BASE_URL}/api/sandbox/reset"
    SANDBOX_STATE_URL = f"{BASE_URL}/api/sandbox/state"
    FINANCE_API_URL = f"{BASE_URL}/api/finance"

    # API Credentials for Agent Authentication
    ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "rl_agent_admin")
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")

    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # or mistral, llama2, etc.

    # RL Hyperparameters (PPO)
    LEARNING_RATE = 0.0003
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    BATCH_SIZE = 64
    N_STEPS = 2048
    TOTAL_TIMESTEPS = 20000

    # Environment Parameters
    MAX_STEPS_PER_EPISODE = 20
    ACTION_DIM = 5  # Discrete actions: [0: Read, 1: Create, 2: Approve, 3: Pay, 4: Reconcile]
    OBS_DIM = 4     # Observation vector size: [pending_invoices, unapproved_invoices, unpaid_invoices, balance]