import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
from config import Config

class FinanceEnv(gym.Env):
    """
    Custom Gymnasium Environment wrapping Express.js Corporate Finance Backend.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(FinanceEnv, self).__init__()

        # Action Space: Discrete actions corresponding to financial endpoints
        # 0: Fetch Invoices, 1: Post Invoice, 2: Approve Invoice, 3: Execute Payment, 4: Reconcile
        self.action_space = spaces.Discrete(Config.ACTION_DIM)

        # Observation Space: Numeric vector representing current sandbox state metrics
        self.observation_space = spaces.Box(
            low=0, high=100000, shape=(Config.OBS_DIM,), dtype=np.float32
        )

        self.token = None
        self.current_step = 0

    def _login(self):
        """Authenticates with Express backend to obtain JWT Bearer Token."""
        try:
            res = requests.post(
                Config.AUTH_LOGIN_URL,
                json={"username": Config.ADMIN_USERNAME, "password": Config.ADMIN_PASSWORD},
                timeout=5
            )
            if res.status_code == 200:
                self.token = res.json().get("token")
            else:
                self.token = "mock_token"  # Fallback for mock backend
        except Exception:
            self.token = "mock_token"

    def _get_headers(self):
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._login()

        # Reset database state in sandbox
        try:
            requests.post(Config.SANDBOX_RESET_URL, headers=self._get_headers(), timeout=5)
        except Exception:
            pass

        obs = self._get_observation()
        info = {}
        return obs, info

    def _get_observation(self):
        try:
            res = requests.get(Config.SANDBOX_STATE_URL, headers=self._get_headers(), timeout=5)
            if res.status_code == 200:
                data = res.json()
                return np.array([
                    data.get("pending_invoices", 0),
                    data.get("unapproved_invoices", 0),
                    data.get("unpaid_invoices", 0),
                    data.get("account_balance", 1000)
                ], dtype=np.float32)
        except Exception:
            pass
        return np.array([5, 3, 2, 1000.0], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        headers = self._get_headers()
        reward = -0.1  # Step penalty to encourage efficiency
        done = False
        truncated = False

        try:
            if action == 0:  # Fetch Invoices
                requests.get(f"{Config.FINANCE_API_URL}/invoices", headers=headers, timeout=5)
                reward += 0.5
            elif action == 1:  # Post Invoice
                requests.post(f"{Config.FINANCE_API_URL}/invoices", json={"amount": 100}, headers=headers, timeout=5)
                reward += 1.0
            elif action == 2:  # Approve Invoice
                requests.patch(f"{Config.FINANCE_API_URL}/invoices/1/status", json={"status": "APPROVED"}, headers=headers, timeout=5)
                reward += 2.0
            elif action == 3:  # Execute Payment
                requests.post(f"{Config.FINANCE_API_URL}/pay", json={"invoiceId": 1}, headers=headers, timeout=5)
                reward += 3.0
            elif action == 4:  # Reconcile Transaction
                requests.post(f"{Config.FINANCE_API_URL}/reconcile", json={"transactionId": 1}, headers=headers, timeout=5)
                reward += 5.0  # Main goal achieved
                done = True
        except Exception:
            reward -= 1.0  # Penalty for failed API call

        if self.current_step >= Config.MAX_STEPS_PER_EPISODE:
            truncated = True

        obs = self._get_observation()
        return obs, reward, done, truncated, {"step": self.current_step}