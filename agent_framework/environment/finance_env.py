"""
finance_env.py

Gymnasium-style wrapper around the Finance RL Backend.

This class connects:
RL Agent <-> API Client <-> Backend Environment

The backend is responsible for:
- state transitions
- rewards
- episode tracking
- audit logging

The environment only manages interaction.
"""

from typing import Any, Dict, Tuple
from environment.api_client import APIClient
from environment.state_encoder import StateEncoder
from environment.reward_processor import RewardProcessor
from config.config import config


class FinanceEnvironment:
    """
    Finance RL Environment.

    Provides:
        reset()
        step()
        render()
        close()
    """

    def __init__(self):

        self.client = APIClient()

        self.encoder = StateEncoder()

        self.reward_processor = RewardProcessor()

        self.episode_id = None

        self.current_state = None

        self.current_step = 0

        self.completed = False

        self.done = False

        # Login once when environment starts
        self._authenticate()

    # ==========================================================
    # Authentication
    # ==========================================================

    def _authenticate(self):

        response = self.client.login()

        if not response["success"]:

            raise Exception("Backend authentication failed")

    # ==========================================================
    # Reset Environment
    # ==========================================================

    def reset(self) -> Dict[str, Any]:
        """
        Reset backend environment.

        Backend:
            - clears previous sandbox data
            - generates new random scenario
            - resets rewards

        Returns:
            Initial observation
        """

        self.current_step = 0
        self.completed = False
        self.done = False

        # Reset backend
        reset_response = self.client.reset_environment()

        if not reset_response["success"]:

            raise Exception("Environment reset failed")

        # Start episode tracking

        episode_payload = {
            "agentType": config.agent.AGENT_TYPE,
            "algorithm": config.agent.ALGORITHM,
            "goal": config.agent.TASK,
            "initialState": reset_response.get("stateCreated"),
        }

        episode_response = self.client.start_episode(episode_payload)

        if episode_response["success"]:

            self.episode_id = episode_response["data"].get("episodeId")

        # Get initial state

        state_response = self.client.get_state()

        self.current_state = state_response["data"]

        observation = self.encoder.encode(
            self.current_state
        )

        return observation

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one agent action.

        Parameters:
            action:
                Action selected by RL agent


        Returns:

            next_state,
            reward,
            done,
            info

        """
        if self.done:
            raise RuntimeError(
                "Episode has already terminated. Call reset()."
            )

        self.current_step += 1

        # ---------------------------------------------------
        # State before action
        # ---------------------------------------------------
        state_before = self.current_state

        # ---------------------------------------------------
        # Execute action
        # ---------------------------------------------------
        response = self.action_space.execute(
            action,
            self.current_state,
        )

        success = response.get("success", False)
        endpoint = response.get("endpoint")
        message = (
            response.get("message")
            or response.get("data", {}).get("message")
        )

        # ---------------------------------------------------
        # Get updated state
        # ---------------------------------------------------
        state_response = self.client.get_state()

        if state_response["success"]:
            self.current_state = state_response["data"]

            # Backend tells us if the goal has been achieved
            self.completed = self.current_state.get("completed", False)

        state_after = self.current_state

        # ---------------------------------------------------
        # Get reward
        # ---------------------------------------------------
        reward_response = self.client.get_reward(
            self.episode_id
        )

        reward, reward_info = (
            self.reward_processor.process(
                reward_response
            )
        )
        # ---------------------------------------------------
        # Record step
        # ---------------------------------------------------
        self.record_step(
            action=action,
            endpoint=endpoint,
            reward=reward,
            success=success,
            state_before=state_before,
            state_after=state_after,
            message=message,
        )

        # ---------------------------------------------------
        # Check episode termination
        # ---------------------------------------------------

        terminated_reason = None

        # Goal achieved
        if self.completed:
            self.done = True
            terminated_reason = "GOAL_COMPLETED"

        # Episode ran out of steps
        elif self.current_step >= config.environment.MAX_STEPS_PER_EPISODE:
            self.done = True
            terminated_reason = "MAX_STEPS"

        # Backend/environment failure
        elif response.get("environment_error", False):
            self.done = True
            terminated_reason = "ENVIRONMENT_ERROR"

        else:
            self.done = False
        # ---------------------------------------------------
        # End episode if required
        # ---------------------------------------------------
        if self.done:
            self.end_episode(terminated_reason)

        info = {
            "backend_response": response,
            "step": self.current_step,
            "success": success,
            "message": message,
            "terminated_reason": terminated_reason,
        }

        info.update(reward_info)

        next_observation = self.encoder.encode(
            self.current_state
        )

        return (
            next_observation,
            reward,
            self.done,
            info
        )

    # ==========================================================
    # Record Episode Step
    # ==========================================================

    def record_step(
        self, action, endpoint, reward, success, state_before, state_after, message=None
    ):

        if not self.episode_id:

            return

        payload = {
            "step": self.current_step,
            "action": action,
            "endpoint": endpoint,
            "reward": reward,
            "success": success,
            "stateBefore": state_before,
            "stateAfter": state_after,
            "message": message,
        }

        self.client.record_step(self.episode_id, payload)

    # ==========================================================
    # End Episode
    # ==========================================================

    def end_episode(self, terminated_reason):

        if not self.episode_id:

            return

        payload = {"finalState": self.current_state, "completed": self.completed, "terminatedReason": terminated_reason}

        self.client.end_episode(self.episode_id, payload)

        self.done = True

    # ==========================================================
    # Get Current State
    # ==========================================================

    def get_state(self):

        response = self.client.get_state()

        self.current_state = response["data"]

        return self.current_state

    # ==========================================================
    # Render
    # ==========================================================

    def render(self):

        print("\n===== Finance Environment =====")

        print(self.current_state)

        print("Step:", self.current_step)

    # ==========================================================
    # Close
    # ==========================================================

    def close(self, terminated_reason):

        if self.episode_id:

            self.end_episode(terminated_reason)

        self.client.session.close()
