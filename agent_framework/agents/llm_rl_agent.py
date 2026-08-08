"""
llm_rl_agent.py

LLM-enhanced PPO agent.

The agent extends PPOAgent and adds an LLM planning component.

Architecture:

    Environment State
            |
            v
        LLM Planner
            |
            v
      Planning Hint
            |
            v
       PPO Policy
            |
            v
       Actual Action
            |
            v
       Environment

The LLM does NOT execute backend actions.
The PPO policy remains responsible for action selection.
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from agents.ppo_agent import PPOAgent

from llm.planner import LLMPlanner
from llm.prompts import PromptBuilder
from llm.parser import PlanningParser


class LLMRLAgent(PPOAgent):
    """
    PPO agent enhanced with LLM-based planning guidance.
    """

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        goal: str,
        llm_planner: Optional[LLMPlanner] = None,
        llm_parser: Optional[PlanningParser] = None,
        llm_temperature: Optional[float] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        hidden_size: int = 256,
        planning_frequency: int = 1,
    ):
        """
        Initialize the LLM + PPO agent.

        Parameters
        ----------
        observation_size : int
            Size of encoded environment observation.

        action_size : int
            Number of available RL actions.

        goal : str
            Current task goal.

        llm_planner : LLMPlanner, optional
            Ollama planner instance.

        llm_parser : PlanningParser, optional
            Parser for LLM responses.

        llm_temperature : float, optional
            LLM temperature if a planner is created internally.

        learning_rate : float
            PPO learning rate.

        gamma : float
            Discount factor.

        gae_lambda : float
            GAE lambda.

        clip_epsilon : float
            PPO clipping parameter.

        epochs : int
            PPO optimization epochs.

        hidden_size : int
            Neural network hidden layer size.

        planning_frequency : int
            Number of environment steps between LLM calls.
        """

        super().__init__(
            observation_size=observation_size,
            action_size=action_size,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
            epochs=epochs,
            hidden_size=hidden_size,
        )

        self.goal = goal

        self.planner = (
            llm_planner
            if llm_planner is not None
            else LLMPlanner(temperature=llm_temperature)
        )

        self.parser = llm_parser if llm_parser is not None else PlanningParser()

        self.planning_frequency = max(
            1,
            planning_frequency,
        )

        self.current_step = 0

        self.current_plan = None

        self.last_llm_response = ""

        self.last_planning_hint = None

        self.llm_calls = 0

        self.llm_failures = 0

        self.total_llm_latency = 0.0

    # ==========================================================
    # Goal
    # ==========================================================

    def set_goal(self, goal: str):
        """
        Update the current task goal.
        """

        self.goal = goal

        self.current_plan = None

        self.last_llm_response = ""

        self.last_planning_hint = None

    # ==========================================================
    # LLM Planning
    # ==========================================================

    def plan(
        self,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM for a planning recommendation.

        The LLM only provides guidance.
        It does not select or execute the RL action.
        """

        prompt = PromptBuilder.build_planning_prompt(
            state=state,
            goal=self.goal,
        )

        result = self.planner.plan(
            prompt=prompt,
            system_prompt=PromptBuilder.SYSTEM_PROMPT,
        )

        self.llm_calls += 1

        latency = result.get(
            "latency",
            0.0,
        )

        self.total_llm_latency += latency

        if not result.get("success", False):

            self.llm_failures += 1

            self.current_plan = {
                "recommendation": "",
                "keywords": [],
                "confidence": 0.0,
                "valid": False,
                "llm_success": False,
                "latency": latency,
                "error": result.get("error"),
            }

            return self.current_plan

        response = result.get(
            "response",
            "",
        )

        self.last_llm_response = response

        parsed = self.parser.parse(response)

        parsed["llm_success"] = True

        parsed["latency"] = latency

        parsed["error"] = None

        self.current_plan = parsed

        self.last_planning_hint = parsed

        return parsed

    # ==========================================================
    # Action Selection
    # ==========================================================

    def select_action(
        self,
        state,
        raw_state: Optional[Dict[str, Any]] = None,
    ):
        """
        Select an action using PPO with optional LLM guidance.

        Parameters
        ----------
        state : array-like
            Encoded state used by PPO.

        raw_state : dict, optional
            Original environment state used by the LLM.

        Returns
        -------
        tuple
            action,
            log_probability,
            value
        """

        self.current_step += 1

        # ------------------------------------------------------
        # Ask LLM according to planning frequency
        # ------------------------------------------------------

        should_plan = self.current_plan is None or (
            self.current_step % self.planning_frequency == 0
        )

        if should_plan and raw_state is not None:

            self.plan(raw_state)

        # ------------------------------------------------------
        # PPO still selects the actual action
        # ------------------------------------------------------

        return super().select_action(state)

    # ==========================================================
    # Episode Reset
    # ==========================================================

    def reset_planning(self):
        """
        Reset LLM planning state at the beginning of an episode.
        """

        self.current_step = 0

        self.current_plan = None

        self.last_llm_response = ""

        self.last_planning_hint = None

    # ==========================================================
    # Planning Information
    # ==========================================================

    def get_planning_info(self) -> Dict[str, Any]:
        """
        Return information about the current LLM planning state.
        """

        return {
            "goal": self.goal,
            "current_plan": self.current_plan,
            "last_response": self.last_llm_response,
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "total_llm_latency": (self.total_llm_latency),
        }

    # ==========================================================
    # Statistics
    # ==========================================================

    def get_llm_statistics(self) -> Dict[str, Any]:
        """
        Return LLM-specific experiment statistics.
        """

        average_latency = 0.0

        if self.llm_calls > 0:

            average_latency = self.total_llm_latency / self.llm_calls

        success_rate = 0.0

        if self.llm_calls > 0:

            success_rate = (self.llm_calls - self.llm_failures) / self.llm_calls

        return {
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "llm_success_rate": success_rate,
            "total_llm_latency": (self.total_llm_latency),
            "average_llm_latency": (average_latency),
        }

    # ==========================================================
    # Save
    # ==========================================================

    def save(
        self,
        path: str,
    ):
        """
        Save PPO networks.

        LLM planning state is intentionally not saved as part
        of the neural-network checkpoint.
        """

        super().save(path)

    # ==========================================================
    # Load
    # ==========================================================

    def load(
        self,
        path: str,
    ):
        """
        Load PPO networks.
        """

        super().load(path)
