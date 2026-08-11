"""
tests/test_agent.py

Tests for the RL agents.

This test module focuses on:

    - PPO agent creation
    - LLM + PPO agent creation
    - policy network output
    - value network output
    - action selection
    - rollout interaction
    - basic agent/environment compatibility

The real backend is not required for these tests.
Backend/API behaviour is tested separately in:

    tests/test_api.py
    tests/test_environment.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ==============================================================
# Project Root
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==============================================================
# Project Imports
# ==============================================================

from agents.ppo_agent import PPOAgent
from agents.llm_rl_agent import LLMRLAgent

from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork

from environment.action_space import ActionSpace

# ==============================================================
# Mock Environment
# ==============================================================


class MockFinanceEnv:
    """
    Lightweight environment used for agent unit tests.

    This avoids making real HTTP requests to the finance backend.
    """

    def __init__(self):

        self.observation_space = None
        self.action_space = ActionSpace()

        self.current_state = {
            "invoices": [],
            "suppliers": [],
            "accounts": [],
            "transactions": [],
        }

        self.done = False
        self.completed = False
        self.current_step = 0

    def reset(self):

        self.current_step = 0
        self.done = False
        self.completed = False

        return self.current_state

    def step(self, action):

        self.current_step += 1

        next_state = self.current_state

        reward = 0.0

        terminated = False

        truncated = self.current_step >= 10

        info = {}

        if truncated:

            self.done = True

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    def close(
        self,
        terminated_reason=None,
    ):
        """
        Mock close method.
        """

        self.done = True


# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture
def mock_env():

    return MockFinanceEnv()


@pytest.fixture
def state_dimension():

    return 32


@pytest.fixture
def action_dimension():

    return 10


# ==============================================================
# Policy Network Tests
# ==============================================================


def test_policy_network_creation(
    state_dimension,
    action_dimension,
):
    """
    Verify that the policy network can be created.
    """

    policy = PolicyNetwork(
        state_dim=state_dimension,
        action_dim=action_dimension,
    )

    assert policy is not None


def test_policy_network_output_shape(
    state_dimension,
    action_dimension,
):
    """
    Verify that the policy produces action logits
    with the expected action dimension.
    """

    policy = PolicyNetwork(
        state_dim=state_dimension,
        action_dim=action_dimension,
    )

    state = torch.randn(
        1,
        state_dimension,
    )

    output = policy(state)

    assert output is not None

    assert output.shape[-1] == (action_dimension)


# ==============================================================
# Value Network Tests
# ==============================================================


def test_value_network_creation(
    state_dimension,
):
    """
    Verify that the value network can be created.
    """

    value_network = ValueNetwork(
        state_dim=state_dimension,
    )

    assert value_network is not None


def test_value_network_output_shape(
    state_dimension,
):
    """
    Verify that the value network produces
    one scalar value per state.
    """

    value_network = ValueNetwork(
        state_dim=state_dimension,
    )

    state = torch.randn(
        1,
        state_dimension,
    )

    output = value_network(state)

    assert output is not None

    assert output.shape[0] == 1

    assert output.shape[-1] == 1


# ==============================================================
# PPO Agent Tests
# ==============================================================


def test_ppo_agent_creation(
    mock_env,
):
    """
    Verify PPO agent creation.
    """

    agent = PPOAgent(env=mock_env)

    assert agent is not None


def test_ppo_agent_has_policy(
    mock_env,
):
    """
    Verify that the PPO agent contains a policy network.
    """

    agent = PPOAgent(env=mock_env)

    assert hasattr(
        agent,
        "policy",
    )

    assert agent.policy is not None


def test_ppo_agent_has_value_network(
    mock_env,
):
    """
    Verify that PPO contains a value function.
    """

    agent = PPOAgent(env=mock_env)

    assert hasattr(
        agent,
        "value_network",
    )

    assert agent.value_network is not None


# ==============================================================
# Action Selection Tests
# ==============================================================


def test_ppo_action_selection(
    mock_env,
):
    """
    Verify that PPO can select an action.
    """

    agent = PPOAgent(env=mock_env)

    state = mock_env.reset()

    action = agent.select_action(state)

    assert action is not None


def test_ppo_action_is_valid(
    mock_env,
):
    """
    Verify that the action selected by PPO belongs
    to the environment action space.
    """

    agent = PPOAgent(env=mock_env)

    state = mock_env.reset()

    action = agent.select_action(state)

    valid_actions = mock_env.action_space.get_valid_actions(state)

    assert action in valid_actions


# ==============================================================
# Environment Interaction
# ==============================================================


def test_ppo_environment_interaction(
    mock_env,
):
    """
    Verify that PPO can perform a basic environment step.
    """

    agent = PPOAgent(env=mock_env)

    state = mock_env.reset()

    action = agent.select_action(state)

    result = mock_env.step(action)

    assert len(result) == 5

    next_state = result[0]
    reward = result[1]
    terminated = result[2]
    truncated = result[3]

    assert next_state is not None

    assert isinstance(
        reward,
        (int, float),
    )

    assert isinstance(
        terminated,
        bool,
    )

    assert isinstance(
        truncated,
        bool,
    )


# ==============================================================
# LLM + PPO Tests
# ==============================================================


def test_llm_ppo_agent_creation(
    mock_env,
):
    """
    Verify LLM + PPO agent creation.

    The test does not make an actual Ollama request.
    """

    agent = LLMRLAgent(env=mock_env)

    assert agent is not None


def test_llm_ppo_has_environment(
    mock_env,
):
    """
    Verify that the LLM + PPO agent keeps a reference
    to the environment.
    """

    agent = LLMRLAgent(env=mock_env)

    assert hasattr(
        agent,
        "env",
    )

    assert agent.env is mock_env


# ==============================================================
# State Handling
# ==============================================================


def test_agent_can_receive_state(
    mock_env,
):
    """
    Verify that the agent can receive an environment state.
    """

    agent = PPOAgent(env=mock_env)

    state = mock_env.reset()

    assert state is not None

    action = agent.select_action(state)

    assert action is not None


# ==============================================================
# Deterministic Policy Test
# ==============================================================


def test_policy_deterministic_output(
    state_dimension,
    action_dimension,
):
    """
    Verify that the policy produces repeatable output
    when evaluated on the same input without stochastic
    sampling.

    This test is intentionally lightweight and only verifies
    tensor generation.
    """

    torch.manual_seed(42)

    policy = PolicyNetwork(
        state_dim=state_dimension,
        action_dim=action_dimension,
    )

    state = torch.randn(
        1,
        state_dimension,
    )

    with torch.no_grad():

        output_1 = policy(state)

        output_2 = policy(state)

    assert torch.allclose(
        output_1,
        output_2,
    )


# ==============================================================
# Batch State Test
# ==============================================================


def test_policy_batch_processing(
    state_dimension,
    action_dimension,
):
    """
    Verify that the policy can process multiple states
    in a batch.
    """

    policy = PolicyNetwork(
        state_dim=state_dimension,
        action_dim=action_dimension,
    )

    batch_size = 8

    states = torch.randn(
        batch_size,
        state_dimension,
    )

    outputs = policy(states)

    assert outputs.shape[0] == (batch_size)

    assert outputs.shape[-1] == (action_dimension)


# ==============================================================
# Numerical Stability
# ==============================================================


def test_policy_output_contains_no_nan(
    state_dimension,
    action_dimension,
):
    """
    Verify that the policy does not immediately produce
    NaN values for a normal input.
    """

    policy = PolicyNetwork(
        state_dim=state_dimension,
        action_dim=action_dimension,
    )

    state = torch.randn(
        4,
        state_dimension,
    )

    output = policy(state)

    assert not torch.isnan(output).any()


def test_value_output_contains_no_nan(
    state_dimension,
):
    """
    Verify that the value network does not immediately
    produce NaN values.
    """

    value_network = ValueNetwork(
        state_dim=state_dimension,
    )

    state = torch.randn(
        4,
        state_dimension,
    )

    output = value_network(state)

    assert not torch.isnan(output).any()


# ==============================================================
# Agent Interface Tests
# ==============================================================


def test_ppo_has_select_action_method(
    mock_env,
):
    """
    Verify the expected agent interface.
    """

    agent = PPOAgent(env=mock_env)

    assert callable(
        getattr(
            agent,
            "select_action",
            None,
        )
    )


def test_llm_ppo_has_select_action_method(
    mock_env,
):
    """
    Verify that LLM + PPO exposes the same basic
    action-selection interface.
    """

    agent = LLMRLAgent(env=mock_env)

    assert callable(
        getattr(
            agent,
            "select_action",
            None,
        )
    )


# ==============================================================
# Smoke Test
# ==============================================================


def test_ppo_smoke_run(
    mock_env,
):
    """
    Small end-to-end smoke test.

    The purpose is not to train PPO. It simply verifies that:

        environment
             ↓
        state
             ↓
        PPO
             ↓
        action
             ↓
        environment

    can execute without an immediate interface error.
    """

    agent = PPOAgent(env=mock_env)

    state = mock_env.reset()

    for _ in range(3):

        action = agent.select_action(state)

        assert action is not None

        (
            state,
            reward,
            terminated,
            truncated,
            info,
        ) = mock_env.step(action)

        assert isinstance(
            reward,
            (int, float),
        )

        if terminated or truncated:

            break

    mock_env.close()
