"""
ppo_agent.py

Proximal Policy Optimization (PPO) agent.

This is the baseline RL agent used for comparison
against the LLM-enhanced PPO agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from memory.rollout_buffer import RolloutBuffer
from config.config import config


class PPOAgent(BaseAgent):

    def __init__(
        self,
        observation_size,
        action_size,
    ):

        super().__init__(
            observation_size,
            action_size,
        )

        # Hyperparameters

        self.gamma = config.training.GAMMA

        self.gae_lambda = config.training.GAE_LAMBDA

        self.clip_epsilon = config.training.CLIP_EPSILON

        self.epochs = config.training.EPOSHS

        # Networks

        self.policy = PolicyNetwork(
            observation_size,
            action_size,
            config.training.HIDDEN_NEURON_SIZE,
        ).to(self.device)

        self.value = ValueNetwork(
            observation_size,
            config.training.HIDDEN_NEURON_SIZE,
        ).to(self.device)

        # Optimizers

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.training.LEARNING_RATE,
        )

        self.value_optimizer = optim.Adam(
            self.value.parameters(),
            lr=config.training.LEARNING_RATE,
        )

        # Rollout storage

        self.buffer = RolloutBuffer()

    # ==========================================================
    # Action Selection
    # ==========================================================

    def select_action(
        self,
        state,
    ):

        state = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        state = state.unsqueeze(0)

        with torch.no_grad():

            distribution = self.policy.get_distribution(state)

            action = distribution.sample()

            log_prob = distribution.log_prob(action)

            value = self.value(state)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    # ==========================================================
    # Store Experience
    # ==========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        log_prob,
        value,
    ):

        self.buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            log_prob,
            value,
        )

    # ==========================================================
    # GAE Calculation
    # ==========================================================

    def compute_advantages(self):

        rewards = [t.reward for t in self.buffer.buffer]

        values = [t.value for t in self.buffer.buffer]

        dones = [t.done for t in self.buffer.buffer]

        advantages = []

        returns = []

        gae = 0

        next_value = 0

        for step in reversed(range(len(rewards))):

            delta = (
                rewards[step]
                + self.gamma * next_value * (1 - dones[step])
                - values[step]
            )

            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae

            advantages.insert(0, gae)

            returns.insert(0, gae + values[step])

            next_value = values[step]

        self.buffer.add_advantages(
            advantages,
            returns,
        )

    # ==========================================================
    # PPO Learning Update
    # ==========================================================

    def learn(
        self,
        *args,
        **kwargs,
    ):

        if len(self.buffer) == 0:

            return

        self.compute_advantages()

        batch = self.buffer.get_tensors(self.device)

        states = batch["states"]

        actions = batch["actions"]

        old_log_probs = batch["old_log_probs"]

        advantages = batch["advantages"]

        returns = batch["returns"]

        # Normalize advantages

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):

            distribution = self.policy.get_distribution(states)

            new_log_probs = distribution.log_prob(actions)

            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)

            clipped_ratio = torch.clamp(
                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
            )

            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()

            value_prediction = self.value(states)

            value_loss = nn.MSELoss()(value_prediction, returns)

            # Actor update

            self.policy_optimizer.zero_grad()

            policy_loss.backward()

            self.policy_optimizer.step()

            # Critic update

            self.value_optimizer.zero_grad()

            value_loss.backward()

            self.value_optimizer.step()

        self.buffer.clear()

    # ==========================================================
    # Save Model
    # ==========================================================

    def save(
        self,
        path,
    ):

        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
            },
            path,
        )

    # ==========================================================
    # Load Model
    # ==========================================================

    def load(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.policy.load_state_dict(checkpoint["policy"])

        self.value.load_state_dict(checkpoint["value"])
