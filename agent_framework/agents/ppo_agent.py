"""
ppo_agent.py

Proximal Policy Optimization (PPO) agent.

This is the baseline reinforcement learning agent used
for comparison with the LLM-enhanced PPO agent.

The FinanceEnvironment is responsible for:
    - state transitions
    - high-level action rewards
    - backend API interaction
    - episode recording
    - environment-error detection

PPO is responsible only for learning a policy from
valid environment transitions.
"""

from pathlib import Path

import numpy as np
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
        observation_size: int,
        action_size: int,
    ):

        super().__init__(
            observation_size=observation_size,
            action_size=action_size,
        )

        # ==========================================================
        # PPO Hyperparameters
        # ==========================================================

        self.gamma = float(config.training.GAMMA)

        self.gae_lambda = float(config.training.GAE_LAMBDA)

        self.clip_epsilon = float(config.training.CLIP_EPSILON)

        self.epochs = int(config.training.EPOCHS)

        self.batch_size = int(config.training.BATCH_SIZE)

        self.update_interval = int(config.training.UPDATE_INTERVAL)

        self.learning_rate = float(config.training.LEARNING_RATE)

        self.hidden_size = int(config.training.HIDDEN_NEURON_SIZE)

        # ==========================================================
        # Networks
        # ==========================================================

        self.policy = PolicyNetwork(
            observation_size,
            action_size,
            self.hidden_size,
        ).to(self.device)

        self.value = ValueNetwork(
            observation_size,
            self.hidden_size,
        ).to(self.device)

        # ==========================================================
        # Optimizers
        # ==========================================================

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
        )

        self.value_optimizer = optim.Adam(
            self.value.parameters(),
            lr=self.learning_rate,
        )

        # ==========================================================
        # Rollout Buffer
        # ==========================================================

        self.buffer = RolloutBuffer(
            observation_size=self.observation_size,
        )

        # ==========================================================
        # Training Statistics
        # ==========================================================

        self.update_count = 0

        self.last_policy_loss = None

        self.last_value_loss = None

        self.last_entropy = None

    # ==========================================================
    # Action Selection
    # ==========================================================

    def select_action(
        self,
        state,
    ):
        """
        Select an action.

        During training:
            sample from the policy distribution.

        During evaluation:
            choose the action with the highest probability.

        Returns
        -------
        action : int
        log_prob : float
        value : float
        """

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        # Ensure batch dimension.
        if state_tensor.dim() == 1:

            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():

            distribution = self.policy.get_distribution(state_tensor)

            # ------------------------------------------------------
            # Training = stochastic exploration
            # Evaluation = deterministic action
            # ------------------------------------------------------

            if self.training:

                action = distribution.sample()

            else:

                action = torch.argmax(
                    distribution.probs,
                    dim=-1,
                )

            log_prob = distribution.log_prob(action)

            value = self.value(state_tensor)

            value = value.squeeze(-1)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
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
        environment_error=False,
    ):
        """
        Store one valid PPO transition.

        Environment/backend failures must not be learned from.

        Parameters
        ----------
        environment_error : bool
            True when the environment/backend failed.
            Such transitions are deliberately ignored.

        Returns
        -------
        bool
            True if stored.
            False if ignored.
        """

        # ==========================================================
        # Infrastructure Failure
        # ==========================================================

        if environment_error:

            self.buffer.mark_last_done()

            return False

        if reward is None:

            return False

        # ==========================================================
        # Store Valid Transition
        # ==========================================================

        self.buffer.add(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
            log_prob=float(log_prob),
            value=float(value),
        )

        self.increment_step()

        return True

    # ==========================================================
    # Rollout Status
    # ==========================================================

    def ready_to_learn(
        self,
    ):
        """
        Return True when enough rollout transitions have
        accumulated for a PPO update.
        """

        return len(self.buffer) >= self.update_interval

    # ==========================================================
    # State Value Helper
    # ==========================================================

    def _estimate_value(
        self,
        state,
    ):
        """
        Estimate V(state) without gradients.
        """

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        )

        if state_tensor.dim() == 1:

            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():

            value = self.value(state_tensor)

            value = value.squeeze()

        return float(value.item())

    # ==========================================================
    # GAE Calculation
    # ==========================================================

    def compute_advantages(
        self,
    ):
        """
        Compute Generalized Advantage Estimation.

        Important:
        If the rollout ends because UPDATE_INTERVAL was reached
        rather than because the episode terminated, the value of
        the final next_state is used for bootstrapping.
        """

        if len(self.buffer) == 0:

            return

        rewards = [float(t.reward) for t in self.buffer.buffer]

        values = [float(t.value) for t in self.buffer.buffer]

        dones = [bool(t.done) for t in self.buffer.buffer]

        # ------------------------------------------------------
        # Bootstrap final transition if rollout did not terminate.
        # ------------------------------------------------------

        last_transition = self.buffer.buffer[-1]

        if last_transition.done:

            next_value = 0.0

        else:

            next_value = self._estimate_value(last_transition.next_state)

        advantages = [0.0] * len(rewards)

        returns = [0.0] * len(rewards)

        gae = 0.0

        # ------------------------------------------------------
        # Reverse-time GAE
        # ------------------------------------------------------

        for step in reversed(range(len(rewards))):

            not_done = 1.0 - float(dones[step])

            delta = rewards[step] + self.gamma * next_value * not_done - values[step]

            gae = delta + self.gamma * self.gae_lambda * not_done * gae

            advantages[step] = gae

            returns[step] = gae + values[step]

            # --------------------------------------------------
            # For the previous transition, current V(s)
            # becomes V(s_{t+1}).
            # --------------------------------------------------

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
        force=False,
    ):
        """
        Perform one PPO update.

        Normally PPO updates when UPDATE_INTERVAL transitions
        have accumulated.

        force=True can be used at the end of training to learn
        from the final incomplete rollout.

        Returns
        -------
        dict | None
            Training statistics, or None if no update occurred.
        """

        if len(self.buffer) == 0:

            return None

        if not force and not self.ready_to_learn():

            return None

        # ------------------------------------------------------
        # Calculate advantages / returns
        # ------------------------------------------------------

        self.compute_advantages()

        batch = self.buffer.get_tensors(self.device)

        states = batch["states"]

        actions = batch["actions"].long()

        old_log_probs = batch["old_log_probs"]

        advantages = batch["advantages"]

        returns = batch["returns"]

        # Flatten fields that should be vectors.
        old_log_probs = old_log_probs.view(-1)

        advantages = advantages.view(-1)

        returns = returns.view(-1)

        # ------------------------------------------------------
        # Advantage normalization
        #
        # unbiased=False avoids NaN when the rollout contains
        # only one transition.
        # ------------------------------------------------------

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        rollout_size = states.size(0)

        batch_size = min(
            self.batch_size,
            rollout_size,
        )

        policy_losses = []

        value_losses = []

        entropies = []

        # ======================================================
        # PPO Epochs
        # ======================================================

        for _ in range(self.epochs):

            # Randomize transition order every epoch.
            indices = torch.randperm(
                rollout_size,
                device=self.device,
            )

            # ==================================================
            # Mini-batches
            # ==================================================

            for start in range(
                0,
                rollout_size,
                batch_size,
            ):

                end = start + batch_size

                batch_indices = indices[start:end]

                batch_states = states[batch_indices]

                batch_actions = actions[batch_indices]

                batch_old_log_probs = old_log_probs[batch_indices]

                batch_advantages = advantages[batch_indices]

                batch_returns = returns[batch_indices]

                # ==============================================
                # Policy Distribution
                # ==============================================

                distribution = self.policy.get_distribution(batch_states)

                new_log_probs = distribution.log_prob(batch_actions)

                entropy = distribution.entropy().mean()

                # ==============================================
                # PPO Probability Ratio
                # ==============================================

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                )

                # ==============================================
                # PPO Clipped Objective
                # ==============================================

                surrogate_1 = ratio * batch_advantages

                surrogate_2 = clipped_ratio * batch_advantages

                policy_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # ==============================================
                # Critic Loss
                # ==============================================

                value_prediction = self.value(batch_states).squeeze(-1)

                value_loss = nn.functional.mse_loss(
                    value_prediction,
                    batch_returns,
                )

                # ==============================================
                # Actor Update
                # ==============================================

                self.policy_optimizer.zero_grad()

                policy_loss.backward()

                self.policy_optimizer.step()

                # ==============================================
                # Critic Update
                # ==============================================

                self.value_optimizer.zero_grad()

                value_loss.backward()

                self.value_optimizer.step()

                # ==============================================
                # Statistics
                # ==============================================

                policy_losses.append(policy_loss.item())

                value_losses.append(value_loss.item())

                entropies.append(entropy.item())

        # ======================================================
        # Update statistics
        # ======================================================

        self.update_count += 1

        self.last_policy_loss = float(np.mean(policy_losses))

        self.last_value_loss = float(np.mean(value_losses))

        self.last_entropy = float(np.mean(entropies))

        metrics = {
            "update": self.update_count,
            "rollout_size": rollout_size,
            "policy_loss": self.last_policy_loss,
            "value_loss": self.last_value_loss,
            "entropy": self.last_entropy,
        }

        # ------------------------------------------------------
        # PPO is on-policy.
        #
        # Once these transitions have been used for an update,
        # they must not be reused.
        # ------------------------------------------------------

        self.buffer.clear()

        return metrics

    # ==========================================================
    # Training Mode
    # ==========================================================

    def train(
        self,
    ):
        """
        Enable PPO training mode.
        """

        super().train()

        self.policy.train()

        self.value.train()

        return self

    # ==========================================================
    # Evaluation Mode
    # ==========================================================

    def eval(
        self,
    ):
        """
        Enable deterministic PPO evaluation mode.
        """

        super().eval()

        self.policy.eval()

        self.value.eval()

        return self

    # ==========================================================
    # Save Model
    # ==========================================================

    def save(
        self,
        path,
    ):
        """
        Save complete PPO checkpoint.
        """

        path = self.ensure_parent_directory(path)

        torch.save(
            {
                # ----------------------------------------------
                # Networks
                # ----------------------------------------------
                "policy": self.policy.state_dict(),
                "value": self.value.state_dict(),
                # ----------------------------------------------
                # Optimizers
                # ----------------------------------------------
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "value_optimizer": self.value_optimizer.state_dict(),
                # ----------------------------------------------
                # Agent progress
                # ----------------------------------------------
                "episode": self.episode,
                "total_steps": self.total_steps,
                "update_count": self.update_count,
                # ----------------------------------------------
                # Architecture
                # ----------------------------------------------
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                # ----------------------------------------------
                # PPO settings
                # ----------------------------------------------
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "learning_rate": self.learning_rate,
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
        """
        Load PPO checkpoint.
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(f"PPO checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        # ------------------------------------------------------
        # Optional architecture validation
        # ------------------------------------------------------

        saved_observation_size = checkpoint.get("observation_size")

        saved_action_size = checkpoint.get("action_size")

        if (
            saved_observation_size is not None
            and saved_observation_size != self.observation_size
        ):

            raise ValueError(
                "Checkpoint observation size "
                f"{saved_observation_size} does not match "
                f"current observation size "
                f"{self.observation_size}."
            )

        if saved_action_size is not None and saved_action_size != self.action_size:

            raise ValueError(
                "Checkpoint action size "
                f"{saved_action_size} does not match "
                f"current action size "
                f"{self.action_size}."
            )

        # ------------------------------------------------------
        # Networks
        # ------------------------------------------------------

        self.policy.load_state_dict(checkpoint["policy"])

        self.value.load_state_dict(checkpoint["value"])

        # ------------------------------------------------------
        # Optimizers
        # ------------------------------------------------------

        if "policy_optimizer" in checkpoint:

            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

        if "value_optimizer" in checkpoint:

            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])

        # ------------------------------------------------------
        # Training progress
        # ------------------------------------------------------

        self.episode = int(
            checkpoint.get(
                "episode",
                0,
            )
        )

        self.total_steps = int(
            checkpoint.get(
                "total_steps",
                0,
            )
        )

        self.update_count = int(
            checkpoint.get(
                "update_count",
                0,
            )
        )

        return self

    # ==========================================================
    # Information
    # ==========================================================

    def get_info(
        self,
    ):

        info = super().get_info()

        info.update(
            {
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "update_interval": self.update_interval,
                "buffer_size": len(self.buffer),
                "update_count": self.update_count,
                "last_policy_loss": self.last_policy_loss,
                "last_value_loss": self.last_value_loss,
                "last_entropy": self.last_entropy,
            }
        )

        return info
