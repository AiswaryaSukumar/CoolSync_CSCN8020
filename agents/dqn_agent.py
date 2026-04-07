# agents/dqn_agent.py

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_network import DQNNetwork
from training.replay_buffer import ReplayBuffer
from training.state_preprocessor import StatePreprocessor


class DQNAgent:
    """
    Deep Q-Network agent for CoolSync.

    Features:
    - epsilon-greedy exploration
    - experience replay
    - target network
    - CUDA / GPU support when available
    """

    def __init__(
        self,
        preprocessor: StatePreprocessor,
        state_dim: int,
        num_actions: int = 3,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 10000,
        target_update_freq: int = 10,
    ) -> None:
        self.preprocessor = preprocessor
        self.state_dim = state_dim
        self.num_actions = num_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self._validate_hyperparameters()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online / policy network
        self.policy_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=num_actions,
        ).to(self.device)

        # Target network
        self.target_net = DQNNetwork(
            input_dim=state_dim,
            output_dim=num_actions,
        ).to(self.device)

        # Start target network equal to policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Count gradient updates performed
        self.training_steps = 0

    def _validate_hyperparameters(self) -> None:
        """
        Validate initialization parameters.
        """
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be > 0, got {self.state_dim}")

        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {self.num_actions}")

        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate must be in (0, 1], got {self.learning_rate}"
            )

        if not (0.0 <= self.discount_factor <= 1.0):
            raise ValueError(
                f"discount_factor must be in [0, 1], got {self.discount_factor}"
            )

        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon}")

        if not (0.0 <= self.epsilon_min <= 1.0):
            raise ValueError(
                f"epsilon_min must be in [0, 1], got {self.epsilon_min}"
            )

        if self.epsilon_min > self.epsilon:
            raise ValueError(
                f"epsilon_min ({self.epsilon_min}) cannot be greater than "
                f"epsilon_start ({self.epsilon})"
            )

        if not (0.0 < self.epsilon_decay <= 1.0):
            raise ValueError(
                f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")

        if self.target_update_freq <= 0:
            raise ValueError(
                f"target_update_freq must be > 0, got {self.target_update_freq}"
            )

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        """
        normalized_state = self.preprocessor.normalize_state(state)

        # Explore during training with probability epsilon
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_tensor = torch.tensor(
            normalized_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store normalized transition in replay buffer.
        """
        if not (0 <= action < self.num_actions):
            raise ValueError(
                f"action must be in [0, {self.num_actions - 1}], got {action}"
            )

        normalized_state = self.preprocessor.normalize_state(state)
        normalized_next_state = self.preprocessor.normalize_state(next_state)

        self.replay_buffer.push(
            normalized_state,
            action,
            reward,
            normalized_next_state,
            done,
        )

    def train_step(self) -> Dict[str, float]:
        """
        Perform one gradient update if enough replay data exists.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states_tensor = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Q(s,a) from current policy network
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        # Bellman target using target network
        with torch.no_grad():
            next_max_q_values = self.target_net(next_states_tensor).max(
                dim=1, keepdim=True
            )[0]
            target_q_values = rewards_tensor + (
                1.0 - dones_tensor
            ) * self.discount_factor * next_max_q_values

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)

        self.optimizer.step()

        self.training_steps += 1

        return {"loss": float(loss.item())}

    def update_target_network(self) -> None:
        """
        Copy policy network weights into target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def maybe_update_target_network(self, episode: int) -> None:
        """
        Update target network every fixed number of episodes.
        """
        if episode % self.target_update_freq == 0:
            self.update_target_network()

    def decay_epsilon(self) -> None:
        """
        Decay exploration rate after each episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self) -> None:
        """
        Disable exploration for evaluation.
        """
        self.epsilon = 0.0
        self.policy_net.eval()
        self.target_net.eval()

    def set_training_mode(self, epsilon: float | None = None) -> None:
        """
        Re-enable training mode.

        Args:
            epsilon: Optional exploration rate override.
        """
        self.policy_net.train()
        self.target_net.eval()

        if epsilon is not None:
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
            self.epsilon = epsilon
        else:
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save DQN checkpoint.
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        checkpoint = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "training_steps": self.training_steps,
        }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load DQN checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"DQN checkpoint not found: {filepath}")

        checkpoint = torch.load(
            filepath,
            map_location=self.device,
            weights_only=False,
        )

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.state_dim = int(checkpoint["state_dim"])
        self.num_actions = int(checkpoint["num_actions"])
        self.learning_rate = float(checkpoint.get("learning_rate", self.learning_rate))
        self.discount_factor = float(checkpoint["discount_factor"])
        self.epsilon = float(checkpoint["epsilon"])
        self.epsilon_min = float(checkpoint["epsilon_min"])
        self.epsilon_decay = float(checkpoint["epsilon_decay"])
        self.batch_size = int(checkpoint["batch_size"])
        self.target_update_freq = int(checkpoint["target_update_freq"])
        self.training_steps = int(checkpoint.get("training_steps", 0))

        self._validate_hyperparameters()

    def summary(self) -> Dict[str, Any]:
        """
        Return a lightweight summary of the current agent state.
        """
        return {
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "training_steps": self.training_steps,
            "replay_buffer_size": len(self.replay_buffer),
            "device": str(self.device),
        }